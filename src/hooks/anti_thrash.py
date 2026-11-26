from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from src.core.context import Message, RunState, ToolCall, ToolResult
from src.core.policy import AntiThrashPolicy, PoliciesConfig, load_policies
from src.hooks.base import PostToolDecision, PostToolUseHook

ANTI_THRASH_METADATA_KEY = "anti_thrash"


def _normalize_path(raw: str) -> str:
    cleaned = raw.replace("\\", "/").strip()
    if cleaned.startswith(("a/", "b/")):
        cleaned = cleaned[2:]
    return cleaned.lstrip("./")


def _normalize_tool_signature(tool_call: ToolCall) -> str:
    """Return a deterministic canonical JSON signature for (tool_name, arguments)."""
    serialized_args = json.dumps(
        tool_call.arguments,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return f"{tool_call.name}:{serialized_args}"


def _extract_patch_targets(tool_call: ToolCall, result: ToolResult) -> list[str]:
    """Resolve target workspace file paths touched by an apply_patch call."""
    if tool_call.name != "apply_patch":
        return []

    patched = result.metadata.get("patched_files")
    if isinstance(patched, list) and patched:
        targets: list[str] = []
        for entry in patched:
            norm = _normalize_path(str(entry))
            if norm and norm not in targets:
                targets.append(norm)
        return targets

    patch_text = str(tool_call.arguments.get("patch") or "")
    targets = []
    for line in patch_text.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw_target = line[4:].split("\t", 1)[0].strip()
        if raw_target == "/dev/null":
            continue
        norm = _normalize_path(raw_target)
        if norm and norm not in targets:
            targets.append(norm)
    if targets:
        return targets

    fallback = tool_call.arguments.get("path") or tool_call.arguments.get("file_path")
    if isinstance(fallback, str) and fallback.strip():
        return [_normalize_path(fallback)]
    return []


class AntiThrashState(BaseModel):
    """Checkpoint-serializable thrash detection history stored in RunState.metadata."""

    step_index: int = 0
    strikes: int = 0
    call_counts: dict[str, int] = Field(default_factory=dict)
    read_sig_targets: dict[str, str] = Field(default_factory=dict)
    consecutive_errors: int = 0
    last_error_output: str = ""
    identical_error_streak: int = 0
    recent_file_edits: list[dict[str, Any]] = Field(default_factory=list)
    strike_reasons: list[str] = Field(default_factory=list)


class AntiThrashHook(PostToolUseHook):
    """Detects repeat calls, error stagnation, and patch oscillation with a 2-strike breaker."""

    def __init__(
        self,
        policy: AntiThrashPolicy | None = None,
        policies: PoliciesConfig | None = None,
        recent_edit_window: int = 15,
    ) -> None:
        if policy is not None:
            self.policy = policy
        elif policies is not None:
            self.policy = policies.anti_thrash
        else:
            self.policy = load_policies().anti_thrash
        self.recent_edit_window = max(recent_edit_window, 1)

    def _load_tracker(self, state: RunState) -> AntiThrashState:
        raw = state.metadata.get(ANTI_THRASH_METADATA_KEY)
        if isinstance(raw, dict):
            return AntiThrashState.model_validate(raw)
        return AntiThrashState()

    def _save_tracker(self, state: RunState, tracker: AntiThrashState) -> None:
        state.metadata[ANTI_THRASH_METADATA_KEY] = tracker.model_dump(mode="json")

    def _check_repeated_call(
        self, tracker: AntiThrashState, tool_call: ToolCall
    ) -> str | None:
        sig = _normalize_tool_signature(tool_call)
        count = tracker.call_counts.get(sig, 0) + 1
        tracker.call_counts[sig] = count

        if tool_call.name == "read_file":
            raw_path = str(tool_call.arguments.get("path") or "")
            if raw_path:
                tracker.read_sig_targets[sig] = _normalize_path(raw_path)

        if count >= self.policy.repeat_call_threshold:
            return (
                f"Repeated identical tool call {tool_call.name!r} "
                f"({count}x >= threshold {self.policy.repeat_call_threshold})"
            )
        return None

    def _check_error_stagnation(
        self, tracker: AntiThrashState, result: ToolResult
    ) -> str | None:
        if not result.is_error:
            tracker.consecutive_errors = 0
            tracker.last_error_output = ""
            tracker.identical_error_streak = 0
            return None

        tracker.consecutive_errors += 1
        normalized_err = result.output.strip()
        if normalized_err and normalized_err == tracker.last_error_output:
            tracker.identical_error_streak += 1
        else:
            tracker.last_error_output = normalized_err
            tracker.identical_error_streak = 1

        if tracker.consecutive_errors >= self.policy.error_stagnation_threshold:
            return (
                f"Error stagnation: {tracker.consecutive_errors} consecutive tool errors "
                f"(>= threshold {self.policy.error_stagnation_threshold})"
            )
        if tracker.identical_error_streak >= self.policy.error_stagnation_threshold:
            return (
                f"Identical error repeated {tracker.identical_error_streak}x "
                f"(>= threshold {self.policy.error_stagnation_threshold})"
            )
        return None

    def _check_file_oscillation(
        self, tracker: AntiThrashState, tool_call: ToolCall, result: ToolResult
    ) -> str | None:
        if result.blocked:
            return None
        targets = _extract_patch_targets(tool_call, result)
        if not targets:
            return None

        if not result.is_error:
            self._invalidate_read_cache_for_targets(tracker, targets)

        cutoff = tracker.step_index - self.recent_edit_window
        tracker.recent_file_edits = [
            entry
            for entry in tracker.recent_file_edits
            if int(entry.get("step", 0)) > cutoff
        ]
        for path in targets:
            tracker.recent_file_edits.append({"path": path, "step": tracker.step_index})

        oscillating = []
        for path in targets:
            count = sum(1 for entry in tracker.recent_file_edits if entry["path"] == path)
            if count >= self.policy.file_oscillation_threshold:
                oscillating.append(f"{path} ({count}x)")

        if oscillating:
            return (
                "File edit oscillation in recent steps for "
                f"{', '.join(oscillating)} "
                f"(>= threshold {self.policy.file_oscillation_threshold})"
            )
        return None

    def _invalidate_read_cache_for_targets(
        self, tracker: AntiThrashState, targets: list[str]
    ) -> None:
        target_set = set(targets)
        stale_sigs = [
            sig
            for sig, read_path in tracker.read_sig_targets.items()
            if read_path in target_set
        ]
        for sig in stale_sigs:
            tracker.call_counts.pop(sig, None)
            tracker.read_sig_targets.pop(sig, None)

    def _build_nudge_message(self, strike: int, reasons: list[str]) -> Message:
        joined = "; ".join(reasons)
        return Message(
            role="user",
            content=(
                f"[ANTI-THRASH WARNING - Strike {strike}/{self.policy.strikes_before_abort}] "
                f"Detected thrashing behavior: {joined}. "
                "Stop repeating the same action or editing the same file blindly. "
                "Switch to a different diagnostic approach (for example, query_lineage, "
                "read_file on a different line range or related module, git_inspect, "
                "or run_sql_readonly with a revised query) before attempting another fix. "
                "A further thrash strike will abort this Run."
            ),
        )

    def on_post_tool_use(
        self, state: RunState, tool_call: ToolCall, result: ToolResult
    ) -> PostToolDecision:
        """Track thrashing signals and enforce the two-strike circuit breaker."""
        tracker = self._load_tracker(state)
        tracker.step_index += 1

        signals = [
            self._check_repeated_call(tracker, tool_call),
            self._check_error_stagnation(tracker, result),
            self._check_file_oscillation(tracker, tool_call, result),
        ]
        reasons = [msg for msg in signals if msg is not None]

        if not reasons:
            self._save_tracker(state, tracker)
            return PostToolDecision(action="allow")

        tracker.strikes += 1
        summary = "; ".join(reasons)
        tracker.strike_reasons.append(summary)
        self._save_tracker(state, tracker)

        if tracker.strikes >= self.policy.strikes_before_abort:
            return PostToolDecision(
                action="allow",
                abort_status="ABORTED_THRASH",
                abort_reason=(
                    f"Anti-thrash circuit breaker tripped "
                    f"(strike {tracker.strikes}/{self.policy.strikes_before_abort}): {summary}"
                ),
            )

        return PostToolDecision(
            action="allow",
            injected_messages=[self._build_nudge_message(tracker.strikes, reasons)],
        )
