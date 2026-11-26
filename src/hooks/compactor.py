from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.context import Message, RunState, RunStatus, ToolCall, ToolResult
from src.core.policy import LimitsPolicy, PoliciesConfig, load_policies
from src.hooks.base import PostToolDecision, PostToolUseHook

FOLDED_MARKER = "[Compacted "
_ERROR_LINE_RE = re.compile(
    r"(?:\b(?:ERROR|FATAL|CRITICAL|FAIL(?:ED)?|Traceback|Exception|OutOfMemoryError|AssertionError)\b|\bCaused by:)",
    re.IGNORECASE,
)


def _sanitize_segment(raw: str, fallback: str) -> str:
    """Sanitize a path segment against traversal, separators, and dot-only names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).replace("..", "_").strip("._")
    return cleaned or fallback


def _parse_started_at(raw_value: Any) -> datetime | None:
    """Parse an ISO timestamp, epoch seconds, or datetime into UTC datetime."""
    if raw_value is None:
        return None
    if isinstance(raw_value, datetime):
        return raw_value.replace(tzinfo=UTC) if raw_value.tzinfo is None else raw_value.astimezone(UTC)
    if isinstance(raw_value, (int, float)):
        return datetime.fromtimestamp(float(raw_value), tz=UTC)
    if isinstance(raw_value, str) and raw_value.strip():
        parsed = datetime.fromisoformat(raw_value.strip())
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    return None


def _truncate_lines_and_chars(
    lines: list[str],
    raw_text: str,
    *,
    from_tail: bool,
    max_lines: int,
    max_chars: int,
) -> str:
    """Extract a bounded head or tail excerpt respecting both line and char caps."""
    if not raw_text:
        return ""
    if len(lines) <= 1:
        return raw_text[-max_chars:] if from_tail else raw_text[:max_chars]
    selected = lines[-max_lines:] if from_tail else lines[:max_lines]
    joined = "\n".join(selected)
    if len(joined) <= max_chars:
        return joined
    return joined[-max_chars:] if from_tail else joined[:max_chars]


def _extract_middle_error_lines(
    lines: list[str],
    start: int,
    end: int,
    *,
    max_lines: int = 10,
    max_chars: int = 800,
) -> str:
    """Extract error/exception lines from the omitted middle region with line numbers."""
    if start >= end or max_lines <= 0 or max_chars <= 0:
        return ""
    matched: list[str] = []
    used_chars = 0
    for idx in range(start, end):
        stripped = lines[idx].strip()
        if not stripped or not _ERROR_LINE_RE.search(stripped):
            continue
        entry = f"L{idx + 1}: {stripped}"
        if len(entry) > 160:
            entry = f"{entry[:157]}..."
        sep = 1 if matched else 0
        if used_chars + sep + len(entry) > max_chars:
            if not matched:
                matched.append(entry[:max_chars])
            break
        matched.append(entry)
        used_chars += sep + len(entry)
        if len(matched) >= max_lines:
            break
    return "\n".join(matched)


def _fold_tool_message(msg: Message) -> str:
    """Produce a concise deterministic one-line summary for an older tool message."""
    tool_label = msg.tool_name or "tool"
    content = msg.content
    if content.startswith("[Output spilled:"):
        first_line = content.splitlines()[0].strip()
        return f"{FOLDED_MARKER}{tool_label}] {first_line}"

    lines = content.splitlines()
    first_non_empty = next((ln.strip() for ln in lines if ln.strip()), "(empty)")
    preview = " ".join(first_non_empty.split())
    if len(preview) > 80:
        preview = f"{preview[:77]}..."
    return f"{FOLDED_MARKER}{tool_label} | {len(lines)}L, {len(content)}C] {preview}"


class ContextCompactorHook(PostToolUseHook):
    """Spills huge tool outputs, folds old tool history prefix-safely, and enforces budgets."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        limits: LimitsPolicy | None = None,
        policies: PoliciesConfig | None = None,
        recent_turns_to_keep: int = 2,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        if limits is not None:
            self.limits = limits
        elif policies is not None:
            self.limits = policies.limits
        else:
            self.limits = load_policies().limits

        self.workspace_root = (workspace_root or Path("workspace")).resolve()
        self.recent_turns_to_keep = max(recent_turns_to_keep, 1)
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def _current_utc(self) -> datetime:
        now = self._now_fn()
        return now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)

    def _ensure_started_at(self, state: RunState, now: datetime) -> datetime:
        raw = getattr(state, "started_at", None)
        if raw is None:
            raw = state.metadata.get("started_at")
        parsed = _parse_started_at(raw)
        if parsed is not None:
            return parsed
        state.metadata["started_at"] = now.isoformat()
        return now

    def _spill_large_output(
        self, state: RunState, tool_call: ToolCall, result: ToolResult
    ) -> str | None:
        lines = result.output.splitlines()
        line_count = len(lines)
        char_count = len(result.output)
        if (
            line_count <= self.limits.compactor_max_lines
            and char_count <= self.limits.compactor_max_chars
        ):
            return None

        scratch_root = (self.workspace_root / ".scratch").resolve()
        safe_run_id = _sanitize_segment(state.run_id, "run")
        safe_call_id = _sanitize_segment(tool_call.id, "call")
        spill_path = (scratch_root / safe_run_id / f"{safe_call_id}.txt").resolve()
        if not spill_path.is_relative_to(scratch_root):
            raise ValueError(f"Spill path escaped .scratch/: {spill_path}")

        spill_path.parent.mkdir(parents=True, exist_ok=True)
        spill_path.write_text(result.output, encoding="utf-8")
        rel_path = spill_path.relative_to(self.workspace_root).as_posix()

        compaction_meta = state.metadata.setdefault("compaction", {})
        compaction_meta["spilled_outputs"] = int(compaction_meta.get("spilled_outputs", 0)) + 1
        spilled_files = state.metadata.setdefault("spilled_files", [])
        if rel_path not in spilled_files:
            spilled_files.append(rel_path)

        return self._format_spill_summary(
            lines=lines,
            raw_output=result.output,
            line_count=line_count,
            char_count=char_count,
            rel_path=rel_path,
        )

    def _format_spill_summary(
        self,
        *,
        lines: list[str],
        raw_output: str,
        line_count: int,
        char_count: int,
        rel_path: str,
    ) -> str:
        body_budget = max(self.limits.compactor_max_chars - 500, 200)
        line_cap = min(15, max(self.limits.compactor_max_lines // 4, 3))
        hint_end = min(100, max(self.limits.compactor_max_lines, 1))

        error_excerpt = ""
        if line_count > 2 * line_cap:
            error_budget = min(800, max(body_budget // 4, 150))
            error_excerpt = _extract_middle_error_lines(
                lines,
                line_cap,
                max(line_count - line_cap, line_cap),
                max_lines=10,
                max_chars=error_budget,
            )

        text_budget = (
            max(body_budget - len(error_excerpt) - 40, 120)
            if error_excerpt
            else body_budget
        )
        head_budget = text_budget // 2
        tail_budget = text_budget - head_budget

        head = _truncate_lines_and_chars(
            lines, raw_output, from_tail=False, max_lines=line_cap, max_chars=head_budget
        )
        tail = _truncate_lines_and_chars(
            lines, raw_output, from_tail=True, max_lines=line_cap, max_chars=tail_budget
        )
        omitted_lines = max(line_count - (2 * line_cap), 0)
        omitted_desc = (
            f"{omitted_lines} middle lines omitted"
            if line_count > 2 * line_cap
            else f"{max(char_count - len(head) - len(tail), 0)} middle chars omitted"
        )
        error_block = (
            f"--- KEY ERROR LINES ---\n{error_excerpt}\n" if error_excerpt else ""
        )
        return (
            f"[Output spilled: {line_count} lines, {char_count} chars -> {rel_path}]\n"
            f'Hint: inspect with read_file(path="{rel_path}", start_line=1, end_line={hint_end}) '
            f'or grep_search(pattern="...", path="{rel_path}").\n'
            f"--- HEAD ---\n{head}\n"
            f"{error_block}"
            f"--- ... ({omitted_desc}) ... ---\n"
            f"--- TAIL ---\n{tail}"
        )

    def _should_fold(self, state: RunState) -> bool:
        turn_ratio = state.turn / max(self.limits.max_turns, 1)
        cost_ratio = state.usage.cost_usd / max(self.limits.max_cost_usd, 1e-9)
        budget_ratio = max(turn_ratio, cost_ratio)
        return (
            state.turn > self.limits.compactor_fold_after_turns
            or budget_ratio >= self.limits.compactor_fold_budget_ratio
        )

    def _fold_window_bounds(self, messages: list[Message]) -> tuple[int, int]:
        """Return (prefix_end, recent_start) defining the foldable middle span."""
        prefix_end = 0
        for msg in messages:
            if msg.role in ("system", "user"):
                prefix_end += 1
            else:
                break

        assistant_indices = [
            idx for idx, msg in enumerate(messages) if msg.role == "assistant"
        ]
        if assistant_indices:
            last_asst = assistant_indices[-1]
            expected_tools = len(messages[last_asst].tool_calls)
            actual_tools = sum(1 for m in messages[last_asst + 1 :] if m.role == "tool")
            current_turn_complete = (
                actual_tools >= expected_tools if expected_tools > 0 else actual_tools > 0
            )
            keep_count = (
                self.recent_turns_to_keep
                if current_turn_complete
                else self.recent_turns_to_keep + 1
            )
            if len(assistant_indices) <= keep_count:
                return prefix_end, 0
            return prefix_end, assistant_indices[-keep_count]

        tool_indices = [idx for idx, msg in enumerate(messages) if msg.role == "tool"]
        if len(tool_indices) <= self.recent_turns_to_keep:
            return prefix_end, 0
        return prefix_end, tool_indices[-self.recent_turns_to_keep]

    def _fold_older_tool_messages(self, state: RunState) -> None:
        if not self._should_fold(state):
            return
        prefix_end, recent_start = self._fold_window_bounds(state.messages)
        if recent_start <= prefix_end:
            return

        newly_folded = 0
        chars_saved = 0
        for idx in range(prefix_end, recent_start):
            msg = state.messages[idx]
            if msg.role != "tool" or msg.content.startswith(FOLDED_MARKER):
                continue
            orig_len = len(msg.content)
            msg.content = _fold_tool_message(msg)
            newly_folded += 1
            chars_saved += max(orig_len - len(msg.content), 0)

        if newly_folded == 0:
            return

        meta = state.metadata.setdefault("compaction", {})
        meta["fold_runs"] = int(meta.get("fold_runs", 0)) + 1
        meta["compacted_messages"] = int(meta.get("compacted_messages", 0)) + newly_folded
        meta["chars_saved"] = int(meta.get("chars_saved", 0)) + chars_saved
        meta["last_compacted_turn"] = state.turn

    def _effective_max_turns(self, state: RunState) -> int:
        if "effective_max_turns" in state.metadata:
            return int(state.metadata["effective_max_turns"])
        capped_rejections = min(
            max(state.rejection_count, 0),
            max(self.limits.max_rejections, 0),
        )
        extension = capped_rejections * self.limits.rejection_turn_extension
        extra_turns = int(state.metadata.get("extra_turns", 0) or 0)
        policy_limit = self.limits.max_turns + extension + extra_turns
        state_limit = state.max_turns + (
            extension if state.max_turns <= self.limits.max_turns else 0
        )
        return min(policy_limit, state_limit)

    def _check_hard_ceilings(self, state: RunState, started_at: datetime, now: datetime) -> str:
        effective_max_turns = self._effective_max_turns(state)
        if state.turn >= effective_max_turns:
            return (
                f"Turn ceiling reached ({state.turn}/{effective_max_turns} "
                f"with rejection_count={state.rejection_count})."
            )

        elapsed_seconds = max((now - started_at).total_seconds(), 0.0)
        if elapsed_seconds >= self.limits.max_wall_clock_seconds:
            return (
                f"Wall-clock budget exceeded ({elapsed_seconds:.1f}s >= "
                f"{self.limits.max_wall_clock_seconds}s)."
            )

        if state.usage.cost_usd >= self.limits.max_cost_usd:
            return (
                f"Cost ceiling reached (${state.usage.cost_usd:.4f} >= "
                f"${self.limits.max_cost_usd:.4f})."
            )
        return ""

    def on_post_tool_use(
        self, state: RunState, tool_call: ToolCall, result: ToolResult
    ) -> PostToolDecision:
        """Spill large outputs, compact older tool messages, and enforce hard ceilings."""
        now = self._current_utc()
        started_at = self._ensure_started_at(state, now)

        modified_output = self._spill_large_output(state, tool_call, result)
        self._fold_older_tool_messages(state)
        abort_reason = self._check_hard_ceilings(state, started_at, now)

        abort_status: RunStatus | None = "ABORTED_BUDGET" if abort_reason else None
        if modified_output is not None:
            return PostToolDecision(
                action="modify",
                modified_output=modified_output,
                abort_status=abort_status,
                abort_reason=abort_reason,
            )
        return PostToolDecision(
            action="allow",
            abort_status=abort_status,
            abort_reason=abort_reason,
        )
