from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.adapters.model_adapter import ModelAdapter, TurnRequest
from src.core.context import (
    CheckpointStore,
    IncidentInput,
    Message,
    RunState,
    ToolResult,
    UserContext,
)
from src.core.policy import PoliciesConfig, load_policies
from src.hooks.base import HookPipeline
from src.tools.registry import ToolRegistry

DEFAULT_STATIC_PROMPT = (
    "You are DataOps-Harness, an autonomous data-platform troubleshooting agent. "
    "Investigate pipeline Incidents across PySpark, Delta Lake, and dbt-duckdb using "
    "the six orthogonal tools, verify fixes via evidence, and output a structured RCA."
)


class HarnessLoop:
    """Minimal Observe -> Act -> Verify single-loop runtime with local checkpointing."""

    def __init__(
        self,
        model_adapter: ModelAdapter,
        tool_registry: ToolRegistry,
        hooks: HookPipeline | None = None,
        runs_dir: Path = Path("runs"),
        static_system_prompt: str = DEFAULT_STATIC_PROMPT,
        policies: PoliciesConfig | None = None,
    ) -> None:
        self.model_adapter = model_adapter
        self.tool_registry = tool_registry
        self.hooks = hooks or HookPipeline()
        self.runs_dir = runs_dir.resolve()
        self.static_system_prompt = static_system_prompt
        self.policies = policies or load_policies()

    @staticmethod
    def _reconcile_pending_tool_calls(state: RunState) -> None:
        """Synthesize missing tool results if a previous turn crashed mid-tool-batch."""
        last_asst_idx = -1
        for idx, msg in enumerate(state.messages):
            if msg.role == "assistant":
                last_asst_idx = idx
        if last_asst_idx < 0:
            return
        pending_calls = state.messages[last_asst_idx].tool_calls
        if not pending_calls:
            return

        insert_pos = last_asst_idx + 1
        seen_ids: set[str] = set()
        for msg in state.messages[last_asst_idx + 1 :]:
            if msg.role == "tool":
                insert_pos += 1
                if msg.tool_call_id:
                    seen_ids.add(msg.tool_call_id)
            else:
                break

        missing = [
            Message(
                role="tool",
                content="Tool execution interrupted before checkpoint completion.",
                tool_call_id=tc.id,
                tool_name=tc.name,
            )
            for tc in pending_calls
            if tc.id not in seen_ids
        ]
        if missing:
            state.messages[insert_pos:insert_pos] = missing

    def _format_initial_user_message(
        self, incident: IncidentInput, user_context: UserContext
    ) -> str:
        payload = {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "summary": incident.summary,
            "affected_assets": incident.affected_assets,
            "shared_ancestors": incident.shared_ancestors,
            "alerts": incident.alerts,
            "principal_role": user_context.role,
            "resolution_mode": user_context.resolution_mode,
        }
        return "Investigate and resolve the following Incident:\n" + json.dumps(
            payload, ensure_ascii=False, indent=2
        )

    def _enforce_role_mode_ceiling(self, user_context: UserContext) -> UserContext:
        role_policy = self.policies.roles.get(user_context.role)
        if role_policy is not None and role_policy.resolution_mode == "diagnosis_only":
            return user_context.model_copy(update={"resolution_mode": "diagnosis_only"})
        return user_context

    def run(
        self,
        incident: IncidentInput,
        user_context: UserContext,
        run_id: str | None = None,
        max_turns: int = 20,
    ) -> RunState:
        """Start a fresh Run for `incident`, checkpointing every turn."""
        resolved_run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        effective_ctx = self._enforce_role_mode_ceiling(user_context)
        state = RunState(
            run_id=resolved_run_id,
            incident=incident,
            user_context=effective_ctx,
            status="RUNNING",
            max_turns=max_turns,
            messages=[
                Message(
                    role="user",
                    content=self._format_initial_user_message(incident, effective_ctx),
                )
            ],
        )
        store = CheckpointStore(self.runs_dir, resolved_run_id)
        store.append("run_started", state)
        return self._execute_loop(state, store)

    def resume(
        self,
        run_id: str,
        extra_message: str | None = None,
        extra_turns: int = 0,
        *,
        is_rejection_extension: bool = False,
    ) -> RunState:
        """Reload the latest checkpointed state for `run_id` and continue execution."""
        store = CheckpointStore(self.runs_dir, run_id)
        state = store.load_latest()
        if state.status in {"RESOLVED", "DIAGNOSED", "ABORTED_THRASH"}:
            raise ValueError(
                f"Cannot resume Run {run_id!r} from terminal status {state.status!r}."
            )
        if state.status == "ABORTED_BUDGET" and extra_turns <= 0:
            raise ValueError(
                f"Cannot resume ABORTED_BUDGET Run {run_id!r} without extra_turns > 0."
            )

        for hook in self.hooks.post_tool_hooks:
            on_resumed = getattr(hook, "on_run_resumed", None)
            if callable(on_resumed):
                on_resumed(state)

        self._reconcile_pending_tool_calls(state)
        state.user_context = self._enforce_role_mode_ceiling(state.user_context)
        if extra_turns > 0:
            state.max_turns += extra_turns
            if not is_rejection_extension:
                state.metadata["extra_turns"] = (
                    int(state.metadata.get("extra_turns", 0) or 0) + extra_turns
                )
        if extra_message:
            state.messages.append(Message(role="user", content=extra_message))
        state.status = "RUNNING"
        store.append(
            "run_resumed",
            state,
            detail={"extra_turns": extra_turns, "has_extra_message": bool(extra_message)},
        )
        return self._execute_loop(state, store)

    def _execute_loop(self, state: RunState, store: CheckpointStore) -> RunState:
        tool_defs = self.tool_registry.definitions()

        while state.status == "RUNNING":
            if state.turn >= state.max_turns:
                state.status = "ABORTED_BUDGET"
                state.metadata["abort_reason"] = (
                    f"Reached max_turns ceiling ({state.max_turns})."
                )
                store.append("run_aborted_budget", state)
                break

            tail_note = (
                f"[Run State] turn={state.turn + 1}/{state.max_turns} "
                f"role={state.user_context.role} "
                f"mode={state.user_context.resolution_mode}"
            )
            request = TurnRequest(
                static_system_prompt=self.static_system_prompt,
                messages=state.messages,
                tools=tool_defs,
                dynamic_tail_note=tail_note,
            )
            response = self.model_adapter.complete(request)
            state.turn += 1
            state.usage.add(response.usage)
            state.messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            if response.tool_calls:
                turn_injected: list[Message] = []
                for idx, tc in enumerate(response.tool_calls):
                    pre_decision, effective_call = self.hooks.run_pre_tool(state, tc)
                    if pre_decision.action == "block":
                        raw_result = ToolResult(
                            tool_call_id=tc.id,
                            name=tc.name,
                            output=f"BLOCKED by hook: {pre_decision.reason}",
                            is_error=True,
                            blocked=True,
                        )
                    else:
                        raw_result = self.tool_registry.execute(effective_call)

                    for patched in raw_result.metadata.get("patched_files") or []:
                        if patched not in state.patched_files:
                            state.patched_files.append(patched)

                    final_result, injected, abort_status, abort_reason = (
                        self.hooks.run_post_tool(state, effective_call, raw_result)
                    )
                    state.messages.append(
                        Message(
                            role="tool",
                            content=final_result.output,
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                        )
                    )
                    turn_injected.extend(injected)
                    store.append(
                        "tool_step",
                        state,
                        detail={
                            "tool": tc.name,
                            "blocked": raw_result.blocked,
                            "is_error": final_result.is_error,
                        },
                    )
                    if abort_status is not None:
                        for remaining_tc in response.tool_calls[idx + 1 :]:
                            state.messages.append(
                                Message(
                                    role="tool",
                                    content="Tool skipped: Run aborted by hook.",
                                    tool_call_id=remaining_tc.id,
                                    tool_name=remaining_tc.name,
                                )
                            )
                        state.messages.extend(turn_injected)
                        state.status = abort_status
                        state.metadata["abort_reason"] = abort_reason
                        store.append("run_aborted", state, detail={"reason": abort_reason})
                        return state

                state.messages.extend(turn_injected)
                store.append("turn_completed", state, detail={"turn": state.turn})
                continue

            for hook in self.hooks.post_tool_hooks:
                turn_budget_fn = getattr(hook, "check_turn_budget", None)
                if callable(turn_budget_fn):
                    budget_reason = turn_budget_fn(state)
                    if budget_reason:
                        state.status = "ABORTED_BUDGET"
                        state.metadata["abort_reason"] = budget_reason
                        store.append("run_aborted_budget", state, detail={"reason": budget_reason})
                        return state

            stop_decision = self.hooks.run_stop(state, response.content)
            if not stop_decision.allow_stop:
                state.messages.append(
                    Message(role="user", content=stop_decision.feedback_message)
                )
                store.append(
                    "stop_rejected",
                    state,
                    detail={"feedback": stop_decision.feedback_message},
                )
                continue

            state.final_output = response.content
            default_terminal = (
                "DIAGNOSED"
                if state.user_context.resolution_mode == "diagnosis_only"
                else "AWAITING_APPROVAL"
            )
            state.status = stop_decision.target_status or default_terminal
            if state.status == "AWAITING_APPROVAL":
                for hook in self.hooks.post_tool_hooks:
                    on_paused = getattr(hook, "on_run_paused", None)
                    if callable(on_paused):
                        on_paused(state)
            store.append("run_completed", state, detail={"status": state.status})
            break

        return state
