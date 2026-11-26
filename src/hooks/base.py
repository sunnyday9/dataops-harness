from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.context import Message, RunState, RunStatus, ToolCall, ToolResult


class PreToolDecision(BaseModel):
    """Outcome of PreToolUse hook evaluation."""

    action: Literal["allow", "block", "modify"] = "allow"
    reason: str = ""
    modified_arguments: dict[str, Any] | None = None


class PostToolDecision(BaseModel):
    """Outcome of PostToolUse hook evaluation."""

    action: Literal["allow", "modify"] = "allow"
    modified_output: str | None = None
    injected_messages: list[Message] = Field(default_factory=list)
    abort_status: RunStatus | None = None
    abort_reason: str = ""


class StopDecision(BaseModel):
    """Outcome of Stop hook evaluation when the model attempts to finish."""

    allow_stop: bool = True
    target_status: RunStatus | None = None
    feedback_message: str = ""


class PreToolUseHook(ABC):
    """Intercepts a tool invocation before execution."""

    @abstractmethod
    def on_pre_tool_use(self, state: RunState, tool_call: ToolCall) -> PreToolDecision:
        """Return allow, block, or modify before the tool handler runs."""


class PostToolUseHook(ABC):
    """Transforms tool output or triggers nudges/aborts after execution."""

    @abstractmethod
    def on_post_tool_use(
        self, state: RunState, tool_call: ToolCall, result: ToolResult
    ) -> PostToolDecision:
        """Inspect and optionally transform `result` or request abort."""


class StopHook(ABC):
    """Validates exit contracts (RCA Report, shadow verification) before finishing."""

    @abstractmethod
    def on_stop(self, state: RunState, candidate_output: str) -> StopDecision:
        """Approve stop or reject with structured corrective feedback."""


class HookPipeline:
    """Ordered, fault-agnostic middleware pipeline for lifecycle hooks."""

    def __init__(
        self,
        pre_tool_hooks: list[PreToolUseHook] | None = None,
        post_tool_hooks: list[PostToolUseHook] | None = None,
        stop_hooks: list[StopHook] | None = None,
    ) -> None:
        self.pre_tool_hooks = list(pre_tool_hooks or [])
        self.post_tool_hooks = list(post_tool_hooks or [])
        self.stop_hooks = list(stop_hooks or [])

    def run_pre_tool(
        self, state: RunState, tool_call: ToolCall
    ) -> tuple[PreToolDecision, ToolCall]:
        current_call = tool_call.model_copy(deep=True)
        for hook in self.pre_tool_hooks:
            decision = hook.on_pre_tool_use(state, current_call)
            if decision.action == "block":
                return decision, current_call
            if decision.action == "modify" and decision.modified_arguments is not None:
                current_call = current_call.model_copy(
                    update={"arguments": dict(decision.modified_arguments)}
                )
        return PreToolDecision(action="allow"), current_call

    def run_post_tool(
        self, state: RunState, tool_call: ToolCall, result: ToolResult
    ) -> tuple[ToolResult, list[Message], RunStatus | None, str]:
        current_result = result.model_copy(deep=True)
        injected: list[Message] = []
        for hook in self.post_tool_hooks:
            decision = hook.on_post_tool_use(state, tool_call, current_result)
            if decision.action == "modify" and decision.modified_output is not None:
                current_result = current_result.model_copy(
                    update={"output": decision.modified_output}
                )
            if decision.injected_messages:
                injected.extend(decision.injected_messages)
            if decision.abort_status is not None:
                return (
                    current_result,
                    injected,
                    decision.abort_status,
                    decision.abort_reason,
                )
        return current_result, injected, None, ""

    def run_stop(self, state: RunState, candidate_output: str) -> StopDecision:
        target_status: RunStatus | None = None
        for hook in self.stop_hooks:
            decision = hook.on_stop(state, candidate_output)
            if not decision.allow_stop:
                return decision
            if decision.target_status is not None:
                target_status = decision.target_status
        return StopDecision(allow_stop=True, target_status=target_status)
