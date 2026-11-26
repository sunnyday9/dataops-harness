from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PrincipalRole = Literal["de", "da", "sre"]
ResolutionMode = Literal["fix", "diagnosis_only"]
RunStatus = Literal[
    "RUNNING",
    "AWAITING_APPROVAL",
    "RESOLVED",
    "DIAGNOSED",
    "ABORTED_THRASH",
    "ABORTED_BUDGET",
    "FAILED_ERROR",
]


class UserContext(BaseModel):
    """Caller identity and governance resolution mode propagated through a Run."""

    user_id: str = "cli-user"
    role: PrincipalRole = "de"
    resolution_mode: ResolutionMode | None = None

    @model_validator(mode="after")
    def _default_resolution_mode(self) -> UserContext:
        if self.resolution_mode is None:
            self.resolution_mode = "diagnosis_only" if self.role == "da" else "fix"
        return self


class IncidentInput(BaseModel):
    """Deduplicated root-cause Incident handed to the Harness loop."""

    incident_id: str
    title: str
    summary: str
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    affected_assets: list[str] = Field(default_factory=list)
    shared_ancestors: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    """Normalized model tool invocation."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result produced by executing a single atomic tool through hooks."""

    tool_call_id: str
    name: str
    output: str
    is_error: bool = False
    blocked: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Stateless local conversation turn stored in RunState."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


class UsageMetrics(BaseModel):
    """Accumulated token and cost metrics across turns."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: UsageMetrics) -> None:
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd = round(self.cost_usd + other.cost_usd, 6)

    @property
    def cache_hit_rate(self) -> float:
        total_input = self.input_tokens + self.cached_input_tokens
        if total_input <= 0:
            return 0.0
        return round(self.cached_input_tokens / total_input, 4)


class RunState(BaseModel):
    """Full serializable state of a single Run persisted at every checkpoint."""

    run_id: str
    incident: IncidentInput
    user_context: UserContext
    status: RunStatus = "RUNNING"
    turn: int = 0
    max_turns: int = 20
    rejection_count: int = 0
    messages: list[Message] = Field(default_factory=list)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    patched_files: list[str] = Field(default_factory=list)
    final_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointStore:
    """Append-only JSONL checkpoint persistence resilient to trailing crash lines."""

    def __init__(self, runs_dir: Path, run_id: str) -> None:
        self.runs_dir = runs_dir.resolve()
        self.run_id = run_id
        self.state_path = self.runs_dir / run_id / "state.jsonl"
        self._seq: int | None = None

    def append(
        self,
        event_type: str,
        state: RunState,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self._seq is None:
            self._seq = len(self.read_events())
        self._seq += 1

        record = {
            "seq": self._seq,
            "ts": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "detail": detail or {},
            "state": state.model_dump(mode="json"),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.state_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read_events(self) -> list[dict[str, Any]]:
        if not self.state_path.is_file():
            return []
        events: list[dict[str, Any]] = []
        with self.state_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    # A crash mid-write can leave an incomplete trailing line.
                    continue
                if isinstance(parsed, dict) and "state" in parsed:
                    events.append(parsed)
        return events

    def load_latest(self) -> RunState:
        if not self.state_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {self.state_path}")
        events = self.read_events()
        if not events:
            raise ValueError(f"No valid checkpoint events in: {self.state_path}")
        return RunState.model_validate(events[-1]["state"])
