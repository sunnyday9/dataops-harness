from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.adapters.base import DataDiffSummary

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


class RcaReport(BaseModel):
    """Structured Root Cause Analysis contract required at Stop time."""

    incident_id: str = Field(min_length=1)
    root_cause_summary: str = Field(min_length=10)
    affected_assets: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    engine: Literal["spark", "dbt", "cross_engine", "governance", "other"] = "cross_engine"
    proposed_fix_summary: str = Field(min_length=5)
    patched_files: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "high"


class OracleVerdict(BaseModel):
    """Outcome of a Hidden Regression Oracle check against shadow execution."""

    passed: bool
    oracle_name: str = "default"
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ShadowVerificationResult(BaseModel):
    """Full Shadow Overlay verification report produced before HiTL approval."""

    passed: bool
    run_id: str
    worktree_path: str
    shadow_silver_dir: str
    shadow_duckdb_path: str
    changed_files: list[str] = Field(default_factory=list)
    affected_spark_jobs: list[str] = Field(default_factory=list)
    affected_dbt_models: list[str] = Field(default_factory=list)
    spark_runs: list[dict[str, Any]] = Field(default_factory=list)
    dbt_run_ok: bool = False
    dbt_test_ok: bool = False
    dbt_stdout: str = ""
    dbt_stderr: str = ""
    data_diffs: list[DataDiffSummary] = Field(default_factory=list)
    oracle_verdict: OracleVerdict | None = None
    failure_reasons: list[str] = Field(default_factory=list)


class ApprovalPacket(BaseModel):
    """Human-in-the-loop review packet persisted when transitioning to AWAITING_APPROVAL."""

    run_id: str
    incident_id: str
    rca_report: RcaReport
    unified_diff: str
    shadow_verification: ShadowVerificationResult
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def _extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_rca_report(candidate_output: str) -> RcaReport:
    """Parse and validate a JSON `RcaReport` from model stop output."""
    text = (candidate_output or "").strip()
    if not text:
        raise ValueError("Candidate final output is empty; must contain an RCA Report JSON.")

    candidates: list[str] = [match.group(1).strip() for match in _FENCED_JSON_RE.finditer(text)]
    if text.startswith("{"):
        candidates.append(text)
    balanced = _extract_balanced_json_object(text)
    if balanced and balanced not in candidates:
        candidates.append(balanced)

    parsed_payload: dict[str, Any] | None = None
    last_json_err: str | None = None
    for blob in candidates:
        if not blob:
            continue
        try:
            loaded = json.loads(blob)
        except json.JSONDecodeError as exc:
            last_json_err = str(exc)
            continue
        if isinstance(loaded, dict):
            parsed_payload = loaded
            break

    if parsed_payload is None:
        detail = f" ({last_json_err})" if last_json_err else ""
        raise ValueError(
            "Output does not contain a valid JSON object matching the RcaReport schema"
            f"{detail}."
        )

    try:
        return RcaReport.model_validate(parsed_payload)
    except ValidationError as exc:
        issues = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ValueError(f"RCA Report validation failed: {issues}") from exc
