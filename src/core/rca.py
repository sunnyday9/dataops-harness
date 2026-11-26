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


def _extract_all_balanced_json_objects(text: str) -> list[str]:
    """Scan `text` and return every top-level balanced `{...}` JSON substring."""
    objs: list[str] = []
    idx = 0
    length = len(text)
    while idx < length:
        start = text.find("{", idx)
        if start < 0:
            break
        depth = 0
        in_string = False
        escape = False
        matched_end: int | None = None
        for cur_idx in range(start, length):
            ch = text[cur_idx]
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
                    matched_end = cur_idx + 1
                    break
        if matched_end is not None:
            objs.append(text[start:matched_end])
            idx = matched_end
        else:
            break
    return objs


def parse_rca_report(candidate_output: str) -> RcaReport:
    """Parse and validate a JSON `RcaReport` from model stop output."""
    text = (candidate_output or "").strip()
    if not text:
        raise ValueError("Candidate final output is empty; must contain an RCA Report JSON.")

    candidates: list[str] = [match.group(1).strip() for match in _FENCED_JSON_RE.finditer(text)]
    if text.startswith("{"):
        candidates.append(text)
    for balanced in _extract_all_balanced_json_objects(text):
        if balanced not in candidates:
            candidates.append(balanced)

    parsed_dicts: list[dict[str, Any]] = []
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
            parsed_dicts.append(loaded)

    if not parsed_dicts:
        detail = f" ({last_json_err})" if last_json_err else ""
        raise ValueError(
            "Output does not contain a valid JSON object matching the RcaReport schema"
            f"{detail}."
        )

    last_validation_err: ValidationError | None = None
    for payload in parsed_dicts:
        try:
            return RcaReport.model_validate(payload)
        except ValidationError as exc:
            last_validation_err = exc

    assert last_validation_err is not None
    issues = "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}"
        for err in last_validation_err.errors()
    )
    raise ValueError(f"RCA Report validation failed: {issues}") from last_validation_err
