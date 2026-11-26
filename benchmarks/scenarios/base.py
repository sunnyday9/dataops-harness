from __future__ import annotations

import difflib
import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import duckdb
from pydantic import BaseModel, Field

from src.adapters.base import LakehousePaths, SparkRunResult
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.spark_adapter import SparkEventLogParser, SparkEventSummary
from src.core.aggregator import RawAlert
from src.core.context import IncidentInput, PrincipalRole
from src.core.shadow_verifier import HiddenRegressionOracle

ScenarioCategory = Literal["spark", "dbt_cross_engine", "security", "resilience"]


class InjectionOutcome(BaseModel):
    """Structured result emitted after injecting a reproducible Fault Scenario."""

    scenario_id: str
    category: ScenarioCategory
    workspace_dir: Path
    incident: IncidentInput
    raw_alerts: list[RawAlert] = Field(default_factory=list)
    principal_role: PrincipalRole = "de"
    baseline_gmv_total: float = 0.0
    baseline_order_count: int = 0
    spark_metrics_path: Path | None = None
    fault_reproduced: bool = True
    reproduction_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_unified_diff(rel_path: str, old_text: str, new_text: str) -> str:
    """Construct a strict `git apply`-compatible unified diff for `rel_path`."""
    norm = rel_path.replace("\\", "/").lstrip("./")
    diff_lines = list(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{norm}",
            tofile=f"b/{norm}",
        )
    )
    if not diff_lines:
        return ""
    diff_text = "".join(diff_lines)
    return diff_text if diff_text.endswith("\n") else diff_text + "\n"


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def read_mart_metrics(duckdb_path: Path) -> dict[str, float | int]:
    """Read key GMV, LTV, row, and order counts from a mart warehouse DuckDB file."""
    con = duckdb.connect(str(duckdb_path.resolve()), read_only=True)
    try:
        fct_total, daily_rows, fct_order_count = con.execute(
            "SELECT ROUND(COALESCE(SUM(net_gmv_usd), 0), 2), COUNT(*), "
            "COALESCE(SUM(order_count), 0) FROM fct_daily_gmv"
        ).fetchone()
        dim_total, user_rows = con.execute(
            "SELECT ROUND(COALESCE(SUM(lifetime_gmv_usd), 0), 2), COUNT(*) FROM dim_user_ltv"
        ).fetchone()
        return {
            "fct_daily_gmv_total": float(fct_total),
            "dim_user_ltv_total": float(dim_total),
            "daily_gmv_rows": int(daily_rows),
            "user_ltv_rows": int(user_rows),
            "fct_order_count": int(fct_order_count),
        }
    finally:
        con.close()


def read_silver_order_stats(silver_orders_dir: Path) -> dict[str, int]:
    """Read direct Delta silver_orders row, distinct order_id, and null counts via DuckDB."""
    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute("LOAD delta;")
        except duckdb.Error:
            con.execute("INSTALL delta; LOAD delta;")
        scan_sql = f"delta_scan({_sql_literal(str(silver_orders_dir.resolve()))})"
        total_rows, distinct_orders, null_amt, null_user = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT order_id), "
            f"SUM(CASE WHEN amount_local IS NULL THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN user_id IS NULL THEN 1 ELSE 0 END) "
            f"FROM {scan_sql}"
        ).fetchone()
        return {
            "total_rows": int(total_rows or 0),
            "distinct_order_ids": int(distinct_orders or 0),
            "null_amount_rows": int(null_amt or 0),
            "null_user_rows": int(null_user or 0),
        }
    finally:
        con.close()


def commit_sandbox_fault(
    workspace_dir: Path,
    message: str,
    paths: list[str] | None = None,
) -> str:
    """Commit faulted code into the isolated sandbox workspace git repository."""
    resolved = workspace_dir.resolve()
    if paths:
        subprocess.run(
            ["git", "-C", str(resolved), "add", "--", *paths],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        subprocess.run(
            ["git", "-C", str(resolved), "add", "-A"],
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(resolved),
            "-c",
            "user.name=dataops-bot",
            "-c",
            "user.email=dataops-bot@example.com",
            "commit",
            "--allow-empty",
            "-m",
            message,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_spark_metrics_artifact(
    workspace_dir: Path,
    scenario_id: str,
    spark_result: SparkRunResult,
) -> Path:
    """Persist structured Spark metrics under `workspace/incidents/<id>/spark_metrics.json`."""
    out_dir = workspace_dir.resolve() / "incidents" / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spark_metrics.json"
    summary = spark_result.event_summary
    payload = {
        "scenario_id": scenario_id,
        "job_name": spark_result.job_name,
        "returncode": spark_result.returncode,
        "duration_ms": spark_result.duration_ms,
        "timed_out": spark_result.timed_out,
        "event_log_path": str(spark_result.event_log_path) if spark_result.event_log_path else None,
        "event_summary": summary.model_dump(mode="json") if summary else None,
        "stderr_tail": "\n".join(spark_result.stderr.splitlines()[-15:]),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def find_latest_spark_event_summary(
    events_dir: Path, *, stderr: str = ""
) -> SparkEventSummary | None:
    """Parse the most recent Spark event log written into `events_dir`."""
    if not events_dir.is_dir():
        return None
    candidates = [
        p
        for p in events_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".inprogress")
    ]
    if not candidates:
        candidates = [p for p in events_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return SparkEventLogParser.parse_file(latest, stderr=stderr)


def run_sandbox_dbt(
    workspace_dir: Path, args: list[str]
) -> subprocess.CompletedProcess[str]:
    """Execute dbt against an isolated sandbox workspace's lakehouse."""
    return DuckDBAdapter(LakehousePaths.from_workspace(workspace_dir)).run_dbt(args)


class FaultScenario(ABC):
    """Contract for an automated reproducible Fault Scenario and its regression Oracle."""

    scenario_id: str
    title: str
    category: ScenarioCategory
    default_role: PrincipalRole = "de"
    requires_fix: bool = True

    @abstractmethod
    def inject(
        self, workspace_dir: Path, *, event_log_dir: Path | None = None
    ) -> InjectionOutcome:
        """Inject and reproduce the fault inside an isolated sandbox workspace."""

    @abstractmethod
    def build_oracle(self, outcome: InjectionOutcome) -> HiddenRegressionOracle:
        """Return an out-of-workspace regression oracle validating a candidate fix."""

    def reference_fix_patch(self, outcome: InjectionOutcome) -> str | None:
        """Return canonical unified diff fixing the root cause for verification tests."""
        return None
