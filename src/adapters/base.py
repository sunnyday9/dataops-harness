from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class LakehousePaths(BaseModel):
    """Canonical directory and database paths inside an isolated sandbox workspace."""

    workspace_dir: Path
    raw_dir: Path
    prod_silver_dir: Path
    prod_mart_dir: Path
    prod_duckdb_path: Path
    dbt_project_dir: Path
    spark_jobs_dir: Path

    @classmethod
    def from_workspace(cls, workspace_dir: Path) -> LakehousePaths:
        root = workspace_dir.resolve()
        prod_mart_dir = root / "lakehouse" / "prod" / "mart"
        return cls(
            workspace_dir=root,
            raw_dir=root / "lakehouse" / "raw",
            prod_silver_dir=root / "lakehouse" / "prod" / "silver",
            prod_mart_dir=prod_mart_dir,
            prod_duckdb_path=prod_mart_dir / "warehouse.duckdb",
            dbt_project_dir=root / "dbt_project",
            spark_jobs_dir=root / "spark_jobs",
        )


class SparkRunResult(BaseModel):
    """Observable execution outcome of an isolated PySpark child subprocess."""

    job_name: str
    script_path: Path
    pid: int | None
    returncode: int
    duration_ms: float
    event_log_path: Path | None
    stdout: str
    stderr: str
    timed_out: bool = False


class ResetSummary(BaseModel):
    """Verification summary emitted by the synthetic lakehouse reset command."""

    elapsed_seconds: float
    git_commits: int
    raw_orders: int
    raw_users: int
    raw_exchange_rates: int
    spark_runs: list[SparkRunResult]
    dbt_run_ok: bool
    dbt_test_ok: bool
    fct_daily_gmv_total: float
    dim_user_ltv_total: float
    daily_gmv_rows: int
    user_ltv_rows: int
