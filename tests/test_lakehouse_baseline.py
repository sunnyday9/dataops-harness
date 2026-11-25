from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

from benchmarks.reset_env import reset_environment
from src.adapters.base import LakehousePaths, ResetSummary
from src.adapters.spark_adapter import discover_java_home

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def baseline_reset(tmp_path_factory: pytest.TempPathFactory) -> ResetSummary:
    event_dir = tmp_path_factory.mktemp("spark_events")
    return reset_environment(repo_root=REPO_ROOT, event_log_dir=event_dir)


def test_toolchain_health() -> None:
    """Verify OpenJDK 17, DuckDB Delta extension, and dbt binary are usable."""
    java_home = discover_java_home()
    proc = subprocess.run(
        [str(java_home / "bin" / "java"), "-version"],
        text=True,
        capture_output=True,
        check=True,
    )
    version_text = proc.stdout + proc.stderr
    assert "17." in version_text

    con = duckdb.connect()
    try:
        con.execute("INSTALL delta; LOAD delta;")
        row = con.execute("SELECT 1").fetchone()
        assert row == (1,)
    finally:
        con.close()

    dbt_exe = Path(sys.executable).parent / "dbt"
    assert dbt_exe.is_file() or shutil.which("dbt") is not None


def test_reset_builds_isolated_git_repo_and_green_lakehouse(
    baseline_reset: ResetSummary,
) -> None:
    """Verify sandbox git isolation, Delta silver outputs, and isolated Spark subprocess event logs."""
    paths = LakehousePaths.from_workspace(REPO_ROOT / "workspace")
    assert (paths.workspace_dir / ".git").is_dir()
    assert baseline_reset.git_commits >= 3

    for table_name in ("silver_users", "silver_exchange_rates", "silver_orders"):
        assert (paths.prod_silver_dir / table_name / "_delta_log").is_dir()

    assert len(baseline_reset.spark_runs) == 3
    for run in baseline_reset.spark_runs:
        assert run.returncode == 0
        assert run.pid is not None and run.pid != os.getpid()
        assert run.event_log_path is not None and run.event_log_path.is_file()
        event_types = {
            json.loads(line).get("Event")
            for line in run.event_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert "SparkListenerApplicationStart" in event_types
        assert "SparkListenerTaskEnd" in event_types


def test_dbt_models_and_gmv_reconciliation(baseline_reset: ResetSummary) -> None:
    """Verify dbt tests pass and mart GMV totals match across fct_daily_gmv and dim_user_ltv."""
    paths = LakehousePaths.from_workspace(REPO_ROOT / "workspace")
    dbt_exe = Path(sys.executable).parent / "dbt"
    env = os.environ.copy()
    env["DATAOPS_DUCKDB_PATH"] = str(paths.prod_duckdb_path)
    env["DATAOPS_PROD_SILVER_DIR"] = str(paths.prod_silver_dir)
    proc = subprocess.run(
        [str(dbt_exe), "test", "--profiles-dir", "."],
        cwd=str(paths.dbt_project_dir),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert baseline_reset.daily_gmv_rows == 7
    assert baseline_reset.user_ltv_rows == 1000
    assert baseline_reset.fct_daily_gmv_total > 0
    assert abs(baseline_reset.fct_daily_gmv_total - baseline_reset.dim_user_ltv_total) <= 0.01


def test_repeated_reset_is_idempotent_with_identical_golden_gmv(
    baseline_reset: ResetSummary,
    tmp_path: Path,
) -> None:
    """Verify repeated reset converges cleanly and yields identical golden GMV totals."""
    second = reset_environment(repo_root=REPO_ROOT, event_log_dir=tmp_path / "events")
    assert second.dbt_run_ok is True
    assert second.dbt_test_ok is True
    assert second.raw_orders == baseline_reset.raw_orders
    assert second.daily_gmv_rows == baseline_reset.daily_gmv_rows
    assert second.user_ltv_rows == baseline_reset.user_ltv_rows
    assert second.fct_daily_gmv_total == baseline_reset.fct_daily_gmv_total
    assert second.dim_user_ltv_total == baseline_reset.dim_user_ltv_total
