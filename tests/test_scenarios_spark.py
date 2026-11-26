from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.scenarios import get_scenario, list_scenarios
from benchmarks.scenarios.base import InjectionOutcome, read_mart_metrics, read_silver_order_stats
from src.adapters.base import LakehousePaths
from src.adapters.spark_adapter import SparkJobRunner
from src.core.shadow_verifier import ShadowOverlayVerifier
from src.core.worktree import RunWorktreeManager

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_WORKSPACE = REPO_ROOT / "workspace"


def _clone_sandbox(tmp_path: Path) -> Path:
    """Create an isolated sandbox workspace copy with clean git repo in ~20ms."""
    ws = tmp_path / "workspace"
    shutil.copytree(
        LIVE_WORKSPACE,
        ws,
        ignore=shutil.ignore_patterns(".git", ".scratch", "logs", ".user.yml", "__pycache__"),
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=ws, check=True, capture_output=True)
    return ws


def _apply_patch(worktree_dir: Path, patch_text: str) -> None:
    subprocess.run(
        ["git", "-C", str(worktree_dir), "apply", "-"],
        input=patch_text if patch_text.endswith("\n") else patch_text + "\n",
        text=True,
        check=True,
        capture_output=True,
    )


def _assert_smoke_registry() -> None:
    ids = {s.scenario_id for s in list_scenarios()}
    assert {
        "spark_join_skew",
        "spark_driver_oom",
        "spark_append_gmv_doubling",
        "spark_json_schema_drift",
    }.issubset(ids)


def test_spark_join_skew_reproduces_over_10x_and_oracle_rejects_cheat_passes_fix(
    tmp_path: Path,
) -> None:
    """Skew fault reproduces >10x ratio; Oracle rejects row-dropping cheat and passes broadcast/AQE fix."""
    _assert_smoke_registry()
    ws = _clone_sandbox(tmp_path)
    runs_dir = tmp_path / "runs"
    scenario = get_scenario("spark_join_skew")

    outcome: InjectionOutcome = scenario.inject(ws, event_log_dir=tmp_path / "events_inj")
    assert outcome.fault_reproduced is True
    assert float(outcome.metadata["max_skew_ratio"]) > 10.0
    assert outcome.spark_metrics_path is not None and outcome.spark_metrics_path.is_file()
    metrics_json = json.loads(outcome.spark_metrics_path.read_text(encoding="utf-8"))
    assert metrics_json["event_summary"]["max_skew_ratio"] > 10.0

    oracle = scenario.build_oracle(outcome)
    verifier = ShadowOverlayVerifier(ws, runs_dir, oracle=oracle)

    # 1. Cheat fix dropping hot key U0001 to eliminate skew must be rejected by Oracle.
    wt_cheat = RunWorktreeManager.create(ws, runs_dir, "run-skew-cheat")
    job_cheat = wt_cheat / "spark_jobs" / "build_silver_orders.py"
    text_cheat = job_cheat.read_text(encoding="utf-8").replace(
        ".filter(F.col(\"order_id\").isNotNull() & F.col(\"order_date\").isNotNull())",
        ".filter(F.col(\"order_id\").isNotNull() & F.col(\"order_date\").isNotNull() & (F.col(\"user_id\") != \"U0001\"))",
    )
    text_cheat = text_cheat.replace(
        'spark = build_spark_session("build_silver_orders", shuffle_partitions=8, aqe_enabled=False)\n'
        '    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")',
        'spark = build_spark_session("build_silver_orders", shuffle_partitions=8, aqe_enabled=True)',
    ).replace(
        'orders_raw.join(users_country, on="user_id", how="left")',
        'orders_raw.join(F.broadcast(users_country), on="user_id", how="left")',
    )
    job_cheat.write_text(text_cheat, encoding="utf-8")
    cheat_res = verifier.verify("run-skew-cheat", wt_cheat)
    assert cheat_res.passed is False
    assert any("row count mismatch" in r or "U0001" in r or "hot-key" in r for r in cheat_res.failure_reasons)

    # 2. Genuine broadcast + AQE fix passes with conserved GMV and shadow skew < 10x.
    wt_fix = RunWorktreeManager.create(ws, runs_dir, "run-skew-fix")
    fix_patch = scenario.reference_fix_patch(outcome)
    assert fix_patch
    _apply_patch(wt_fix, fix_patch)
    fix_res = verifier.verify("run-skew-fix", wt_fix)
    assert fix_res.passed is True, fix_res.failure_reasons
    assert fix_res.oracle_verdict is not None
    assert float(fix_res.oracle_verdict.metrics["shadow_etl_skew_ratio"]) < 10.0

    RunWorktreeManager.cleanup(ws, wt_cheat)
    RunWorktreeManager.cleanup(ws, wt_fix)


def test_spark_driver_oom_reproduces_isolated_jvm_oom_and_heals_at_384m(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """384m Driver OOM fails child JVM cleanly without crashing pytest and heals via distributed write."""
    monkeypatch.setenv("SPARK_DRIVER_EXTRA_JAVA_OPTIONS", "-Dspark.testing.memory=471859200")
    ws = _clone_sandbox(tmp_path)
    runs_dir = tmp_path / "runs"
    scenario = get_scenario("spark_driver_oom")

    outcome = scenario.inject(ws, event_log_dir=tmp_path / "events_oom")
    assert outcome.fault_reproduced is True
    assert outcome.spark_metrics_path is not None
    metrics_json = json.loads(outcome.spark_metrics_path.read_text(encoding="utf-8"))
    assert metrics_json["returncode"] != 0
    assert metrics_json["event_summary"]["oom_detected"] is True

    oracle = scenario.build_oracle(outcome)
    shadow_events = runs_dir / "run-oom-fix" / "shadow" / "spark_events"
    verifier = ShadowOverlayVerifier(
        ws,
        runs_dir,
        oracle=oracle,
        spark_runner=SparkJobRunner(event_log_dir=shadow_events, default_driver_memory="384m"),
    )

    wt_fix = RunWorktreeManager.create(ws, runs_dir, "run-oom-fix")
    fix_patch = scenario.reference_fix_patch(outcome)
    assert fix_patch
    _apply_patch(wt_fix, fix_patch)
    res = verifier.verify("run-oom-fix", wt_fix)
    assert res.passed is True, res.failure_reasons
    assert res.dbt_test_ok is True

    RunWorktreeManager.cleanup(ws, wt_fix)


def test_spark_append_gmv_doubling_rejects_distinct_cheat_and_passes_overwrite_fix(
    tmp_path: Path,
) -> None:
    """Append bug doubles GMV; Oracle rejects downstream SELECT DISTINCT cheat and passes overwrite fix."""
    ws = _clone_sandbox(tmp_path)
    runs_dir = tmp_path / "runs"
    scenario = get_scenario("spark_append_gmv_doubling")

    outcome = scenario.inject(ws, event_log_dir=tmp_path / "events_append")
    assert outcome.fault_reproduced is True
    assert abs(float(outcome.metadata["doubled_gmv_total"]) - 2.0 * outcome.baseline_gmv_total) <= 0.1

    oracle = scenario.build_oracle(outcome)
    verifier = ShadowOverlayVerifier(ws, runs_dir, oracle=oracle)

    # 1. Cheat fix adding SELECT DISTINCT in stg_orders.sql without fixing Spark append fails Oracle.
    wt_cheat = RunWorktreeManager.create(ws, runs_dir, "run-append-cheat")
    stg_orders = wt_cheat / "dbt_project" / "models" / "staging" / "stg_orders.sql"
    stg_orders.write_text(
        stg_orders.read_text(encoding="utf-8").replace("SELECT\n", "SELECT DISTINCT\n", 1),
        encoding="utf-8",
    )
    cheat_res = verifier.verify("run-append-cheat", wt_cheat)
    assert cheat_res.passed is False
    assert any("Cheat rejected" in r or "append" in r for r in cheat_res.failure_reasons)

    # 2. Genuine .mode("overwrite") fix restores exact 1x rows & baseline GMV.
    wt_fix = RunWorktreeManager.create(ws, runs_dir, "run-append-fix")
    fix_patch = scenario.reference_fix_patch(outcome)
    assert fix_patch
    _apply_patch(wt_fix, fix_patch)
    fix_res = verifier.verify("run-append-fix", wt_fix)
    assert fix_res.passed is True, fix_res.failure_reasons
    mart = read_mart_metrics(Path(fix_res.shadow_duckdb_path))
    assert abs(float(mart["fct_daily_gmv_total"]) - outcome.baseline_gmv_total) <= 0.01

    RunWorktreeManager.cleanup(ws, wt_cheat)
    RunWorktreeManager.cleanup(ws, wt_fix)


def test_spark_json_schema_drift_rejects_drop_cheat_and_recovers_valid_revenue(
    tmp_path: Path,
) -> None:
    """Dirty JSON currency strings cause NULLs; Oracle rejects silent drops and passes parsing fix."""
    ws = _clone_sandbox(tmp_path)
    runs_dir = tmp_path / "runs"
    scenario = get_scenario("spark_json_schema_drift")

    outcome = scenario.inject(ws, event_log_dir=tmp_path / "events_drift")
    assert outcome.fault_reproduced is True
    drift_stats = read_silver_order_stats(LakehousePaths.from_workspace(ws).prod_silver_dir / "silver_orders")
    assert drift_stats["null_amount_rows"] == 25

    oracle = scenario.build_oracle(outcome)
    verifier = ShadowOverlayVerifier(ws, runs_dir, oracle=oracle)

    # 1. Cheat fix filtering NULLs without parsing "$100.00" strings drops valid rows & GMV -> rejected.
    wt_cheat = RunWorktreeManager.create(ws, runs_dir, "run-drift-cheat")
    job_cheat = wt_cheat / "spark_jobs" / "build_silver_orders.py"
    job_cheat.write_text(
        job_cheat.read_text(encoding="utf-8").replace(
            '.filter(F.col("order_id").isNotNull() & F.col("order_date").isNotNull())',
            '.filter(F.col("order_id").isNotNull() & F.col("order_date").isNotNull() & F.col("amount_local").isNotNull())',
        ),
        encoding="utf-8",
    )
    cheat_res = verifier.verify("run-drift-cheat", wt_cheat)
    assert cheat_res.passed is False
    assert any("20 valid" in r or "Expected" in r for r in cheat_res.failure_reasons)

    # 2. Genuine fix parses valid currency strings and excludes corrupt rows -> passes Oracle.
    wt_fix = RunWorktreeManager.create(ws, runs_dir, "run-drift-fix")
    fix_patch = scenario.reference_fix_patch(outcome)
    assert fix_patch
    _apply_patch(wt_fix, fix_patch)
    fix_res = verifier.verify("run-drift-fix", wt_fix)
    assert fix_res.passed is True, fix_res.failure_reasons

    RunWorktreeManager.cleanup(ws, wt_cheat)
    RunWorktreeManager.cleanup(ws, wt_fix)
