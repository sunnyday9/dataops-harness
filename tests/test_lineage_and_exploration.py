from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from src.adapters.base import LakehousePaths
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.spark_adapter import SparkEventLogParser

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = REPO_ROOT / "workspace"


@pytest.fixture(scope="module")
def lineage_adapter() -> LineageAdapter:
    return LineageAdapter.from_workspace(WORKSPACE_DIR)


@pytest.fixture(scope="module")
def duckdb_adapter() -> DuckDBAdapter:
    return DuckDBAdapter(LakehousePaths.from_workspace(WORKSPACE_DIR))


def test_cross_engine_lineage_upstream_and_downstream(
    lineage_adapter: LineageAdapter,
) -> None:
    """Lineage traversal crosses Spark-to-dbt boundaries and includes columns and tests."""
    upstream = lineage_adapter.query_lineage(
        "fct_daily_gmv", direction="upstream", depth=5
    )
    assert {
        "fct_daily_gmv",
        "int_orders_usd",
        "stg_orders",
        "stg_exchange_rates",
        "silver_orders",
        "silver_exchange_rates",
        "silver_users",
        "raw_orders",
        "raw_exchange_rates",
        "raw_users",
    }.issubset(upstream.nodes.keys())

    assert upstream.nodes["silver_orders"].engine == "spark"
    assert upstream.nodes["raw_orders"].engine == "raw"
    assert "order_id" in upstream.nodes["silver_orders"].columns
    assert "order_date" in upstream.nodes["silver_orders"].partition_columns

    fct_tests = {t.name for t in upstream.nodes["fct_daily_gmv"].tests}
    assert "assert_gmv_reconciliation" in fct_tests
    assert any(t.test_type == "unique" for t in upstream.nodes["fct_daily_gmv"].tests)

    downstream = lineage_adapter.query_lineage(
        "silver_orders", direction="downstream", depth=4
    )
    assert set(downstream.nodes.keys()) == {
        "silver_orders",
        "stg_orders",
        "int_orders_usd",
        "fct_daily_gmv",
        "dim_user_ltv",
    }


def test_lca_and_shared_ancestors_for_sibling_marts(
    lineage_adapter: LineageAdapter,
) -> None:
    """Lowest Common Ancestor identifies shared upstream causes across sibling marts."""
    assert lineage_adapter.lowest_common_ancestors(
        ["fct_daily_gmv", "dim_user_ltv"]
    ) == ["int_orders_usd"]

    shared = lineage_adapter.shared_ancestors(["fct_daily_gmv", "dim_user_ltv"])
    assert "int_orders_usd" in shared
    assert "silver_orders" in shared
    assert "silver_exchange_rates" in shared

    assert lineage_adapter.lowest_common_ancestors(
        ["stg_orders", "stg_users"]
    ) == ["silver_users"]
    assert lineage_adapter.lowest_common_ancestors(
        ["silver_orders", "fct_daily_gmv"]
    ) == ["silver_orders"]


def test_affected_downstream_for_spark_job_macro_and_dbt_model(
    lineage_adapter: LineageAdapter,
) -> None:
    """Changed workspace paths resolve to exact impacted Spark jobs and dbt models."""
    spark_impact = lineage_adapter.affected_downstream(
        ["spark_jobs/build_silver_orders.py"]
    )
    assert spark_impact.spark_jobs == ["spark_jobs/build_silver_orders.py"]
    assert set(spark_impact.dbt_models) == {
        "stg_orders",
        "int_orders_usd",
        "fct_daily_gmv",
        "dim_user_ltv",
    }

    macro_impact = lineage_adapter.affected_downstream(
        ["dbt_project/macros/currency_utils.sql"]
    )
    assert macro_impact.spark_jobs == []
    assert set(macro_impact.dbt_models) == {
        "int_orders_usd",
        "fct_daily_gmv",
        "dim_user_ltv",
    }

    mart_impact = lineage_adapter.affected_downstream(
        ["dbt_project/models/marts/fct_daily_gmv.sql"]
    )
    assert mart_impact.spark_jobs == []
    assert mart_impact.dbt_models == ["fct_daily_gmv"]


def test_duckdb_readonly_reads_delta_silver_and_mart_subsecond(
    duckdb_adapter: DuckDBAdapter,
) -> None:
    """DuckDBAdapter probes Delta silver and DuckDB mart tables sub-second and read-only."""
    silver_res = duckdb_adapter.execute_readonly(
        "SELECT COUNT(*) AS cnt FROM silver_orders"
    )
    assert silver_res.rows == [[6000]]
    assert silver_res.duration_ms < 1000.0

    mart_res = duckdb_adapter.execute_readonly(
        "SELECT "
        "(SELECT ROUND(SUM(net_gmv_usd), 2) FROM fct_daily_gmv) AS fct_gmv, "
        "(SELECT ROUND(SUM(lifetime_gmv_usd), 2) FROM dim_user_ltv) AS dim_gmv"
    )
    fct_gmv, dim_gmv = mart_res.rows[0]
    assert fct_gmv > 0
    assert abs(fct_gmv - dim_gmv) <= 0.01
    assert mart_res.duration_ms < 1000.0

    with pytest.raises(duckdb.Error):
        duckdb_adapter.execute_readonly("DELETE FROM mart.main.fct_daily_gmv")


def test_duckdb_data_diff_detects_identical_and_changed_tables(
    tmp_path: Path,
) -> None:
    """data_diff distinguishes identical tables from added, removed, and modified keys."""
    prod_db = tmp_path / "prod.duckdb"
    shadow_same = tmp_path / "shadow_same.duckdb"
    shadow_diff = tmp_path / "shadow_diff.duckdb"

    for db_path, rows in (
        (prod_db, [("2026-11-01", 100.0), ("2026-11-02", 200.0), ("2026-11-03", 300.0)]),
        (shadow_same, [("2026-11-01", 100.0), ("2026-11-02", 200.0), ("2026-11-03", 300.0)]),
        (shadow_diff, [("2026-11-01", 100.0), ("2026-11-02", 250.0), ("2026-11-04", 400.0)]),
    ):
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                "CREATE TABLE fct_daily_gmv (order_date VARCHAR, net_gmv_usd DOUBLE)"
            )
            con.executemany("INSERT INTO fct_daily_gmv VALUES (?, ?)", rows)
        finally:
            con.close()

    adapter = DuckDBAdapter(LakehousePaths.from_workspace(WORKSPACE_DIR))

    same_diff = adapter.data_diff(prod_db, shadow_same, "fct_daily_gmv", "order_date")
    assert same_diff.is_identical is True
    assert same_diff.added_keys == []
    assert same_diff.removed_keys == []
    assert same_diff.changed_keys == []

    changed_diff = adapter.data_diff(prod_db, shadow_diff, "fct_daily_gmv", "order_date")
    assert changed_diff.is_identical is False
    assert changed_diff.added_keys == ["2026-11-04"]
    assert changed_diff.removed_keys == ["2026-11-03"]
    assert changed_diff.changed_keys == ["2026-11-02"]


def test_spark_event_log_parser_balanced_skew_and_oom(tmp_path: Path) -> None:
    """Zero-JVM SparkEventLogParser flags balanced stages, >10x skew, and heap OOMs."""
    balanced_log = tmp_path / "balanced.jsonl"
    balanced_events = [
        {"Event": "SparkListenerApplicationStart", "App ID": "local-1", "App Name": "ok"},
        {"Event": "SparkListenerStageSubmitted", "Stage Info": {"Stage ID": 0, "Stage Name": "scan"}},
        *[
            {
                "Event": "SparkListenerTaskEnd",
                "Stage ID": 0,
                "Task Info": {"Launch Time": 1000, "Finish Time": 1000 + ms, "Failed": False},
                "Task End Reason": {"Reason": "Success"},
            }
            for ms in (100, 110, 105, 120)
        ],
    ]
    balanced_log.write_text(
        "\n".join(json.dumps(e) for e in balanced_events), encoding="utf-8"
    )
    balanced_summary = SparkEventLogParser.parse_file(balanced_log)
    assert balanced_summary.max_skew_ratio < 2.0
    assert balanced_summary.skewed_stages == []
    assert balanced_summary.oom_detected is False

    skew_log = tmp_path / "skew.jsonl"
    skew_events = [
        {"Event": "SparkListenerApplicationStart", "App ID": "local-2", "App Name": "skew"},
        {"Event": "SparkListenerStageSubmitted", "Stage Info": {"Stage ID": 1, "Stage Name": "join"}},
        *[
            {
                "Event": "SparkListenerTaskEnd",
                "Stage ID": 1,
                "Task Info": {"Launch Time": 2000, "Finish Time": 2000 + ms, "Failed": False},
                "Task End Reason": {"Reason": "Success"},
            }
            for ms in (100, 105, 110, 2500)
        ],
    ]
    skew_log.write_text("\n".join(json.dumps(e) for e in skew_events), encoding="utf-8")
    skew_summary = SparkEventLogParser.parse_file(skew_log)
    assert skew_summary.max_skew_ratio > 10.0
    assert skew_summary.skewed_stages == [1]
    assert skew_summary.oom_detected is False

    oom_log = tmp_path / "oom.jsonl"
    oom_events = [
        {"Event": "SparkListenerApplicationStart", "App ID": "local-3", "App Name": "oom"},
        {
            "Event": "SparkListenerTaskEnd",
            "Stage ID": 0,
            "Task Info": {"Launch Time": 1000, "Finish Time": 1400, "Failed": True},
            "Task End Reason": {
                "Reason": "ExceptionFailure",
                "Class Name": "java.lang.OutOfMemoryError",
                "Description": "Java heap space",
            },
        },
    ]
    oom_log.write_text(
        "\n".join(json.dumps(e) for e in oom_events) + "\n{truncated_json",
        encoding="utf-8",
    )
    oom_summary = SparkEventLogParser.parse_file(
        oom_log, stderr="ERROR Executor: Exception in task: java.lang.OutOfMemoryError: Java heap space"
    )
    assert oom_summary.oom_detected is True
    assert oom_summary.failed_tasks == 1
    assert any("Java heap space" in r for r in oom_summary.oom_reasons)
