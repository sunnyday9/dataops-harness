from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.adapters.alert_ingestion import (
    ingest_and_converge,
    ingest_dbt_run_results,
    ingest_openlineage_alerts,
    ingest_text_alert,
)
from src.adapters.base import LakehousePaths
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.openlineage import (
    extract_openlineage_lineage,
    infer_dialect_from_namespace,
    load_openlineage_events,
    normalize_dataset_name,
)
from src.core.context import IncidentInput, RunState, ToolCall, ToolResult, UserContext
from src.core.policy import resolve_policies
from src.core.project import discover_project, normalize_sql_dialect
from src.core.shadow_verifier import (
    DefaultIntegrityOracle,
    MetricReconciliationRule,
    _discover_mart_diff_targets,
    order_spark_jobs,
)
from src.hooks.pii_redactor import PiiRedactorHook
from src.hooks.policy_rbac import PolicyRbacHook
from src.hooks.sql_guard import SqlGuardHook
from src.tools.registry import ToolRegistry


def _dbt_exe() -> str:
    venv_dbt = Path(sys.executable).parent / "dbt"
    return str(venv_dbt) if venv_dbt.exists() else (shutil.which("dbt") or "dbt")


def _run_dbt_cli(project_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_dbt_exe(), "--no-version-check", *args, "--profiles-dir", str(project_dir)],
        cwd=str(project_dir),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def _create_external_saas_dbt_repo(base_dir: Path) -> Path:
    """Create an independent root-level dbt-duckdb repository completely outside workspace/."""
    repo = base_dir / "external_saas_dbt"
    (repo / "seeds").mkdir(parents=True)
    (repo / "models" / "staging").mkdir(parents=True)
    (repo / "models" / "marts").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)

    (repo / "dbt_project.yml").write_text(
        """name: saas_billing
version: "1.0.0"
config-version: 2
profile: saas_billing

model-paths: ["models"]
seed-paths: ["seeds"]
test-paths: ["tests"]
target-path: "target"

models:
  saas_billing:
    staging:
      +materialized: view
    marts:
      +materialized: table
""",
        encoding="utf-8",
    )
    (repo / "profiles.yml").write_text(
        """saas_billing:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: "warehouse.duckdb"
      threads: 2
""",
        encoding="utf-8",
    )

    # Seed includes a negative MRR bug on sub-3 so `dbt test` genuinely fails and emits real artifacts.
    (repo / "seeds" / "raw_subscriptions.csv").write_text(
        "subscription_id,customer_id,subscriber_email,billing_ssn,billing_date,mrr_usd\n"
        "sub-1,cust-1,ceo@acme.io,123-45-6789,2026-11-01,120.0\n"
        "sub-2,cust-2,ops@beta.io,987-65-4321,2026-11-01,80.0\n"
        "sub-3,cust-3,ceo@acme.io,555-55-5555,2026-11-02,-25.0\n",
        encoding="utf-8",
    )

    (repo / "models" / "staging" / "stg_subscriptions.sql").write_text(
        "SELECT\n"
        "    subscription_id,\n"
        "    customer_id,\n"
        "    subscriber_email,\n"
        "    billing_ssn,\n"
        "    CAST(billing_date AS VARCHAR) AS billing_date,\n"
        "    CAST(mrr_usd AS DOUBLE) AS mrr_usd\n"
        "FROM {{ ref('raw_subscriptions') }}\n",
        encoding="utf-8",
    )

    (repo / "models" / "marts" / "fct_mrr.sql").write_text(
        "SELECT\n"
        "    billing_date,\n"
        "    COUNT(*) AS subscription_count,\n"
        "    ANY_VALUE(subscriber_email) AS subscriber_email,\n"
        "    ANY_VALUE(billing_ssn) AS billing_ssn,\n"
        "    ROUND(SUM(mrr_usd), 2) AS total_mrr_usd\n"
        "FROM {{ ref('stg_subscriptions') }}\n"
        "GROUP BY billing_date\n",
        encoding="utf-8",
    )

    (repo / "models" / "schema.yml").write_text(
        """version: 2
models:
  - name: stg_subscriptions
    columns:
      - name: subscription_id
        tests:
          - not_null
          - unique
      - name: subscriber_email
        tags: ["pii"]
      - name: billing_ssn
        meta:
          pii: true
  - name: fct_mrr
    config:
      meta:
        partition_columns: ["billing_date"]
    columns:
      - name: billing_date
        meta:
          partition: true
        tests:
          - not_null
          - unique
      - name: subscriber_email
        tags: ["pii"]
      - name: billing_ssn
        meta:
          pii: true
""",
        encoding="utf-8",
    )

    (repo / "tests" / "assert_positive_mrr.sql").write_text(
        "SELECT billing_date, total_mrr_usd\n"
        "FROM {{ ref('fct_mrr') }}\n"
        "WHERE total_mrr_usd <= 0\n",
        encoding="utf-8",
    )

    seed_proc = _run_dbt_cli(repo, ["seed"])
    assert seed_proc.returncode == 0, seed_proc.stderr or seed_proc.stdout
    run_proc = _run_dbt_cli(repo, ["run"])
    assert run_proc.returncode == 0, run_proc.stderr or run_proc.stdout
    test_proc = _run_dbt_cli(repo, ["test"])
    assert test_proc.returncode != 0, "Expected intentionally broken MRR test to fail"

    return repo


def _make_state(role: str, resolution_mode: str | None = None, run_id: str = "run-ext-1") -> RunState:
    return RunState(
        run_id=run_id,
        incident=IncidentInput(
            incident_id="INC-EXT",
            title="Negative MRR in external dbt repo",
            summary="Investigate fct_mrr failure",
        ),
        user_context=UserContext(role=role, resolution_mode=resolution_mode),  # type: ignore[arg-type]
    )


def _sample_patch(target_path: str) -> str:
    return (
        f"--- a/{target_path}\n"
        f"+++ b/{target_path}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )


def test_external_dbt_repo_convention_discovery_manifest_overrides_and_alerts(
    tmp_path: Path,
) -> None:
    """Verify real external dbt repo discovery, Diagnosis-Only fallback, Fix Mode manifest, and alert convergence."""
    repo = _create_external_saas_dbt_repo(tmp_path)

    # 1. Manifest-less auto-discovery enters Diagnosis-Only Mode without leaking workspace/ samples.
    project = discover_project(repo)
    assert project.has_manifest is False
    assert project.default_resolution_mode == "diagnosis_only"
    assert project.dbt_project_rel == ""
    assert project.dbt_project_dir == repo.resolve()
    assert project.dbt_manifest_path == (repo / "target" / "manifest.json").resolve()
    assert project.dbt_run_results_path == (repo / "target" / "run_results.json").resolve()
    assert project.duckdb_path == (repo / "warehouse.duckdb").resolve()
    assert project.sql_dialect == "duckdb"

    assert set(project.discovered.mart_tables) == {"fct_mrr"}
    assert {"subscriber_email", "billing_ssn"}.issubset(set(project.discovered.pii_columns))
    assert project.discovered.partitioned_tables.get("fct_mrr") == ["billing_date"]
    assert "models/" in project.discovered.patchable_paths

    lineage = LineageAdapter.from_project(project)
    assert lineage.sql_dialect == "duckdb"
    upstream = lineage.query_lineage("fct_mrr", direction="upstream", depth=3)
    assert set(upstream.nodes.keys()) == {"raw_subscriptions", "stg_subscriptions", "fct_mrr"}
    impact = lineage.affected_downstream(["models/staging/stg_subscriptions.sql"])
    assert impact.dbt_models == ["stg_subscriptions", "fct_mrr"]
    assert _discover_mart_diff_targets(lineage) == [("fct_mrr", "billing_date")]

    diag_policies = resolve_policies(project, lineage=lineage)
    assert diag_policies.sql_guard.sql_dialect == "duckdb"
    assert diag_policies.roles["de"].resolution_mode == "diagnosis_only"
    assert diag_policies.roles["da"].resolution_mode == "diagnosis_only"
    assert diag_policies.roles["sre"].resolution_mode == "diagnosis_only"
    assert diag_policies.roles["da"].allowed_tables == ["fct_mrr"]
    assert "silver_orders" not in diag_policies.partitioned_tables
    assert diag_policies.partitioned_tables == {"fct_mrr": ["billing_date"]}
    assert "subscriber_email" in diag_policies.pii_columns
    assert "billing_ssn" in diag_policies.pii_columns

    # ADR-0004: ToolRegistry still exposes strictly the 6 orthogonal tools.
    paths = LakehousePaths.from_project(project)
    registry = ToolRegistry(
        workspace_dir=repo,
        lineage_adapter=lineage,
        duckdb_adapter=DuckDBAdapter(paths),
    )
    assert [t.name for t in registry.definitions()] == [
        "read_file",
        "grep_search",
        "query_lineage",
        "run_sql_readonly",
        "git_inspect",
        "apply_patch",
    ]

    # Governance enforcement on discovered rules.
    rbac = PolicyRbacHook(policies=diag_policies, workspace_dir=repo)
    sql_guard = SqlGuardHook(policies=diag_policies, workspace_dir=repo)
    redactor = PiiRedactorHook(policies=diag_policies)
    de_state = _make_state("de", resolution_mode="diagnosis_only")

    assert (
        rbac.on_pre_tool_use(
            de_state,
            ToolCall(
                id="p1",
                name="apply_patch",
                arguments={"patch": _sample_patch("models/marts/fct_mrr.sql")},
            ),
        ).action
        == "block"
    )
    assert (
        sql_guard.on_pre_tool_use(
            de_state,
            ToolCall(id="q1", name="run_sql_readonly", arguments={"sql": "SELECT * FROM fct_mrr LIMIT 10"}),
        ).action
        == "block"
    )
    allowed_sql = sql_guard.on_pre_tool_use(
        de_state,
        ToolCall(
            id="q2",
            name="run_sql_readonly",
            arguments={
                "sql": "SELECT billing_date, subscriber_email, billing_ssn, total_mrr_usd "
                "FROM fct_mrr WHERE billing_date = '2026-11-01'"
            },
        ),
    )
    assert allowed_sql.action == "modify"
    assert allowed_sql.modified_arguments is not None
    assert "LIMIT 50" in allowed_sql.modified_arguments["sql"].upper()

    # Run real read-only SQL probe via DuckDBAdapter + PiiRedactorHook on external warehouse.
    sql_res = registry.execute(
        ToolCall(id="q2", name="run_sql_readonly", arguments=allowed_sql.modified_arguments)
    )
    assert sql_res.is_error is False
    redacted_dec = redactor.on_post_tool_use(
        de_state,
        ToolCall(id="q2", name="run_sql_readonly", arguments=allowed_sql.modified_arguments),
        ToolResult(tool_call_id="q2", name="run_sql_readonly", output=sql_res.output),
    )
    assert redacted_dec.action == "modify" and redacted_dec.modified_output is not None
    assert "ceo@acme.io" not in redacted_dec.modified_output
    assert "123-45-6789" not in redacted_dec.modified_output

    # Oracle ignores missing sample GMV tables on external repos while still guarding external tests.
    wt_mock = tmp_path / "wt_mock"
    shutil.copytree(repo, wt_mock)
    oracle = DefaultIntegrityOracle()
    assert (
        oracle.verify(
            prod_workspace=repo,
            worktree_dir=wt_mock,
            shadow_silver_dir=tmp_path / "none",
            shadow_duckdb_path=repo / "warehouse.duckdb",
            changed_files=[],
            unified_diff="",
        ).passed
        is True
    )
    (wt_mock / "tests" / "assert_positive_mrr.sql").unlink()
    assert (
        oracle.verify(
            prod_workspace=repo,
            worktree_dir=wt_mock,
            shadow_silver_dir=tmp_path / "none",
            shadow_duckdb_path=repo / "warehouse.duckdb",
            changed_files=["tests/assert_positive_mrr.sql"],
            unified_diff="",
        ).passed
        is False
    )

    # 2. Adding OpenLineage FAIL + Project Manifest unlocks Fix Mode and custom overrides.
    (repo / "openlineage").mkdir(parents=True)
    ol_event = {
        "eventType": "FAIL",
        "eventTime": "2026-11-27T10:05:00Z",
        "run": {
            "runId": "ol-run-sub-1",
            "facets": {
                "errorMessage": {
                    "message": "Null constraint check failed in upstream ingest",
                    "programmingLanguage": "SQL",
                }
            },
        },
        "job": {"namespace": "duckdb://saas-billing", "name": "ingest_raw_subscriptions"},
        "inputs": [],
        "outputs": [{"namespace": "duckdb://saas-billing", "name": "main.raw_subscriptions"}],
    }
    (repo / "openlineage" / "events.jsonl").write_text(
        json.dumps(ol_event) + "\n", encoding="utf-8"
    )
    (repo / "dataops-harness.yaml").write_text(
        """version: 1
name: saas-billing
openlineage_paths:
  - openlineage/events.jsonl
shadow:
  backend: duckdb
  duckdb_path: warehouse.duckdb
governance:
  default_resolution_mode: fix
  pii_columns:
    - tax_id
  partitioned_tables:
    fct_mrr:
      - billing_date
      - tenant_id
""",
        encoding="utf-8",
    )

    fix_project = discover_project(repo)
    assert fix_project.has_manifest is True
    assert fix_project.default_resolution_mode == "fix"
    fix_lineage = LineageAdapter.from_project(fix_project)
    fix_policies = resolve_policies(fix_project, lineage=fix_lineage)

    assert fix_policies.roles["de"].resolution_mode == "fix"
    assert fix_policies.roles["da"].resolution_mode == "diagnosis_only"
    assert "tax_id" in fix_policies.pii_columns
    assert fix_policies.partitioned_tables["fct_mrr"] == ["billing_date", "tenant_id"]

    fix_rbac = PolicyRbacHook(policies=fix_policies, workspace_dir=repo)
    de_fix_state = _make_state("de", resolution_mode="fix")
    assert (
        fix_rbac.on_pre_tool_use(
            de_fix_state,
            ToolCall(
                id="p_ok",
                name="apply_patch",
                arguments={"patch": _sample_patch("models/marts/fct_mrr.sql")},
            ),
        ).action
        == "allow"
    )
    assert (
        fix_rbac.on_pre_tool_use(
            de_fix_state,
            ToolCall(
                id="p_manifest",
                name="apply_patch",
                arguments={"patch": _sample_patch("dataops-harness.yaml")},
            ),
        ).action
        == "block"
    )

    # 3. Alert & Incident normalization from real dbt run_results.json + OpenLineage FAIL + text.
    dbt_alerts = ingest_dbt_run_results(
        repo / "target" / "run_results.json",
        manifest=repo / "target" / "manifest.json",
    )
    assert len(dbt_alerts) >= 1
    assert any(a.asset == "fct_mrr" and a.alert_type == "dbt_test_failure" for a in dbt_alerts)

    ol_alerts = ingest_openlineage_alerts(repo / "openlineage" / "events.jsonl")
    assert len(ol_alerts) == 1
    assert ol_alerts[0].asset == "raw_subscriptions"
    assert ol_alerts[0].alert_type == "openlineage_fail"

    # Converge all three heterogeneous alert signals into one root-cause Incident.
    now_ts = datetime.now(UTC)
    ol_recent = [a.model_copy(update={"timestamp": now_ts}) for a in ol_alerts]
    dbt_recent = [a.model_copy(update={"timestamp": now_ts}) for a in dbt_alerts]
    text_recent = ingest_text_alert(
        "Finance reported negative total_mrr_usd in fct_mrr dashboard",
        lineage=fix_lineage,
        timestamp=now_ts,
    )
    assert [a.asset for a in text_recent] == ["fct_mrr"]

    all_alerts, incidents = ingest_and_converge(
        lineage=fix_lineage,
        raw_alerts=[*ol_recent, *dbt_recent, *text_recent],
        window_seconds=900,
    )
    assert len(all_alerts) == len(ol_recent) + len(dbt_recent) + len(text_recent)
    assert len(incidents) == 1
    assert incidents[0].root_cause_candidates == ["raw_subscriptions"]
    inc_input = incidents[0].to_incident_input()
    assert "raw_subscriptions" in inc_input.shared_ancestors


def test_openlineage_discovery_postgres_dialect_and_security_hooks(tmp_path: Path) -> None:
    """OpenLineage discovers lineage/PII/partitions and drives Postgres-dialect SQL & RBAC hooks."""
    repo = tmp_path / "pg_ol_repo"
    (repo / "openlineage").mkdir(parents=True)
    events = [
        {
            "eventType": "COMPLETE",
            "eventTime": "2026-11-27T08:00:00Z",
            "run": {"runId": "ol-1"},
            "producer": "https://github.com/OpenLineage/OpenLineage",
            "job": {
                "namespace": "postgres://prod-pg:5432/analytics",
                "name": "transform_stg_payments",
                "facets": {
                    "sourceCodeLocation": {"path": "sql/staging/stg_payments.sql"}
                },
            },
            "inputs": [
                {
                    "namespace": "postgres://prod-pg:5432/analytics",
                    "name": "analytics.raw.raw_payments",
                    "facets": {
                        "schema": {
                            "fields": [
                                {"name": "payment_id", "type": "varchar"},
                                {"name": "payer_email", "type": "varchar", "tags": ["pii"]},
                                {"name": "event_date", "type": "date"},
                            ]
                        }
                    },
                }
            ],
            "outputs": [
                {
                    "namespace": "postgres://prod-pg:5432/analytics",
                    "name": "analytics.staging.stg_payments",
                    "facets": {
                        "schema": {
                            "fields": [
                                {"name": "payment_id", "type": "varchar"},
                                {"name": "payer_email", "type": "varchar", "tags": ["pii"]},
                                {"name": "event_date", "type": "date"},
                            ]
                        }
                    },
                }
            ],
        },
        {
            "eventType": "COMPLETE",
            "eventTime": "2026-11-27T08:05:00Z",
            "run": {"runId": "ol-2"},
            "job": {
                "namespace": "postgres://prod-pg:5432/analytics",
                "name": "build_mart_revenue",
                "facets": {
                    "sourceCodeLocation": {"path": "sql/marts/mart_revenue.sql"}
                },
            },
            "inputs": [
                {
                    "namespace": "postgres://prod-pg:5432/analytics",
                    "name": "analytics.staging.stg_payments",
                }
            ],
            "outputs": [
                {
                    "namespace": "postgres://prod-pg:5432/analytics",
                    "name": "analytics.marts.mart_revenue",
                    "facets": {
                        "dataops": {"layer": "mart", "pii_columns": ["customer_iban"]},
                        "storage": {"partitionColumns": ["event_date"]},
                        "schema": {
                            "fields": [
                                {"name": "event_date", "type": "date"},
                                {"name": "account_id", "type": "bigint"},
                                {"name": "customer_iban", "type": "varchar"},
                                {"name": "revenue_usd", "type": "numeric"},
                            ]
                        },
                    },
                }
            ],
        },
    ]
    (repo / "openlineage" / "lineage.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    (repo / "dataops-harness.yaml").write_text(
        "version: 1\n"
        "name: pg-analytics\n"
        "shadow:\n"
        "  backend: postgres\n"
        "  connection_env: PG_DSN\n",
        encoding="utf-8",
    )

    assert infer_dialect_from_namespace("postgresql://db:5432/main") == "postgres"
    assert normalize_dataset_name("analytics.marts.mart_revenue") == "mart_revenue"
    assert normalize_sql_dialect("postgresql") == "postgres"

    project = discover_project(repo)
    assert project.sql_dialect == "postgres"
    assert project.discovered.mart_tables == ["mart_revenue"]
    assert project.discovered.partitioned_tables == {"mart_revenue": ["event_date"]}
    assert {"payer_email", "customer_iban"}.issubset(set(project.discovered.pii_columns))

    lineage = LineageAdapter.from_project(project)
    assert lineage.get_node("analytics.marts.mart_revenue").name == "mart_revenue"
    assert lineage.query_lineage("mart_revenue", "upstream", depth=3).nodes.keys() == {
        "raw_payments",
        "stg_payments",
        "mart_revenue",
    }

    policies = resolve_policies(project, lineage=lineage)
    assert policies.sql_guard.sql_dialect == "postgres"
    assert policies.roles["da"].allowed_tables == ["mart_revenue"]
    assert "silver_orders" not in policies.partitioned_tables

    sql_guard = SqlGuardHook(policies=policies, workspace_dir=repo)
    rbac = PolicyRbacHook(policies=policies, workspace_dir=repo)
    de_state = _make_state("de", resolution_mode="fix")
    da_state = _make_state("da")
    sre_state = _make_state("sre")

    # Postgres-specific type cast + INTERVAL syntax parses cleanly and rewrites LIMIT 50.
    pg_ok = sql_guard.on_pre_tool_use(
        de_state,
        ToolCall(
            id="pg-1",
            name="run_sql_readonly",
            arguments={
                "sql": "SELECT account_id::TEXT, revenue_usd FROM mart_revenue "
                "WHERE event_date >= CURRENT_DATE - INTERVAL '7 days'"
            },
        ),
    )
    assert pg_ok.action == "modify"
    assert pg_ok.modified_arguments is not None
    assert "LIMIT 50" in pg_ok.modified_arguments["sql"].upper()

    # Postgres EXPLAIN (ANALYZE, BUFFERS) requires partition predicate, non-ANALYZE does not.
    assert (
        sql_guard.on_pre_tool_use(
            de_state,
            ToolCall(
                id="pg-exp-bad",
                name="run_sql_readonly",
                arguments={"sql": "EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM mart_revenue"},
            ),
        ).action
        == "block"
    )
    assert (
        sql_guard.on_pre_tool_use(
            de_state,
            ToolCall(
                id="pg-exp-ok",
                name="run_sql_readonly",
                arguments={
                    "sql": "EXPLAIN (ANALYZE, BUFFERS) SELECT COUNT(*) FROM mart_revenue "
                    "WHERE event_date = '2026-11-27'"
                },
            ),
        ).action
        == "allow"
    )

    # Postgres server escape functions & COPY PROGRAM are blocked.
    for bad_sql in (
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT lo_import('/etc/passwd')",
        "COPY mart_revenue TO PROGRAM 'id'",
        "SELECT dblink('host=evil', 'SELECT 1')",
    ):
        assert (
            sql_guard.on_pre_tool_use(
                de_state, ToolCall(id="pg-esc", name="run_sql_readonly", arguments={"sql": bad_sql})
            ).action
            == "block"
        ), bad_sql

    # DA mart_only + SRE PII aggregation restrictions use discovered Postgres tables & PII columns.
    assert (
        rbac.on_pre_tool_use(
            da_state,
            ToolCall(
                id="da-bad",
                name="run_sql_readonly",
                arguments={"sql": "SELECT * FROM stg_payments WHERE event_date = '2026-11-27' LIMIT 5"},
            ),
        ).action
        == "block"
    )
    assert (
        rbac.on_pre_tool_use(
            da_state,
            ToolCall(
                id="da-ok",
                name="run_sql_readonly",
                arguments={
                    "sql": "SELECT event_date, revenue_usd FROM analytics.marts.mart_revenue "
                    "WHERE event_date = '2026-11-27' LIMIT 5"
                },
            ),
        ).action
        == "allow"
    )
    assert (
        rbac.on_pre_tool_use(
            sre_state,
            ToolCall(
                id="sre-pii",
                name="run_sql_readonly",
                arguments={
                    "sql": "SELECT customer_iban, COUNT(*) FROM mart_revenue "
                    "WHERE event_date = '2026-11-27' GROUP BY customer_iban"
                },
            ),
        ).action
        == "block"
    )


def test_boundary_errors_and_dynamic_spark_topological_order(tmp_path: Path) -> None:
    """Validate boundary traversal checks, OOM severity, dynamic Spark order, and metric rules."""
    # 1. Manifest path traversal rejected at boundary.
    bad_repo = tmp_path / "bad_repo"
    bad_repo.mkdir()
    (bad_repo / "dataops-harness.yaml").write_text(
        "version: 1\ndbt_project_dir: ../../outside\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="escapes repository root"):
        discover_project(bad_repo)

    # 2. Malformed OpenLineage JSONL rejected at boundary.
    with pytest.raises(ValueError, match="Malformed OpenLineage"):
        load_openlineage_events('{"eventType": "FAIL"}\n{not valid json')

    # 3. Critical severity for OpenLineage OOM FAIL events even when outputs is empty.
    oom_alerts = ingest_openlineage_alerts(
        {
            "eventType": "FAIL",
            "eventTime": "2026-11-27T11:00:00Z",
            "run": {
                "runId": "r-oom",
                "facets": {
                    "errorMessage": {
                        "message": "java.lang.OutOfMemoryError: Java heap space"
                    }
                },
            },
            "job": {"namespace": "spark://cluster", "name": "custom_silver_step"},
            "inputs": [],
            "outputs": [],
        }
    )
    assert len(oom_alerts) == 1
    assert oom_alerts[0].severity == "critical"
    assert oom_alerts[0].asset == "custom_silver_step"

    # 4. Custom Spark jobs order purely by lineage graph (zero hardcoded sample names).
    spark_repo = tmp_path / "spark_repo"
    (spark_repo / "spark_jobs").mkdir(parents=True)
    (spark_repo / "spark_jobs" / "lineage.yaml").write_text(
        """assets:
  src_events:
    engine: raw
    layer: raw
    upstream: []
  step_one:
    engine: spark
    layer: silver
    job_path: spark_jobs/custom_step_one.py
    upstream: [src_events]
  step_two:
    engine: spark
    layer: silver
    job_path: spark_jobs/custom_step_two.py
    upstream: [step_one]
""",
        encoding="utf-8",
    )
    custom_lineage = LineageAdapter.from_workspace(spark_repo)
    assert order_spark_jobs(
        ["spark_jobs/custom_step_two.py", "spark_jobs/custom_step_one.py"],
        lineage_adapter=custom_lineage,
    ) == ["spark_jobs/custom_step_one.py", "spark_jobs/custom_step_two.py"]

    # 5. Extract OpenLineage graph directly & validate custom MetricReconciliationRule.
    ol_graph = extract_openlineage_lineage(
        [
            {
                "eventType": "COMPLETE",
                "job": {"namespace": "duckdb://db", "name": "job_a"},
                "inputs": [{"name": "raw_t"}],
                "outputs": [{"name": "fct_t"}],
            }
        ]
    )
    assert ol_graph.nodes["fct_t"].upstream == ["raw_t"]
    assert MetricReconciliationRule(
        left_table="a", left_column="x", right_table="b", right_column="y"
    ).tolerance == 0.01
