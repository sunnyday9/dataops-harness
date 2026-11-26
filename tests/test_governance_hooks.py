from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb

from src.adapters.base import LakehousePaths
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.model_adapter import FakeReplayModelAdapter, TurnResponse
from src.core.context import (
    IncidentInput,
    PrincipalRole,
    RunState,
    ToolCall,
    ToolResult,
    UserContext,
)
from src.core.loop import HarnessLoop
from src.core.policy import load_policies
from src.hooks.base import HookPipeline
from src.hooks.pii_redactor import PiiRedactorHook
from src.hooks.policy_rbac import PolicyRbacHook
from src.hooks.sql_guard import SqlGuardHook
from src.tools.registry import ToolRegistry


def _make_state(role: PrincipalRole, run_id: str = "run-gov-1") -> RunState:
    return RunState(
        run_id=run_id,
        incident=IncidentInput(
            incident_id="INC-GOV",
            title="GMV discrepancy",
            summary="Daily GMV reconciliation failed",
        ),
        user_context=UserContext(role=role),
    )


def _sample_patch(target_path: str) -> str:
    return (
        f"--- a/{target_path}\n"
        f"+++ b/{target_path}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )


def test_rbac_matrix_across_de_da_and_sre_roles(tmp_path: Path) -> None:
    """Verify exact de/da/sre matrix across apply_patch, Spark logs, skills, and SQL scopes."""
    policies = load_policies()
    hook = PolicyRbacHook(policies=policies, workspace_dir=tmp_path)

    de_state = _make_state("de")
    da_state = _make_state("da")
    sre_state = _make_state("sre")

    # 1. Patch scope + resolution_mode
    spark_patch = ToolCall(
        id="p1",
        name="apply_patch",
        arguments={"patch": _sample_patch("spark_jobs/build_silver_orders.py")},
    )
    dbt_patch = ToolCall(
        id="p2",
        name="apply_patch",
        arguments={"patch": _sample_patch("dbt_project/models/marts/fct_daily_gmv.sql")},
    )
    incidents_patch = ToolCall(
        id="p3",
        name="apply_patch",
        arguments={"patch": _sample_patch("incidents/note.md")},
    )
    traversal_patch = ToolCall(
        id="p4",
        name="apply_patch",
        arguments={"patch": _sample_patch("../spark_jobs/escape.py")},
    )

    assert hook.on_pre_tool_use(de_state, spark_patch).action == "allow"
    assert hook.on_pre_tool_use(de_state, dbt_patch).action == "allow"
    assert hook.on_pre_tool_use(de_state, incidents_patch).action == "block"
    assert hook.on_pre_tool_use(de_state, traversal_patch).action == "block"

    assert hook.on_pre_tool_use(da_state, spark_patch).action == "block"
    assert hook.on_pre_tool_use(da_state, dbt_patch).action == "block"

    assert hook.on_pre_tool_use(sre_state, spark_patch).action == "allow"
    assert hook.on_pre_tool_use(sre_state, dbt_patch).action == "block"

    # 2. Spark event logs vs skills/*.md and workspace source files
    spark_log_read = ToolCall(
        id="r1",
        name="read_file",
        arguments={"path": "logs/spark_events/app-123.json"},
    )
    external_spark_log_read = ToolCall(
        id="r2",
        name="read_file",
        arguments={"path": "/tmp/spark-events/app-123"},
    )
    spark_log_grep = ToolCall(
        id="r3",
        name="grep_search",
        arguments={"pattern": "OOM", "path": "logs/spark_events"},
    )
    skill_read = ToolCall(
        id="r4",
        name="read_file",
        arguments={"path": "skills/rca_playbook.md"},
    )
    spark_code_read = ToolCall(
        id="r5",
        name="read_file",
        arguments={"path": "spark_jobs/build_silver_orders.py"},
    )

    for call in (spark_log_read, external_spark_log_read, spark_log_grep, skill_read):
        assert hook.on_pre_tool_use(de_state, call).action == "allow"
        assert hook.on_pre_tool_use(sre_state, call).action == "allow"

    assert hook.on_pre_tool_use(da_state, spark_log_read).action == "block"
    assert hook.on_pre_tool_use(da_state, external_spark_log_read).action == "block"
    assert hook.on_pre_tool_use(da_state, spark_log_grep).action == "block"
    assert hook.on_pre_tool_use(da_state, skill_read).action == "allow"
    assert hook.on_pre_tool_use(da_state, spark_code_read).action == "allow"

    # 3. DA mart_only SQL scope
    da_allowed = [
        "SELECT * FROM fct_daily_gmv WHERE order_date = '2026-01-01' LIMIT 10",
        "SELECT * FROM mart.main.dim_user_ltv LIMIT 10",
    ]
    da_blocked = [
        "SELECT * FROM silver_orders WHERE order_date = '2026-01-01' LIMIT 10",
        "SELECT * FROM stg_orders LIMIT 10",
        (
            "WITH c AS (SELECT * FROM silver_orders WHERE order_date = '2026-01-01') "
            "SELECT * FROM fct_daily_gmv WHERE order_date = '2026-01-01'"
        ),
        "EXPLAIN SELECT * FROM silver_orders",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM read_parquet('workspace/lakehouse/prod/silver/silver_orders/*.parquet')",
    ]
    for sql in da_allowed:
        assert hook.on_pre_tool_use(
            da_state,
            ToolCall(id="q", name="run_sql_readonly", arguments={"sql": sql}),
        ).action == "allow", sql
    for sql in da_blocked:
        assert hook.on_pre_tool_use(
            da_state,
            ToolCall(id="q", name="run_sql_readonly", arguments={"sql": sql}),
        ).action == "block", sql

    # 4. SRE aggregate_or_metadata_only SQL mode
    sre_allowed = [
        "DESCRIBE silver_orders",
        "DESCRIBE HISTORY silver_orders",
        "EXPLAIN SELECT * FROM silver_orders",
        "EXPLAIN FORMATTED SELECT * FROM silver_orders",
        "SELECT COUNT(*) AS cnt, SUM(amount_usd) FROM silver_orders WHERE order_date = '2026-01-01'",
        (
            "SELECT order_date, COUNT(*) AS cnt FROM silver_orders "
            "WHERE order_date = '2026-01-01' GROUP BY order_date"
        ),
        (
            "SELECT (SELECT COUNT(*) FROM silver_orders WHERE order_date = '2026-01-01') AS s_cnt, "
            "COUNT(*) FROM fct_daily_gmv WHERE order_date = '2026-01-01'"
        ),
        (
            "SELECT COUNT(*) FROM silver_orders WHERE order_date = '2026-01-01' "
            "UNION SELECT COUNT(*) FROM fct_daily_gmv WHERE order_date = '2026-01-01'"
        ),
        "EXPLAIN ANALYZE SELECT COUNT(*) FROM silver_orders WHERE order_date = '2026-01-01'",
    ]
    sre_blocked = [
        "SELECT * FROM silver_orders WHERE order_date = '2026-01-01' LIMIT 10",
        "SELECT order_id, amount_usd FROM silver_orders WHERE order_date = '2026-01-01'",
        "SELECT DISTINCT order_date FROM silver_orders",
        "SELECT order_id FROM silver_orders WHERE order_date = '2026-01-01' GROUP BY order_id",
        (
            "SELECT customer_email, COUNT(*) FROM silver_orders "
            "WHERE order_date = '2026-01-01' GROUP BY customer_email"
        ),
        (
            "WITH c AS (SELECT customer_email AS contact FROM silver_orders "
            "WHERE order_date = '2026-01-01') "
            "SELECT contact, COUNT(*) FROM c GROUP BY contact"
        ),
        (
            "SELECT (SELECT customer_email FROM silver_orders WHERE order_date = '2026-01-01' "
            "LIMIT 1), COUNT(*) FROM fct_daily_gmv WHERE order_date = '2026-01-01'"
        ),
        "EXPLAIN ANALYZE SELECT * FROM silver_orders WHERE order_date = '2026-01-01'",
    ]
    for sql in sre_allowed:
        assert hook.on_pre_tool_use(
            sre_state,
            ToolCall(id="q", name="run_sql_readonly", arguments={"sql": sql}),
        ).action == "allow", sql
    for sql in sre_blocked:
        assert hook.on_pre_tool_use(
            sre_state,
            ToolCall(id="q", name="run_sql_readonly", arguments={"sql": sql}),
        ).action == "block", sql


def test_sql_guard_ast_allowlist_escapes_partitions_and_limit(tmp_path: Path) -> None:
    """Validate SQLGlot AST allowlist, DuckDB escape blocking, partition rules, and LIMIT 50."""
    ws = tmp_path / "workspace"
    (ws / "lakehouse" / "prod" / "silver" / "silver_orders").mkdir(parents=True)
    guard = SqlGuardHook(policies=load_policies(), workspace_dir=ws)
    state = _make_state("de")

    def eval_sql(sql: str):
        return guard.on_pre_tool_use(
            state, ToolCall(id="sql", name="run_sql_readonly", arguments={"sql": sql})
        )

    # Blocked statements, empty, multi-statement, and DuckDB filesystem escapes
    blocked_sqls = [
        "",
        "   ",
        "SELECT FROM WHERE",
        "SELECT 1; DROP TABLE fct_daily_gmv",
        "INSERT INTO fct_daily_gmv VALUES ('2026-01-01', 1.0)",
        "UPDATE fct_daily_gmv SET gmv_usd = 0 WHERE order_date = '2026-01-01'",
        "DELETE FROM fct_daily_gmv WHERE order_date = '2026-01-01'",
        "DROP TABLE fct_daily_gmv",
        "CREATE TABLE x (a INT)",
        "ALTER TABLE fct_daily_gmv ADD COLUMN z INT",
        "COPY fct_daily_gmv TO '/tmp/out.parquet'",
        "ATTACH '/tmp/other.duckdb' AS ext",
        "DETACH mart",
        "PRAGMA table_info('fct_daily_gmv')",
        "INSTALL httpfs",
        "LOAD delta",
        "SET threads = 4",
        "CALL duckdb_tables()",
        "SELECT * INTO temp_tbl FROM dim_user_ltv",
        "SUMMARIZE dim_user_ltv",
        "SELECT read_text('/etc/passwd')",
        "SELECT read_blob('/etc/hosts')",
        "SELECT * FROM glob('/etc/*')",
        "SELECT * FROM read_parquet('/etc/passwd')",
        "SELECT * FROM read_csv('workspace/lakehouse/../../etc/passwd')",
        "SELECT * FROM '/etc/passwd'",
        "SELECT * FROM dim_user_ltv LIMIT 51",
        "SELECT * FROM dim_user_ltv LIMIT 100",
    ]
    for sql in blocked_sqls:
        dec = eval_sql(sql)
        assert dec.action == "block", f"Expected block for {sql!r}, got {dec}"

    # Partition enforcement & exemptions
    partition_blocked = [
        "SELECT * FROM silver_orders LIMIT 10",
        "SELECT order_id FROM fct_daily_gmv LIMIT 10",
        "SELECT * FROM silver_orders WHERE order_date = order_date LIMIT 10",
        "SELECT * FROM silver_orders WHERE order_date != '2026-01-01' LIMIT 10",
        "SELECT * FROM silver_orders WHERE order_date IS NOT NULL LIMIT 10",
        "SELECT * FROM silver_orders WHERE order_date = '2026-01-01' OR 1 = 1 LIMIT 10",
        "SELECT * FROM silver_orders o JOIN dim_users u ON o.order_date = u.signup_date LIMIT 10",
        (
            "SELECT * FROM silver_orders o LEFT JOIN dim_users u "
            "ON o.order_date = u.signup_date WHERE u.signup_date = '2026-01-01' OR 1 = 1 LIMIT 10"
        ),
        "EXPLAIN ANALYZE SELECT * FROM silver_orders",
    ]
    for sql in partition_blocked:
        dec = eval_sql(sql)
        assert dec.action == "block", f"Expected partition block for {sql!r}, got {dec}"

    partition_allowed_verbatim = [
        "SELECT * FROM silver_orders WHERE order_date = '2026-01-01' LIMIT 10",
        "SELECT * FROM silver_orders WHERE order_date BETWEEN '2026-01-01' AND '2026-01-02' LIMIT 20",
        "SELECT * FROM fct_daily_gmv WHERE order_date IN ('2026-01-01', '2026-01-02') LIMIT 50",
        (
            "SELECT * FROM silver_orders o JOIN dim_users u ON o.order_date = u.signup_date "
            "WHERE u.signup_date = '2026-01-01' LIMIT 15"
        ),
        "EXPLAIN SELECT * FROM silver_orders",
        "EXPLAIN FORMATTED SELECT * FROM fct_daily_gmv",
        "DESCRIBE silver_orders",
        "SELECT MIN(order_date), MAX(order_date) FROM silver_orders",
        "SELECT COUNT(*) FROM silver_orders WHERE order_date = '2026-01-01'",
    ]
    for sql in partition_allowed_verbatim:
        dec = eval_sql(sql)
        assert dec.action == "allow", f"Expected allow for {sql!r}, got {dec}"

    # Default policies block read_parquet; when unblocked, lakehouse containment still applies.
    assert eval_sql(
        "SELECT * FROM read_parquet('workspace/lakehouse/prod/silver/silver_orders/*.parquet') "
        "WHERE order_date = '2026-01-01' LIMIT 10"
    ).action == "block"
    relaxed = load_policies().model_copy(deep=True)
    relaxed.sql_guard.blocked_functions = ["read_text", "read_blob", "glob"]
    relaxed_guard = SqlGuardHook(policies=relaxed, workspace_dir=ws)
    assert relaxed_guard.on_pre_tool_use(
        state,
        ToolCall(
            id="lh-ok",
            name="run_sql_readonly",
            arguments={
                "sql": "SELECT * FROM read_parquet('workspace/lakehouse/prod/silver/silver_orders/*.parquet') "
                "WHERE order_date = '2026-01-01' LIMIT 10"
            },
        ),
    ).action == "allow"
    assert relaxed_guard.on_pre_tool_use(
        state,
        ToolCall(
            id="lh-bad",
            name="run_sql_readonly",
            arguments={"sql": "SELECT * FROM read_parquet('/etc/passwd') LIMIT 10"},
        ),
    ).action == "block"

    # Pure partition-discovery query allowed (rewritten only for LIMIT if omitted)
    discovery = eval_sql("SELECT DISTINCT order_date FROM silver_orders")
    assert discovery.action in {"allow", "modify"}
    if discovery.action == "modify":
        assert "LIMIT 50" in discovery.modified_arguments["sql"].upper()

    # Non-aggregate SELECT & UNION without LIMIT are rewritten to LIMIT 50
    for unbounded_sql in [
        "SELECT * FROM dim_user_ltv",
        "SELECT * FROM silver_orders WHERE order_date = '2026-01-01'",
        "WITH d AS (SELECT * FROM dim_user_ltv) SELECT * FROM d",
        (
            "SELECT order_date FROM silver_orders WHERE order_date = '2026-01-01' "
            "UNION SELECT order_date FROM fct_daily_gmv WHERE order_date = '2026-01-01'"
        ),
    ]:
        dec = eval_sql(unbounded_sql)
        assert dec.action == "modify", f"Expected LIMIT rewrite for {unbounded_sql!r}, got {dec}"
        assert dec.modified_arguments is not None
        assert "LIMIT 50" in dec.modified_arguments["sql"].upper()


def test_pii_redactor_two_layer_deterministic_and_run_unlinkable() -> None:
    """Verify column-tag + regex redaction, within-run consistency, cross-run unlinkability, and idempotency."""
    redactor = PiiRedactorHook(policies=load_policies())
    run1_a = _make_state("de", run_id="run-1")
    run1_b = _make_state("de", run_id="run-2")

    sql_json = json.dumps(
        {
            "sql": "SELECT customer_email, phone, national_id, notes FROM silver_orders "
            "WHERE customer_email = 'Alice@Example.com' LIMIT 10",
            "columns": ["customer_email", "phone", "national_id", "notes", "created_at"],
            "rows": [
                [
                    "alice@example.com",
                    "13812345678",
                    "11010519900101123x",
                    "backup contact: bob@corp.org, phone 415-555-0199, ssn 123-45-6789",
                    "2026-01-15 12:30:45",
                ],
                [
                    "Alice@Example.com",
                    "+86-138-1234-5678",
                    "11010519900101123X",
                    "repeat mail alice@example.com on 2026-01-15",
                    "2026-01-15 13:00:00",
                ],
            ],
            "row_count": 2,
        }
    )

    call = ToolCall(id="c1", name="run_sql_readonly", arguments={"sql": "SELECT 1"})
    dec1 = redactor.on_post_tool_use(
        run1_a, call, ToolResult(tool_call_id="c1", name="run_sql_readonly", output=sql_json)
    )
    assert dec1.action == "modify" and dec1.modified_output is not None

    # Zero plaintext PII anywhere
    for secret in [
        "alice@example.com",
        "Alice@Example.com",
        "bob@corp.org",
        "13812345678",
        "415-555-0199",
        "11010519900101123x",
        "11010519900101123X",
        "123-45-6789",
    ]:
        assert secret not in dec1.modified_output

    parsed1 = json.loads(dec1.modified_output)
    assert parsed1["rows"][0][0] == parsed1["rows"][1][0]
    assert re.fullmatch(r"<EMAIL_[0-9a-f]{8}>", parsed1["rows"][0][0])
    assert re.fullmatch(r"<PHONE_[0-9a-f]{8}>", parsed1["rows"][0][1])
    assert re.fullmatch(r"<PHONE_[0-9a-f]{8}>", parsed1["rows"][1][1])
    assert parsed1["rows"][0][2] == parsed1["rows"][1][2]
    assert re.fullmatch(r"<ID_[0-9a-f]{8}>", parsed1["rows"][0][2])
    assert parsed1["rows"][0][0] in parsed1["rows"][1][3]
    # Normal timestamps and ISO dates remain intact
    assert parsed1["rows"][0][4] == "2026-01-15 12:30:45"
    assert "on 2026-01-15" in parsed1["rows"][1][3]

    # Free-text Layer 2 consistency with Layer 1 inside the same run
    log_text = "User alice@example.com failed auth with ID 11010519900101123X from 13812345678."
    dec_log = redactor.on_post_tool_use(
        run1_a,
        ToolCall(id="c2", name="read_file", arguments={"path": "logs/app.log"}),
        ToolResult(tool_call_id="c2", name="read_file", output=log_text),
    )
    assert dec_log.action == "modify" and dec_log.modified_output is not None
    assert parsed1["rows"][0][0] in dec_log.modified_output
    assert parsed1["rows"][0][2] in dec_log.modified_output
    assert parsed1["rows"][0][1] in dec_log.modified_output

    # Idempotency: running redactor a second time returns allow without mutating placeholders
    dec_idem = redactor.on_post_tool_use(
        run1_a, call, ToolResult(tool_call_id="c1", name="run_sql_readonly", output=dec1.modified_output)
    )
    assert dec_idem.action == "allow"

    # Cross-run unlinkability: a distinct Run produces distinct HMAC pseudonyms
    dec2 = redactor.on_post_tool_use(
        run1_b, call, ToolResult(tool_call_id="c1", name="run_sql_readonly", output=sql_json)
    )
    assert dec2.action == "modify" and dec2.modified_output is not None
    parsed2 = json.loads(dec2.modified_output)
    assert parsed2["rows"][0][0] != parsed1["rows"][0][0]
    assert parsed2["rows"][0][2] != parsed1["rows"][0][2]


def test_e2e_harness_loop_governance_rbac_sql_guard_and_pii_redaction(tmp_path: Path) -> None:
    """Full HarnessLoop integration proving live RBAC blocking, LIMIT 50 rewrite, and PII redaction."""
    ws = tmp_path / "workspace"
    (ws / "spark_jobs").mkdir(parents=True)
    (ws / "spark_jobs" / "job.py").write_text("x = 1\n", encoding="utf-8")
    (ws / "dbt_project" / "models" / "marts").mkdir(parents=True)
    (ws / "dbt_project" / "models" / "marts" / "fct_daily_gmv.sql").write_text(
        "SELECT 1\n", encoding="utf-8"
    )
    (ws / "logs" / "spark_events").mkdir(parents=True)
    (ws / "logs" / "spark_events" / "event.log").write_text(
        "spark stage failed\n", encoding="utf-8"
    )

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "da_guide.md").write_text("# DA Playbook\n", encoding="utf-8")

    paths = LakehousePaths.from_workspace(ws)
    paths.prod_mart_dir.mkdir(parents=True)
    con = duckdb.connect(str(paths.prod_duckdb_path))
    try:
        con.execute(
            "CREATE TABLE fct_daily_gmv (order_date VARCHAR, customer_email VARCHAR, "
            "phone VARCHAR, national_id VARCHAR, gmv_usd DOUBLE)"
        )
        rows = [
            (
                "2026-01-01",
                "alice@example.com" if i % 2 == 0 else "bob@example.com",
                "13812345678",
                "110105199001011234",
                100.0 + i,
            )
            for i in range(60)
        ]
        con.executemany("INSERT INTO fct_daily_gmv VALUES (?, ?, ?, ?, ?)", rows)
    finally:
        con.close()

    policies = load_policies()
    registry = ToolRegistry(
        workspace_dir=ws,
        skills_dir=skills_dir,
        duckdb_adapter=DuckDBAdapter(paths),
    )
    hooks = HookPipeline(
        pre_tool_hooks=[
            PolicyRbacHook(policies=policies, workspace_dir=ws),
            SqlGuardHook(policies=policies, workspace_dir=ws),
        ],
        post_tool_hooks=[
            PiiRedactorHook(policies=policies),
        ],
    )

    model = FakeReplayModelAdapter(
        [
            TurnResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="apply_patch",
                        arguments={"patch": _sample_patch("dbt_project/models/marts/fct_daily_gmv.sql")},
                    ),
                    ToolCall(
                        id="c2",
                        name="read_file",
                        arguments={"path": "logs/spark_events/event.log"},
                    ),
                    ToolCall(
                        id="c3",
                        name="read_file",
                        arguments={"path": "skills/da_guide.md"},
                    ),
                    ToolCall(
                        id="c4",
                        name="run_sql_readonly",
                        arguments={
                            "sql": "SELECT customer_email, phone, national_id, gmv_usd "
                            "FROM fct_daily_gmv WHERE order_date = '2026-01-01'"
                        },
                    ),
                ]
            ),
            TurnResponse(
                content="RCA Report: DA verified fct_daily_gmv anomaly using redacted mart rows."
            ),
        ]
    )

    loop = HarnessLoop(
        model_adapter=model,
        tool_registry=registry,
        hooks=hooks,
        runs_dir=tmp_path / "runs",
    )
    incident = IncidentInput(
        incident_id="INC-E2E-5",
        title="GMV mismatch",
        summary="Investigate fct_daily_gmv",
    )
    state = loop.run(incident, UserContext(role="da"), run_id="e2e-gov-run")

    assert state.status == "DIAGNOSED"
    tool_msgs = [m for m in state.messages if m.role == "tool"]
    assert "diagnosis_only" in tool_msgs[0].content
    assert "Spark event logs" in tool_msgs[1].content
    assert "# DA Playbook" in tool_msgs[2].content

    sql_payload = json.loads(tool_msgs[3].content)
    assert sql_payload["row_count"] == 50  # Missing LIMIT rewritten to LIMIT 50
    assert sql_payload["rows"][0][0] == sql_payload["rows"][2][0]  # Duplicate Alice pseudonym match
    assert sql_payload["rows"][0][0] != sql_payload["rows"][1][0]  # Distinct Bob pseudonym

    serialized_requests = json.dumps(
        [req.model_dump(mode="json") for req in model.recorded_requests]
    )
    for raw_pii in ("alice@example.com", "bob@example.com", "13812345678", "110105199001011234"):
        assert raw_pii not in serialized_requests
