from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from benchmarks.ablation_runner import (
    AblationBenchmarkRunner,
    ArmConfig,
    _aggregate_usage,
    _ArmToolRegistry,
    _clone_git_workspace,
)
from benchmarks.scenarios.base import InjectionOutcome
from src.adapters.base import LakehousePaths
from src.adapters.model_adapter import (
    FakeReplayModelAdapter,
    PricingConfig,
    ProviderConfig,
    TurnResponse,
)
from src.core.context import (
    CheckpointStore,
    IncidentInput,
    Message,
    RunState,
    ToolCall,
    ToolResult,
    UsageMetrics,
    UserContext,
)
from src.core.loop import HarnessLoop
from src.core.policy import LimitsPolicy, load_policies
from src.core.promotion import PromotionService
from src.core.rca import ApprovalPacket, RcaReport, ShadowVerificationResult, parse_rca_report
from src.core.shadow_verifier import DefaultIntegrityOracle
from src.core.worktree import RunWorktreeManager
from src.hooks.anti_thrash import ANTI_THRASH_METADATA_KEY, AntiThrashHook
from src.hooks.base import HookPipeline
from src.hooks.compactor import ContextCompactorHook
from src.hooks.pii_redactor import PiiRedactorHook
from src.hooks.policy_rbac import PolicyRbacHook
from src.hooks.sql_guard import SqlGuardHook
from src.hooks.stop_gate import StopGateHook
from src.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_WORKSPACE = REPO_ROOT / "workspace"
SKILLS_DIR = REPO_ROOT / "skills"


def _make_state(
    role: str = "de",
    *,
    run_id: str = "run-interrogate",
    resolution_mode: str = "fix",
    max_turns: int = 20,
) -> RunState:
    return RunState(
        run_id=run_id,
        incident=IncidentInput(
            incident_id="INC-INT-1",
            title="Regression test incident",
            summary="Verify security and lifecycle invariants.",
        ),
        user_context=UserContext(role=role, resolution_mode=resolution_mode),  # type: ignore[arg-type]
        max_turns=max_turns,
    )


def _init_git_repo(ws: Path, files: dict[str, str], *, message: str = "baseline") -> str:
    for rel, content in files.items():
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=ws, check=True, capture_output=True)
    return subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _valid_rca_json(incident_id: str = "INC-INT-1", patched_files: list[str] | None = None) -> str:
    return json.dumps(
        {
            "incident_id": incident_id,
            "root_cause_summary": "Verified root cause summary exceeding minimum length.",
            "affected_assets": ["fct_daily_gmv"],
            "evidence": ["Reconciled upstream silver and dbt mart metrics."],
            "engine": "dbt",
            "proposed_fix_summary": "Applied minimal surgical fix.",
            "patched_files": patched_files if patched_files is not None else [],
            "confidence": "high",
        }
    )


def test_f1_sql_guard_and_rbac_block_escape_functions(tmp_path: Path) -> None:
    """F1: SqlGuardHook and PolicyRbacHook block DuckDB escape/metadata table functions."""
    policies = load_policies()
    guard = SqlGuardHook(policies=policies, workspace_dir=tmp_path)
    rbac = PolicyRbacHook(policies=policies, workspace_dir=tmp_path)
    de_state = _make_state("de")
    da_state = _make_state("da", resolution_mode="diagnosis_only")

    escape_sqls = [
        "SELECT * FROM query('SELECT 1')",
        "SELECT * FROM query_table('silver_orders')",
        "SELECT * FROM json_execute_serialized_sql(json_serialize_sql('SELECT 1'))",
        "SELECT * FROM sniff_csv('workspace/lakehouse/raw/users/users.csv')",
        "SELECT * FROM pragma_table_info('silver_orders')",
        "SELECT * FROM duckdb_tables()",
        "SELECT * FROM st_read('/etc/passwd')",
        "SELECT current_setting('access_mode')",
    ]
    for sql in escape_sqls:
        call = ToolCall(id="f1", name="run_sql_readonly", arguments={"sql": sql})
        assert guard.on_pre_tool_use(de_state, call).action == "block", sql
        assert rbac.on_pre_tool_use(da_state, call).action == "block", sql


def test_f2_git_inspect_and_apply_patch_hardening(tmp_path: Path) -> None:
    """F2: Block dangerous git flags/paths and rename/symlink apply_patch escapes."""
    ws = tmp_path / "ws"
    _init_git_repo(
        ws,
        {
            "spark_jobs/job.py": "x = 1\n",
            "dbt_project/models/m.sql": "SELECT 1\n",
        },
    )
    policies = load_policies()
    rbac = PolicyRbacHook(policies=policies, workspace_dir=ws)
    reg = ToolRegistry(ws)
    sre_state = _make_state("sre")

    bad_git_calls = [
        ToolCall(
            id="g1",
            name="git_inspect",
            arguments={"subcommand": "diff", "args": ["--no-index", "spark_jobs/job.py", "dbt_project/models/m.sql"]},
        ),
        ToolCall(id="g2", name="git_inspect", arguments={"subcommand": "log", "args": ["--output=/tmp/x"]}),
        ToolCall(id="g3", name="git_inspect", arguments={"subcommand": "log", "args": ["-o", "/tmp/x"]}),
        ToolCall(id="g4", name="git_inspect", arguments={"subcommand": "log", "args": ["-c", "core.pager=cat"]}),
        ToolCall(id="g5", name="git_inspect", arguments={"subcommand": "log", "args": ["-C", "/etc"]}),
        ToolCall(id="g6", name="git_inspect", arguments={"subcommand": "log", "args": ["--git-dir=/tmp/g"]}),
        ToolCall(id="g7", name="git_inspect", arguments={"subcommand": "log", "args": ["--work-tree=/etc"]}),
        ToolCall(id="g8", name="git_inspect", arguments={"subcommand": "log", "args": ["/etc/hostname"]}),
        ToolCall(id="g9", name="git_inspect", arguments={"subcommand": "log", "args": ["~/.ssh/id_rsa"]}),
        ToolCall(id="g10", name="git_inspect", arguments={"subcommand": "log", "args": ["../escape"]}),
    ]
    for call in bad_git_calls:
        assert rbac.on_pre_tool_use(sre_state, call).action == "block", call.arguments
        res = reg.execute(call)
        assert res.is_error and res.blocked, call.arguments

    # Benign git flags & ancestry expressions still work.
    good_log = ToolCall(id="g-ok1", name="git_inspect", arguments={"subcommand": "log", "args": ["--oneline", "-n", "1"]})
    good_diff = ToolCall(id="g-ok2", name="git_inspect", arguments={"subcommand": "diff", "args": ["HEAD~0..HEAD"]})
    for call in (good_log, good_diff):
        assert rbac.on_pre_tool_use(sre_state, call).action == "allow", call.arguments
        assert not reg.execute(call).is_error, call.arguments

    # Patch target extraction catches rename/copy targets and rejects symlinks.
    escape_patches = [
        (
            "diff --git a/dbt_project/models/m.sql b/spark_jobs/moved.sql\n"
            "similarity index 100%\n"
            "rename from dbt_project/models/m.sql\n"
            "rename to spark_jobs/moved.sql\n"
        ),
        (
            "diff --git a/spark_jobs/job.py b/dbt_project/models/escaped.py\n"
            "copy from spark_jobs/job.py\n"
            "copy to dbt_project/models/escaped.py\n"
        ),
        (
            "diff --git a/spark_jobs/sym b/spark_jobs/sym\n"
            "new file mode 120000\n"
            "--- /dev/null\n"
            "+++ b/spark_jobs/sym\n"
            "@@ -0,0 +1 @@\n"
            "+/etc/passwd\n"
        ),
    ]
    for patch_text in escape_patches:
        call = ToolCall(id="p-bad", name="apply_patch", arguments={"patch": patch_text})
        assert rbac.on_pre_tool_use(sre_state, call).action == "block", patch_text
        if "120000" in patch_text:
            res = reg.execute(call)
            assert res.is_error, patch_text


def test_f3_spark_event_log_traversal_and_grep_exclusion(tmp_path: Path) -> None:
    """F3: Block skills/../ Spark log traversal and exclude spark_events from grep_search."""
    ws = tmp_path / "ws"
    (ws / "logs" / "spark_events").mkdir(parents=True)
    (ws / "logs" / "spark_events" / "event.log").write_text("SECRET_EVENT_TOKEN\n", encoding="utf-8")
    (ws / "spark_jobs").mkdir(parents=True)
    (ws / "spark_jobs" / "job.py").write_text("# SAFE_JOB_TOKEN\n", encoding="utf-8")

    rbac = PolicyRbacHook(policies=load_policies(), workspace_dir=ws)
    da_state = _make_state("da", resolution_mode="diagnosis_only")

    traversal_call = ToolCall(
        id="r-trav",
        name="read_file",
        arguments={"path": "skills/../logs/spark_events/event.log"},
    )
    assert rbac.on_pre_tool_use(da_state, traversal_call).action == "block"

    playbook_call = ToolCall(
        id="r-skill",
        name="read_file",
        arguments={"path": "skills/rca_playbook.md"},
    )
    assert rbac.on_pre_tool_use(da_state, playbook_call).action == "allow"

    res = ToolRegistry(ws).execute(
        ToolCall(id="g-all", name="grep_search", arguments={"pattern": "_TOKEN", "path": "."})
    )
    assert not res.is_error
    assert "SAFE_JOB_TOKEN" in res.output
    assert "SECRET_EVENT_TOKEN" not in res.output
    assert "spark_events" not in res.output


def test_f4_pii_redactor_nid_and_nested_sql_json() -> None:
    """F4: Redact NID-XXXXXX-XX identifiers and nested list/struct PII inside SQL JSON."""
    redactor = PiiRedactorHook(policies=load_policies())
    state = _make_state("de", run_id="run-f4")

    # Plain-text CSV & benign SQL column alias both redact NID-877572-01.
    plain_csv = "U0001,alice@synthetic-shop.test,NID-877572-01"
    dec_csv = redactor.on_post_tool_use(
        state,
        ToolCall(id="r1", name="read_file", arguments={"path": "users.csv"}),
        ToolResult(tool_call_id="r1", name="read_file", output=plain_csv),
    )
    assert dec_csv.action == "modify" and dec_csv.modified_output is not None
    assert "NID-877572-01" not in dec_csv.modified_output
    assert "alice@synthetic-shop.test" not in dec_csv.modified_output

    aliased_sql = json.dumps(
        {
            "columns": ["nid"],
            "rows": [["NID-877572-01"]],
            "row_count": 1,
        }
    )
    dec_alias = redactor.on_post_tool_use(
        state,
        ToolCall(id="s1", name="run_sql_readonly", arguments={"sql": "SELECT national_id AS nid"}),
        ToolResult(tool_call_id="s1", name="run_sql_readonly", output=aliased_sql),
    )
    assert dec_alias.action == "modify" and dec_alias.modified_output is not None
    assert "NID-877572-01" not in dec_alias.modified_output

    # Nested list (e.g. LIST(email)) and struct (e.g. {'p': phone}) values are redacted recursively.
    nested_sql = json.dumps(
        {
            "columns": ["email_list", "contact_struct"],
            "rows": [
                [
                    ["alice@example.com", "bob@corp.org"],
                    {"p": "13812345678", "phone": "+1-415-555-0199", "doc": "NID-123456-99"},
                ]
            ],
            "row_count": 1,
        }
    )
    dec_nested = redactor.on_post_tool_use(
        state,
        ToolCall(id="s2", name="run_sql_readonly", arguments={"sql": "SELECT 1"}),
        ToolResult(tool_call_id="s2", name="run_sql_readonly", output=nested_sql),
    )
    assert dec_nested.action == "modify" and dec_nested.modified_output is not None
    for secret in ("alice@example.com", "bob@corp.org", "13812345678", "+1-415-555-0199", "NID-123456-99"):
        assert secret not in dec_nested.modified_output
    assert re.search(r"<EMAIL_[0-9a-f]{8}>", dec_nested.modified_output)
    assert re.search(r"<PHONE_[0-9a-f]{8}>", dec_nested.modified_output)
    assert re.search(r"<ID_[0-9a-f]{8}>", dec_nested.modified_output)


def test_f5_sre_aggregate_only_hardening() -> None:
    """F5: Block row-returning aggregates, non-COUNT PII aggs, CTE/derived/positional PII grouping."""
    rbac = PolicyRbacHook(policies=load_policies())
    sre_state = _make_state("sre")

    blocked_sqls = [
        "SELECT LIST(email) FROM silver_users",
        "SELECT ARRAY_AGG(email) FROM silver_users",
        "SELECT STRING_AGG(national_id, ',') FROM silver_users",
        "SELECT MIN(national_id) FROM silver_users",
        "SELECT MAX(email) FROM silver_users",
        "SELECT n, COUNT(*) FROM (SELECT national_id AS n FROM silver_users) AS sub GROUP BY n",
        (
            "WITH a AS (SELECT email AS e FROM silver_users), "
            "b AS (SELECT e AS x FROM a) SELECT x, COUNT(*) FROM b GROUP BY x"
        ),
        "SELECT email, COUNT(*) FROM silver_users GROUP BY 1",
        "SELECT REPLACE(email, '@', '_at_'), COUNT(*) FROM silver_users GROUP BY 1",
    ]
    for sql in blocked_sqls:
        dec = rbac.on_pre_tool_use(
            sre_state, ToolCall(id="sre-bad", name="run_sql_readonly", arguments={"sql": sql})
        )
        assert dec.action == "block", f"Expected SRE block for {sql!r}"

    allowed_sqls = [
        "SELECT COUNT(*) FROM silver_users",
        "SELECT COUNT(email) FROM silver_users",
        "SELECT order_date, COUNT(*) FROM silver_orders GROUP BY order_date",
        "DESCRIBE HISTORY silver_orders",
    ]
    for sql in allowed_sqls:
        dec = rbac.on_pre_tool_use(
            sre_state, ToolCall(id="sre-ok", name="run_sql_readonly", arguments={"sql": sql})
        )
        assert dec.action == "allow", f"Expected SRE allow for {sql!r}, got {dec}"


def test_f6_sql_guard_fetch_percent_and_unnest_limit() -> None:
    """F6: Enforce max_limit on FETCH FIRST & LIMIT PERCENT and cap UNNEST(LIST(...)) queries."""
    guard = SqlGuardHook(policies=load_policies())
    state = _make_state("de")

    assert guard.on_pre_tool_use(
        state,
        ToolCall(id="q1", name="run_sql_readonly", arguments={"sql": "SELECT * FROM dim_user_ltv FETCH FIRST 100 ROWS ONLY"}),
    ).action == "block"
    assert guard.on_pre_tool_use(
        state,
        ToolCall(id="q2", name="run_sql_readonly", arguments={"sql": "SELECT * FROM dim_user_ltv LIMIT 50 PERCENT"}),
    ).action == "block"
    assert guard.on_pre_tool_use(
        state,
        ToolCall(id="q3", name="run_sql_readonly", arguments={"sql": "SELECT * FROM dim_user_ltv FETCH FIRST 10 ROWS ONLY"}),
    ).action == "allow"

    unnest_dec = guard.on_pre_tool_use(
        state,
        ToolCall(id="q4", name="run_sql_readonly", arguments={"sql": "SELECT UNNEST(LIST(user_id)) FROM dim_user_ltv"}),
    )
    assert unnest_dec.action == "modify"
    assert unnest_dec.modified_arguments is not None
    rewritten_sql = unnest_dec.modified_arguments["sql"]
    assert "LIMIT 50" in rewritten_sql.upper()

    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE TABLE dim_user_ltv (user_id INTEGER)")
        con.executemany("INSERT INTO dim_user_ltv VALUES (?)", [(i,) for i in range(120)])
        rows = con.execute(rewritten_sql).fetchall()
        assert len(rows) == 50
    finally:
        con.close()


def test_f7_sql_guard_partition_pruning_and_discovery() -> None:
    """F7: Reject expression/self-join tautologies on partition columns while allowing discovery."""
    guard = SqlGuardHook(policies=load_policies())
    state = _make_state("de")

    tautologies = [
        "SELECT * FROM silver_orders WHERE order_date BETWEEN order_date AND order_date LIMIT 10",
        "SELECT * FROM silver_orders WHERE order_date IN (order_date) LIMIT 10",
        "SELECT * FROM silver_orders WHERE order_date = COALESCE(order_date, DATE '2026-03-01') LIMIT 10",
        "SELECT * FROM silver_orders WHERE order_date = order_date + INTERVAL 0 DAY LIMIT 10",
        "SELECT * FROM silver_orders WHERE order_date IN (SELECT DISTINCT order_date FROM silver_orders) LIMIT 10",
        "SELECT * FROM silver_orders a JOIN silver_orders b ON a.order_id = b.order_id WHERE a.order_date >= b.order_date LIMIT 10",
    ]
    for sql in tautologies:
        dec = guard.on_pre_tool_use(
            state, ToolCall(id="p-bad", name="run_sql_readonly", arguments={"sql": sql})
        )
        assert dec.action == "block", f"Expected partition tautology block for {sql!r}"

    # Pure top-level partition discovery and concrete partition filters stay permitted.
    discovery = guard.on_pre_tool_use(
        state,
        ToolCall(id="p-disc", name="run_sql_readonly", arguments={"sql": "SELECT DISTINCT order_date FROM silver_orders"}),
    )
    assert discovery.action in {"allow", "modify"}
    assert (
        guard.on_pre_tool_use(
            state,
            ToolCall(
                id="p-ok",
                name="run_sql_readonly",
                arguments={"sql": "SELECT * FROM silver_orders WHERE order_date = DATE '2026-03-01' LIMIT 10"},
            ),
        ).action
        == "allow"
    )
    assert (
        guard.on_pre_tool_use(
            state,
            ToolCall(
                id="p-join-ok",
                name="run_sql_readonly",
                arguments={
                    "sql": "SELECT * FROM silver_orders a JOIN silver_orders b "
                    "ON a.order_id = b.order_id WHERE b.order_date = DATE '2026-03-01' "
                    "AND a.order_date = b.order_date LIMIT 10"
                },
            ),
        ).action
        == "allow"
    )


def test_f8_promotion_approve_atomic_rollback(tmp_path: Path) -> None:
    """F8: PromotionService.approve() rolls back git HEAD, new files, and DuckDB on verification failure."""
    prod_ws = tmp_path / "prod"
    runs_dir = tmp_path / "runs"
    orig_model = "SELECT 1 AS bad_val\n"
    pre_head = _init_git_repo(
        prod_ws,
        {"dbt_project/models/marts/fct_daily_gmv.sql": orig_model},
        message="initial prod",
    )
    # Create an untracked note that atomic rollback MUST preserve (no `git clean -fd`).
    keep_note = prod_ws / "notes_keep.txt"
    keep_note.write_text("user local note\n", encoding="utf-8")

    paths = LakehousePaths.from_workspace(prod_ws)
    paths.prod_mart_dir.mkdir(parents=True, exist_ok=True)
    orig_duckdb_bytes = b"ORIGINAL-DUCKDB-SNAPSHOT"
    paths.prod_duckdb_path.write_bytes(orig_duckdb_bytes)

    run_id = "run-rollback-1"
    wt = RunWorktreeManager.create(prod_ws, runs_dir, run_id)
    (wt / "dbt_project/models/marts/fct_daily_gmv.sql").write_text("SELECT 2 AS fixed_val\n", encoding="utf-8")
    (wt / "dbt_project/models/marts/new_model.sql").write_text("SELECT 42 AS extra\n", encoding="utf-8")
    diff_text = RunWorktreeManager.unified_diff(wt)
    changed = ["dbt_project/models/marts/fct_daily_gmv.sql", "dbt_project/models/marts/new_model.sql"]

    shadow_res = ShadowVerificationResult(
        passed=True,
        run_id=run_id,
        worktree_path=str(wt),
        shadow_silver_dir=str(runs_dir / run_id / "shadow" / "silver"),
        shadow_duckdb_path=str(runs_dir / run_id / "shadow" / "mart" / "warehouse.duckdb"),
        changed_files=changed,
    )
    packet = ApprovalPacket(
        run_id=run_id,
        incident_id="INC-RB-1",
        rca_report=RcaReport.model_validate_json(_valid_rca_json("INC-RB-1", changed)),
        unified_diff=diff_text,
        shadow_verification=shadow_res,
    )
    state = RunState(
        run_id=run_id,
        incident=IncidentInput(incident_id="INC-RB-1", title="Rollback", summary="Rollback"),
        user_context=UserContext(role="de", resolution_mode="fix"),
        status="AWAITING_APPROVAL",
        metadata={"approval_packet": packet.model_dump(mode="json")},
    )
    CheckpointStore(runs_dir, run_id).append("run_completed", state)

    def _failing_dbt(
        dbt_paths: LakehousePaths,
        silver_overrides: dict[str, Path],
        mart_duckdb_path: Path,
        args: list[str],
    ) -> subprocess.CompletedProcess[str]:
        # Simulate destructive dbt mutation right before post-commit failure.
        mart_duckdb_path.write_bytes(b"CORRUPTED-BY-FAILED-DBT")
        return subprocess.CompletedProcess(["dbt", *args], 1, stdout="", stderr="simulated dbt failure")

    service = PromotionService(prod_ws, runs_dir, dbt_executor=_failing_dbt)
    with pytest.raises(RuntimeError, match="Production post-commit verification failed"):
        service.approve(run_id)

    cur_head = subprocess.run(
        ["git", "-C", str(prod_ws), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert cur_head == pre_head
    assert (prod_ws / "dbt_project/models/marts/fct_daily_gmv.sql").read_text(encoding="utf-8") == orig_model
    assert not (prod_ws / "dbt_project/models/marts/new_model.sql").exists()
    assert paths.prod_duckdb_path.read_bytes() == orig_duckdb_bytes
    assert keep_note.read_text(encoding="utf-8") == "user local note\n"
    assert CheckpointStore(runs_dir, run_id).load_latest().status == "AWAITING_APPROVAL"
    RunWorktreeManager.cleanup(prod_ws, wt)


def test_f9_integrity_oracle_blocks_neutralized_sql_and_yaml_tests(tmp_path: Path) -> None:
    """F9: DefaultIntegrityOracle rejects neutralized SQL tests and weakened schema YAML configs."""
    prod_ws = tmp_path / "prod"
    wt_ws = tmp_path / "wt"
    orig_sql = "SELECT * FROM {{ ref('fct_daily_gmv') }} WHERE net_gmv_usd < 0\n"
    orig_yml = (
        "version: 2\n"
        "models:\n"
        "  - name: fct_daily_gmv\n"
        "    columns:\n"
        "      - name: order_date\n"
        "        tests:\n"
        "          - not_null\n"
    )
    for root in (prod_ws, wt_ws):
        (root / "dbt_project/tests").mkdir(parents=True)
        (root / "dbt_project/models/marts").mkdir(parents=True)
        (root / "dbt_project/tests/assert_gmv_reconciliation.sql").write_text(orig_sql, encoding="utf-8")
        (root / "dbt_project/models/marts/schema.yml").write_text(orig_yml, encoding="utf-8")

    oracle = DefaultIntegrityOracle()

    def _run_oracle() -> bool:
        return oracle.verify(
            prod_workspace=prod_ws,
            worktree_dir=wt_ws,
            shadow_silver_dir=tmp_path / "none_silver",
            shadow_duckdb_path=tmp_path / "missing.duckdb",
            changed_files=[],
            unified_diff="",
        ).passed

    assert _run_oracle() is True

    # Neutralized singular SQL tests must be rejected.
    wt_test_sql = wt_ws / "dbt_project/tests/assert_gmv_reconciliation.sql"
    for bad_sql in (
        "SELECT * FROM {{ ref('fct_daily_gmv') }} WHERE 1 = 0\n",
        "SELECT * FROM {{ ref('fct_daily_gmv') }} WHERE FALSE\n",
        "SELECT * FROM {{ ref('fct_daily_gmv') }} LIMIT 0\n",
        "SELECT 1 AS ok\n",
    ):
        wt_test_sql.write_text(bad_sql, encoding="utf-8")
        assert _run_oracle() is False, bad_sql
    wt_test_sql.write_text(orig_sql, encoding="utf-8")

    # Weakened schema.yml test configs must be rejected.
    wt_yml = wt_ws / "dbt_project/models/marts/schema.yml"
    for weakened_entry in (
        "          - not_null:\n              enabled: false\n",
        "          - not_null:\n              severity: warn\n",
        "          - not_null:\n              limit: 0\n",
        "          - not_null:\n              where: \"1 = 0\"\n",
        "          - not_null:\n              config:\n                where: \"order_date IS NULL AND 1=0\"\n",
    ):
        wt_yml.write_text(orig_yml.replace("          - not_null\n", weakened_entry), encoding="utf-8")
        assert _run_oracle() is False, weakened_entry


def test_f10_real_spark_describe_history(tmp_path: Path) -> None:
    """F10: Real Spark DESCRIBE HISTORY resolves Silver Delta table directory without catalog errors."""
    ws = tmp_path / "ws"
    shutil.copytree(
        LIVE_WORKSPACE,
        ws,
        ignore=shutil.ignore_patterns(".git", ".scratch", "logs", ".user.yml", "__pycache__"),
    )
    reg = ToolRegistry(ws)
    res = reg.execute(
        ToolCall(
            id="hist-1",
            name="run_sql_readonly",
            arguments={"sql": "DESCRIBE HISTORY silver_orders", "engine": "spark"},
        )
    )
    assert res.is_error is False, res.output
    payload = json.loads(res.output)
    assert "version" in payload["columns"]
    assert "operation" in payload["columns"]
    assert len(payload["rows"]) >= 1


def test_f11_loop_wire_integrity_crash_reconciliation_and_terminal_resume_guard(
    tmp_path: Path,
) -> None:
    """F11: Preserve sibling tool adjacency on Strike 1, reconcile crashed tool turns, guard terminal resume."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("alpha\n", encoding="utf-8")
    (ws / "b.txt").write_text("beta\n", encoding="utf-8")

    policies = load_policies()
    # Part A: Strike 1 nudge appears AFTER all sibling tool messages of the same turn.
    reg = ToolRegistry(ws)
    relaxed_thrash = AntiThrashHook(
        policy=policies.anti_thrash.model_copy(update={"strikes_before_abort": 3})
    )
    adapter2 = FakeReplayModelAdapter(
        [
            TurnResponse(
                tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a.txt"})]
            ),
            TurnResponse(
                tool_calls=[
                    ToolCall(id="t2", name="read_file", arguments={"path": "a.txt"}),  # repeat -> Strike 1
                    ToolCall(id="t3", name="read_file", arguments={"path": "b.txt"}),  # sibling
                ]
            ),
            TurnResponse(content="Done after warning."),
        ]
    )
    runs_dir = tmp_path / "runs"
    loop = HarnessLoop(
        model_adapter=adapter2,
        tool_registry=reg,
        hooks=HookPipeline(post_tool_hooks=[relaxed_thrash]),
        runs_dir=runs_dir,
        policies=policies,
    )
    state = loop.run(
        IncidentInput(incident_id="INC-F11", title="Wire order", summary="Check wire order"),
        UserContext(role="da", resolution_mode="diagnosis_only"),
        run_id="run-f11-wire",
    )
    assert state.status == "DIAGNOSED"
    # Find the turn-2 assistant message with [t2, t3] and verify t2 and t3 precede the nudge.
    turn2_idx = next(
        i for i, m in enumerate(state.messages) if m.role == "assistant" and len(m.tool_calls) == 2
    )
    assert state.messages[turn2_idx + 1].role == "tool"
    assert state.messages[turn2_idx + 1].tool_call_id == "t2"
    assert state.messages[turn2_idx + 2].role == "tool"
    assert state.messages[turn2_idx + 2].tool_call_id == "t3"
    assert state.messages[turn2_idx + 3].role == "user"
    assert "[ANTI-THRASH WARNING" in state.messages[turn2_idx + 3].content

    # Part B: Crash mid-tool-batch recovery synthesizes missing tool result on resume().
    crash_run = "run-f11-crash"
    crashed_state = RunState(
        run_id=crash_run,
        incident=IncidentInput(incident_id="INC-CRASH", title="Crash", summary="Crash"),
        user_context=UserContext(role="da", resolution_mode="diagnosis_only"),
        status="RUNNING",
        turn=1,
        max_turns=5,
        messages=[
            Message(role="user", content="Start"),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"}),
                    ToolCall(id="c2", name="read_file", arguments={"path": "b.txt"}),
                ],
            ),
            Message(role="tool", content="alpha\n", tool_call_id="c1", tool_name="read_file"),
        ],
    )
    CheckpointStore(runs_dir, crash_run).append("tool_step", crashed_state)
    res_adapter = FakeReplayModelAdapter(
        [TurnResponse(content="Recovered."), TurnResponse(content="Resumed.")]
    )
    res_loop = HarnessLoop(
        model_adapter=res_adapter, tool_registry=reg, runs_dir=runs_dir, policies=policies
    )
    recovered = res_loop.resume(crash_run)
    assert recovered.status == "DIAGNOSED"
    sent_msgs = res_adapter.recorded_requests[0].messages
    assert [m.role for m in sent_msgs[:4]] == ["user", "assistant", "tool", "tool"]
    assert sent_msgs[3].tool_call_id == "c2"

    # Part C: Terminal states block resume(), except ABORTED_BUDGET with extra_turns > 0.
    for idx, term_status in enumerate(["DIAGNOSED", "RESOLVED", "ABORTED_THRASH"]):
        rid = f"run-term-{idx}"
        st = _make_state("de", run_id=rid)
        st.status = term_status  # type: ignore[assignment]
        CheckpointStore(runs_dir, rid).append("done", st)
        with pytest.raises(ValueError, match="terminal status"):
            res_loop.resume(rid, extra_turns=3)

    budget_rid = "run-term-budget"
    bst = _make_state("da", run_id=budget_rid, resolution_mode="diagnosis_only", max_turns=1)
    bst.turn = 1
    bst.status = "ABORTED_BUDGET"
    CheckpointStore(runs_dir, budget_rid).append("budget", bst)
    with pytest.raises(ValueError, match="extra_turns > 0"):
        res_loop.resume(budget_rid, extra_turns=0)
    resumed_ok = res_loop.resume(budget_rid, extra_turns=2)
    assert resumed_ok.status == "DIAGNOSED"


def test_f12_role_ceiling_cost_budget_extra_turns_and_pause_clock(tmp_path: Path) -> None:
    """F12: Enforce role ceiling, non-tool cost budget, resume extra_turns, and pause clock exclusion."""
    ws = tmp_path / "ws"
    _init_git_repo(ws, {"dbt_project/models/marts/fct_daily_gmv.sql": "SELECT 1\n"})
    runs_dir = tmp_path / "runs"
    policies = load_policies()
    reg = ToolRegistry(ws)

    # Part A: `da` role requesting `fix` is clamped to `diagnosis_only`.
    da_adapter = FakeReplayModelAdapter([TurnResponse(content="Diagnosis complete.")])
    da_loop = HarnessLoop(
        model_adapter=da_adapter, tool_registry=reg, runs_dir=runs_dir, policies=policies
    )
    da_state = da_loop.run(
        IncidentInput(incident_id="INC-DA", title="Check", summary="Check"),
        UserContext(role="da", resolution_mode="fix"),
        run_id="run-f12-da",
    )
    assert da_state.user_context.resolution_mode == "diagnosis_only"
    assert da_state.status == "DIAGNOSED"

    # Part B: Non-tool turn hits max_cost_usd before StopGateHook.
    t0 = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
    clock = [t0]
    tight_limits = LimitsPolicy(max_turns=20, max_wall_clock_seconds=300, max_cost_usd=0.05)
    compactor = ContextCompactorHook(workspace_root=ws, limits=tight_limits, now_fn=lambda: clock[0])
    cost_adapter = FakeReplayModelAdapter(
        [
            TurnResponse(
                content="Summary",
                usage=UsageMetrics(input_tokens=100, output_tokens=20, cost_usd=0.06),
            )
        ]
    )
    cost_loop = HarnessLoop(
        model_adapter=cost_adapter,
        tool_registry=reg,
        hooks=HookPipeline(post_tool_hooks=[compactor]),
        runs_dir=runs_dir,
        policies=policies,
    )
    cost_state = cost_loop.run(
        IncidentInput(incident_id="INC-COST", title="Cost", summary="Cost"),
        UserContext(role="de", resolution_mode="diagnosis_only"),
        run_id="run-f12-cost",
    )
    assert cost_state.status == "ABORTED_BUDGET"
    assert "Cost ceiling" in str(cost_state.metadata.get("abort_reason"))

    # Part C & D: Wait 360s in AWAITING_APPROVAL (> 300s max_wall_clock_seconds), then reject + resume.
    pause_limits = LimitsPolicy(
        max_turns=2, max_wall_clock_seconds=300, max_cost_usd=2.0, rejection_turn_extension=8
    )
    pause_compactor = ContextCompactorHook(
        workspace_root=ws, limits=pause_limits, now_fn=lambda: clock[0]
    )
    wt = RunWorktreeManager.create(ws, runs_dir, "run-f12-pause")

    class _PassVerifier:
        def __init__(self, verifier_runs_dir: Path) -> None:
            self.runs_dir = verifier_runs_dir

        def verify(self, run_id: str, worktree_dir: Path) -> ShadowVerificationResult:
            return ShadowVerificationResult(
                passed=True,
                run_id=run_id,
                worktree_path=str(worktree_dir),
                shadow_silver_dir=str(self.runs_dir / run_id / "silver"),
                shadow_duckdb_path=str(self.runs_dir / run_id / "mart.duckdb"),
                changed_files=[],
            )

    stop_gate = StopGateHook(
        verifier=_PassVerifier(runs_dir), worktree_dir=wt, runs_dir=runs_dir  # type: ignore[arg-type]
    )
    pause_hooks = HookPipeline(post_tool_hooks=[pause_compactor], stop_hooks=[stop_gate])
    p_adapter1 = FakeReplayModelAdapter(
        [
            TurnResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "dbt_project/models/marts/fct_daily_gmv.sql"})]),
            TurnResponse(content=_valid_rca_json("INC-PAUSE")),
        ]
    )
    p_loop1 = HarnessLoop(
        model_adapter=p_adapter1,
        tool_registry=ToolRegistry(wt),
        hooks=pause_hooks,
        runs_dir=runs_dir,
        policies=policies,
    )
    paused = p_loop1.run(
        IncidentInput(incident_id="INC-PAUSE", title="Pause", summary="Pause"),
        UserContext(role="de", resolution_mode="fix"),
        run_id="run-f12-pause",
        max_turns=2,
    )
    assert paused.status == "AWAITING_APPROVAL"

    # Simulate 360 seconds of human review delay before rejecting and resuming.
    clock[0] = t0 + timedelta(seconds=360)
    p_adapter2 = FakeReplayModelAdapter(
        [
            TurnResponse(tool_calls=[ToolCall(id="r2", name="read_file", arguments={"path": "dbt_project/models/marts/fct_daily_gmv.sql"})]),
            TurnResponse(content=_valid_rca_json("INC-PAUSE")),
        ]
    )
    p_loop2 = HarnessLoop(
        model_adapter=p_adapter2,
        tool_registry=ToolRegistry(wt),
        hooks=pause_hooks,
        runs_dir=runs_dir,
        policies=policies,
    )
    resumed = PromotionService(ws, runs_dir).reject("run-f12-pause", "Check again", loop=p_loop2)
    assert resumed.status == "AWAITING_APPROVAL"
    assert float(resumed.metadata.get("paused_seconds", 0.0)) >= 360.0
    assert int(resumed.metadata.get("extra_turns", 0)) == 0

    # Non-rejection extra_turns increments metadata["extra_turns"].
    ext_run = "run-f12-ext"
    est = _make_state("da", run_id=ext_run, resolution_mode="diagnosis_only", max_turns=1)
    est.turn = 1
    est.status = "ABORTED_BUDGET"
    CheckpointStore(runs_dir, ext_run).append("budget", est)
    ext_loop = HarnessLoop(
        model_adapter=FakeReplayModelAdapter([TurnResponse(content="Done.")]),
        tool_registry=reg,
        hooks=HookPipeline(post_tool_hooks=[pause_compactor]),
        runs_dir=runs_dir,
        policies=policies,
    )
    ext_res = ext_loop.resume(ext_run, extra_turns=4)
    assert ext_res.status == "DIAGNOSED"
    assert int(ext_res.metadata.get("extra_turns", 0)) == 4
    RunWorktreeManager.cleanup(ws, wt)


def test_f13_parse_rca_report_with_leading_json_object() -> None:
    """F13: parse_rca_report finds valid RcaReport even after an earlier non-RCA JSON object."""
    prose = (
        'Intermediate metric check: {"row_count": 42, "status": "ok"}\n\n'
        "Final structured report:\n"
        f"{_valid_rca_json('INC-F13', ['dbt_project/models/marts/fct_daily_gmv.sql'])}"
    )
    parsed = parse_rca_report(prose)
    assert parsed.incident_id == "INC-F13"
    assert parsed.patched_files == ["dbt_project/models/marts/fct_daily_gmv.sql"]


def test_f14_f15_ablation_runner_git_history_tools_hitl_and_pricing(tmp_path: Path) -> None:
    """F14/F15: Ablation preserves fault git history, wires tools/skills, avoids HITL 2x count & uses config pricing."""
    # Check Group C end-to-end on dbt_broken_macro_git, ops_hitl_reject_refix_promote, ops_alert_storm_convergence.
    recorded_group_c: dict[str, FakeReplayModelAdapter] = {}

    runner = AblationBenchmarkRunner(provider="fake_replay", max_turns=20)
    report = runner.run(
        scenario_ids=[
            "dbt_broken_macro_git",
            "ops_hitl_reject_refix_promote",
            "ops_alert_storm_convergence",
        ],
        arms=("group_c",),
    )
    assert not recorded_group_c  # sanity
    recs = {r.scenario_id: r for r in report.arms[0].records}

    macro_rec = recs["dbt_broken_macro_git"]
    assert macro_rec.scenario_passed is True
    assert macro_rec.verified_fix is True

    hitl_rec = recs["ops_hitl_reject_refix_promote"]
    assert hitl_rec.scenario_passed is True
    assert hitl_rec.runs_dispatched == 1
    assert hitl_rec.turns == 4

    storm_rec = recs["ops_alert_storm_convergence"]
    assert storm_rec.scenario_passed is True
    assert storm_rec.runs_dispatched == 1

    # Verify AlertStorm custom pricing uses configured PricingConfig via Harness adapter.
    custom_pricing = PricingConfig(input=10.0, cached_input=2.0, output=40.0)
    custom_cfg = ProviderConfig(
        api_protocol="fake_replay",
        model="custom-priced",
        pricing_per_1m_tokens=custom_pricing,
    )

    def _custom_priced_factory(arm: ArmConfig, outcome: InjectionOutcome):
        return FakeReplayModelAdapter(
            [
                TurnResponse(
                    content=_valid_rca_json(outcome.incident.incident_id),
                    usage=UsageMetrics(
                        input_tokens=1_000_000,
                        cached_input_tokens=500_000,
                        output_tokens=250_000,
                    ),
                )
            ],
            config=custom_cfg,
        )

    priced_runner = AblationBenchmarkRunner(
        provider="fake_replay",
        max_turns=20,
        model_factory=_custom_priced_factory,
    )
    priced_report = priced_runner.run(
        scenario_ids=["ops_alert_storm_convergence"], arms=("group_c",)
    )
    priced_rec = priced_report.arms[0].records[0]
    assert priced_rec.projected_cost_usd == pytest.approx(21.0)  # 10*1.0 + 2*0.5 + 40*0.25

    # Verify _clone_git_workspace(..., preserve_git=True) retains injected commit history.
    src_ws = tmp_path / "src_ws"
    _init_git_repo(src_ws, {"a.txt": "1\n"}, message="fault commit: invert FX macro")
    cloned = _clone_git_workspace(src_ws, tmp_path / "cloned_ws", preserve_git=True)
    git_log = subprocess.run(
        ["git", "-C", str(cloned), "log", "-n", "1", "--pretty=%s"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert git_log == "fault commit: invert FX macro"

    # Verify _ArmToolRegistry works cleanly when wired with skills_dir on real workspace clone.
    arm_ws = _clone_git_workspace(LIVE_WORKSPACE, tmp_path / "arm_ws", preserve_git=False)
    reg = _ArmToolRegistry(arm_ws, enable_lineage_tool=True, skills_dir=SKILLS_DIR)
    skill_res = reg.execute(
        ToolCall(
            id="sk1",
            name="read_file",
            arguments={"path": "skills/dbt_troubleshooting.md", "start_line": 1, "end_line": 20},
        )
    )
    assert not skill_res.is_error
    lin_res = reg.execute(
        ToolCall(id="ln1", name="query_lineage", arguments={"table_name": "fct_daily_gmv"})
    )
    assert lin_res.is_error is True  # proves explicit lineage_adapter injection matters
    _, _, _, _, proj_cost, _ = _aggregate_usage(
        [
            RunState(
                run_id="r-price",
                incident=IncidentInput(incident_id="INC-P", title="P", summary="P"),
                user_context=UserContext(role="da", resolution_mode="diagnosis_only"),
                usage=UsageMetrics(
                    input_tokens=1_000_000, cached_input_tokens=500_000, output_tokens=250_000
                ),
            )
        ],
        custom_pricing,
    )
    assert proj_cost == pytest.approx(21.0)


def test_f14_group_c_dbt_broken_macro_git_uses_real_git_and_skills(tmp_path: Path) -> None:
    """F14: Verify Group C in dbt_broken_macro_git executes git_inspect, query_lineage, and read_file(skills) cleanly."""
    observed_msgs: list[list[Message]] = []

    def _capturing_factory(arm: ArmConfig, outcome: InjectionOutcome):
        from benchmarks.scenarios import get_scenario

        scenario = get_scenario("dbt_broken_macro_git")
        canonical_patch = scenario.reference_fix_patch(outcome)
        assert canonical_patch is not None
        adapter = FakeReplayModelAdapter(
            [
                TurnResponse(
                    tool_calls=[
                        ToolCall(
                            id="c-lineage",
                            name="query_lineage",
                            arguments={"table_name": "fct_daily_gmv", "direction": "upstream", "depth": 3},
                        ),
                        ToolCall(
                            id="c-playbook",
                            name="read_file",
                            arguments={"path": "skills/dbt_troubleshooting.md", "start_line": 1, "end_line": 80},
                        ),
                        ToolCall(
                            id="c-git-log",
                            name="git_inspect",
                            arguments={"subcommand": "log", "args": ["-n", "2", "--oneline"]},
                        ),
                    ]
                ),
                TurnResponse(
                    tool_calls=[
                        ToolCall(id="c-fix", name="apply_patch", arguments={"patch": canonical_patch})
                    ]
                ),
                TurnResponse(content=_valid_rca_json(outcome.incident.incident_id, ["dbt_project/macros/currency_utils.sql"])),
            ]
        )
        observed_msgs.append([])  # placeholder synced below via recorded_requests

        orig_complete = adapter.complete

        def _wrapped(req):
            resp = orig_complete(req)
            observed_msgs[-1] = list(req.messages)
            return resp

        adapter.complete = _wrapped  # type: ignore[method-assign]
        return adapter

    runner = AblationBenchmarkRunner(
        provider="fake_replay",
        max_turns=20,
        model_factory=_capturing_factory,
    )
    report = runner.run(scenario_ids=["dbt_broken_macro_git"], arms=("group_c",))
    rec = report.arms[0].records[0]
    assert rec.scenario_passed is True
    assert observed_msgs
    tool_outputs = {m.tool_call_id: m.content for m in observed_msgs[-1] if m.role == "tool"}
    assert "dbt_project" in tool_outputs["c-lineage"]
    assert "# " in tool_outputs["c-playbook"]
    assert "invert FX normalization in to_usd" in tool_outputs["c-git-log"]
    assert not any(m.role == "user" and "[ANTI-THRASH WARNING" in m.content for m in observed_msgs[-1])
    assert ANTI_THRASH_METADATA_KEY  # reference constant cleanly imported
