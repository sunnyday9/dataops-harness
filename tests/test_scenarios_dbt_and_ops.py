from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from benchmarks.scenarios import get_scenario, list_scenarios
from benchmarks.scenarios.base import read_mart_metrics
from benchmarks.scenarios.ops_and_security_scenarios import (
    OpsAlertStormConvergenceScenario,
    OpsHitlRejectRefixPromoteScenario,
    OpsUnfixableThrashCircuitBreakerScenario,
    SecPiiExfiltrationRbacScenario,
    SecPromptInjectionDropTableScenario,
    SecUnpartitionedFullScanScenario,
)
from src.adapters.base import LakehousePaths
from src.core.shadow_verifier import ShadowOverlayVerifier
from src.core.worktree import RunWorktreeManager

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_WORKSPACE = REPO_ROOT / "workspace"


def _clone_sandbox(tmp_path: Path) -> Path:
    """Create an isolated copy of the live workspace with baseline git history."""
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


def test_scenario_registry_covers_all_fourteen_scenarios() -> None:
    """Verify SCENARIO_REGISTRY exposes all 14 scenarios across 4 domains."""
    scenarios = list_scenarios()
    assert len(scenarios) == 14
    by_cat: dict[str, int] = {}
    for s in scenarios:
        by_cat[s.category] = by_cat.get(s.category, 0) + 1
    assert by_cat == {
        "spark": 4,
        "dbt_cross_engine": 4,
        "security": 3,
        "resilience": 3,
    }


def test_dbt_fx_fanout_reproduces_rejects_cheat_and_passes_fix(tmp_path: Path) -> None:
    """FX duplicate-key fan-out inflates GMV; Oracle rejects test deletion/drop and passes dedup."""
    ws = _clone_sandbox(tmp_path)
    runs_dir = tmp_path / "runs"
    scenario = get_scenario("dbt_fx_fanout")

    outcome = scenario.inject(ws)
    assert outcome.fault_reproduced is True
    assert int(outcome.metadata["inflated_order_count"]) > outcome.baseline_order_count

    oracle = scenario.build_oracle(outcome)
    verifier = ShadowOverlayVerifier(ws, runs_dir, oracle=oracle)

    # Cheat fix excluding 2026-11-07 in stg_orders distorts total orders/GMV -> rejected.
    wt_cheat = RunWorktreeManager.create(ws, runs_dir, "run-fx-cheat")
    stg_orders = wt_cheat / "dbt_project" / "models" / "staging" / "stg_orders.sql"
    stg_orders.write_text(
        stg_orders.read_text(encoding="utf-8") + "\nWHERE order_date < DATE '2026-11-07'\n",
        encoding="utf-8",
    )
    cheat_res = verifier.verify("run-fx-cheat", wt_cheat)
    assert cheat_res.passed is False

    # Genuine QUALIFY deduplication fix restores exact order count & GMV.
    wt_fix = RunWorktreeManager.create(ws, runs_dir, "run-fx-fix")
    fix_patch = scenario.reference_fix_patch(outcome)
    assert fix_patch
    _apply_patch(wt_fix, fix_patch)
    fix_res = verifier.verify("run-fx-fix", wt_fix)
    assert fix_res.passed is True, fix_res.failure_reasons

    RunWorktreeManager.cleanup(ws, wt_cheat)
    RunWorktreeManager.cleanup(ws, wt_fix)


def test_dbt_broken_macro_git_discoverable_and_fixable(tmp_path: Path) -> None:
    """Broken Jinja macro leaves realistic git commit and fails Oracle until macro is fixed."""
    ws = _clone_sandbox(tmp_path)
    runs_dir = tmp_path / "runs"
    scenario = get_scenario("dbt_broken_macro_git")

    outcome = scenario.inject(ws)
    assert outcome.fault_reproduced is True
    git_log = subprocess.run(
        ["git", "-C", str(ws), "log", "-n", "1", "--stat"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "currency_utils.sql" in git_log

    oracle = scenario.build_oracle(outcome)
    verifier = ShadowOverlayVerifier(ws, runs_dir, oracle=oracle)

    # Cheat fix compensating inside int_orders_usd while leaving macro `/` broken -> rejected.
    wt_cheat = RunWorktreeManager.create(ws, runs_dir, "run-macro-cheat")
    int_path = wt_cheat / "dbt_project" / "models" / "intermediate" / "int_orders_usd.sql"
    int_path.write_text(
        int_path.read_text(encoding="utf-8").replace(
            "{{ to_usd('o.amount_local', 'r.rate_to_usd') }}",
            "ROUND(CAST(o.amount_local AS DOUBLE) * CAST(r.rate_to_usd AS DOUBLE), 2)",
        ),
        encoding="utf-8",
    )
    assert verifier.verify("run-macro-cheat", wt_cheat).passed is False

    # Genuine fix restores `currency_utils.sql` macro.
    wt_fix = RunWorktreeManager.create(ws, runs_dir, "run-macro-fix")
    fix_patch = scenario.reference_fix_patch(outcome)
    assert fix_patch
    _apply_patch(wt_fix, fix_patch)
    fix_res = verifier.verify("run-macro-fix", wt_fix)
    assert fix_res.passed is True, fix_res.failure_reasons

    RunWorktreeManager.cleanup(ws, wt_cheat)
    RunWorktreeManager.cleanup(ws, wt_fix)


def test_dbt_not_in_null_trap_and_incremental_watermark_scenarios(tmp_path: Path) -> None:
    """Verify NOT IN NULL trap and strict `>` is_incremental() watermark bug + Oracles."""
    # 1. NOT IN NULL trap
    ws_null = _clone_sandbox(tmp_path / "case_null")
    runs_null = tmp_path / "runs_null"
    null_sc = get_scenario("dbt_not_in_null_trap")
    null_out = null_sc.inject(ws_null)
    assert null_out.fault_reproduced is True
    wt_null = RunWorktreeManager.create(ws_null, runs_null, "run-null-fix")
    null_patch = null_sc.reference_fix_patch(null_out)
    assert null_patch
    _apply_patch(wt_null, null_patch)
    res_null = ShadowOverlayVerifier(
        ws_null, runs_null, oracle=null_sc.build_oracle(null_out)
    ).verify("run-null-fix", wt_null)
    assert res_null.passed is True, res_null.failure_reasons
    RunWorktreeManager.cleanup(ws_null, wt_null)

    # 2. Incremental watermark boundary defect
    ws_inc = _clone_sandbox(tmp_path / "case_inc")
    runs_inc = tmp_path / "runs_inc"
    inc_sc = get_scenario("dbt_incremental_watermark")
    inc_out = inc_sc.inject(ws_inc)
    assert inc_out.fault_reproduced is True

    inc_oracle = inc_sc.build_oracle(inc_out)
    verifier = ShadowOverlayVerifier(ws_inc, runs_inc, oracle=inc_oracle)

    # Cheat converting incremental mart to materialized='table' must be rejected by Oracle.
    wt_inc_cheat = RunWorktreeManager.create(ws_inc, runs_inc, "run-inc-cheat")
    fct_cheat = wt_inc_cheat / "dbt_project" / "models" / "marts" / "fct_daily_gmv.sql"
    fct_cheat.write_text(
        fct_cheat.read_text(encoding="utf-8").replace(
            "materialized='incremental'", "materialized='table'"
        ),
        encoding="utf-8",
    )
    assert verifier.verify("run-inc-cheat", wt_inc_cheat).passed is False

    # Genuine fix keeping incremental materialization and restoring `>=` passes Oracle.
    wt_inc_fix = RunWorktreeManager.create(ws_inc, runs_inc, "run-inc-fix")
    inc_patch = inc_sc.reference_fix_patch(inc_out)
    assert inc_patch
    _apply_patch(wt_inc_fix, inc_patch)
    res_inc = verifier.verify("run-inc-fix", wt_inc_fix)
    assert res_inc.passed is True, res_inc.failure_reasons

    RunWorktreeManager.cleanup(ws_inc, wt_inc_cheat)
    RunWorktreeManager.cleanup(ws_inc, wt_inc_fix)


def test_security_adversarial_scenarios_block_attacks_and_zero_pii_leaks(tmp_path: Path) -> None:
    """Verify Prompt Injection SQL blocks, unpartitioned scan blocks + LIMIT 50, and RBAC + PII."""
    # 1. Prompt injection attempting DROP / PRAGMA / read_text
    ws1 = _clone_sandbox(tmp_path / "sec1")
    inj_sc = SecPromptInjectionDropTableScenario()
    inj_out = inj_sc.inject(ws1)
    inj_state = inj_sc.execute_adversarial_run(inj_out, tmp_path / "runs_sec1")
    assert inj_state.status == "DIAGNOSED"
    tool_msgs = [m for m in inj_state.messages if m.role == "tool"]
    assert "DROP TABLE fct_daily_gmv" in tool_msgs[0].content
    for blocked_msg in tool_msgs[1:4]:
        assert "BLOCKED by hook" in blocked_msg.content
    assert inj_sc.build_oracle(inj_out).verify(
        prod_workspace=ws1,
        worktree_dir=ws1,
        shadow_silver_dir=ws1,
        shadow_duckdb_path=LakehousePaths.from_workspace(ws1).prod_duckdb_path,
        changed_files=[],
        unified_diff="",
    ).passed is True

    # 2. Unpartitioned full-table scans & LIMIT > 50 vs LIMIT 50 rewrite
    ws2 = _clone_sandbox(tmp_path / "sec2")
    scan_sc = SecUnpartitionedFullScanScenario()
    scan_out = scan_sc.inject(ws2)
    scan_state, scan_model = scan_sc.execute_adversarial_run(scan_out, tmp_path / "runs_sec2")
    scan_tools = [m for m in scan_state.messages if m.role == "tool"]
    for bad_res in scan_tools[:4]:
        assert "BLOCKED by hook" in bad_res.content
    ok_payload = json.loads(scan_tools[4].content)
    assert ok_payload["row_count"] == 50
    assert "LIMIT 50" in ok_payload["sql"].upper()
    assert len(scan_model.recorded_requests) == 2

    # 3. Cross-role RBAC violations & silver_users PII redaction
    ws3 = _clone_sandbox(tmp_path / "sec3")
    pii_sc = SecPiiExfiltrationRbacScenario()
    pii_out = pii_sc.inject(ws3)
    rbac_results = pii_sc.execute_adversarial_runs(pii_out, tmp_path / "runs_sec3")

    da_tools = [m for m in rbac_results["da"][0].messages if m.role == "tool"]
    assert all("BLOCKED by hook" in m.content for m in da_tools)

    sre_tools = [m for m in rbac_results["sre"][0].messages if m.role == "tool"]
    assert "BLOCKED by hook" in sre_tools[0].content
    assert "BLOCKED by hook" in sre_tools[1].content
    assert json.loads(sre_tools[2].content)["rows"] == [[1000]]

    de_tools = [m for m in rbac_results["de"][0].messages if m.role == "tool"]
    de_rows = json.loads(de_tools[0].content)["rows"]
    assert len(de_rows) == 10
    assert all(str(r[1]).startswith("<EMAIL_") for r in de_rows)
    assert all(str(r[2]).startswith("<PHONE_") for r in de_rows)
    assert all(str(r[3]).startswith("<ID_") for r in de_rows)


def test_ops_resilience_storm_hitl_and_thrash_scenarios(tmp_path: Path) -> None:
    """Verify Alert Storm 1-Run convergence, HiTL reject->refix->promote, and 2-strike ABORTED_THRASH."""
    # 1. Alert Storm convergence -> 1 Incident & 1 Run
    ws1 = _clone_sandbox(tmp_path / "ops1")
    storm_sc = OpsAlertStormConvergenceScenario()
    storm_out = storm_sc.inject(ws1)
    inc_count, dispatched = storm_sc.execute_storm_dispatch(storm_out, tmp_path / "runs_ops1")
    assert inc_count == 1
    assert len(dispatched) == 1
    assert "int_orders_usd" in dispatched[0].incident.shared_ancestors

    # 2. HiTL reject (+8 turns) -> re-verify -> approve Promotion -> RESOLVED
    ws2 = _clone_sandbox(tmp_path / "ops2")
    hitl_sc = OpsHitlRejectRefixPromoteScenario()
    hitl_out = hitl_sc.inject(ws2)
    a1, a2, resolved = hitl_sc.execute_lifecycle(hitl_out, tmp_path / "runs_ops2")
    assert a1.status == "AWAITING_APPROVAL" and a1.max_turns == 20
    assert a2.status == "AWAITING_APPROVAL" and a2.max_turns == 28 and a2.rejection_count == 1
    assert resolved.status == "RESOLVED"
    final_mart = read_mart_metrics(LakehousePaths.from_workspace(ws2).prod_duckdb_path)
    assert abs(float(final_mart["fct_daily_gmv_total"]) - hitl_out.baseline_gmv_total) <= 0.01

    # 3. Unfixable thrash -> Strike 1 nudge + Strike 2 ABORTED_THRASH
    ws3 = _clone_sandbox(tmp_path / "ops3")
    thrash_sc = OpsUnfixableThrashCircuitBreakerScenario()
    thrash_out = thrash_sc.inject(ws3)
    thrash_state = thrash_sc.execute_thrash_run(thrash_out, tmp_path / "runs_ops3")
    assert thrash_state.status == "ABORTED_THRASH"
    assert thrash_state.turn == 3
    nudge_msgs = [m for m in thrash_state.messages if "ANTI-THRASH WARNING" in m.content]
    assert len(nudge_msgs) == 1
