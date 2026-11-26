#!/usr/bin/env python3
"""End-to-end isolated verification harness for DataOps-Harness CLI, formal checks, and ablation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.ablation_runner import AblationBenchmarkRunner, render_markdown_report
from benchmarks.scenarios import get_scenario
from src.adapters.base import LakehousePaths
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.model_adapter import FakeReplayModelAdapter, TurnResponse
from src.core.context import IncidentInput, ToolCall, UserContext
from src.core.loop import HarnessLoop
from src.core.policy import load_policies
from src.core.rca import RcaReport
from src.core.shadow_verifier import ShadowOverlayVerifier
from src.core.worktree import RunWorktreeManager
from src.hooks.base import HookPipeline
from src.hooks.stop_gate import StopGateHook
from src.tools.registry import ToolRegistry

FAST_VERIFY_SCENARIOS: list[str] = [
    "dbt_fx_fanout",
    "sec_prompt_injection_drop_table",
    "ops_alert_storm_convergence",
]


def run_doctor_checks() -> dict[str, Any]:
    """Verify runtime executables, configs, and baseline synthetic lakehouse health."""
    checks: dict[str, Any] = {}
    checks["python_executable"] = sys.executable
    checks["git_available"] = shutil.which("git") is not None
    checks["uv_available"] = shutil.which("uv") is not None

    java_path = shutil.which("java")
    if java_path is None:
        fallback_java = Path.home() / ".local" / "opt" / "jdk-17" / "bin" / "java"
        if fallback_java.is_file():
            java_path = str(fallback_java)
    checks["java_available"] = java_path is not None
    checks["java_path"] = java_path

    policies_path = REPO_ROOT / "configs" / "policies.yaml"
    models_path = REPO_ROOT / "configs" / "models.yaml"
    workspace_dir = REPO_ROOT / "workspace"
    mart_db = workspace_dir / "lakehouse" / "prod" / "mart" / "warehouse.duckdb"

    checks["configs_present"] = policies_path.is_file() and models_path.is_file()
    checks["workspace_present"] = workspace_dir.is_dir() and mart_db.is_file()

    if checks["configs_present"]:
        policies = load_policies(policies_path)
        checks["policy_roles"] = sorted(policies.roles.keys())

    mart_orders = 0
    if mart_db.is_file():
        con = duckdb.connect(str(mart_db), read_only=True)
        try:
            row = con.execute("SELECT COALESCE(SUM(order_count), 0) FROM fct_daily_gmv").fetchone()
            mart_orders = int(row[0] or 0) if row else 0
        finally:
            con.close()
    checks["baseline_mart_order_count"] = mart_orders

    passed = bool(
        checks["git_available"]
        and checks["configs_present"]
        and checks["workspace_present"]
        and mart_orders > 0
    )
    return {
        "passed": passed,
        "checked_at": datetime.now(UTC).isoformat(),
        "checks": checks,
    }


def _clone_isolated_workspace(source_ws: Path, target_ws: Path) -> Path:
    shutil.copytree(
        source_ws,
        target_ws,
        ignore=shutil.ignore_patterns(".git", ".scratch", "logs", ".user.yml", "__pycache__"),
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=target_ws, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "verify-harness"],
        cwd=target_ws,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "verify@dataops.local"],
        cwd=target_ws,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=target_ws, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "verify base"],
        cwd=target_ws,
        check=True,
        capture_output=True,
    )
    return target_ws


def _run_cli(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CLI command failed ({proc.returncode}): {' '.join(args)}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return json.loads(proc.stdout)


def drive_cli_lifecycle(isolated_root: Path) -> dict[str, Any]:
    """Drive Fix Run -> CLI status -> CLI reject -> resume -> CLI approve -> CLI status."""
    ws = _clone_isolated_workspace(REPO_ROOT / "workspace", isolated_root / "workspace")
    runs_dir = isolated_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    scenario = get_scenario("ops_hitl_reject_refix_promote")
    outcome = scenario.inject(ws)
    patch = scenario.reference_fix_patch(outcome)
    assert patch is not None

    run_id = "verify-hitl-run"
    wt = RunWorktreeManager.create(ws, runs_dir, run_id)
    oracle = scenario.build_oracle(outcome)
    verifier = ShadowOverlayVerifier(
        ws,
        runs_dir,
        lineage_adapter=LineageAdapter.from_workspace(wt),
        oracle=oracle,
    )
    stop_hook = StopGateHook(verifier=verifier, worktree_dir=wt, runs_dir=runs_dir)
    rca_json = RcaReport(
        incident_id=outcome.incident.incident_id,
        root_cause_summary="Removed accidental 2x multiplier in fct_daily_gmv.sql.",
        affected_assets=["fct_daily_gmv"],
        evidence=["assert_gmv_reconciliation failed after multiplier injection."],
        engine="dbt",
        proposed_fix_summary="Restore exact SUM(amount_usd) calculation.",
        patched_files=["dbt_project/models/marts/fct_daily_gmv.sql"],
        confidence="high",
    ).model_dump_json()

    loop = HarnessLoop(
        model_adapter=FakeReplayModelAdapter(
            [
                TurnResponse(
                    tool_calls=[
                        ToolCall(id="v-patch", name="apply_patch", arguments={"patch": patch})
                    ]
                ),
                TurnResponse(content=rca_json),
            ]
        ),
        tool_registry=ToolRegistry(wt, lineage_adapter=LineageAdapter.from_workspace(wt)),
        hooks=HookPipeline(stop_hooks=[stop_hook]),
        runs_dir=runs_dir,
    )
    first_state = loop.run(
        IncidentInput(
            incident_id=outcome.incident.incident_id,
            title=outcome.incident.title,
            summary=outcome.incident.summary,
            affected_assets=outcome.incident.affected_assets,
        ),
        UserContext(role="de", resolution_mode="fix"),
        run_id=run_id,
    )
    if first_state.status != "AWAITING_APPROVAL":
        raise RuntimeError(f"Expected AWAITING_APPROVAL, got {first_state.status}")

    status_initial = _run_cli(["status", run_id, "--runs-dir", str(runs_dir)])
    if status_initial["status"] != "AWAITING_APPROVAL":
        raise RuntimeError(f"Unexpected initial CLI status: {status_initial}")

    after_reject = _run_cli(
        [
            "resume",
            run_id,
            "--workspace",
            str(ws),
            "--runs-dir",
            str(runs_dir),
            "--reject",
            "--reason",
            "Confirm dim_user_ltv conservation alongside fct_daily_gmv.",
        ]
    )
    if after_reject["rejection_count"] != 1 or after_reject["max_turns"] != 28:
        raise RuntimeError(f"Unexpected CLI reject output: {after_reject}")

    resumed_loop = HarnessLoop(
        model_adapter=FakeReplayModelAdapter([TurnResponse(content=rca_json)]),
        tool_registry=ToolRegistry(wt, lineage_adapter=LineageAdapter.from_workspace(wt)),
        hooks=HookPipeline(stop_hooks=[stop_hook]),
        runs_dir=runs_dir,
    )
    resumed_state = resumed_loop.resume(run_id)
    if resumed_state.status != "AWAITING_APPROVAL":
        raise RuntimeError(f"Expected AWAITING_APPROVAL after resume, got {resumed_state.status}")

    after_approve = _run_cli(
        [
            "resume",
            run_id,
            "--workspace",
            str(ws),
            "--runs-dir",
            str(runs_dir),
            "--approve",
        ]
    )
    if after_approve["status"] != "RESOLVED" or not after_approve["dbt_test_ok"]:
        raise RuntimeError(f"Unexpected CLI approve output: {after_approve}")

    final_status = _run_cli(["status", run_id, "--runs-dir", str(runs_dir)])
    if final_status["status"] != "RESOLVED":
        raise RuntimeError(f"Unexpected final CLI status: {final_status}")

    con = duckdb.connect(str(LakehousePaths.from_workspace(ws).prod_duckdb_path), read_only=True)
    try:
        promoted_row = con.execute(
            "SELECT ROUND(COALESCE(SUM(net_gmv_usd), 0), 2) FROM fct_daily_gmv"
        ).fetchone()
        promoted_gmv = float(promoted_row[0]) if promoted_row else 0.0
    finally:
        con.close()

    return {
        "passed": abs(promoted_gmv - outcome.baseline_gmv_total) <= 0.01,
        "isolated_workspace": str(ws),
        "run_id": run_id,
        "baseline_gmv_total": outcome.baseline_gmv_total,
        "promoted_gmv_total": promoted_gmv,
        "initial_status": status_initial,
        "after_reject": after_reject,
        "after_resumed_reverify_status": resumed_state.status,
        "after_approve": after_approve,
        "final_status": final_status,
    }


def run_formal_verification(proof_dir: Path) -> dict[str, Any]:
    """Run formal RunLifecycle verifier and write proof into `proof_dir/formal_check.json`."""
    script_path = (
        REPO_ROOT
        / ".agents"
        / "skills"
        / "verify-run-state-machine"
        / "scripts"
        / "check_run_state_machine.py"
    )
    formal_out = proof_dir / "formal_check.json"
    proc = subprocess.run(
        [sys.executable, str(script_path), "--json-output", str(formal_out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Formal check failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    return json.loads(formal_out.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch isolated DataOps-Harness verification, capture proofs, and clean up."
    )
    parser.add_argument(
        "--proof-dir",
        type=Path,
        default=Path("/tmp/dataops-harness-proof"),
        help="Durable output directory outside the ephemeral sandbox root.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all 14 scenarios (including PySpark JVM jobs) in ablation verification.",
    )
    args = parser.parse_args(argv)

    proof_dir = args.proof_dir.resolve()
    proof_dir.mkdir(parents=True, exist_ok=True)
    isolated_root = Path(f"/tmp/dataops-verify-{uuid.uuid4().hex[:10]}")
    if isolated_root.exists():
        shutil.rmtree(isolated_root)
    isolated_root.mkdir(parents=True, exist_ok=True)

    cli_proof: dict[str, Any] = {}
    doctor_proof: dict[str, Any] = {}
    formal_proof: dict[str, Any] = {}
    ablation_summary: dict[str, Any] = {}

    try:
        doctor_proof = run_doctor_checks()
        (proof_dir / "doctor.json").write_text(
            json.dumps(doctor_proof, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not doctor_proof["passed"]:
            raise RuntimeError(f"Doctor check failed: {doctor_proof}")

        cli_proof = drive_cli_lifecycle(isolated_root)
        (proof_dir / "cli_lifecycle.json").write_text(
            json.dumps(cli_proof, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not cli_proof["passed"]:
            raise RuntimeError(f"CLI lifecycle proof failed: {cli_proof}")

        formal_proof = run_formal_verification(proof_dir)
        scenario_subset = None if args.all_scenarios else FAST_VERIFY_SCENARIOS
        runner = AblationBenchmarkRunner(provider="fake_replay", max_turns=20)
        ablation_report = runner.run(scenario_ids=scenario_subset)
        (proof_dir / "ablation_report.md").write_text(
            render_markdown_report(ablation_report), encoding="utf-8"
        )
        arm_c = ablation_report.arms[-1]
        ablation_summary = {
            "scenarios_evaluated": len(ablation_report.scenario_ids),
            "arm_c_passed": arm_c.scenarios_passed,
            "arm_c_verified_fix_rate": arm_c.verified_fix_rate,
            "arm_c_safety_interception_rate": arm_c.safety_interception_rate,
        }
    finally:
        if isolated_root.exists():
            shutil.rmtree(isolated_root)

    isolated_removed = not isolated_root.exists()
    summary_lines = [
        "# DataOps-Harness End-to-End Verification Proof Summary",
        "",
        f"- **Verified At**: `{datetime.now(UTC).isoformat()}`",
        f"- **Ephemeral Sandbox Torn Down**: `{isolated_root}` (`removed={isolated_removed}`)",
        f"- **Doctor Check**: `passed={doctor_proof.get('passed')}` ([doctor.json](doctor.json))",
        (
            f"- **CLI HiTL Lifecycle**: `status={cli_proof.get('final_status', {}).get('status')}`, "
            f"`gmv=${cli_proof.get('promoted_gmv_total')}` ([cli_lifecycle.json](cli_lifecycle.json))"
        ),
        (
            f"- **Formal RunLifecycle Verification**: `passed={formal_proof.get('passed')}`, "
            f"`states_explored={formal_proof.get('bfs_model_check', {}).get('states_explored')}` "
            "([formal_check.json](formal_check.json))"
        ),
        (
            f"- **Ablation Verification (`fake_replay`)**: `scenarios={ablation_summary.get('scenarios_evaluated')}`, "
            f"`arm_c_passed={ablation_summary.get('arm_c_passed')}` ([ablation_report.md](ablation_report.md))"
        ),
        "",
    ]
    (proof_dir / "verification_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    expected_artifacts = (
        "doctor.json",
        "cli_lifecycle.json",
        "formal_check.json",
        "ablation_report.md",
        "verification_summary.md",
    )
    missing = [
        name
        for name in expected_artifacts
        if not (proof_dir / name).is_file() or (proof_dir / name).stat().st_size == 0
    ]
    passed = isolated_removed and not missing
    print(
        json.dumps(
            {
                "passed": passed,
                "proof_dir": str(proof_dir),
                "isolated_root": str(isolated_root),
                "isolated_removed": isolated_removed,
                "artifacts": list(expected_artifacts),
                "missing_artifacts": missing,
                "ablation_summary": ablation_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
