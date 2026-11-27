#!/usr/bin/env python3
"""Formal state-space BFS model checker + live Python conformance verifier for Run lifecycle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scenarios.base import build_unified_diff
from src.adapters.base import LakehousePaths, SparkRunResult
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.model_adapter import FakeReplayModelAdapter, TurnResponse
from src.adapters.spark_adapter import SparkJobRunner
from src.core.context import (
    CheckpointStore,
    IncidentInput,
    Message,
    RunState,
    ToolCall,
    UserContext,
)
from src.core.loop import HarnessLoop
from src.core.policy import PoliciesConfig, load_policies
from src.core.promotion import PromotionService
from src.core.rca import RcaReport
from src.core.shadow_verifier import DefaultIntegrityOracle, ShadowOverlayVerifier
from src.core.worktree import RunWorktreeManager
from src.hooks.base import HookPipeline
from src.hooks.compactor import ContextCompactorHook
from src.hooks.stop_gate import StopGateHook
from src.tools.registry import ToolRegistry

TERMINAL_STATUSES = frozenset(
    {
        "RESOLVED",
        "DIAGNOSED",
        "ABORTED_THRASH",
        "ABORTED_BUDGET",
        "FAILED_ERROR",
    }
)


@dataclass(frozen=True)
class ModelState:
    """Finite abstract state matching specs/RunLifecycle.tla variables."""

    status: str
    resolution_mode: str
    turn: int
    max_turns: int
    rejection_count: int
    thrash_strikes: int
    shadow_green: bool
    approval_packet_ready: bool
    prod_mutated: bool


@dataclass(frozen=True)
class ModelBounds:
    """Exhaustive model-checking bounds."""

    init_max_turns: int = 2
    max_rejections: int = 2
    ext_turns: int = 2
    max_strikes: int = 2


def check_invariants(state: ModelState, bounds: ModelBounds) -> list[str]:
    """Evaluate all safety invariants declared in RunLifecycle.tla."""
    violations: list[str] = []
    if state.prod_mutated and state.status != "RESOLVED":
        violations.append(
            f"NoUnapprovedProdMutation violated: prod_mutated=True while status={state.status}"
        )
    if state.status == "RESOLVED" and (
        not state.shadow_green
        or not state.approval_packet_ready
        or state.resolution_mode != "fix"
    ):
        violations.append(
            "PromotionRequiresGreenShadow violated: RESOLVED reached without "
            "green shadow verification and ApprovalPacket in fix mode."
        )
    if state.status == "AWAITING_APPROVAL" and (
        not state.shadow_green
        or not state.approval_packet_ready
        or state.resolution_mode != "fix"
    ):
        violations.append(
            "AwaitingApprovalRequiresGreenShadow violated: entered AWAITING_APPROVAL "
            "without green shadow verification and ApprovalPacket."
        )
    if state.resolution_mode == "diagnosis_only" and (
        state.prod_mutated or state.status in ("AWAITING_APPROVAL", "RESOLVED")
    ):
        violations.append(
            f"DiagnosisOnlyNeverPromotes violated: mode=diagnosis_only reached {state.status}"
        )
    if state.rejection_count > bounds.max_rejections:
        violations.append(
            f"BoundedRejections violated: rejection_count={state.rejection_count} > {bounds.max_rejections}"
        )
    max_cap = bounds.init_max_turns + bounds.max_rejections * bounds.ext_turns
    if state.max_turns > max_cap or (state.status == "RUNNING" and state.turn > state.max_turns):
        violations.append(
            f"TurnBudgetBounded violated: turn={state.turn}, max_turns={state.max_turns}"
        )
    return violations


def next_states(
    state: ModelState,
    bounds: ModelBounds,
    *,
    inject_unapproved_promote_bug: bool = False,
    inject_rollback_leak_bug: bool = False,
) -> list[tuple[str, ModelState]]:
    """Enumerate all enabled actions from `state` matching RunLifecycle.tla Next."""
    if state.status in TERMINAL_STATUSES:
        return []

    out: list[tuple[str, ModelState]] = []
    if state.status == "RUNNING":
        if state.turn < state.max_turns:
            out.append(
                (
                    "ToolStepClean",
                    ModelState(
                        status="RUNNING",
                        resolution_mode=state.resolution_mode,
                        turn=state.turn + 1,
                        max_turns=state.max_turns,
                        rejection_count=state.rejection_count,
                        thrash_strikes=0,
                        shadow_green=False,
                        approval_packet_ready=False,
                        prod_mutated=state.prod_mutated,
                    ),
                )
            )
            if state.thrash_strikes + 1 < bounds.max_strikes:
                out.append(
                    (
                        "ToolStepThrashStrike",
                        ModelState(
                            status="RUNNING",
                            resolution_mode=state.resolution_mode,
                            turn=state.turn + 1,
                            max_turns=state.max_turns,
                            rejection_count=state.rejection_count,
                            thrash_strikes=state.thrash_strikes + 1,
                            shadow_green=False,
                            approval_packet_ready=False,
                            prod_mutated=state.prod_mutated,
                        ),
                    )
                )
            else:
                out.append(
                    (
                        "AntiThrashAbort",
                        ModelState(
                            status="ABORTED_THRASH",
                            resolution_mode=state.resolution_mode,
                            turn=state.turn + 1,
                            max_turns=state.max_turns,
                            rejection_count=state.rejection_count,
                            thrash_strikes=state.thrash_strikes + 1,
                            shadow_green=state.shadow_green,
                            approval_packet_ready=state.approval_packet_ready,
                            prod_mutated=state.prod_mutated,
                        ),
                    )
                )
            out.append(
                (
                    "CostOrWallClockBudgetAbort",
                    ModelState(
                        status="ABORTED_BUDGET",
                        resolution_mode=state.resolution_mode,
                        turn=state.turn + 1,
                        max_turns=state.max_turns,
                        rejection_count=state.rejection_count,
                        thrash_strikes=state.thrash_strikes,
                        shadow_green=state.shadow_green,
                        approval_packet_ready=state.approval_packet_ready,
                        prod_mutated=state.prod_mutated,
                    ),
                )
            )
            if state.resolution_mode == "diagnosis_only":
                out.append(
                    (
                        "StopDiagnosisOnly",
                        ModelState(
                            status="DIAGNOSED",
                            resolution_mode=state.resolution_mode,
                            turn=state.turn + 1,
                            max_turns=state.max_turns,
                            rejection_count=state.rejection_count,
                            thrash_strikes=state.thrash_strikes,
                            shadow_green=False,
                            approval_packet_ready=False,
                            prod_mutated=False,
                        ),
                    )
                )
            else:
                out.append(
                    (
                        "StopFixShadowFail",
                        ModelState(
                            status="RUNNING",
                            resolution_mode="fix",
                            turn=state.turn + 1,
                            max_turns=state.max_turns,
                            rejection_count=state.rejection_count,
                            thrash_strikes=state.thrash_strikes,
                            shadow_green=False,
                            approval_packet_ready=False,
                            prod_mutated=state.prod_mutated,
                        ),
                    )
                )
                out.append(
                    (
                        "StopFixShadowPass",
                        ModelState(
                            status="AWAITING_APPROVAL",
                            resolution_mode="fix",
                            turn=state.turn + 1,
                            max_turns=state.max_turns,
                            rejection_count=state.rejection_count,
                            thrash_strikes=state.thrash_strikes,
                            shadow_green=True,
                            approval_packet_ready=True,
                            prod_mutated=state.prod_mutated,
                        ),
                    )
                )
        else:
            out.append(
                (
                    "BudgetAbort",
                    ModelState(
                        status="ABORTED_BUDGET",
                        resolution_mode=state.resolution_mode,
                        turn=state.turn,
                        max_turns=state.max_turns,
                        rejection_count=state.rejection_count,
                        thrash_strikes=state.thrash_strikes,
                        shadow_green=state.shadow_green,
                        approval_packet_ready=state.approval_packet_ready,
                        prod_mutated=state.prod_mutated,
                    ),
                )
            )

        out.append(
            (
                "UnexpectedErrorAbort",
                ModelState(
                    status="FAILED_ERROR",
                    resolution_mode=state.resolution_mode,
                    turn=state.turn,
                    max_turns=state.max_turns,
                    rejection_count=state.rejection_count,
                    thrash_strikes=state.thrash_strikes,
                    shadow_green=state.shadow_green,
                    approval_packet_ready=state.approval_packet_ready,
                    prod_mutated=state.prod_mutated,
                ),
            )
        )
        if inject_unapproved_promote_bug:
            out.append(
                (
                    "BuggyDirectPromote",
                    ModelState(
                        status="RESOLVED",
                        resolution_mode=state.resolution_mode,
                        turn=state.turn,
                        max_turns=state.max_turns,
                        rejection_count=state.rejection_count,
                        thrash_strikes=state.thrash_strikes,
                        shadow_green=False,
                        approval_packet_ready=False,
                        prod_mutated=True,
                    ),
                )
            )

    if state.status == "AWAITING_APPROVAL":
        if state.rejection_count < bounds.max_rejections:
            out.append(
                (
                    "HiTLReject",
                    ModelState(
                        status="RUNNING",
                        resolution_mode=state.resolution_mode,
                        turn=state.turn,
                        max_turns=state.max_turns + bounds.ext_turns,
                        rejection_count=state.rejection_count + 1,
                        thrash_strikes=state.thrash_strikes,
                        shadow_green=False,
                        approval_packet_ready=False,
                        prod_mutated=state.prod_mutated,
                    ),
                )
            )
        if state.shadow_green and state.approval_packet_ready and state.resolution_mode == "fix":
            out.append(
                (
                    "HiTLApprovePromote",
                    ModelState(
                        status="RESOLVED",
                        resolution_mode=state.resolution_mode,
                        turn=state.turn,
                        max_turns=state.max_turns,
                        rejection_count=state.rejection_count,
                        thrash_strikes=state.thrash_strikes,
                        shadow_green=True,
                        approval_packet_ready=True,
                        prod_mutated=True,
                    ),
                )
            )
            out.append(
                (
                    "HiTLApproveRollback",
                    ModelState(
                        status="AWAITING_APPROVAL",
                        resolution_mode=state.resolution_mode,
                        turn=state.turn,
                        max_turns=state.max_turns,
                        rejection_count=state.rejection_count,
                        thrash_strikes=state.thrash_strikes,
                        shadow_green=True,
                        approval_packet_ready=True,
                        prod_mutated=False,
                    ),
                )
            )
            if inject_rollback_leak_bug:
                out.append(
                    (
                        "BuggyApproveRollbackLeak",
                        ModelState(
                            status="AWAITING_APPROVAL",
                            resolution_mode=state.resolution_mode,
                            turn=state.turn,
                            max_turns=state.max_turns,
                            rejection_count=state.rejection_count,
                            thrash_strikes=state.thrash_strikes,
                            shadow_green=True,
                            approval_packet_ready=True,
                            prod_mutated=True,
                        ),
                    )
                )

    return out


def run_bfs_model_check(
    bounds: ModelBounds,
    *,
    inject_unapproved_promote_bug: bool = False,
    inject_rollback_leak_bug: bool = False,
) -> dict[str, Any]:
    """Run exhaustive BFS over all reachable states and report statistics or counterexample."""
    roots = [
        ModelState(
            status="RUNNING",
            resolution_mode=mode,
            turn=0,
            max_turns=bounds.init_max_turns,
            rejection_count=0,
            thrash_strikes=0,
            shadow_green=False,
            approval_packet_ready=False,
            prod_mutated=False,
        )
        for mode in ("fix", "diagnosis_only")
    ]
    visited: set[ModelState] = set(roots)
    parent: dict[ModelState, tuple[ModelState | None, str]] = {r: (None, "Init") for r in roots}
    queue: deque[tuple[ModelState, int]] = deque((r, 0) for r in roots)
    transitions_explored = 0
    max_depth = 0
    reachable_statuses: set[str] = set()

    while queue:
        cur, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        reachable_statuses.add(cur.status)

        violations = check_invariants(cur, bounds)
        if violations:
            trace: list[str] = []
            cursor: ModelState | None = cur
            while cursor is not None:
                prev, act = parent[cursor]
                trace.append(
                    f"{act} -> status={cursor.status}, mode={cursor.resolution_mode}, "
                    f"turn={cursor.turn}/{cursor.max_turns}, rejections={cursor.rejection_count}, "
                    f"shadow_green={cursor.shadow_green}, prod_mutated={cursor.prod_mutated}"
                )
                cursor = prev
            trace.reverse()
            return {
                "passed": False,
                "states_explored": len(visited),
                "transitions_explored": transitions_explored,
                "max_depth": max_depth,
                "violations": violations,
                "counterexample": trace,
            }

        for action, nxt in next_states(
            cur,
            bounds,
            inject_unapproved_promote_bug=inject_unapproved_promote_bug,
            inject_rollback_leak_bug=inject_rollback_leak_bug,
        ):
            transitions_explored += 1
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = (cur, action)
                queue.append((nxt, depth + 1))

    return {
        "passed": True,
        "states_explored": len(visited),
        "transitions_explored": transitions_explored,
        "max_depth": max_depth,
        "reachable_statuses": sorted(reachable_statuses),
        "violations": [],
        "counterexample": [],
    }


def _ok_spark_result(script_path: Path) -> SparkRunResult:
    return SparkRunResult(
        job_name=script_path.name,
        script_path=script_path,
        pid=0,
        returncode=0,
        duration_ms=1.0,
        event_log_path=None,
        stdout="ok",
        stderr="",
    )


class _NoOpSparkRunner(SparkJobRunner):
    def __init__(self, event_log_dir: Path) -> None:
        super().__init__(event_log_dir=event_log_dir)

    def run_job(
        self,
        script_path: Path,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SparkRunResult:
        return _ok_spark_result(script_path)


class _MutatingSparkRunner(SparkJobRunner):
    """Test double that writes a marker into `--output-dir/silver_orders/` to simulate Silver mutation."""

    def __init__(self, event_log_dir: Path, marker_name: str = "_formal_mutated_marker") -> None:
        super().__init__(event_log_dir=event_log_dir)
        self.marker_name = marker_name

    def run_job(
        self,
        script_path: Path,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SparkRunResult:
        arg_list = args or []
        if "--output-dir" in arg_list:
            out_dir = Path(arg_list[arg_list.index("--output-dir") + 1])
            target = out_dir / "silver_orders"
            target.mkdir(parents=True, exist_ok=True)
            (target / self.marker_name).write_text("mutated_during_promotion", encoding="utf-8")
        return _ok_spark_result(script_path)


def _ok_dbt(
    dbt_paths: LakehousePaths,
    silver_overrides: dict[str, Path],
    mart_duckdb_path: Path,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["dbt", *args], 0, stdout="ok", stderr="")


def _fail_dbt_mutating(
    dbt_paths: LakehousePaths,
    silver_overrides: dict[str, Path],
    mart_duckdb_path: Path,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    mart_duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    mart_duckdb_path.write_bytes(b"CORRUPTED-BY-FAILED-PROMOTE-DBT")
    return subprocess.CompletedProcess(["dbt", *args], 1, stdout="", stderr="simulated dbt failure")


def _init_mini_git_workspace(target_ws: Path) -> Path:
    source_ws = REPO_ROOT / "workspace"
    shutil.copytree(
        source_ws,
        target_ws,
        ignore=shutil.ignore_patterns(".git", ".scratch", "logs", ".user.yml", "__pycache__"),
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=target_ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "formal-check"], cwd=target_ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "formal@dataops.local"], cwd=target_ws, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=target_ws, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=target_ws, check=True, capture_output=True)
    return target_ws


def _head_sha(ws: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_named_check(
    name: str, checks: list[dict[str, Any]], fn: Callable[[], None]
) -> None:
    try:
        fn()
    except (AssertionError, ValueError, RuntimeError, IndexError) as exc:
        checks.append({"name": name, "passed": False, "error": str(exc)})
    else:
        checks.append({"name": name, "passed": True})


def _prepare_awaiting_approval_run(
    ws: Path,
    runs_dir: Path,
    run_id: str,
    rel_file: str,
    *,
    incident_id: str,
    policies: PoliciesConfig,
    spark_runner: SparkJobRunner,
    compactor: ContextCompactorHook | None = None,
) -> tuple[Path, StopGateHook, RunState]:
    wt = RunWorktreeManager.create(ws, runs_dir, run_id)
    target = wt / rel_file
    orig_src = target.read_text(encoding="utf-8")
    patch = build_unified_diff(
        rel_file, orig_src, orig_src + "\n# formal verification harmless comment\n"
    )
    verifier = ShadowOverlayVerifier(
        ws,
        runs_dir,
        lineage_adapter=LineageAdapter.from_workspace(ws),
        oracle=DefaultIntegrityOracle(),
        spark_runner=spark_runner,
        dbt_executor=_ok_dbt,
    )
    stop_hook = StopGateHook(verifier=verifier, worktree_dir=wt, runs_dir=runs_dir)
    rca = RcaReport(
        incident_id=incident_id,
        root_cause_summary="Added harmless documentation comment",
        affected_assets=["fct_daily_gmv"],
        evidence=["verified"],
        engine="spark" if rel_file.startswith("spark_jobs/") else "dbt",
        proposed_fix_summary=f"Patch {rel_file}",
        patched_files=[rel_file],
        confidence="high",
    ).model_dump_json()

    post_hooks = [compactor] if compactor is not None else []
    loop = HarnessLoop(
        model_adapter=FakeReplayModelAdapter(
            [
                TurnResponse(
                    tool_calls=[
                        ToolCall(id="p-init", name="apply_patch", arguments={"patch": patch})
                    ]
                ),
                TurnResponse(content=rca),
            ]
        ),
        tool_registry=ToolRegistry(wt),
        hooks=HookPipeline(post_tool_hooks=post_hooks, stop_hooks=[stop_hook]),
        runs_dir=runs_dir,
        policies=policies,
    )
    state = loop.run(
        IncidentInput(incident_id=incident_id, title="HiTL check", summary="Formal check"),
        UserContext(role="de", resolution_mode="fix"),
        run_id=run_id,
    )
    assert state.status == "AWAITING_APPROVAL", f"Expected AWAITING_APPROVAL, got {state.status}"
    return wt, stop_hook, state


def run_live_code_conformance() -> dict[str, Any]:
    """Exercise real CheckpointStore, HarnessLoop, ContextCompactorHook, StopGateHook, and PromotionService."""
    policies = load_policies()
    checks: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="dataops-formal-") as tmp_str:
        tmp = Path(tmp_str)
        ws = _init_mini_git_workspace(tmp / "workspace")
        runs_dir = tmp / "runs"
        runs_dir.mkdir(parents=True)
        base_sha = _head_sha(ws)
        svc = PromotionService(
            ws,
            runs_dir,
            policies=policies,
            spark_runner=_NoOpSparkRunner(tmp / "sp"),
            dbt_executor=_ok_dbt,
        )

        # Check 1: Unapproved RUNNING promotion blocked + dbt & Spark+dbt rollback atomicity.
        def _check_no_unapproved_or_partial_rollback() -> None:
            run1 = "formal-unapproved-running"
            loop1 = HarnessLoop(
                model_adapter=FakeReplayModelAdapter(
                    [
                        TurnResponse(
                            tool_calls=[
                                ToolCall(
                                    id="r1",
                                    name="read_file",
                                    arguments={"path": "dbt_project/models/marts/fct_daily_gmv.sql"},
                                )
                            ]
                        ),
                        TurnResponse(content="Reached turn 2 limit."),
                    ]
                ),
                tool_registry=ToolRegistry(ws),
                runs_dir=runs_dir,
                policies=policies,
            )
            st1 = loop1.run(
                IncidentInput(incident_id="INC-F1", title="Formal check 1", summary="Budget pause check"),
                UserContext(role="de", resolution_mode="fix"),
                run_id=run1,
                max_turns=1,
            )
            assert st1.status == "ABORTED_BUDGET", f"Setup expected ABORTED_BUDGET, got {st1.status}"
            store1 = CheckpointStore(runs_dir, run1)
            st1.status = "RUNNING"
            store1.append("crash_simulated", st1)
            blocked_running = False
            try:
                svc.approve(run1)
            except ValueError:
                blocked_running = True
            assert blocked_running and _head_sha(ws) == base_sha, (
                "NoUnapprovedProdMutation / approve() allowed promotion from RUNNING."
            )

            # Subcheck 1b: Dbt-only failed post-commit promotion rolls back git + DuckDB.
            ws_dbt = _init_mini_git_workspace(tmp / "ws_rb_dbt")
            pre_dbt_sha = _head_sha(ws_dbt)
            dbt_paths = LakehousePaths.from_workspace(ws_dbt)
            pre_dbt_bytes = dbt_paths.prod_duckdb_path.read_bytes()
            wt_dbt, _, _ = _prepare_awaiting_approval_run(
                ws_dbt,
                runs_dir,
                "formal-rb-dbt",
                "dbt_project/models/marts/fct_daily_gmv.sql",
                incident_id="INC-RB-DBT",
                policies=policies,
                spark_runner=_NoOpSparkRunner(tmp / "sp_rb_dbt"),
            )
            svc_fail_dbt = PromotionService(
                ws_dbt,
                runs_dir,
                policies=policies,
                spark_runner=_NoOpSparkRunner(tmp / "sp_rb_dbt2"),
                dbt_executor=_fail_dbt_mutating,
            )
            failed_dbt = False
            try:
                svc_fail_dbt.approve("formal-rb-dbt")
            except RuntimeError:
                failed_dbt = True
            assert failed_dbt, "Expected RuntimeError when post-commit dbt verification fails."
            assert _head_sha(ws_dbt) == pre_dbt_sha, "Dbt failed promotion did not roll back git HEAD."
            assert dbt_paths.prod_duckdb_path.read_bytes() == pre_dbt_bytes, (
                "Dbt failed promotion did not restore warehouse.duckdb."
            )
            assert CheckpointStore(runs_dir, "formal-rb-dbt").load_latest().status == "AWAITING_APPROVAL"
            RunWorktreeManager.cleanup(ws_dbt, wt_dbt)

            # Subcheck 1c: Spark+dbt failed promotion must roll back git, DuckDB, AND Silver Delta writes.
            ws_spark = _init_mini_git_workspace(tmp / "ws_rb_spark")
            pre_spark_sha = _head_sha(ws_spark)
            sp_paths = LakehousePaths.from_workspace(ws_spark)
            pre_sp_db_bytes = sp_paths.prod_duckdb_path.read_bytes()
            wt_sp, _, _ = _prepare_awaiting_approval_run(
                ws_spark,
                runs_dir,
                "formal-rb-spark",
                "spark_jobs/build_silver_orders.py",
                incident_id="INC-RB-SPARK",
                policies=policies,
                spark_runner=_NoOpSparkRunner(tmp / "sp_rb_sp"),
            )
            svc_fail_spark = PromotionService(
                ws_spark,
                runs_dir,
                policies=policies,
                spark_runner=_MutatingSparkRunner(tmp / "sp_rb_sp2"),
                dbt_executor=_fail_dbt_mutating,
            )
            failed_sp = False
            try:
                svc_fail_spark.approve("formal-rb-spark")
            except RuntimeError:
                failed_sp = True
            marker = sp_paths.prod_silver_dir / "silver_orders" / "_formal_mutated_marker"
            assert failed_sp, "Expected RuntimeError when post-commit Spark+dbt verification fails."
            assert _head_sha(ws_spark) == pre_spark_sha, "Spark+dbt failure leaked git commit."
            assert sp_paths.prod_duckdb_path.read_bytes() == pre_sp_db_bytes, (
                "Spark+dbt failure leaked warehouse.duckdb mutation."
            )
            assert not marker.exists(), (
                "NoUnapprovedProdMutation violated: PromotionService.approve() wrote into "
                "prod_silver_dir during failed Spark+dbt promotion and did not roll back Silver storage."
            )
            RunWorktreeManager.cleanup(ws_spark, wt_sp)

        _run_named_check("no_unapproved_promote_from_running", checks, _check_no_unapproved_or_partial_rollback)

        # Check 2: Diagnosis-only Run + DA role mode ceiling clamping cannot promote.
        def _check_diagnosis_only_and_role_ceiling() -> None:
            diag_rca = RcaReport(
                incident_id="INC-DIAG",
                root_cause_summary="Read-only diagnosis",
                affected_assets=["fct_daily_gmv"],
                evidence=["checked SQL"],
                engine="dbt",
                proposed_fix_summary="No code mutation required.",
                patched_files=[],
                confidence="high",
            ).model_dump_json()
            for idx, req_mode in enumerate(("diagnosis_only", "fix")):
                rid = f"formal-diag-{idx}"
                loop_diag = HarnessLoop(
                    model_adapter=FakeReplayModelAdapter(
                        [TurnResponse(content=diag_rca), TurnResponse(content=diag_rca)]
                    ),
                    tool_registry=ToolRegistry(ws),
                    hooks=HookPipeline(stop_hooks=[StopGateHook(runs_dir=runs_dir)]),
                    runs_dir=runs_dir,
                    policies=policies,
                )
                st_diag = loop_diag.run(
                    IncidentInput(incident_id="INC-DIAG", title="Diag", summary="Diag mode"),
                    UserContext(role="da", resolution_mode=req_mode),
                    run_id=rid,
                    max_turns=2,
                )
                diag_blocked = False
                try:
                    svc.approve(rid)
                except ValueError:
                    diag_blocked = True
                assert (
                    st_diag.user_context.resolution_mode == "diagnosis_only"
                    and st_diag.status == "DIAGNOSED"
                    and diag_blocked
                    and _head_sha(ws) == base_sha
                ), (
                    f"DiagnosisOnlyNeverPromotes / _enforce_role_mode_ceiling failed for "
                    f"role='da' resolution_mode={req_mode!r}: effective={st_diag.user_context.resolution_mode!r}, "
                    f"status={st_diag.status!r}."
                )

        _run_named_check("diagnosis_only_never_promotes", checks, _check_diagnosis_only_and_role_ceiling)

        # Shared HiTL Run used across Check 3, Check 4, and Check 5 (run independently inside Checks 3-5).
        t0 = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
        clock = [t0]
        compactor = ContextCompactorHook(workspace_root=ws, policies=policies, now_fn=lambda: clock[0])
        wt_hitl, stop_hitl, st_await = _prepare_awaiting_approval_run(
            ws,
            runs_dir,
            "formal-hitl",
            "dbt_project/models/marts/fct_daily_gmv.sql",
            incident_id="INC-HITL",
            policies=policies,
            spark_runner=_NoOpSparkRunner(tmp / "sp2"),
            compactor=compactor,
        )
        valid_rca = RcaReport(
            incident_id="INC-HITL",
            root_cause_summary="Added harmless documentation comment",
            affected_assets=["fct_daily_gmv"],
            evidence=["verified"],
            engine="dbt",
            proposed_fix_summary="Patch fct_daily_gmv",
            patched_files=["dbt_project/models/marts/fct_daily_gmv.sql"],
            confidence="high",
        ).model_dump_json()

        # Check 3: Tampering ApprovalPacket.shadow_verification.passed=False blocks approve().
        def _check_promotion_requires_green_shadow() -> None:
            store_hitl = CheckpointStore(runs_dir, "formal-hitl")
            tampered = store_hitl.load_latest().model_copy(deep=True)
            tampered.metadata["approval_packet"]["shadow_verification"]["passed"] = False
            store_hitl.append("tamper_shadow_test", tampered)
            tamper_blocked = False
            try:
                svc.approve("formal-hitl")
            except ValueError:
                tamper_blocked = True
            finally:
                store_hitl.append("restore_green_shadow", st_await)
            assert tamper_blocked and _head_sha(ws) == base_sha, (
                "PromotionRequiresGreenShadow violated: approve() allowed passed=False shadow verification."
            )

        _run_named_check("promotion_requires_green_shadow", checks, _check_promotion_requires_green_shadow)

        # Check 4: Bounded rejections (+8 turns), HiTL pause clock exclusion, and non-inflated extra_turns.
        def _check_bounded_rejections_and_pause_clock() -> None:
            resume_adapter = FakeReplayModelAdapter(
                [TurnResponse(content=valid_rca), TurnResponse(content=valid_rca)]
            )
            resume_loop = HarnessLoop(
                model_adapter=resume_adapter,
                tool_registry=ToolRegistry(wt_hitl),
                hooks=HookPipeline(post_tool_hooks=[compactor], stop_hooks=[stop_hitl]),
                runs_dir=runs_dir,
                policies=policies,
            )
            clock[0] = t0 + timedelta(seconds=360)
            st_r1 = svc.reject("formal-hitl", "First review note", loop=resume_loop)
            assert st_r1.status == "AWAITING_APPROVAL", (
                f"Expected AWAITING_APPROVAL after HiTL pause resume, got {st_r1.status} "
                f"(reason={st_r1.metadata.get('abort_reason')})"
            )
            assert float(st_r1.metadata.get("paused_seconds", 0.0)) >= 360.0, (
                f"Pause clock exclusion failed: paused_seconds={st_r1.metadata.get('paused_seconds')}"
            )
            assert int(st_r1.metadata.get("extra_turns", 0)) == 0, (
                f"Rejection extension leaked into extra_turns={st_r1.metadata.get('extra_turns')}"
            )
            expected_m1 = policies.limits.max_turns + policies.limits.rejection_turn_extension
            assert compactor._effective_max_turns(st_r1) == st_r1.max_turns == expected_m1, (
                f"Expected max_turns={expected_m1}, got max_turns={st_r1.max_turns}, "
                f"effective={compactor._effective_max_turns(st_r1)}"
            )

            st_r2 = svc.reject("formal-hitl", "Second review note", loop=resume_loop)
            assert (
                st_r2.max_turns
                == policies.limits.max_turns + 2 * policies.limits.rejection_turn_extension
                and st_r2.rejection_count == policies.limits.max_rejections
            ), "Second rejection did not preserve bounded +8 extension."

            third_blocked = False
            try:
                svc.reject("formal-hitl", "Third exceeding max_rejections", loop=resume_loop)
            except ValueError:
                third_blocked = True
            assert third_blocked, "BoundedRejections violated: 3rd rejection succeeded."

        _run_named_check("bounded_rejections_and_turn_extension", checks, _check_bounded_rejections_and_pause_clock)

        # Check 5: Terminal RESOLVED absorbing (no re-approve) + HarnessLoop.resume() terminal guards.
        def _check_terminal_states_absorbing() -> None:
            promo = svc.approve("formal-hitl")
            reapprove_blocked = False
            try:
                svc.approve("formal-hitl")
            except ValueError:
                reapprove_blocked = True
            assert promo.status == "RESOLVED" and reapprove_blocked and _head_sha(ws) != base_sha, (
                "Terminal RESOLVED promotion or idempotency guard failed."
            )

            guard_loop = HarnessLoop(
                model_adapter=FakeReplayModelAdapter([TurnResponse(content=valid_rca)]),
                tool_registry=ToolRegistry(ws),
                hooks=HookPipeline(stop_hooks=[StopGateHook(runs_dir=runs_dir)]),
                runs_dir=runs_dir,
                policies=policies,
            )
            # Budget zero-extension blocked, positive extension permitted (documented operator override).
            rid_budget = "formal-term-budget"
            st_budget = RunState(
                run_id=rid_budget,
                incident=IncidentInput(incident_id="INC-TB", title="Budget", summary="Budget"),
                user_context=UserContext(role="da", resolution_mode="diagnosis_only"),
                status="ABORTED_BUDGET",
                turn=1,
                max_turns=1,
            )
            CheckpointStore(runs_dir, rid_budget).append("budget", st_budget)
            zero_blocked = False
            try:
                guard_loop.resume(rid_budget, extra_turns=0)
            except ValueError:
                zero_blocked = True
            assert zero_blocked, "HarnessLoop.resume() allowed ABORTED_BUDGET with extra_turns=0."
            resumed_budget = guard_loop.resume(rid_budget, extra_turns=2)
            assert resumed_budget.status == "DIAGNOSED", (
                "Operator override resume(ABORTED_BUDGET, extra_turns=2) did not succeed."
            )

            # All other terminal RunStatus values must refuse resume() even with extra_turns=2.
            for idx, term_status in enumerate(
                ("RESOLVED", "DIAGNOSED", "ABORTED_THRASH", "FAILED_ERROR")
            ):
                rid = f"formal-term-{idx}"
                st = RunState(
                    run_id=rid,
                    incident=IncidentInput(incident_id="INC-T", title="Term", summary="Term"),
                    user_context=UserContext(role="da", resolution_mode="diagnosis_only"),
                    status=term_status,  # type: ignore[arg-type]
                )
                CheckpointStore(runs_dir, rid).append("term", st)
                term_loop = HarnessLoop(
                    model_adapter=FakeReplayModelAdapter([TurnResponse(content="Should not run")]),
                    tool_registry=ToolRegistry(ws),
                    runs_dir=runs_dir,
                    policies=policies,
                )
                refused = False
                try:
                    term_loop.resume(rid, extra_turns=2)
                except ValueError:
                    refused = True
                assert refused, (
                    f"TerminalStatesAbsorbing violated: HarnessLoop.resume() allowed continuing "
                    f"terminal status {term_status!r}."
                )

        _run_named_check("terminal_resolved_absorbing", checks, _check_terminal_states_absorbing)

        # Check 6: Crash/resume continuity + mid-tool-batch crash reconciliation.
        def _check_crash_resume_invariants() -> None:
            run_crash = "formal-crash-resume"
            crash_wt = RunWorktreeManager.create(ws, runs_dir, run_crash)
            crash_sql = crash_wt / "dbt_project" / "models" / "marts" / "fct_daily_gmv.sql"
            crash_orig = crash_sql.read_text(encoding="utf-8")
            crash_patch = build_unified_diff(
                "dbt_project/models/marts/fct_daily_gmv.sql",
                crash_orig,
                crash_orig + "\n-- crash resume verification marker\n",
            )
            crash_verifier = ShadowOverlayVerifier(
                ws,
                runs_dir,
                lineage_adapter=LineageAdapter.from_workspace(ws),
                oracle=DefaultIntegrityOracle(),
                spark_runner=_NoOpSparkRunner(tmp / "sp_crash"),
                dbt_executor=_ok_dbt,
            )
            crash_stop = StopGateHook(verifier=crash_verifier, worktree_dir=crash_wt, runs_dir=runs_dir)
            crash_rca = RcaReport(
                incident_id="INC-CRASH",
                root_cause_summary="Added harmless marker after crash recovery",
                affected_assets=["fct_daily_gmv"],
                evidence=["resumed from local checkpoint"],
                engine="dbt",
                proposed_fix_summary="Patch fct_daily_gmv",
                patched_files=["dbt_project/models/marts/fct_daily_gmv.sql"],
                confidence="high",
            ).model_dump_json()

            pre_crash_loop = HarnessLoop(
                model_adapter=FakeReplayModelAdapter(
                    [
                        TurnResponse(
                            tool_calls=[
                                ToolCall(
                                    id="crash-read",
                                    name="read_file",
                                    arguments={"path": "dbt_project/models/marts/fct_daily_gmv.sql"},
                                )
                            ]
                        ),
                        TurnResponse(
                            tool_calls=[
                                ToolCall(
                                    id="crash-read-2",
                                    name="read_file",
                                    arguments={"path": "dbt_project/models/marts/dim_user_ltv.sql"},
                                )
                            ]
                        ),
                    ]
                ),
                tool_registry=ToolRegistry(crash_wt),
                hooks=HookPipeline(stop_hooks=[crash_stop]),
                runs_dir=runs_dir,
                policies=policies,
            )
            aborted_crash = pre_crash_loop.run(
                IncidentInput(incident_id="INC-CRASH", title="Crash check", summary="Resume invariant check"),
                UserContext(role="de", resolution_mode="fix"),
                run_id=run_crash,
                max_turns=1,
            )
            assert (
                aborted_crash.turn == 1
                and aborted_crash.max_turns == 1
                and aborted_crash.rejection_count == 0
            ), "Crash simulation setup failed to preserve turn=1/max_turns=1."

            crash_store = CheckpointStore(runs_dir, run_crash)
            reloaded = crash_store.load_latest()
            reloaded.status = "RUNNING"
            reloaded.max_turns = policies.limits.max_turns
            crash_store.append("crash_recovered", reloaded)

            resumed_crash_loop = HarnessLoop(
                model_adapter=FakeReplayModelAdapter(
                    [
                        TurnResponse(
                            tool_calls=[
                                ToolCall(
                                    id="crash-patch",
                                    name="apply_patch",
                                    arguments={"patch": crash_patch},
                                )
                            ]
                        ),
                        TurnResponse(content=crash_rca),
                    ]
                ),
                tool_registry=ToolRegistry(crash_wt),
                hooks=HookPipeline(stop_hooks=[crash_stop]),
                runs_dir=runs_dir,
                policies=policies,
            )
            post_crash = resumed_crash_loop.resume(run_crash)
            assert (
                post_crash.status == "AWAITING_APPROVAL"
                and post_crash.turn >= 3
                and post_crash.max_turns == policies.limits.max_turns
                and post_crash.rejection_count == 0
                and len(post_crash.messages) > len(aborted_crash.messages)
            ), "CrashResumePreservesInvariants violated: turn/max_turns/rejection_count continuity broke."

            # Mid-tool-batch crash reconciliation check.
            mid_rid = "formal-mid-tool-crash"
            mid_state = RunState(
                run_id=mid_rid,
                incident=IncidentInput(incident_id="INC-MID", title="Mid crash", summary="Mid crash"),
                user_context=UserContext(role="da", resolution_mode="diagnosis_only"),
                status="RUNNING",
                turn=1,
                max_turns=5,
                messages=[
                    Message(role="user", content="Start"),
                    Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id="m1",
                                name="read_file",
                                arguments={"path": "dbt_project/models/marts/fct_daily_gmv.sql"},
                            ),
                            ToolCall(
                                id="m2",
                                name="read_file",
                                arguments={"path": "dbt_project/models/marts/dim_user_ltv.sql"},
                            ),
                        ],
                    ),
                    Message(role="tool", content="first ok", tool_call_id="m1", tool_name="read_file"),
                ],
            )
            CheckpointStore(runs_dir, mid_rid).append("mid_tool_step", mid_state)
            rec_adapter = FakeReplayModelAdapter([TurnResponse(content="Reconciled.")])
            rec_loop = HarnessLoop(
                model_adapter=rec_adapter,
                tool_registry=ToolRegistry(ws),
                runs_dir=runs_dir,
                policies=policies,
            )
            rec_done = rec_loop.resume(mid_rid)
            sent = rec_adapter.recorded_requests[0].messages
            assert (
                rec_done.status == "DIAGNOSED"
                and [m.role for m in sent[:4]] == ["user", "assistant", "tool", "tool"]
                and sent[2].tool_call_id == "m1"
                and sent[3].tool_call_id == "m2"
            ), "Mid-tool-batch crash reconciliation did not synthesize contiguous tool results."

        _run_named_check("crash_resume_preserves_invariants", checks, _check_crash_resume_invariants)

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exhaustive BFS formal check and live conformance verifier for Run lifecycle."
    )
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Run only finite BFS model check without Python runtime conformance.",
    )
    args = parser.parse_args(argv)

    bounds = ModelBounds()
    bfs_clean = run_bfs_model_check(bounds, inject_unapproved_promote_bug=False)
    bfs_buggy = run_bfs_model_check(bounds, inject_unapproved_promote_bug=True)
    bfs_rollback_bug = run_bfs_model_check(bounds, inject_rollback_leak_bug=True)
    live = {"passed": True, "checks": []} if args.skip_live else run_live_code_conformance()

    overall_passed = bool(
        bfs_clean["passed"]
        and (not bfs_buggy["passed"])
        and (not bfs_rollback_bug["passed"])
        and live["passed"]
    )
    report = {
        "passed": overall_passed,
        "bounds": {
            "init_max_turns": bounds.init_max_turns,
            "max_rejections": bounds.max_rejections,
            "ext_turns": bounds.ext_turns,
            "max_strikes": bounds.max_strikes,
        },
        "bfs_model_check": bfs_clean,
        "mutation_negative_control": {
            "bug_caught": not bfs_buggy["passed"],
            "violations": bfs_buggy["violations"],
            "counterexample": bfs_buggy["counterexample"],
            "rollback_leak_caught": not bfs_rollback_bug["passed"],
            "rollback_leak_violations": bfs_rollback_bug["violations"],
            "rollback_leak_counterexample": bfs_rollback_bug["counterexample"],
        },
        "live_code_conformance": live,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
