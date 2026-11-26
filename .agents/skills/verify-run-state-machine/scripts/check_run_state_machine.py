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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adapters.base import LakehousePaths, SparkRunResult
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.model_adapter import FakeReplayModelAdapter, TurnResponse
from src.adapters.spark_adapter import SparkJobRunner
from src.core.context import (
    CheckpointStore,
    IncidentInput,
    ToolCall,
    UserContext,
)
from src.core.loop import HarnessLoop
from src.core.policy import load_policies
from src.core.promotion import PromotionService
from src.core.rca import RcaReport
from src.core.shadow_verifier import DefaultIntegrityOracle, ShadowOverlayVerifier
from src.core.worktree import RunWorktreeManager
from src.hooks.base import HookPipeline
from src.hooks.stop_gate import StopGateHook
from src.tools.registry import ToolRegistry

TERMINAL_STATUSES = frozenset(
    {
        "RESOLVED",
        "DIAGNOSED",
        "ABORTED_THRASH",
        "ABORTED_BUDGET",
        "ABORTED_ERROR",
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
                    status="ABORTED_ERROR",
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

    return out


def run_bfs_model_check(
    bounds: ModelBounds, *, inject_unapproved_promote_bug: bool = False
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
            cur, bounds, inject_unapproved_promote_bug=inject_unapproved_promote_bug
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


class _NoOpSparkRunner(SparkJobRunner):
    def __init__(self, event_log_dir: Path) -> None:
        super().__init__(event_log_dir=event_log_dir)

    def run_job(self, script_path: Path, args: list[str] | None = None, env_overrides: dict[str, str] | None = None) -> SparkRunResult:
        return SparkRunResult(job_name=script_path.name, returncode=0)


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


def run_live_code_conformance() -> dict[str, Any]:
    """Exercise real CheckpointStore, HarnessLoop, StopGateHook, and PromotionService guards."""
    policies = load_policies()
    checks: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="dataops-formal-") as tmp_str:
        tmp = Path(tmp_str)
        ws = _init_mini_git_workspace(tmp / "workspace")
        runs_dir = tmp / "runs"
        runs_dir.mkdir(parents=True)

        def _ok_dbt(
            dbt_paths: LakehousePaths,
            silver_overrides: dict[str, Path],
            mart_duckdb_path: Path,
            args: list[str],
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(["dbt", *args], 0, stdout="ok", stderr="")

        def _head_sha() -> str:
            return subprocess.run(
                ["git", "-C", str(ws), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        base_sha = _head_sha()

        # Check 1: Unapproved Run in RUNNING must NOT be promoted via PromotionService.approve().
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
        )
        inc1 = IncidentInput(incident_id="INC-F1", title="Formal check 1", summary="Budget pause check")
        st1 = loop1.run(inc1, UserContext(role="de", resolution_mode="fix"), run_id=run1, max_turns=1)
        if st1.status != "ABORTED_BUDGET":
            return _conformance_fail("Check 1 setup failed: expected ABORTED_BUDGET.", checks)

        # Force status to RUNNING directly in checkpoint and confirm approve() raises ValueError.
        store1 = CheckpointStore(runs_dir, run1)
        st1.status = "RUNNING"
        store1.append("crash_simulated", st1)
        svc = PromotionService(
            ws,
            runs_dir,
            policies=policies,
            spark_runner=_NoOpSparkRunner(tmp / "sp"),
            dbt_executor=_ok_dbt,
        )
        blocked_running = False
        try:
            svc.approve(run1)
        except ValueError:
            blocked_running = True
        if not blocked_running or _head_sha() != base_sha:
            return _conformance_fail(
                "NoUnapprovedProdMutation / PromotionService.approve() allowed promotion outside AWAITING_APPROVAL!",
                checks,
            )
        checks.append({"name": "no_unapproved_promote_from_running", "passed": True})

        # Check 2: Diagnosis-only Run reaches DIAGNOSED and cannot be promoted.
        run_diag = "formal-diag"
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
        loop_diag = HarnessLoop(
            model_adapter=FakeReplayModelAdapter([TurnResponse(content=diag_rca)]),
            tool_registry=ToolRegistry(ws),
            hooks=HookPipeline(stop_hooks=[StopGateHook(runs_dir=runs_dir)]),
            runs_dir=runs_dir,
        )
        st_diag = loop_diag.run(
            IncidentInput(incident_id="INC-DIAG", title="Diag", summary="Diag mode"),
            UserContext(role="da", resolution_mode="diagnosis_only"),
            run_id=run_diag,
        )
        diag_blocked = False
        try:
            svc.approve(run_diag)
        except ValueError:
            diag_blocked = True
        if st_diag.status != "DIAGNOSED" or not diag_blocked or _head_sha() != base_sha:
            return _conformance_fail(
                "DiagnosisOnlyNeverPromotes violated on live code.", checks
            )
        checks.append({"name": "diagnosis_only_never_promotes", "passed": True})

        # Check 3: Green shadow required, rejection bound enforced (max 2), approve -> RESOLVED,
        # and terminal RESOLVED cannot be re-approved.
        run_hitl = "formal-hitl"
        wt = RunWorktreeManager.create(ws, runs_dir, run_hitl)
        sql_path = wt / "dbt_project" / "models" / "marts" / "fct_daily_gmv.sql"
        orig_sql = sql_path.read_text(encoding="utf-8")
        from benchmarks.scenarios.base import build_unified_diff

        patch = build_unified_diff(
            "dbt_project/models/marts/fct_daily_gmv.sql",
            orig_sql,
            orig_sql + "\n-- formal verification harmless comment\n",
        )
        verifier = ShadowOverlayVerifier(
            ws,
            runs_dir,
            lineage_adapter=LineageAdapter.from_workspace(ws),
            oracle=DefaultIntegrityOracle(),
            spark_runner=_NoOpSparkRunner(tmp / "sp2"),
            dbt_executor=_ok_dbt,
        )
        stop_hook = StopGateHook(verifier=verifier, worktree_dir=wt, runs_dir=runs_dir)
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

        loop_hitl = HarnessLoop(
            model_adapter=FakeReplayModelAdapter(
                [
                    TurnResponse(
                        tool_calls=[
                            ToolCall(id="p-hitl", name="apply_patch", arguments={"patch": patch})
                        ]
                    ),
                    TurnResponse(content=valid_rca),
                ]
            ),
            tool_registry=ToolRegistry(wt),
            hooks=HookPipeline(stop_hooks=[stop_hook]),
            runs_dir=runs_dir,
        )
        st_await = loop_hitl.run(
            IncidentInput(incident_id="INC-HITL", title="HiTL check", summary="Formal HiTL check"),
            UserContext(role="de", resolution_mode="fix"),
            run_id=run_hitl,
        )
        if st_await.status != "AWAITING_APPROVAL":
            return _conformance_fail(
                f"Expected AWAITING_APPROVAL after green StopGate, got {st_await.status}",
                checks,
            )

        # Verify tampering ApprovalPacket.shadow_verification.passed=False blocks approve().
        store_hitl = CheckpointStore(runs_dir, run_hitl)
        tampered = store_hitl.load_latest().model_copy(deep=True)
        tampered.metadata["approval_packet"]["shadow_verification"]["passed"] = False
        store_hitl.append("tamper_shadow_test", tampered)
        tamper_blocked = False
        try:
            svc.approve(run_hitl)
        except ValueError:
            tamper_blocked = True
        if not tamper_blocked or _head_sha() != base_sha:
            return _conformance_fail(
                "PromotionRequiresGreenShadow violated: PromotionService.approve() allowed passed=False shadow verification!",
                checks,
            )
        # Restore valid green ApprovalPacket state before continuing reject->approve checks.
        store_hitl.append("restore_green_shadow", st_await)
        checks.append({"name": "promotion_requires_green_shadow", "passed": True})

        # Verify bounded rejections (max_rejections = 2 -> 3rd rejects with ValueError) & +8 turns.
        resume_adapter = FakeReplayModelAdapter(
            [TurnResponse(content=valid_rca), TurnResponse(content=valid_rca)]
        )
        resume_loop = HarnessLoop(
            model_adapter=resume_adapter,
            tool_registry=ToolRegistry(wt),
            hooks=HookPipeline(stop_hooks=[stop_hook]),
            runs_dir=runs_dir,
        )
        st_r1 = svc.reject(run_hitl, "First review note", loop=resume_loop)
        st_r2 = svc.reject(run_hitl, "Second review note", loop=resume_loop)
        third_blocked = False
        try:
            svc.reject(run_hitl, "Third exceeding max_rejections", loop=resume_loop)
        except ValueError:
            third_blocked = True
        if (
            st_r1.max_turns != policies.limits.max_turns + policies.limits.rejection_turn_extension
            or st_r2.max_turns
            != policies.limits.max_turns + 2 * policies.limits.rejection_turn_extension
            or st_r2.rejection_count != policies.limits.max_rejections
            or not third_blocked
        ):
            return _conformance_fail("BoundedRejections / +8 extension violated.", checks)
        checks.append({"name": "bounded_rejections_and_turn_extension", "passed": True})

        # Approve -> RESOLVED; second approve -> ValueError (terminal absorbing).
        promo = svc.approve(run_hitl)
        reapprove_blocked = False
        try:
            svc.approve(run_hitl)
        except ValueError:
            reapprove_blocked = True
        if promo.status != "RESOLVED" or not reapprove_blocked or _head_sha() == base_sha:
            return _conformance_fail(
                "Terminal RESOLVED promotion or idempotency guard failed.", checks
            )
        checks.append({"name": "terminal_resolved_absorbing", "passed": True})

        # Check 4: Crash/resume preserves turn count, max_turns, rejection_count, and reaches AWAITING_APPROVAL.
        run_crash = "formal-crash-resume"
        crash_wt = RunWorktreeManager.create(ws, runs_dir, run_crash)
        crash_sql = crash_wt / "dbt_project" / "models" / "marts" / "fct_daily_gmv.sql"
        crash_orig = crash_sql.read_text(encoding="utf-8")
        crash_patch = build_unified_diff(
            "dbt_project/models/marts/fct_daily_gmv.sql",
            crash_orig,
            crash_orig + "\n-- crash resume verification marker\n",
        )
        crash_stop = StopGateHook(verifier=verifier, worktree_dir=crash_wt, runs_dir=runs_dir)
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
        )
        aborted_crash = pre_crash_loop.run(
            IncidentInput(incident_id="INC-CRASH", title="Crash check", summary="Resume invariant check"),
            UserContext(role="de", resolution_mode="fix"),
            run_id=run_crash,
            max_turns=1,
        )
        if aborted_crash.turn != 1 or aborted_crash.max_turns != 1 or aborted_crash.rejection_count != 0:
            return _conformance_fail("Crash simulation setup failed to preserve turn=1/max_turns=1.", checks)

        # Simulate crash recovery in RUNNING status and resume with a fresh CheckpointStore/HarnessLoop.
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
        )
        post_crash = resumed_crash_loop.resume(run_crash)
        if (
            post_crash.status != "AWAITING_APPROVAL"
            or post_crash.turn < 3
            or post_crash.max_turns != policies.limits.max_turns
            or post_crash.rejection_count != 0
            or len(post_crash.messages) <= len(aborted_crash.messages)
        ):
            return _conformance_fail(
                "CrashResumePreservesInvariants violated: turn/max_turns/rejection_count continuity broke.",
                checks,
            )
        checks.append({"name": "crash_resume_preserves_invariants", "passed": True})

    return {"passed": True, "checks": checks}


def _conformance_fail(message: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"passed": False, "error": message, "checks": checks}


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
    live = {"passed": True, "checks": []} if args.skip_live else run_live_code_conformance()

    overall_passed = bool(
        bfs_clean["passed"] and (not bfs_buggy["passed"]) and live["passed"]
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
