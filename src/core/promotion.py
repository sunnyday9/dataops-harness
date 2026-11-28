from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.adapters.base import LineageAdapterBase
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.shadow_backend import (
    DbtExecutor,
    ProductionReplayBackend,
    ShadowBackend,
    capture_incremental_blocks,
    create_shadow_backend,
)
from src.adapters.spark_adapter import SparkJobRunner
from src.core.context import CheckpointStore, Message, RunState, RunStatus
from src.core.loop import HarnessLoop
from src.core.policy import PoliciesConfig, load_policies
from src.core.project import ResolvedProject, discover_project
from src.core.rca import ApprovalPacket
from src.core.worktree import RunWorktreeManager


class PromotionResult(BaseModel):
    """Summary of production patch application, commit, and healed pipeline run."""

    run_id: str
    status: RunStatus
    commit_sha: str
    applied_files: list[str] = Field(default_factory=list)
    spark_runs: list[dict[str, Any]] = Field(default_factory=list)
    dbt_run_ok: bool = False
    dbt_test_ok: bool = False


class PromotionService:
    """Handle HiTL approval Promotion to production and bounded Rejection resume."""

    def __init__(
        self,
        prod_workspace: Path,
        runs_dir: Path,
        *,
        policies: PoliciesConfig | None = None,
        lineage_adapter: LineageAdapterBase | None = None,
        spark_runner: SparkJobRunner | None = None,
        dbt_executor: DbtExecutor | None = None,
        project: ResolvedProject | None = None,
        shadow_backend: ShadowBackend | None = None,
    ) -> None:
        self.prod_workspace = prod_workspace.resolve()
        self.runs_dir = runs_dir.resolve()
        self.policies = policies or load_policies()
        self.lineage_adapter = lineage_adapter
        self._custom_spark_runner = spark_runner
        self._custom_dbt_executor = dbt_executor
        self._project = project
        self._shadow_backend = shadow_backend

    def _load_approval_packet(self, state: RunState) -> ApprovalPacket:
        raw_packet = state.metadata.get("approval_packet")
        if isinstance(raw_packet, dict):
            return ApprovalPacket.model_validate(raw_packet)
        packet_file = self.runs_dir / state.run_id / "approval_packet.json"
        if packet_file.is_file():
            return ApprovalPacket.model_validate_json(packet_file.read_text(encoding="utf-8"))
        raise ValueError(f"ApprovalPacket not found for Run {state.run_id!r}.")

    def approve(self, run_id: str, *, cleanup_worktree: bool = True) -> PromotionResult:
        """Promote an approved Run patch to production sandbox git and local replay backend."""
        store = CheckpointStore(self.runs_dir, run_id)
        state = store.load_latest()
        if state.status != "AWAITING_APPROVAL":
            raise ValueError(
                f"Run {run_id!r} cannot be approved in status {state.status!r}; "
                "expected 'AWAITING_APPROVAL'."
            )

        packet = self._load_approval_packet(state)
        shadow = packet.shadow_verification
        if not shadow.passed:
            raise ValueError(f"Cannot promote Run {run_id!r}: shadow verification is not green.")
        diff_text = packet.unified_diff
        if not diff_text.strip():
            raise ValueError(f"Cannot promote Run {run_id!r}: unified diff is empty.")

        project = self._project or discover_project(self.prod_workspace)
        lineage = self.lineage_adapter or LineageAdapter.from_project(project)
        backend = self._shadow_backend or create_shadow_backend(
            project,
            lineage_adapter=lineage,
            spark_runner=self._custom_spark_runner,
            dbt_executor=self._custom_dbt_executor,
        )
        replay_backend = backend if isinstance(backend, ProductionReplayBackend) else None
        if replay_backend is None:
            raise ValueError(
                f"Shadow backend {backend.backend_name!r} does not support local in-place replay Promotion."
            )

        changed_files = list(shadow.changed_files)
        dbt_rel = project.dbt_project_rel if project.dbt_project_dir is not None else None
        before_inc_blocks = capture_incremental_blocks(
            self.prod_workspace, changed_files, dbt_project_rel=dbt_rel
        )
        affected = lineage.affected_downstream(changed_files)
        pre_head = subprocess.run(
            ["git", "-C", str(self.prod_workspace), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        normalized_diff = diff_text if diff_text.endswith("\n") else diff_text + "\n"
        check_proc = subprocess.run(
            ["git", "-C", str(self.prod_workspace), "apply", "--check", "-"],
            input=normalized_diff,
            text=True,
            capture_output=True,
            check=False,
        )
        if check_proc.returncode != 0:
            raise RuntimeError(
                f"Production git apply --check failed: {check_proc.stderr.strip()}"
            )

        snapshot = replay_backend.snapshot_production(run_id, self.runs_dir, affected=affected)
        try:
            apply_proc = subprocess.run(
                ["git", "-C", str(self.prod_workspace), "apply", "-"],
                input=normalized_diff,
                text=True,
                capture_output=True,
                check=False,
            )
            if apply_proc.returncode != 0:
                raise RuntimeError(f"Production git apply failed: {apply_proc.stderr.strip()}")

            if changed_files:
                subprocess.run(
                    ["git", "-C", str(self.prod_workspace), "add", "--", *changed_files],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                subprocess.run(
                    ["git", "-C", str(self.prod_workspace), "add", "-A"],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            commit_msg = (
                f"fix(dataops): resolve {state.incident.incident_id} (run {run_id})"
            )
            commit_proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.prod_workspace),
                    "-c",
                    "user.name=DataOps-Harness",
                    "-c",
                    "user.email=harness@dataops.local",
                    "commit",
                    "-m",
                    commit_msg,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if commit_proc.returncode != 0:
                raise RuntimeError(
                    "Production git commit failed: "
                    f"{commit_proc.stderr.strip() or commit_proc.stdout.strip()}"
                )

            commit_sha = subprocess.run(
                ["git", "-C", str(self.prod_workspace), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            after_inc_blocks = capture_incremental_blocks(
                self.prod_workspace, changed_files, dbt_project_rel=dbt_rel
            )
            incremental_changed = before_inc_blocks != after_inc_blocks

            exec_res = replay_backend.replay_production(
                changed_files=changed_files,
                affected=affected,
                full_refresh=not incremental_changed,
                run_id=run_id,
                runs_dir=self.runs_dir,
            )
            if exec_res.failure_reasons or not exec_res.dbt_run_ok or not exec_res.dbt_test_ok:
                raise RuntimeError(
                    "Production post-commit verification failed: "
                    + "; ".join(exec_res.failure_reasons)
                )
        except (RuntimeError, OSError, subprocess.SubprocessError):
            subprocess.run(
                ["git", "-C", str(self.prod_workspace), "reset", "--hard", pre_head],
                check=False,
                capture_output=True,
                text=True,
            )
            replay_backend.restore_production(snapshot)
            replay_backend.cleanup_snapshot(snapshot)
            raise

        replay_backend.cleanup_snapshot(snapshot)

        result = PromotionResult(
            run_id=run_id,
            status="RESOLVED",
            commit_sha=commit_sha,
            applied_files=changed_files,
            spark_runs=exec_res.spark_runs,
            dbt_run_ok=exec_res.dbt_run_ok,
            dbt_test_ok=exec_res.dbt_test_ok,
        )
        state.status = "RESOLVED"
        state.metadata["promotion"] = result.model_dump(mode="json")
        store.append("run_promoted", state, detail={"commit_sha": commit_sha})

        if cleanup_worktree:
            wt_raw = state.metadata.get("worktree_path") or shadow.worktree_path
            if wt_raw:
                RunWorktreeManager.cleanup(self.prod_workspace, Path(wt_raw))

        return result

    def reject(
        self,
        run_id: str,
        reason: str,
        *,
        loop: HarnessLoop | None = None,
    ) -> RunState:
        """Record a reviewer rejection, grant +8 turns (max 2), and optionally resume loop."""
        cleaned_reason = reason.strip()
        if not cleaned_reason:
            raise ValueError("Rejection reason must not be empty.")

        store = CheckpointStore(self.runs_dir, run_id)
        state = store.load_latest()
        if state.status != "AWAITING_APPROVAL":
            raise ValueError(
                f"Run {run_id!r} cannot be rejected in status {state.status!r}; "
                "expected 'AWAITING_APPROVAL'."
            )

        limits = self.policies.limits
        if state.rejection_count >= limits.max_rejections:
            raise ValueError(
                f"Run {run_id!r} reached max rejections ({limits.max_rejections})."
            )

        ext = limits.rejection_turn_extension
        state.rejection_count += 1
        rejections = state.metadata.setdefault("rejections", [])
        rejections.append(
            {
                "attempt": state.rejection_count,
                "reason": cleaned_reason,
                "ts": datetime.now(UTC).isoformat(),
            }
        )

        feedback = (
            f"[HiTL REJECTION {state.rejection_count}/{limits.max_rejections}] "
            f"Reviewer rejected proposed fix: {cleaned_reason}\n"
            f"Granted +{ext} extra turns; revise your investigation and shadow patch cleanly."
        )

        if loop is not None:
            store.append(
                "hitl_rejected",
                state,
                detail={"attempt": state.rejection_count, "reason": cleaned_reason},
            )
            return loop.resume(
                run_id,
                extra_message=feedback,
                extra_turns=ext,
                is_rejection_extension=True,
            )

        state.max_turns += ext
        state.messages.append(Message(role="user", content=feedback))
        state.status = "RUNNING"
        store.append(
            "hitl_rejected",
            state,
            detail={
                "attempt": state.rejection_count,
                "reason": cleaned_reason,
                "extra_turns": ext,
            },
        )
        return state
