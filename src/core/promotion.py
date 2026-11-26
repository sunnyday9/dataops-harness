from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.adapters.base import LakehousePaths, LineageAdapterBase
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.spark_adapter import SparkJobRunner
from src.core.context import CheckpointStore, Message, RunState, RunStatus
from src.core.loop import HarnessLoop
from src.core.policy import PoliciesConfig, load_policies
from src.core.rca import ApprovalPacket
from src.core.shadow_verifier import (
    DbtExecutor,
    _default_dbt_executor,
    execute_pipeline_subset,
    extract_incremental_blocks,
)
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
    ) -> None:
        self.prod_workspace = prod_workspace.resolve()
        self.runs_dir = runs_dir.resolve()
        self.policies = policies or load_policies()
        self.lineage_adapter = lineage_adapter
        self._custom_spark_runner = spark_runner
        self._dbt_executor: DbtExecutor = dbt_executor or _default_dbt_executor

    def _load_approval_packet(self, state: RunState) -> ApprovalPacket:
        raw_packet = state.metadata.get("approval_packet")
        if isinstance(raw_packet, dict):
            return ApprovalPacket.model_validate(raw_packet)
        packet_file = self.runs_dir / state.run_id / "approval_packet.json"
        if packet_file.is_file():
            return ApprovalPacket.model_validate_json(packet_file.read_text(encoding="utf-8"))
        raise ValueError(f"ApprovalPacket not found for Run {state.run_id!r}.")

    def approve(self, run_id: str, *, cleanup_worktree: bool = True) -> PromotionResult:
        """Promote an approved Run patch to production sandbox git and lakehouse."""
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

        changed_files = list(shadow.changed_files)
        before_inc_blocks = {
            rel: extract_incremental_blocks((self.prod_workspace / rel).read_text(encoding="utf-8"))
            if rel.startswith("dbt_project/models/")
            and rel.endswith(".sql")
            and (self.prod_workspace / rel).is_file()
            else []
            for rel in changed_files
        }

        pre_head = subprocess.run(
            ["git", "-C", str(self.prod_workspace), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        prod_paths = LakehousePaths.from_workspace(self.prod_workspace)
        backup_duckdb = self.runs_dir / run_id / ".prod_warehouse_backup.duckdb"
        if prod_paths.prod_duckdb_path.is_file():
            backup_duckdb.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prod_paths.prod_duckdb_path, backup_duckdb)

        backup_silver_dir = self.runs_dir / run_id / ".prod_silver_backup"
        silver_existed_before = prod_paths.prod_silver_dir.exists()
        silver_backed_up = False

        def _cleanup_backups() -> None:
            if backup_duckdb.is_file():
                backup_duckdb.unlink()
            if backup_silver_dir.exists():
                shutil.rmtree(backup_silver_dir)

        def _rollback_prod() -> None:
            subprocess.run(
                ["git", "-C", str(self.prod_workspace), "reset", "--hard", pre_head],
                check=False,
                capture_output=True,
                text=True,
            )
            if backup_duckdb.is_file():
                prod_paths.prod_duckdb_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_duckdb, prod_paths.prod_duckdb_path)
            if silver_backed_up:
                if prod_paths.prod_silver_dir.exists():
                    shutil.rmtree(prod_paths.prod_silver_dir)
                if silver_existed_before and backup_silver_dir.is_dir():
                    prod_paths.prod_silver_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(backup_silver_dir, prod_paths.prod_silver_dir)

        normalized_diff = diff_text if diff_text.endswith("\n") else diff_text + "\n"
        check_proc = subprocess.run(
            ["git", "-C", str(self.prod_workspace), "apply", "--check", "-"],
            input=normalized_diff,
            text=True,
            capture_output=True,
            check=False,
        )
        if check_proc.returncode != 0:
            _cleanup_backups()
            raise RuntimeError(
                f"Production git apply --check failed: {check_proc.stderr.strip()}"
            )

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
                    f"Production git commit failed: {commit_proc.stderr.strip() or commit_proc.stdout.strip()}"
                )

            commit_sha = subprocess.run(
                ["git", "-C", str(self.prod_workspace), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            lineage = self.lineage_adapter or LineageAdapter.from_workspace(self.prod_workspace)
            affected = lineage.affected_downstream(changed_files)
            if affected.spark_jobs:
                if backup_silver_dir.exists():
                    shutil.rmtree(backup_silver_dir)
                if silver_existed_before:
                    backup_silver_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(prod_paths.prod_silver_dir, backup_silver_dir)
                silver_backed_up = True
            events_dir = self.runs_dir / run_id / "prod_spark_events"
            spark_runner = self._custom_spark_runner or SparkJobRunner(event_log_dir=events_dir)

            incremental_changed = any(
                rel.startswith("dbt_project/models/")
                and rel.endswith(".sql")
                and before_inc_blocks.get(rel, [])
                != (
                    extract_incremental_blocks(
                        (self.prod_workspace / rel).read_text(encoding="utf-8")
                    )
                    if (self.prod_workspace / rel).is_file()
                    else []
                )
                for rel in changed_files
            )

            (
                spark_runs,
                _overrides,
                dbt_run_ok,
                dbt_test_ok,
                _dbt_out,
                _dbt_err,
                failures,
            ) = execute_pipeline_subset(
                code_workspace=self.prod_workspace,
                raw_dir=prod_paths.raw_dir,
                prod_silver_dir=prod_paths.prod_silver_dir,
                target_silver_dir=prod_paths.prod_silver_dir,
                target_duckdb_path=prod_paths.prod_duckdb_path,
                changed_files=changed_files,
                affected_spark_jobs=affected.spark_jobs,
                affected_dbt_models=affected.dbt_models,
                spark_runner=spark_runner,
                dbt_executor=self._dbt_executor,
                full_refresh=not incremental_changed,
            )
            if failures or not dbt_run_ok or not dbt_test_ok:
                raise RuntimeError(
                    "Production post-commit verification failed: " + "; ".join(failures)
                )
        except (RuntimeError, OSError, subprocess.SubprocessError):
            _rollback_prod()
            _cleanup_backups()
            raise

        _cleanup_backups()

        result = PromotionResult(
            run_id=run_id,
            status="RESOLVED",
            commit_sha=commit_sha,
            applied_files=changed_files,
            spark_runs=spark_runs,
            dbt_run_ok=dbt_run_ok,
            dbt_test_ok=dbt_test_ok,
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
