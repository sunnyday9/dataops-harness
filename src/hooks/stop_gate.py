from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.core.context import RunState
from src.core.rca import ApprovalPacket, parse_rca_report
from src.core.shadow_verifier import ShadowOverlayVerifier
from src.core.worktree import RunWorktreeManager
from src.hooks.base import StopDecision, StopHook


class StopGateHook(StopHook):
    """Stop gate enforcing RCA schema validation and green Shadow Overlay verification."""

    def __init__(
        self,
        verifier: ShadowOverlayVerifier | None = None,
        worktree_dir: Path | None = None,
        worktree_resolver: Callable[[RunState], Path | None] | None = None,
        runs_dir: Path | None = None,
    ) -> None:
        self.verifier = verifier
        self.worktree_dir = worktree_dir.resolve() if worktree_dir is not None else None
        self.worktree_resolver = worktree_resolver
        self.runs_dir = runs_dir.resolve() if runs_dir is not None else None

    def _resolve_worktree(self, state: RunState) -> Path | None:
        if self.worktree_resolver is not None:
            resolved = self.worktree_resolver(state)
            if resolved is not None:
                return resolved.resolve()
        meta_path = state.metadata.get("worktree_path")
        if isinstance(meta_path, str) and meta_path.strip():
            return Path(meta_path).resolve()
        return self.worktree_dir

    def on_stop(self, state: RunState, candidate_output: str) -> StopDecision:
        """Validate `RcaReport`, run Shadow Overlay in Fix Mode, and emit ApprovalPacket."""
        try:
            rca = parse_rca_report(candidate_output)
        except ValueError as exc:
            return StopDecision(
                allow_stop=False,
                feedback_message=(
                    f"Stop rejected: invalid RCA Report ({exc}). "
                    "Output valid JSON with incident_id, root_cause_summary, "
                    "affected_assets, evidence, engine, proposed_fix_summary, "
                    "patched_files, and confidence."
                ),
            )

        state.metadata["rca_report"] = rca.model_dump(mode="json")

        if state.user_context.resolution_mode == "diagnosis_only":
            return StopDecision(allow_stop=True, target_status="DIAGNOSED")

        worktree = self._resolve_worktree(state)
        if self.verifier is None or worktree is None:
            return StopDecision(
                allow_stop=False,
                feedback_message=(
                    "Stop rejected: fix mode requires ShadowOverlayVerifier and an "
                    "isolated worktree before finishing."
                ),
            )

        verification = self.verifier.verify(state.run_id, worktree)
        state.metadata["shadow_verification"] = verification.model_dump(mode="json")

        if not verification.passed:
            reasons = "; ".join(verification.failure_reasons) or "Shadow verification failed."
            if len(reasons) > 1200:
                reasons = f"{reasons[:1197]}..."
            return StopDecision(
                allow_stop=False,
                feedback_message=(
                    f"Stop rejected by Shadow Overlay gate: {reasons} "
                    "Inspect the failing job/model/test and revise your patch cleanly."
                ),
            )

        unified_diff = RunWorktreeManager.unified_diff(worktree)
        packet = ApprovalPacket(
            run_id=state.run_id,
            incident_id=state.incident.incident_id,
            rca_report=rca,
            unified_diff=unified_diff,
            shadow_verification=verification,
        )
        runs_root = self.runs_dir or self.verifier.runs_dir
        packet_path = runs_root / state.run_id / "approval_packet.json"
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")

        state.metadata["approval_packet"] = packet.model_dump(mode="json")
        state.metadata["approval_packet_path"] = str(packet_path)
        state.metadata["worktree_path"] = str(worktree)

        return StopDecision(allow_stop=True, target_status="AWAITING_APPROVAL")
