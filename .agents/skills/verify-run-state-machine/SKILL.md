---
name: verify-run-state-machine
description: Formally verify the DataOps-Harness Run lifecycle state machine, crash/resume checkpoints, bounded HiTL rejections, and Promotion safety invariants using exhaustive BFS model checking and live Python conformance checks. Use whenever modifying src/core/context.py, src/core/loop.py, src/core/promotion.py, or src/hooks/stop_gate.py.
---

# Verify Run State Machine & Promotion Invariants

Formally verify that no `Run` can mutate production git/lakehouse storage without green `ShadowOverlayVerifier` verification and explicit HiTL approval, crash/resume preserves budget and turn counters, and HiTL rejections remain strictly bounded.

**Last Verified Sync Commit**: `71f18fc`

## What This Skill Covers

| Component | Exact Symbols Covered | Safety Responsibility |
| --- | --- | --- |
| State & Checkpoints | `src/core/context.py::RunStatus`, `RunState`, `CheckpointStore` | Append-only local checkpoint persistence for `turn`, `max_turns`, `rejection_count`, and terminal `status`. |
| Single-Loop Execution | `src/core/loop.py::HarnessLoop.run`, `HarnessLoop.resume`, `HarnessLoop._execute_loop` | Turn progression, stop-hook gating, crash/resume continuity, `ABORTED_BUDGET` & `ABORTED_THRASH`. |
| Stop & Shadow Gate | `src/hooks/stop_gate.py::StopGateHook.on_stop`, `src/core/shadow_verifier.py::ShadowOverlayVerifier.verify` | Enforces `DIAGNOSED` exit in `diagnosis_only` mode and green shadow verification + `ApprovalPacket` before `AWAITING_APPROVAL`. |
| HiTL Promotion & Reject | `src/core/promotion.py::PromotionService.approve`, `PromotionService.reject` | Requires `status == "AWAITING_APPROVAL"` and `shadow_verification.passed == True` before applying production diff; enforces `rejection_count < max_rejections`. |
| Configuration Bounds | `configs/policies.yaml::limits`, `anti_thrash` | `max_turns=20`, `max_rejections=2`, `rejection_turn_extension=8`, `strikes_before_abort=2`. |

Read [specs/RunLifecycle.tla](specs/RunLifecycle.tla) and [specs/RunLifecycle.cfg](specs/RunLifecycle.cfg) for the full formal specification, or execute [scripts/check_run_state_machine.py](scripts/check_run_state_machine.py) directly.

## Verification Bounds

The finite model checker explores all reachable states exhaustively via BFS over scaled bounds (`InitMaxTurns = 2`, `MaxRejections = 2`, `ExtTurns = 2`, `MaxStrikes = 2`, covering both `fix` and `diagnosis_only` modes up to maximum depth `InitMaxTurns + MaxRejections * ExtTurns`), while the live Python conformance suite validates exact production policy limits (`max_turns=20`, `max_rejections=2`, `rejection_turn_extension=8`, `strikes_before_abort=2`).

## Run Verification

Execute after editing state transitions, stop gates, or promotion logic:

```bash
uv run python .agents/skills/verify-run-state-machine/scripts/check_run_state_machine.py
```

Pass `--json-output <path>` to save the structured model-check and conformance proof artifact. Exit code `0` means:
1. All reachable states satisfy all 7 safety invariants.
2. Seeded mutation negative control (`BuggyDirectPromote`) fails with a shortest-path counterexample.
3. Live Python conformance suite confirms 6 end-to-end properties against real `CheckpointStore`, `HarnessLoop`, `StopGateHook`, and `PromotionService`: `no_unapproved_promote_from_running`, `diagnosis_only_never_promotes`, `promotion_requires_green_shadow`, `bounded_rejections_and_turn_extension`, `terminal_resolved_absorbing`, and `crash_resume_preserves_invariants`.

## Verified Invariants & Plain-Language Counterexample Guide

| Invariant Name | Plain-Language Guarantee | What a Counterexample Looks Like & How to Fix |
| --- | --- | --- |
| `NoUnapprovedProdMutation` | Production git/lakehouse files never change unless the `Run` finished `PromotionService.approve` into `RESOLVED`. | **Failure example**: `Run` crashes in `RUNNING`, or exits in `ABORTED_BUDGET`, yet production git `HEAD` changes because `PromotionService.approve` skipped `if state.status != "AWAITING_APPROVAL"`. **Fix**: Restore the `AWAITING_APPROVAL` status precondition guard in `src/core/promotion.py`. |
| `PromotionRequiresGreenShadow` | A `Run` cannot enter `RESOLVED` unless both `shadow_verification.passed` and an `ApprovalPacket` exist in `fix` mode. | **Failure example**: Reviewer approves a `Run` whose `ApprovalPacket.shadow_verification.passed` is `False` (or missing). **Fix**: Require `packet.shadow_verification.passed == True` before running `git apply` in `PromotionService.approve`. |
| `AwaitingApprovalRequiresGreenShadow` | No `Run` may pause in `AWAITING_APPROVAL` with failed tests or a missing `ApprovalPacket`. | **Failure example**: `StopGateHook.on_stop` returns `allow_stop=True` in `fix` mode despite `verification.passed == False`. **Fix**: Ensure `StopGateHook.on_stop` rejects stop whenever shadow verification fails. |
| `DiagnosisOnlyNeverPromotes` | Read-only `da` / `diagnosis_only` Runs end in `DIAGNOSED` (or `ABORTED_*`) and never touch production or enter approval. | **Failure example**: A `diagnosis_only` Run emits an `ApprovalPacket` or mutates production. **Fix**: Return `DIAGNOSED` directly in `StopGateHook.on_stop` when `resolution_mode == "diagnosis_only"`. |
| `BoundedRejections` | A Run can be rejected at most `max_rejections` (`2`) times; a 3rd rejection is refused. | **Failure example**: `PromotionService.reject` increments `rejection_count` to `3`. **Fix**: Enforce `state.rejection_count < limits.max_rejections` before extending `max_turns`. |
| `TurnBudgetBounded` | Each `RUNNING` step stays within `max_turns <= 20 + 2 * 8 = 36` turns; exceeding hits `ABORTED_BUDGET`. | **Failure example**: Agent keeps calling tools when `turn >= max_turns` or `reject` grants turns past limit. **Fix**: Check `turn >= state.max_turns` at the top of `HarnessLoop._execute_loop`. |
| `TerminalStatesAbsorbing` | Once `RESOLVED`, `DIAGNOSED`, or `ABORTED_*`, no action mutates state again (re-approve fails). | **Failure example**: Calling `PromotionService.approve` twice re-applies diffs on `RESOLVED`. **Fix**: Reject `approve` unless status is strictly `AWAITING_APPROVAL`. |
