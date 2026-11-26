---
name: verify-run-state-machine
description: Formally verify the DataOps-Harness Run lifecycle state machine, crash/resume checkpoints, bounded HiTL rejections, and Promotion safety invariants using exhaustive BFS model checking and live Python conformance checks. Use whenever modifying src/core/context.py, src/core/loop.py, src/core/promotion.py, src/hooks/stop_gate.py, or src/hooks/compactor.py — and audit/update the spec and conformance checks when transition, rollback, or guard rules change (never rely solely on rerunning an un-updated script).
---

# Verify Run State Machine & Promotion Invariants

Formally verify that no `Run` can mutate production git/lakehouse storage without green `ShadowOverlayVerifier` verification and explicit HiTL approval, crash/resume preserves budget and turn counters, and HiTL rejections remain strictly bounded.

**Last Verified Sync Commit**: `42e4456`

## What This Skill Covers

| Component | Exact Symbols Covered | Safety Responsibility |
| --- | --- | --- |
| State & Checkpoints | `src/core/context.py::RunStatus`, `RunState`, `CheckpointStore` | Append-only local checkpoint persistence for `turn`, `max_turns`, `rejection_count`, and terminal `status` (`RESOLVED`, `DIAGNOSED`, `ABORTED_THRASH`, `ABORTED_BUDGET`, `FAILED_ERROR`). |
| Single-Loop Execution | `src/core/loop.py::HarnessLoop.run`, `HarnessLoop.resume`, `HarnessLoop._execute_loop`, `HarnessLoop._enforce_role_mode_ceiling`, `HarnessLoop._reconcile_pending_tool_calls` | Role mode clamping, non-tool budget enforcement, HiTL pause clock integration, mid-turn crash tool-call reconciliation, and terminal `resume()` guards (`RESOLVED`, `DIAGNOSED`, `ABORTED_THRASH`, `FAILED_ERROR`, and `ABORTED_BUDGET` without `extra_turns > 0`). |
| Compactor & Budgets | `src/hooks/compactor.py::ContextCompactorHook._active_elapsed_seconds`, `on_run_paused`, `on_run_resumed`, `check_turn_budget` | Active wall-clock calculation excluding `AWAITING_APPROVAL` human review waits, non-tool turn cost/time checks, and `_effective_max_turns` accounting. |
| Stop & Shadow Gate | `src/hooks/stop_gate.py::StopGateHook.on_stop`, `src/core/shadow_verifier.py::ShadowOverlayVerifier.verify` | Enforces `DIAGNOSED` exit in `diagnosis_only` mode and green shadow verification + `ApprovalPacket` before `AWAITING_APPROVAL`. |
| HiTL Promotion & Reject | `src/core/promotion.py::PromotionService.approve`, `PromotionService.reject` | Requires `status == "AWAITING_APPROVAL"` and `shadow_verification.passed == True` before applying production diff; atomic rollback on failed post-commit verification; enforces `rejection_count < max_rejections` and `is_rejection_extension=True`. |
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
1. All reachable states satisfy all 7 safety invariants (covering `RUNNING`, `AWAITING_APPROVAL`, `RESOLVED`, `DIAGNOSED`, `ABORTED_THRASH`, `ABORTED_BUDGET`, `FAILED_ERROR`, `CostOrWallClockBudgetAbort`, and `HiTLApproveRollback`).
2. Seeded mutation negative controls (`BuggyDirectPromote` and `BuggyApproveRollbackLeak`) fail with shortest-path counterexamples.
3. Live Python conformance suite confirms 6 end-to-end properties against real `CheckpointStore`, `HarnessLoop`, `ContextCompactorHook`, `StopGateHook`, and `PromotionService`: `no_unapproved_promote_from_running` (including failed dbt and Spark+dbt rollback), `diagnosis_only_never_promotes` (including role mode ceiling clamping), `promotion_requires_green_shadow`, `bounded_rejections_and_turn_extension` (including HiTL pause clock exclusion and `is_rejection_extension`), `terminal_resolved_absorbing` (including `HarnessLoop.resume()` terminal status guards across `RESOLVED`, `DIAGNOSED`, `ABORTED_THRASH`, `FAILED_ERROR`, and `ABORTED_BUDGET` without `extra_turns > 0`), and `crash_resume_preserves_invariants` (including mid-tool-batch crash reconciliation). Note: `ABORTED_BUDGET` intentionally permits out-of-band operator `resume(..., extra_turns > 0)`.

## Verified Invariants & Plain-Language Counterexample Guide

| Invariant Name | Plain-Language Guarantee | What a Counterexample Looks Like & How to Fix |
| --- | --- | --- |
| `NoUnapprovedProdMutation` | Production git/lakehouse files never change unless the `Run` finished `PromotionService.approve` into `RESOLVED` (and any failed promotion rolls back atomically across git, DuckDB, and Silver Delta storage). | **Failure example**: `Run` crashes in `RUNNING`, or `PromotionService.approve` fails mid-verification after writing Spark/dbt outputs and leaves production mutated while status stays `AWAITING_APPROVAL`. **Fix**: Enforce `AWAITING_APPROVAL` guard and atomic rollback across git, DuckDB, and Spark Silver Delta tables in `src/core/promotion.py`. |
| `PromotionRequiresGreenShadow` | A `Run` cannot enter `RESOLVED` unless both `shadow_verification.passed` and an `ApprovalPacket` exist in `fix` mode. | **Failure example**: Reviewer approves a `Run` whose `ApprovalPacket.shadow_verification.passed` is `False` (or missing). **Fix**: Require `packet.shadow_verification.passed == True` before running `git apply` in `PromotionService.approve`. |
| `AwaitingApprovalRequiresGreenShadow` | No `Run` may pause in `AWAITING_APPROVAL` with failed tests or a missing `ApprovalPacket`. | **Failure example**: `StopGateHook.on_stop` returns `allow_stop=True` in `fix` mode despite `verification.passed == False`. **Fix**: Ensure `StopGateHook.on_stop` rejects stop whenever shadow verification fails. |
| `DiagnosisOnlyNeverPromotes` | Read-only `da` / `diagnosis_only` Runs (even when initialized with `resolution_mode="fix"`) clamp to `diagnosis_only`, end in `DIAGNOSED` (or `ABORTED_*`), and never touch production or enter approval. | **Failure example**: A `da` role Run bypasses role mode ceiling, emits an `ApprovalPacket`, or mutates production. **Fix**: Clamp role ceiling in `HarnessLoop._enforce_role_mode_ceiling` and return `DIAGNOSED` in `StopGateHook.on_stop`. |
| `BoundedRejections` | A Run can be rejected at most `max_rejections` (`2`) times without HiTL pause waits tripping the wall-clock ceiling; a 3rd rejection is refused. | **Failure example**: `PromotionService.reject` increments `rejection_count` to `3`, double-counts `extra_turns`, or human wait time triggers `ABORTED_BUDGET`. **Fix**: Enforce `state.rejection_count < limits.max_rejections`, pass `is_rejection_extension=True`, and track active elapsed time via `ContextCompactorHook.on_run_paused`/`on_run_resumed`. |
| `TurnBudgetBounded` | Each `RUNNING` step stays within `max_turns <= 20 + 2 * 8 = 36` turns (unless an explicit operator `resume(..., extra_turns > 0)` overrides `ABORTED_BUDGET`). | **Failure example**: Agent keeps calling tools when `turn >= max_turns` or `reject` grants turns past limit. **Fix**: Check `turn >= state.max_turns`, `check_turn_budget`, and `_effective_max_turns` in `src/core/loop.py` and `src/hooks/compactor.py`. |
| `TerminalStatesAbsorbing` | Once `RESOLVED`, `DIAGNOSED`, `ABORTED_THRASH`, `FAILED_ERROR` (or `ABORTED_BUDGET` with `extra_turns <= 0`), neither `approve` nor `resume` continues the Run. | **Failure example**: Calling `PromotionService.approve` twice re-applies diffs on `RESOLVED`, or `HarnessLoop.resume` continues a `FAILED_ERROR` Run. **Fix**: Reject `approve` outside `AWAITING_APPROVAL` and reject `resume` on all non-budget terminal statuses including `FAILED_ERROR`. |
