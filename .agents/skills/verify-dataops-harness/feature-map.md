# DataOps-Harness User-POV Feature Map

This feature map documents every runnable DataOps-Harness workflow exercised end-to-end by [scripts/run_verification.py](scripts/run_verification.py) inside an isolated temporary workspace.

## 1. Environment & Lakehouse Doctor

- **User Command / Interface**: Runtime check of `python`, `uv`, `git`, `java`, `configs/policies.yaml`, `configs/models.yaml`, and seeded lakehouse `workspace/lakehouse/prod/mart/warehouse.duckdb`.
- **Expected Outcome**: All prerequisites resolve on disk and DuckDB contains populated baseline tables (`fct_daily_gmv`, `dim_user_ltv`).
- **Durable Proof Artifact**: `doctor.json` (`"passed": true`).

## 2. Isolated Fault Reproduction & StopGate Approval Pause

- **User Command / Interface**: Clone production workspace into `/tmp/dataops-verify-<id>/workspace`, reproduce `ops_hitl_reject_refix_promote` (`fct_daily_gmv` 2x GMV defect), execute Fix Mode `HarnessLoop` + `StopGateHook` -> `ShadowOverlayVerifier` + `DefaultIntegrityOracle`, then query CLI:
  ```bash
  uv run python -m src.cli status <run_id> --runs-dir /tmp/dataops-verify-<id>/runs
  ```
- **Expected Outcome**: Exit code `0`, JSON output shows `"status": "AWAITING_APPROVAL"` and `"has_approval_packet": true`, while production lakehouse in `/tmp/dataops-verify-<id>/workspace` remains untouched prior to approval.
- **Durable Proof Artifact**: `cli_lifecycle.json` (`initial_status`).

## 3. HiTL Rejection Feedback & Turn Extension

- **User Command / Interface**: Reject the candidate fix via CLI with mandatory reviewer reason, granting `+8` extra turns and re-entering the repair loop:
  ```bash
  uv run python -m src.cli resume <run_id> \
    --workspace /tmp/dataops-verify-<id>/workspace \
    --runs-dir /tmp/dataops-verify-<id>/runs \
    --reject --reason "Confirm dim_user_ltv reconciliation after promotion."
  ```
- **Expected Outcome**: Exit code `0`, JSON shows `"status": "RUNNING"`, `"rejection_count": 1`, and `"max_turns": 28`. Resuming the agent loop re-verifies shadow outputs back to `AWAITING_APPROVAL`.
- **Durable Proof Artifact**: `cli_lifecycle.json` (`after_reject`, `after_resumed_reverify`).

## 4. HiTL Approval & Production Promotion

- **User Command / Interface**: Approve and promote the shadow-verified patch via CLI, then inspect final Run status:
  ```bash
  uv run python -m src.cli resume <run_id> \
    --workspace /tmp/dataops-verify-<id>/workspace \
    --runs-dir /tmp/dataops-verify-<id>/runs \
    --approve
  uv run python -m src.cli status <run_id> --runs-dir /tmp/dataops-verify-<id>/runs
  ```
- **Expected Outcome**: Promotion commits cleanly inside the isolated sandbox git repo, replays affected dbt models & tests cleanly, and transitions Run status to `"RESOLVED"`.
- **Durable Proof Artifact**: `cli_lifecycle.json` (`after_approve`, `final_status`).

## 5. Run State Machine Formal Verification & Live Conformance

- **User Command / Interface**:
  ```bash
  uv run python .agents/skills/verify-run-state-machine/scripts/check_run_state_machine.py \
    --json-output <proof_dir>/formal_check.json
  ```
- **Expected Outcome**: Exit code `0`; exhaustive BFS checks all reachable states across safety invariants, mutation negative control catches `BuggyDirectPromote`, and all 6 live conformance checks pass.
- **Durable Proof Artifact**: `formal_check.json`.

## 6. 3-Arm Ablation Benchmark Execution

- **User Command / Interface**:
  ```bash
  uv run python -m benchmarks.ablation_runner --provider fake_replay --output <proof_dir>/ablation_report.md
  ```
- **Expected Outcome**: Exit code `0`; compares Group A (Bare ReAct), Group B (+Context & Lineage), and Group C (Full DataOps-Harness) on `Verified Fix Rate`, `Safety Interception Rate`, `Avg Turns`, `Token Cost`, `Prompt Cache Hit Rate`, and `Failure Attribution`.
- **Durable Proof Artifact**: `ablation_report.md`.

## 7. Ephemeral Instance Teardown & Evidence Persistence

- **Expected Outcome**: Temporary `/tmp/dataops-verify-<id>` sandbox is deleted completely while all 5 durable proof files remain intact inside `<proof_dir>`, summarized in `verification_summary.md`.
