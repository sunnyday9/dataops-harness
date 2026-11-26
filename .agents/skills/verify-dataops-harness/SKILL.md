---
name: verify-dataops-harness
description: Run end-to-end verification of DataOps-Harness in an isolated /tmp sandbox by checking prerequisites, driving real CLI status and resume HiTL flows, running formal and ablation checks, capturing durable proof artifacts, and cleaning up ephemeral state without deleting evidence. Use when verifying the repository or a release end-to-end.
---

# Verify DataOps-Harness End-to-End

Launch an isolated temporary sandbox clone of `workspace/`, verify runtime health, drive real CLI `status` and `resume` (`--reject` -> `--approve`) lifecycles, run formal state-machine verification and the 3-arm ablation benchmark, persist 5 durable proof artifacts, and remove all ephemeral sandbox files while leaving evidence intact.

## Mapped Features

See [feature-map.md](feature-map.md) for user-POV commands, assertions, and durable proof artifacts across all 7 verified capabilities:
1. Environment & Lakehouse Doctor (`doctor.json`)
2. Fix Mode + StopGate Shadow Overlay + `src.cli status` (`AWAITING_APPROVAL`)
3. `src.cli resume --reject --reason` (`rejection_count=1`, `max_turns=28`) + resume re-verify
4. `src.cli resume --approve` (`RESOLVED`, post-commit dbt parity -> `cli_lifecycle.json`)
5. Formal Run State Machine Verification (`formal_check.json`)
6. 3-Arm Ablation Benchmark (`ablation_report.md`)
7. Ephemeral `/tmp/dataops-verify-*` Cleanup & Evidence Survival (`verification_summary.md`)

## Runtime Prerequisites

- In non-interactive shells, prepend `$HOME/.local/bin` and `$HOME/.local/opt/jdk-17/bin` to `PATH` before invoking `uv`, `pytest`, or PySpark commands:
  ```bash
  export PATH="$HOME/.local/bin:$HOME/.local/opt/jdk-17/bin:$PATH"
  ```
- Execute PySpark and dbt test suites or ablation runs **sequentially** (never launch overlapping heavy JVM processes in parallel).
- Use `grep` rather than `rg` when searching repository files in this shell environment.

## Run End-to-End Verification

Run the default fast end-to-end verification suite (cross-category 3-scenario ablation + full HiTL CLI lifecycle + formal BFS/conformance checks) and inspect the durable proof directory:

```bash
uv run python .agents/skills/verify-dataops-harness/scripts/run_verification.py \
  --proof-dir /tmp/dataops-verify-proof
```

Pass `--all-scenarios` to include all 4 heavy PySpark JVM Fault Scenarios alongside dbt, security, and ops scenarios inside `ablation_report.md`.

## Completion Criteria

Verification passes when `run_verification.py` exits with code `0` and confirms:
1. Ephemeral `/tmp/dataops-verify-<id>` sandbox workspace and `runs/` directory were completely deleted.
2. Live repository `workspace/` and `runs/` remain clean (`git status` unchanged).
3. All 5 proof artifacts survive non-empty inside `--proof-dir`: `doctor.json`, `cli_lifecycle.json`, `formal_check.json`, `ablation_report.md`, and `verification_summary.md`.
