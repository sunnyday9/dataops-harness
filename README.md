# DataOps-Harness

DataOps-Harness executes an autonomous troubleshooting loop across PySpark, Delta Lake, and dbt-duckdb data pipelines. It enforces SQL safety, role permissions, PII redaction, context compaction, anti-thrash circuit breaking, isolated shadow verification, and human approval before any patch modifies production storage.

## Architecture

A single generic loop runs all diagnosis and remediation work while deterministic hooks guard data access and production state.

- **Single-loop runtime.** [HarnessLoop](src/core/loop.py) drives an Observe, Act, and Verify cycle and persists every step in append-only local checkpoints through [CheckpointStore](src/core/context.py).
- **Six orthogonal tools.** [ToolRegistry](src/tools/registry.py) exposes only `read_file`, `grep_search`, `query_lineage`, `run_sql_readonly`, `git_inspect`, and `apply_patch`, keeping tool schemas small across turns and hiding workflow mechanics from the model.
- **Role and SQL guardrails.** [PolicyRbacHook](src/hooks/policy_rbac.py) enforces `de`, `da`, and `sre` rules from [policies configuration](configs/policies.yaml), while [SqlGuardHook](src/hooks/sql_guard.py) inspects SQLGlot ASTs to block DML or escape functions, enforce partition pruning, and cap result rows at 50.
- **Run-scoped PII redaction.** [PiiRedactorHook](src/hooks/pii_redactor.py) scrubs sensitive values in SQL results and plain-text tool outputs with Run-scoped HMAC-SHA256 pseudonyms, preserving join cardinality while hiding plaintext across Runs.
- **Alert convergence and budget control.** [AlertStormAggregator](src/core/aggregator.py) groups cascading downstream alerts into one root Incident by lowest common ancestor, [ContextCompactorHook](src/hooks/compactor.py) spills large outputs into workspace scratch storage and tracks active budgets excluding human review pauses, and [AntiThrashHook](src/hooks/anti_thrash.py) trips after repeated calls or error stagnation.
- **Shadow Overlay verification and atomic promotion.** [RunWorktreeManager](src/core/worktree.py) isolates candidate patches in detached git worktrees, [StopGateHook](src/hooks/stop_gate.py) runs affected Spark jobs and dbt models through [ShadowOverlayVerifier](src/core/shadow_verifier.py), and [PromotionService](src/core/promotion.py) rolls back production git, DuckDB, and Silver Delta directories on post-commit verification failure.

Read [domain glossary](GLOSSARY.md) and the architectural decision records starting at [single-loop architecture](docs/adr/0001-single-loop-harness-over-workflow-graph.md), [dual-engine isolation](docs/adr/0002-subprocess-spark-and-duckdb-dual-engine.md), [stateless local checkpoints](docs/adr/0003-local-stateless-checkpoint-across-llm-apis.md), and [orthogonal tools with stop hooks](docs/adr/0004-six-orthogonal-tools-with-stop-hook-hitl.md) for rationale and constraints.

## Quickstart

Install Python 3.11 or Python 3.12, OpenJDK 17, Git, and `uv`.

1. Export your tool and Java paths in non-interactive shells before running `uv`, pytest, or PySpark commands.

	```bash
	export PATH="$HOME/.local/bin:$HOME/.local/opt/jdk-17/bin:$PATH"
	```

2. Sync dependencies and editable workspace packages.

	```bash
	uv sync
	```

3. Run static lint checks across source code, benchmarks, tests, and local skills.

	```bash
	uv run ruff check src benchmarks tests .agents/skills
	```

4. Reset or regenerate the synthetic e-commerce lakehouse inside `workspace/` when testing on a clean baseline.

	```bash
	uv run python benchmarks/reset_env.py --seed 42 --orders 6000 --users 1000
	```

	The default seed generates 6,000 raw orders, 1,000 users, and daily exchange rates into Delta Silver tables (`silver_users`, `silver_exchange_rates`, `silver_orders`), builds the DuckDB mart tables (`fct_daily_gmv`, `dim_user_ltv`), and commits baseline history to the sandbox git repository.

5. Execute unit and integration tests sequentially so local Spark subprocesses do not contend on disk I/O.

	```bash
	uv run pytest tests/test_harness_core.py tests/test_governance_hooks.py tests/test_shadow_and_hitl.py -v
	```

## Human-in-the-loop CLI workflow

When a Fix Mode Run passes [DefaultIntegrityOracle](src/core/shadow_verifier.py) and shadow execution, [StopGateHook](src/hooks/stop_gate.py) writes an approval packet and pauses the Run in `AWAITING_APPROVAL`. Operators review and finish the Run through [the CLI module](src/cli.py).

1. Inspect the paused Run checkpoint before approving or rejecting.

	```bash
	uv run python -m src.cli status <run_id> --runs-dir runs
	```

	The command prints JSON showing `"status": "AWAITING_APPROVAL"`, `"rejection_count"`, `"patched_files"`, and `"has_approval_packet": true`.

2. Reject a candidate patch to return feedback to the model and grant 8 extra turns, up to `max_rejections: 2` in [policies configuration](configs/policies.yaml).

	```bash
	uv run python -m src.cli resume <run_id> \
		--workspace workspace \
		--runs-dir runs \
		--reject \
		--reason "Confirm dim_user_ltv conservation alongside fct_daily_gmv."
	```

3. Approve a verified candidate patch to apply it to production sandbox git, replay affected Spark and dbt assets, and transition the Run to `RESOLVED`.

	```bash
	uv run python -m src.cli resume <run_id> \
		--workspace workspace \
		--runs-dir runs \
		--approve
	```

4. Run the isolated end-to-end verifier to exercise toolchain doctor checks, CLI rejection and promotion lifecycle, formal checks, and ablation execution inside `/tmp` without altering the repository `workspace/`.

	```bash
	uv run python .agents/skills/verify-dataops-harness/scripts/run_verification.py \
		--proof-dir /tmp/dataops-verify-proof
	```

	On the current codebase, `doctor.json` reports `passed: true` with `baseline_mart_order_count` equal to `5116`. In `cli_lifecycle.json`, `--reject` updates `max_turns` from `20` to `28`, and `--approve` validates exact mart conservation with both `baseline_gmv_total` and `promoted_gmv_total` equal to `734033.52`. See [end-to-end verification skill](.agents/skills/verify-dataops-harness/SKILL.md) for all proof artifacts.

## Formal state-machine verification

The [Run lifecycle TLA+ specification](.agents/skills/verify-run-state-machine/specs/RunLifecycle.tla) and [model checker script](.agents/skills/verify-run-state-machine/scripts/check_run_state_machine.py) verify safety properties across `RUNNING`, `AWAITING_APPROVAL`, `RESOLVED`, `DIAGNOSED`, `ABORTED_THRASH`, `ABORTED_BUDGET`, and `FAILED_ERROR`.

```bash
uv run python .agents/skills/verify-run-state-machine/scripts/check_run_state_machine.py
```

Running this command exits with code `0` and verifies these guarantees:

- **Exhaustive BFS exploration.** Explores 110 reachable states and 159 transitions up to depth 9 with zero violations across `NoUnapprovedProdMutation`, `PromotionRequiresGreenShadow`, `AwaitingApprovalRequiresGreenShadow`, `DiagnosisOnlyNeverPromotes`, `BoundedRejections`, `TurnBudgetBounded`, and `TerminalStatesAbsorbing`.
- **Mutation negative controls.** Confirms injected faults (`BuggyDirectPromote` and `BuggyApproveRollbackLeak`) fail immediately with counterexample traces.
- **Live Python conformance.** Exercises real [CheckpointStore](src/core/context.py), [HarnessLoop](src/core/loop.py), [ContextCompactorHook](src/hooks/compactor.py), [StopGateHook](src/hooks/stop_gate.py), and [PromotionService](src/core/promotion.py) implementations across all 6 conformance checks, including Spark and dbt failed-promotion rollback and `FAILED_ERROR` resume blocking.

See [formal verification skill guide](.agents/skills/verify-run-state-machine/SKILL.md) whenever modifying Run lifecycle code.

## Three-arm ablation benchmark

[AblationBenchmarkRunner](benchmarks/ablation_runner.py) compares three progressively equipped configurations against 14 Fault Scenarios across Spark execution failures (`spark`), dbt and cross-engine bugs (`dbt_cross_engine`), adversarial security probes (`security`), and operational resilience tests (`resilience`):

```bash
uv run python -m benchmarks.ablation_runner \
	--provider fake_replay \
	--max-turns 20 \
	--output /tmp/ablation_report.md \
	--json-output /tmp/ablation_report.json
```

Observed results across all 14 scenarios using the deterministic `fake_replay` driver (`max_turns=20`, regenerated by the command above):

| Arm | Verified Fix Rate | Safety Interception Rate | Avg Turns | Token Cost | Prompt Cache Hit Rate | Failure Attribution |
| --- | --- | --- | --- | --- | --- | --- |
| Group A (Bare ReAct Baseline) | 11.1% (1/9) | 0.0% (0/13) | 5.29 | 139081 tok / $0.0000 (proj $0.4455) | 0.0% | alert_storm_amplification:1, cheat_or_test_bypass:6, hitl_lifecycle_incomplete:1, missing_lineage_or_wrong_layer:1, pii_leakage:1, safety_policy_bypass:2, thrash_budget_exhausted:1 |
| Group B (+Context & Lineage) | 66.7% (6/9) | 0.0% (0/13) | 4.64 | 111146 tok / $0.0000 (proj $0.2492) | 37.9% | alert_storm_amplification:1, cheat_or_test_bypass:2, hitl_lifecycle_incomplete:1, pii_leakage:1, safety_policy_bypass:2, thrash_budget_exhausted:1 |
| Group C (Full DataOps-Harness) | 100.0% (9/9) | 100.0% (13/13) | 3.21 | 74204 tok / $0.0000 (proj $0.1786) | 34.7% | none (0 failures) |

### Filtering arms and saving traces

Pass `--scenarios`, `--arms`, and `--trace-dir` to run a targeted subset and inspect turn-by-turn tool calls:

```bash
uv run python -m benchmarks.ablation_runner \
	--provider fake_replay \
	--scenarios dbt_broken_macro_git \
	--arms group_b group_c \
	--max-turns 20 \
	--trace-dir /tmp/ablation-traces \
	--output /tmp/ablation_subset.md \
	--json-output /tmp/ablation_subset.json
```

- `--arms` accepts any subset of `group_a`, `group_b`, and `group_c`, and defaults to all three arms.
- `--trace-dir` saves per-run `checkpoint.jsonl` logs under `<trace_dir>/<scenario_id>/<arm_id>/<run_id>/checkpoint.jsonl`.
- The Markdown report appends a `## Tool Call Traces` section with compact per-turn tool steps, and the JSON report includes `tool_trace` and `trace_paths` on each scenario record.

### Model providers

Configure credentials for any provider defined in [models configuration](configs/models.yaml) and pass its key to `--provider`. All live providers run under identical scenario isolation and hidden regression oracles.

| Provider key | Protocol | Model | API key env | Base URL override env | Notes |
| --- | --- | --- | --- | --- | --- |
| `fake_replay` | `fake_replay` | `fake-replay-v1` | None | None | Default deterministic zero-token driver |
| `anthropic_messages` | `anthropic_messages` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` | Ephemeral prompt caching |
| `openai_chat` | `openai_chat` | `gpt-4o` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` | Automatic prefix caching |
| `openai_responses` | `openai_responses` | `gpt-4.1` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` | Stateless (`store: false`) |
| `gemini_free` | `openai_chat` | `gemini-3.5-flash-lite` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` | Free tier, retries `429` and `5xx` up to 4 times, preserves Gemini 3.5 `thought_signature` on tool calls |
| `openrouter_free` | `openai_chat` | `nvidia/nemotron-3-ultra-550b-a55b:free` | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` | Free tier via OpenRouter (`X-Title: DataOps-Harness`), subject to daily free request limits |
| `openrouter_free_router` | `openai_chat` | `openrouter/free` | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` | OpenRouter automatic free-model router, subject to daily free request limits |
| `opencode_free` | `openai_chat` | `deepseek-v4.1-flash-free` | `OPENCODE_API_KEY` | `OPENCODE_BASE_URL` | Free tier via OpenCode Zen, subject to free usage quota limits |

Run a live free-tier evaluation with `gemini_free` after exporting `GEMINI_API_KEY`:

```bash
export GEMINI_API_KEY="<your-key>"
uv run python -m benchmarks.ablation_runner \
	--provider gemini_free \
	--scenarios dbt_broken_macro_git \
	--arms group_c \
	--max-turns 12 \
	--trace-dir /tmp/gemini-traces \
	--output /tmp/gemini_report.md
```

## Repository layout

- [`src/`](src/) holds the core runtime loop ([`src/core/`](src/core/)), lifecycle hooks ([`src/hooks/`](src/hooks/)), wire and engine adapters ([`src/adapters/`](src/adapters/)), atomic tool registry ([`src/tools/`](src/tools/)), and [CLI entrypoint](src/cli.py).
- [`benchmarks/`](benchmarks/) holds synthetic environment reset scripts ([reset script](benchmarks/reset_env.py)), Fault Scenarios, and the 3-arm ablation benchmark runner ([ablation runner](benchmarks/ablation_runner.py)).
- [`configs/`](configs/) defines LLM provider settings ([models configuration](configs/models.yaml)) and governance bounds ([policies configuration](configs/policies.yaml)).
- [`skills/`](skills/) holds read-only operational playbooks ([global playbook](skills/GLOBAL_PLAYBOOK.md), [Spark troubleshooting](skills/spark_troubleshooting.md), [dbt troubleshooting](skills/dbt_troubleshooting.md)).
- [`docs/`](docs/) holds architectural decision records ([ADR directory](docs/adr/)). Root documentation files include [domain glossary](GLOSSARY.md) and [agents guide](AGENTS.md).
- [`.agents/skills/`](.agents/skills/) holds runnable verification and formal state-machine checking skills.
- [`tests/`](tests/) contains 13 pytest modules covering core runtime, governance hooks, shadow isolation, fault scenarios, verification skills, and security hardening.
