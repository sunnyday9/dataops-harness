---
status: accepted
---

# Six Orthogonal Tools With Stop-Hook HiTL and Artifact-Based Spark Visibility

DataOps-Harness exposes strictly six orthogonal atomic tools to the model (`read_file`, `grep_search`, `query_lineage`, `run_sql_readonly`, `git_inspect`, `apply_patch`) rather than expanding the tool surface with workflow-specific tools (`propose_production_fix`, `load_skill`, `inspect_spark_events`). Human-in-the-loop approval is triggered deterministically by the `Stop` hook once shadow verification succeeds; `read_file` grants read-only access to `skills/*.md` alongside `workspace/` while `apply_patch` stays jailed to `workspace/`; Spark event-log summaries are emitted into `workspace/incidents/<id>/spark_metrics.json` for standard file inspection; and `run_sql_readonly` accepts an optional `engine: "duckdb" | "spark"` parameter (defaulting to `"duckdb"`).

## Considered Options

- **Expanded 8–9 tool surface**: Adds explicit `propose_production_fix`, `load_skill`, and `inspect_spark_events` tools, increasing tool-schema tokens and inviting model-side workflow coupling.
- **Six orthogonal tools + lifecycle hooks (chosen)**: Keeps the tool schema minimal and cache-friendly, enforces HiTL verification unconditionally in middleware rather than relying on the model calling a proposal tool, and reuses standard file/SQL tools across engines.
