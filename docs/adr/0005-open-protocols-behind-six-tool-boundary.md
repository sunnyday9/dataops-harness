---
status: accepted
---

# Adopt Open DataOps Protocols Behind the Six-Tool Security Boundary

DataOps-Harness integrates external pipeline metadata, failure alerts, SQL dialects, and audit traces through open standards (OpenLineage, versioned dbt Artifacts `manifest.json` and `run_results.json`, SQLGlot multi-dialect AST parsing, and OpenTelemetry-compatible spans) while placing every protocol client behind internal adapters rather than exposing raw external MCP servers directly to the model. Preserving ADR-0004's six-tool surface guarantees that `PolicyRbacHook`, `SqlGuardHook`, and `PiiRedactorHook` remain unconditional middleware ceilings across all connected engines and prevents unbounded tool-schema bloat.

## Considered Options

- **Direct external MCP tool passthrough to the model**: Fast to wire new integrations, but violates ADR-0004, inflates tool-schema tokens, and lets arbitrary MCP tools bypass SQL AST inspection and Run-scoped PII redaction.
- **Custom per-engine JSON schemas and hardcoded manifests**: Simple for one demo repo, but forces per-project glue code and breaks portability across DuckDB, PostgreSQL, cloud warehouses, and remote Spark clusters.
- **Open protocols behind internal adapters with six-tool facade (chosen)**: Eliminates hardcoded project bindings while keeping security, PII redaction, and prompt-cache stability enforced deterministically in middleware.
