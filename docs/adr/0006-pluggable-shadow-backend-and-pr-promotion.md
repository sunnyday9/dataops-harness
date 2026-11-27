---
status: accepted
---

# Pluggable Shadow Backends and Pull-Request Promotion for External Repositories

DataOps-Harness separates shadow verification into a pluggable `ShadowBackend` lifecycle (`prepare_shadow`, `execute_subset`, `diff_against_prod`, `teardown`) so `HarnessLoop` and `StopGateHook` remain agnostic to whether isolation uses local DuckDB files, temporary PostgreSQL schemas, cloud warehouse clones, or remote Spark namespaces. For external repositories in Fix Mode, human-approved `Promotion` defaults to pushing a remediation branch and opening a GitHub Pull Request containing the `RCA Report`, unified diff, and shadow data diff. Final merge and production backfill remain in existing team CI/CD, while direct local replay remains available for isolated sandboxes and `Diagnosis-Only Mode` supports zero-risk read-only rollout.

## Considered Options

- **Direct production branch commit and in-place warehouse replay**: Works in the local synthetic sandbox, but bypasses team code review and makes production rollback hazardous on shared PostgreSQL, cloud warehouses, and remote clusters.
- **Diagnosis-Only Mode exclusively**: Zero mutation risk, but prevents the harness from delivering verified code patches and shadow diffs.
- **Pluggable `ShadowBackend` with PR-based Promotion (chosen)**: Unlocks DuckDB and PostgreSQL in `v0.2.0` while preserving clean extension seams for cloud warehouses and remote Spark clusters under standard Git review controls.
