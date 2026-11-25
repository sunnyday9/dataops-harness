---
status: accepted
---

# Single-Loop Agent Harness Over Hardcoded Workflow Graph

DataOps-Harness uses a single generic `Observe → Act → Verify` agent loop paired with fault-agnostic runtime hooks (RBAC, SQL AST guard, PII redaction, context compaction, shadow verification, anti-thrash) rather than a hardcoded multi-node workflow graph (e.g., LangGraph state machines that fix the sequence "read log → check lineage → query table → patch"). Faults across heterogeneous Spark and dbt pipelines rarely follow a fixed diagnostic path, so keeping diagnostic sequencing in the model and playbooks while pushing safety, cost, and verification guarantees into deterministic middleware prevents brittle workflow branching while preserving hard operational invariants.

## Considered Options

- **Hardcoded multi-node workflow (LangGraph / DAG state machine)**: Predictable per-fault flow, but requires new branches for every fault class and couples safety/verification logic to specific nodes.
- **Unconstrained ReAct loop**: Flexible, but unsafe on production-like data, prone to context blowup, and vulnerable to cheat fixes or thrashing.
- **Single-loop harness with lifecycle hooks (chosen)**: Preserves autonomous diagnosis across novel failures while enforcing safety, PII, budget, and verification deterministically outside the model.
