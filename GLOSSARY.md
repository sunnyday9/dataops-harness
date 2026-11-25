# DataOps-Harness

A runtime harness that equips a general-purpose troubleshooting agent to diagnose, safely remediate, and verify failures across heterogeneous data pipelines.

## Language

**Alert**:
A single raw failure notification emitted by a pipeline job, data quality test, or monitor.
_Avoid_: alarm, raw error, event

**Alert Storm**:
A burst of multiple Alerts triggered downstream by a single upstream failure.
_Avoid_: alert flood, cascade

**Incident**:
A deduplicated root-cause troubleshooting unit produced by converging one or more lineage-related Alerts.
_Avoid_: fault, issue, ticket

**Run**:
A single end-to-end troubleshooting lifecycle attempting to resolve an Incident, from initial observation through terminal outcome or approval pause.
_Avoid_: session, execution, job

**Playbook**:
A static troubleshooting handbook—global or engine-specific—defining diagnostic conventions and output contracts for the agent.
_Avoid_: prompt, AGENTS.md, runbook

**Resolution Mode**:
The exit contract governing a Run: either Fix Mode (requiring verified remediation and human approval) or Diagnosis-Only Mode (requiring only an evidence-backed root-cause report).
_Avoid_: run mode, execution mode

**Promotion**:
The human-approved transition of a verified remediation from the isolated shadow area into production code and production data.
_Avoid_: deploy, merge, apply

**Fault Scenario**:
A reproducible benchmark fault paired with a hidden regression oracle used to evaluate remediation quality.
_Avoid_: case, bug, test fixture

**Principal Role**:
The caller identity (`de`, `da`, or `sre`) propagated through a Run to intersect user permissions with harness policy ceilings.
_Avoid_: user type, profile, persona

**Shadow Overlay**:
A read-through data view used during shadow verification in which patched upstream tables resolve from a Run's isolated shadow storage while unmodified tables fall back to read-only production storage.
_Avoid_: staging, sandbox copy

**Hidden Regression Oracle**:
An out-of-workspace verification specification that asserts test integrity, metric conservation, and root-cause-layer invariants against shadow outputs without being visible to the agent.
_Avoid_: test suite, evaluator script

**Keyed Pseudonym**:
A Run-scoped deterministic placeholder replacing a sensitive value so plaintext never leaves the harness while duplicate and join-cardinality patterns remain observable.
_Avoid_: static mask, hash, anonymized value

**RCA Report**:
The structured root-cause diagnosis artifact—citing root-cause layer, affected assets, concrete evidence, and proposed remediation—required before a Run can complete or request approval.
_Avoid_: summary, diagnosis text, final output

