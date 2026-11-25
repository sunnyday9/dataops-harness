---
status: accepted
---

# Local Stateless Checkpointing Across Heterogeneous LLM Wire APIs

DataOps-Harness normalizes multiple LLM wire APIs (Anthropic Messages, OpenAI Chat Completions, OpenAI Responses, and a deterministic fake/replay driver) behind a thin adapter protocol and keeps all Run conversation state 100% local in its append-only checkpoint, issuing stateless requests with stable static prefixes rather than relying on server-side conversation state (such as OpenAI's `previous_response_id`). Context compaction, `.scratch/` output spilling, prompt-cache hit accounting, and crash/HiTL resume require full local control over message history; coupling to provider-managed threads would break compaction and make cross-provider ablation non-comparable.

## Consequences

- Every provider adapter must serialize the compacted local message history into its wire format and normalize tool calls and cached-token usage metrics back into a common domain model.
- Prompt caching relies on keeping the static prefix byte-identical across turns (plus explicit `cache_control` markers where the provider requires them).
