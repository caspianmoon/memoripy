# Concepts and Mental Model

Memoripy is easiest to understand if you separate its memory system into five layers:

1. ingestion inputs
2. raw evidence
3. durable memory records and versions
4. retrieval and ranking
5. grounded outputs such as search results, context packs, and chat completions

## Evidence Comes First

Memoripy does not treat memory as only a list of facts. It keeps the source material that produced those facts.

Examples of evidence:

- user and assistant messages
- tool calls
- tool results
- assistant actions
- standalone ingestion items such as documents or assets

Evidence lets Memoripy preserve provenance. A memory can later explain:

- where it came from
- what evidence supports it
- what later contradicted it

This is one of the main differences between Memoripy and a simple vector-memory wrapper.

## Memory Records and Versions

Memoripy stores extracted memory as durable records plus immutable versions.

### Memory Record

A memory record is the current live view of a memory. It includes fields such as:

- `record_id`
- `kind`
- `key`
- `value`
- `summary`
- `state`
- `scope`
- `layer`
- `confidence`
- `salience`
- `evidence_ids`
- `citation_evidence_ids`

### Memory Version

A version is the historical write that produced or changed a memory. Versions preserve:

- action taken: `ADD`, `UPDATE`, `SUPERSEDE`, `DELETE`, or `NONE`
- state at that point in time
- evidence used
- reasoning trace
- contradiction links

This means Memoripy can preserve both the current truth and how that truth changed.

## Semantic vs Episodic Memory

Memoripy distinguishes between two memory layers.

### Semantic Memory

Semantic memory is durable knowledge that should be reusable later. Typical examples:

- name
- location
- employer
- favorite city
- stable preferences
- relationships

Semantic memories are usually what you want when answering questions like:

- "What do you know about me?"
- "Where do I live?"
- "What tools do you know I use?"

### Episodic Memory

Episodic memory is the recent or salient memory of what happened. Typical examples:

- recent user requests
- tool results from a session
- assistant actions
- conversation events worth remembering even if they are not durable profile facts

Episodic memory is useful for questions like:

- "What happened earlier?"
- "What did the tool just return?"
- "What did we do in this run?"

## Scope Hierarchy

Every memory and evidence item can be scoped by:

- `user_id`
- `agent_id`
- `run_id`

These scopes let Memoripy decide how broad or narrow a memory is.

Examples:

- `user_id="u1"` means this memory belongs to a user broadly
- `user_id="u1", agent_id="jarvis"` means it belongs to a specific assistant for that user
- `user_id="u1", agent_id="jarvis", run_id="trip-1"` means it belongs to one specific run or session

By default, retrieval supports hierarchical scope. A run-scoped memory can outrank broader memories when both match.

## Context Packs

`context.build(...)` returns a `ContextPack`, which is a structured memory bundle built for a query.

A context pack can contain:

- `working_memory`
- `profile`
- `preferences`
- `relationships`
- `recent_episodes`
- `tool_observations`
- `citations`
- `trace`

This is the main v3 grounding format. It is more useful than a flat search result when you want to build prompts or agent context deliberately.

## Citations and Trace

Memoripy has two different explainability surfaces.

### Citations

Citations show the evidence that supports a memory in the current result or context pack.

Use citations when you care about:

- provenance
- auditability
- debugging why a result appeared

### Trace

Trace is optional and only returned when requested with `include_trace=True` or enabled by pipeline defaults.

Trace can include:

- pipeline configuration used
- ranking and rank breakdowns
- reconciliation reasoning
- grounding inclusion and omission
- activation data in `attention_fast`
- maintenance and consolidation metadata

Trace is for debugging the system, not just consuming the result.

## `attention_fast` Brain Mode

Memoripy supports a configurable brain mode through `BrainConfig`.

Classic mode keeps the existing retrieval behavior.

`attention_fast` adds:

- activation-aware ranking
- working-memory selection
- dormant memories that can still be recovered
- explicit episodic-to-semantic consolidation
- richer trace output for activation and maintenance behavior

This mode is designed to feel more human-brain-like without introducing LLM calls into the hot path.

## Dormancy

In `attention_fast`, a memory can become `MemoryState.DORMANT`.

Dormant means:

- it is not deleted
- it still has evidence and history
- it is usually deprioritized in retrieval and grounding
- it can be reactivated by a direct cue

This gives Memoripy a soft-forgetting behavior while keeping auditability intact.

## Consolidation

Consolidation is how episodic memories can strengthen into more durable semantic memories.

Memoripy exposes this explicitly through:

```python
client.maintenance.consolidate(...)
```

Consolidation is user-triggered. It does not run on a built-in scheduler.

## Where To Go Next

- [Getting started](./getting-started.md)
- [Assistant-first memory guide](./assistant-memory.md)
- [Brain mode and maintenance](./brain-mode-and-maintenance.md)
