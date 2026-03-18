# Architecture Appendix

This appendix explains how Memoripy works internally at a high level. It is intentionally descriptive rather than exhaustive.

## System Shape

The core runtime is centered around:

- `MemoryClient` and `AsyncMemoryClient`
- `MemoryEngine`
- repositories
- extractors and pipeline components
- types such as `EvidenceItem`, `MemoryRecord`, `MemoryVersion`, `SearchFilters`, and `ContextPack`

The service layer is a thin wrapper around the same client behavior.

## Ingestion Flow

At a high level, ingestion works like this:

1. input is normalized into messages, events, or items
2. evidence items are built and persisted
3. the configured extractor produces memory candidates
4. the reconciler decides whether each candidate should add, update, supersede, delete, or no-op
5. records and versions are written
6. projections are rebuilt

Important outcome: Memoripy stores raw evidence and durable memory separately.

## Extractor, Reconciler, Reranker Pipeline

The pipeline is configurable through `MemoryPipelineConfig`.

### Extractor

The extractor is responsible for turning evidence into candidates.

The default extractor handles:

- profile attributes
- preferences
- relations
- episodic summaries

### Reconciler

The reconciler decides how a new candidate interacts with existing memory.

The default reconciler handles:

- matching related semantic slots
- profile field updates
- preference contradictions
- version transitions and reasoning traces

### Reranker

The reranker is optional and runs after the base ranking flow.

Use it when you need a second stage of ranking without rewriting the engine.

### Asset Processor

The asset processor is optional and runs before evidence is built from ingestion items.

The built-in `LocalAssetProcessor` can derive text from supported document-like inputs and metadata.

## Projections

Memoripy maintains projections that support retrieval and ranking.

Current projection categories include:

- lexical
- graph
- activation
- consolidation metadata
- projection status

These are rebuilt or refreshed as memory changes.

## Ranking

Base ranking uses a mix of signals such as:

- lexical overlap
- vector similarity
- recency
- access behavior
- scope match
- salience
- intent
- graph connectivity

If configured, a reranker contributes after the base ranking stage.

## Trace Output

Trace is the debugging surface that makes ranking and grounding explainable.

Depending on the call and configuration, trace can include:

- pipeline description
- ranking results and rank breakdowns
- reasoning trace for current versions
- grounding inclusion and omission
- activation details
- working-memory selection
- consolidation metadata

## Attention and Activation Lifecycle

When `attention_fast` is enabled:

- activation data is tracked in projections
- retrieval can update access and activation counters
- context building can choose a working-memory subset
- low-activation memories can become dormant
- direct cues can reactivate dormant memories

This is what gives the system a more brain-like feel without turning it into a reflective or LLM-heavy architecture.

## Consolidation

Explicit maintenance scans relevant episodic memory and can promote repeated patterns into semantic memory.

Consolidation is constrained by:

- scope
- support count
- confidence
- contradiction safety
- budget limits

It is exposed through `maintenance.consolidate(...)` and the equivalent HTTP route.

## Contradictions and History

Memoripy does not hide contradictions by overwriting history.

Instead, it preserves:

- versions
- contradiction links
- reasoning trace
- supporting evidence

This makes it suitable for systems that need auditable memory changes.

## Benchmarks

The benchmark harness in [benchmarks/README.md](../benchmarks/README.md) exercises:

- fact extraction
- reconciliation
- retrieval
- grounding
- maintenance traceability
- multimodal recall

The repo also includes a synthetic latency probe for comparing local timing on the same machine.

## Related Guides

- [Concepts and mental model](./concepts.md)
- [API reference](./api-reference.md)
