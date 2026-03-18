# API Reference

This page documents the main public interfaces exported by `memoripy`.

## Clients

## `MemoryClient`

Main synchronous client.

Construction:

```python
from memoripy import MemoryClient

client = MemoryClient()
client = MemoryClient.from_path("./.memoripy")
```

Constructor inputs:

- `repository`
- `chat_model`
- `embedding_model`
- `extractor`
- `pipeline`

Namespaces:

- `client.chat.completions.create(...)`
- `client.context.build(...)`
- `client.maintenance.consolidate(...)`

Direct methods:

- `add(...)`
- `capture(...)`
- `search(...)`
- `get(...)`
- `get_all(...)`
- `update(...)`
- `delete(...)`
- `delete_all(...)`
- `history(...)`
- `export()`
- `import_(...)`

## `AsyncMemoryClient`

Async wrapper around the same surface.

It provides async versions of:

- all direct methods
- `chat.completions.create(...)`
- `context.build(...)`
- `maintenance.consolidate(...)`

## Configuration Types

## `MemoryPipelineConfig`

Pipeline-level configuration object.

Fields:

- `extractor`
- `reconciler`
- `reranker`
- `asset_processor`
- `brain`
- `semantic_promotion_threshold`
- `pending_confidence_threshold`
- `default_include_trace`
- `max_trace_results`

Use it when you want to plug in:

- custom extraction
- custom reconciliation
- custom reranking
- custom asset processing
- brain-mode behavior

## `BrainConfig`

Brain-mode configuration.

Fields:

- `mode`
- `working_memory_size`
- `attention_decay_half_life_hours`
- `dormancy_threshold`
- `activation_spread`
- `fast_path_candidate_limit`
- `consolidation_window_hours`
- `consolidation_min_support`

Most users only need:

```python
BrainConfig(mode="attention_fast")
```

## `SearchFilters`

Search constraint object.

Fields:

- `scope`
- `kinds`
- `states`
- `tags`
- `metadata`
- `layers`
- `source_types`
- `include_pending`
- `limit`
- `hierarchical_scope`

## `ContextPack`

Structured grounding object returned by `context.build(...)`.

Fields:

- `query`
- `scope`
- `intent`
- `working_memory`
- `profile`
- `preferences`
- `relationships`
- `recent_episodes`
- `tool_observations`
- `citations`
- `projection_status`
- `debug`
- `trace`

## `MemoryState`

Public states:

- `active`
- `dormant`
- `pending`
- `superseded`
- `deleted`

## Repositories

## `InMemoryRepository`

Default repository used when none is supplied.

## `FileMemoryRepository`

Used by `MemoryClient.from_path(...)`.

## `PostgresRepository`

SQL-backed repository that requires the `postgres` extras.

## Service Interfaces

## `MemoryService`

Request-handling facade used by both the lightweight HTTP server and FastAPI integration.

## `serve_http()`

Creates a basic threaded HTTP server.

## `create_fastapi_app()`

Creates a FastAPI app if the optional service dependencies are installed.

## Important Method Shapes

## `capture(...)`

Assistant-first ingestion.

Common inputs:

- `messages`
- `events`
- `items`
- `user_id`
- `agent_id`
- `run_id`
- `idempotency_key`

Common outputs:

- `id`
- `strategy`
- `scope`
- `evidence_ids`
- `created`
- `updated`
- `pending`
- `noop`
- `memory_ids`
- `semantic_memory_ids`
- `episodic_memory_ids`
- `projection_status`

## `search(...)`

Common inputs:

- `query`
- `user_id`
- `agent_id`
- `run_id`
- `limit`
- `filters`
- `include_trace`

Common outputs:

- `query`
- `filters`
- `results`
- `projection_status`
- optional `trace`

Each `results` entry includes:

- `memory`
- `score`
- `rank_breakdown`
- `evidence`
- `projection_status`

## `context.build(...)`

Common inputs:

- `query`
- `messages`
- `user_id`
- `agent_id`
- `run_id`
- `limit`
- `max_tokens`
- `filters`
- `include_debug`
- `include_trace`
- `context_policy`

Returns a `ContextPack`.

## `chat.completions.create(...)`

Common inputs:

- `messages`
- `user_id`
- `agent_id`
- `run_id`
- `model`
- `limit`
- `store`
- `idempotency_key`
- `tool_events`
- `memory_strategy`
- `include_memory_pack`
- `include_trace`
- `context_policy`

Common outputs:

- OpenAI-style `id`, `object`, `created`, `model`, and `choices`
- `memory`
- optional `memory_pack`
- optional `trace`

## `maintenance.consolidate(...)`

Common inputs:

- `scope`
- `user_id`
- `agent_id`
- `run_id`
- `limit`
- `budget_ms`
- `idempotency_key`

Common outputs:

- `status`
- `scope`
- `processed_records`
- `promotions`
- `skipped`
- `dormancy_transitions`
- `projection_status`

## Related Guides

- [Getting started](./getting-started.md)
- [Brain mode and maintenance](./brain-mode-and-maintenance.md)
- [Service and storage](./service-and-storage.md)
