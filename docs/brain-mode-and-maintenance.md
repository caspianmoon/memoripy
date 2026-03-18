# Brain Mode and Maintenance

Memoripy can run in a classic retrieval mode or in an activation-aware brain-like mode. This guide explains how to configure and use that behavior.

## `BrainConfig`

`BrainConfig` controls the optional brain-like fast mode.

Example:

```python
from memoripy import BrainConfig, MemoryClient, MemoryPipelineConfig

client = MemoryClient(
    pipeline=MemoryPipelineConfig(
        brain=BrainConfig(mode="attention_fast"),
        default_include_trace=True,
    )
)
```

Current fields:

- `mode`
- `working_memory_size`
- `attention_decay_half_life_hours`
- `dormancy_threshold`
- `activation_spread`
- `fast_path_candidate_limit`
- `consolidation_window_hours`
- `consolidation_min_support`

## Classic vs `attention_fast`

### Classic

Classic mode is the default.

Use it when:

- you want the baseline ranking behavior
- you do not need working memory
- you want the simplest behavior profile

### `attention_fast`

Use `attention_fast` when you want:

- working-memory selection
- activation-aware ranking
- dormant memories that can later reactivate
- explicit consolidation of episodic into semantic memory
- richer trace output for attention and maintenance

This mode is designed to feel more brain-like while keeping the critical path local.

## `MemoryPipelineConfig`

Brain mode is part of the broader pipeline config.

Example:

```python
from memoripy import BrainConfig, LocalAssetProcessor, MemoryClient, MemoryPipelineConfig

client = MemoryClient(
    pipeline=MemoryPipelineConfig(
        brain=BrainConfig(mode="attention_fast"),
        asset_processor=LocalAssetProcessor(),
        default_include_trace=True,
        semantic_promotion_threshold=0.72,
        pending_confidence_threshold=0.75,
    )
)
```

Public pipeline fields:

- `extractor`
- `reconciler`
- `reranker`
- `asset_processor`
- `brain`
- `semantic_promotion_threshold`
- `pending_confidence_threshold`
- `default_include_trace`
- `max_trace_results`

## What `working_memory` Means

When `attention_fast` is enabled, `ContextPack` can include a `working_memory` section.

This is not the same as the full retrieved result set. It is the small set of highly activated memories selected before the rest of the section-filling logic.

Example:

```python
pack = client.context.build(
    query="What matters most right now?",
    user_id="u1",
    include_trace=True,
)

print(pack.working_memory)
print(pack.trace["working_memory"])
```

## Dormancy and Reactivation

`MemoryState.DORMANT` is a soft-forgetting state in `attention_fast`.

Dormant memories:

- remain stored
- keep their evidence and version history
- are usually deprioritized in grounding
- can come back through a strong direct cue

This lets the system forget softly without losing auditability.

## Maintenance and Consolidation

Memoripy exposes explicit maintenance through:

```python
summary = client.maintenance.consolidate(
    user_id="u1",
    limit=200,
    budget_ms=25,
    idempotency_key="maint-1",
)
```

The async client exposes the same surface:

```python
await async_client.maintenance.consolidate(user_id="u1")
```

The maintenance summary can include:

- `status`
- `scope`
- `processed_records`
- `promotions`
- `skipped`
- `dormancy_transitions`
- `projection_status`

## What Consolidation Does

At a high level, consolidation:

- scans eligible episodic memories
- groups related evidence into clusters
- promotes repeated or sufficiently supported patterns into semantic memory
- preserves contradiction safety
- updates maintenance metadata and activation state

It is explicit and user-triggered. Memoripy does not ship a scheduler.

## Service Surface

The same functionality is available over HTTP:

- `POST /v3/maintenance/consolidate`

See [Service and storage](./service-and-storage.md) for the full service view.

## When To Stay on Classic Mode

Stay on classic mode when:

- you do not need working memory
- you prefer the smallest behavioral surface
- you are validating baseline retrieval first

Move to `attention_fast` when you want the system to behave more like an attention-and-consolidation layer.

## Trace in Brain Mode

With `include_trace=True`, search and context can expose:

- activation scores
- working-memory selections
- consolidation metadata
- dormancy transitions

This is the best way to understand how `attention_fast` behaved on a particular query.

## Next Steps

- [Service and storage](./service-and-storage.md)
- [Architecture appendix](./architecture.md)
