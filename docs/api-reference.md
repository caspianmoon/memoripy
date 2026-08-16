# Memoripy v4 API reference

## `Memory`

A small facade for the common workflow.

```python
Memory(path=None, **client_kwargs)
```

Methods:

- `capture(text=None, **kwargs)`
- `recall(query, **kwargs)`
- `search(query, **kwargs)`
- `write(key=..., value=..., **kwargs)`
- `explain(memory_id)`
- `correct(memory_id, value, **kwargs)`
- `forget(memory_id, **kwargs)`
- `audit()`

## `MemoryClient`

### Construction

```python
MemoryClient(
    repository=None,
    chat_model=None,
    embedding_model=None,
    extractor=None,
    pipeline=None,
)

MemoryClient.from_path(path, **kwargs)
```

### Capture and writing

```python
client.capture(
    messages=None,
    events=None,
    items=None,
    user_id=None,
    agent_id=None,
    run_id=None,
    project_id=None,
    organization_id=None,
    namespace=None,
    idempotency_key=None,
)
```

Returns evidence IDs, created and updated records, pending records, quarantined candidates, rejected candidates, admission decisions, and projection status.

```python
client.write(
    kind,
    key,
    value,
    summary=None,
    subject=None,
    metadata=None,
    tags=None,
    entity_names=None,
    layer="semantic",
    observed_at=None,
    durability="durable",
    trust_level="authoritative",
    valid_from=None,
    valid_to=None,
    **scope,
)
```

### Search

```python
client.search(
    query,
    limit=5,
    filters=None,
    include_trace=False,
    include_historical=False,
    as_of=None,
    expand_scope=True,
    track_usage=True,
    **scope,
)
```

Search results contain:

- `memory`
- `score`
- `rank_breakdown`
- `retrieval_receipt`
- `evidence`
- `projection_status`

### Context

```python
client.context.build(
    query=None,
    messages=None,
    limit=8,
    max_tokens=480,
    include_debug=False,
    include_trace=False,
    include_historical=False,
    as_of=None,
    context_policy="compact",
    **scope,
)
```

Returns a `ContextPack`.

### Record operations

- `get(memory_id=...)`
- `get_all(filters=None, **scope)`
- `update(memory_id=..., data=..., idempotency_key=None)`
- `correct(memory_id=..., value=..., reason=None, valid_from=None, idempotency_key=None)`
- `delete(memory_id=..., idempotency_key=None)`
- `forget(memory_id=..., idempotency_key=None)`
- `delete_all(filters=None, idempotency_key=None, **scope)`
- `history(memory_id=...)`
- `explain(memory_id=...)`

### Quality and lifecycle

- `audit()`
- `feedback(memory_id=..., outcome=...)`
- `maintenance.consolidate(...)`
- `recover()` for file repositories

Supported feedback outcomes:

- `included`
- `used`
- `success`
- `confirmed`
- `corrected`
- `rejected`
- `failure`

### Import and export

- `export()`
- `import_(payload, mode="merge" | "replace", idempotency_key=None)`

## `AsyncMemoryClient`

The asynchronous client mirrors `MemoryClient` and runs synchronous repository operations through `asyncio.to_thread`.

## Configuration

### `AdmissionConfig`

Controls confidence thresholds, assistant semantic writes, generated-summary writes, sensitive-data quarantine, external-instruction quarantine, evidence-span requirements, and admission-log retention.

### `RetrievalConfig`

Controls reciprocal-rank fusion and lane weights for lexical, semantic, exact, entity, temporal, authority, activation, and policy retrieval.

### `BrainConfig`

Controls attention mode, working-memory size, decay, dormancy, consolidation, and utility weighting.

### `MemoryPipelineConfig`

Combines extractor, admission policy, reconciler, reranker, asset processor, brain, retrieval, and trace defaults.

## Service

- `MemoryService`
- `serve_http(...)`
- `create_fastapi_app(...)`

The service is a local-development surface unless the caller supplies deployment authentication, authorization, tenancy, and operational controls.
