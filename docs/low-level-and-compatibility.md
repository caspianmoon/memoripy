# Low-Level and Compatibility Guide

Memoripy supports a lower-level memory API alongside the v3 assistant-first surface. This guide covers the direct memory operations and older compatibility paths.

## When To Use the Low-Level API

Choose the low-level API when:

- you want direct CRUD-style memory control
- your application is not naturally organized as assistant turns
- you want to inspect or manage memories explicitly
- you are integrating with older v2-style workflows

## Core Low-Level Methods

The main low-level methods on `MemoryClient` are:

- `add(...)`
- `search(...)`
- `get(...)`
- `get_all(...)`
- `update(...)`
- `delete(...)`
- `delete_all(...)`
- `history(...)`
- `export()`
- `import_(...)`

The same surface exists on `AsyncMemoryClient`.

## `add(...)`

Use `add(...)` to ingest text, messages, or items directly.

```python
client.add(text="I live in Istanbul", user_id="u1")
```

You can also pass messages:

```python
client.add(
    messages=[
        {"role": "user", "content": "My favorite city is Tokyo"},
        {"role": "assistant", "content": "Tokyo is a great city."},
    ],
    user_id="u1",
)
```

## `search(...)`

Use `search(...)` for ranked flat retrieval.

```python
result = client.search(query="favorite city", user_id="u1", include_trace=True)
```

The result is useful when you want:

- ranked top-k memory
- evidence alongside each match
- a simple retrieval API instead of a sectioned context pack

## `get(...)`, `get_all(...)`, and `history(...)`

Retrieve explicit memory objects:

```python
record = client.get(memory_id="memory_...")
records = client.get_all(user_id="u1")
history = client.history(memory_id="memory_...")
```

Use these for inspection, debugging, and admin workflows.

## `update(...)`

Update the current value or metadata for an existing memory.

```python
client.update(
    memory_id="memory_...",
    data={
        "value": "Berlin",
        "summary": "Location: Berlin",
    },
)
```

Because Memoripy is versioned, an update creates or changes version state rather than silently replacing history.

## `delete(...)` and `delete_all(...)`

Delete one memory:

```python
client.delete(memory_id="memory_...")
```

Delete multiple memories with scope or filters:

```python
client.delete_all(user_id="u1")
```

`delete_all(...)` protects against unsafe calls by requiring scope or filters.

## `export()` and `import_(...)`

Export the full engine state:

```python
snapshot = client.export()
```

Import a snapshot:

```python
client.import_(snapshot, mode="merge")
```

Supported modes:

- `merge`
- `replace`

This is useful for:

- backup
- migration
- local debugging
- test fixtures

## `SearchFilters`

`SearchFilters` lets you control low-level retrieval behavior.

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

Example:

```python
from memoripy import SearchFilters

filters = SearchFilters(
    kinds=["preference"],
    layers=["semantic"],
    limit=3,
)

result = client.search(query="city", user_id="u1", filters=filters)
```

## When To Choose `v2` vs `v3`

Choose `v2`-style usage when:

- you want direct control
- you do not need assistant-turn grounding
- you prefer flat retrieval and explicit CRUD

Choose `v3` when:

- your app is an assistant or agent
- you need `capture(...)` and `context.build(...)`
- you want tool observation support and context sections
- you want `chat.completions.create(..., memory_strategy="v3")`

## Legacy Compatibility

The repo still exports compatibility helpers:

- `MemoryManager`
- `JSONStorage`
- `InMemoryStorage`
- `MemoryStore`

Example:

```python
from memoripy import JSONStorage, MemoryManager

manager = MemoryManager(storage=JSONStorage("memory.json"))
manager.add_interaction("My name is Khazar", "Nice to meet you")
print(manager.retrieve_relevant_interactions("name"))
```

Use these only when you need backward compatibility with older integration styles.

## Import Alias

`MemoryClient.import_()` is also exposed as `MemoryClient.import` for convenience, but `import_` is the clearer spelling in Python source.

## Next Steps

- [Assistant-first memory guide](./assistant-memory.md)
- [API reference](./api-reference.md)
