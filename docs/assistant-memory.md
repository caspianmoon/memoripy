# Assistant-First Memory Guide

This guide explains how to use Memoripy as the memory layer behind an assistant or agent.

## When To Use the Assistant-First Surface

Prefer the assistant-first APIs when your application naturally operates in turns:

- user says something
- assistant replies
- tools are called
- tool results come back
- the next turn should reuse the most relevant context

The assistant-first surface is:

- `capture(...)`
- `context.build(...)`
- `chat.completions.create(..., memory_strategy="v3")`

## Modeling Inputs

Memoripy accepts three main kinds of assistant-turn input.

### Messages

Messages are usually the main source of user and assistant context.

```python
messages = [
    {"role": "user", "content": "My favorite city is Tokyo"},
    {"role": "assistant", "content": "Tokyo is a great city."},
]
```

Supported message fields are intentionally simple in the examples:

- `role`
- `content`
- optional `metadata`
- optional timestamps through `metadata` or message fields used by the implementation

### Events

Events represent tool or assistant activity.

Common event types:

- `tool_call`
- `tool_result`
- `assistant_action`

Example:

```python
events = [
    {
        "event_type": "tool_result",
        "name": "calendar.lookup",
        "content": "Dinner with Mert is tomorrow at 7 PM",
    }
]
```

### Items

Items are ingestion objects that can carry content, modality, metadata, or asset references.

Example:

```python
items = [
    {
        "modality": "document",
        "metadata": {"text": "My favorite city is Tokyo"},
    }
]
```

Use items when the source is not naturally a chat turn.

## `capture(...)` vs `add(...)`

Use `capture(...)` when:

- you are ingesting assistant turns
- you have messages and events
- you want the v3 workflow
- you expect to use `context.build(...)` or v3 chat grounding later

Use `add(...)` when:

- you want lower-level direct memory ingestion
- you are doing simpler memory writes
- you do not need the assistant-turn structure

If your application is a chat assistant, start with `capture(...)`.

## What `capture(...)` Produces

`capture(...)` writes evidence first, then derives memory candidates from that evidence.

Typical output fields include:

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

This gives you both a write result and visibility into what kind of memory was created.

## Building Grounding with `context.build(...)`

`context.build(...)` turns memory into a sectioned structure instead of a flat ranked list.

Example:

```python
pack = client.context.build(
    query="What do you remember about me?",
    user_id="u1",
    agent_id="jarvis",
    include_trace=True,
)
```

Why use it:

- it separates profile, preferences, relationships, episodes, and tool observations
- it includes citations
- it is better suited to prompt construction
- it can expose grounding and ranking trace

## `chat.completions.create(..., memory_strategy="v3")`

This is the highest-level assistant-facing API in the package.

Example:

```python
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "What do you remember about me?"}],
    user_id="u1",
    agent_id="jarvis",
    memory_strategy="v3",
    include_memory_pack=True,
    include_trace=True,
)
```

Useful fields:

- `memory_strategy="v3"`: use v3 grounding
- `include_memory_pack=True`: return the structured memory pack
- `include_trace=True`: return trace output
- `context_policy="compact"` or `"verbose"`: control prompt-grounding format
- `tool_events=[...]`: include current tool state in the grounding
- `store=True`: write the interaction back into memory

## Using `include_trace=True`

Turn trace on when you need to debug behavior, not for every production response by default.

Trace can help answer:

- why a memory ranked highly
- why a memory was omitted from grounding
- which pipeline configuration was active
- what activation and maintenance state affected retrieval in `attention_fast`

## Working Memory in `attention_fast`

When `BrainConfig(mode="attention_fast")` is enabled, `ContextPack` can include `working_memory`.

This is the small set of highly activated memories selected before the normal section-filling logic.

Example:

```python
from memoripy import BrainConfig, MemoryClient, MemoryPipelineConfig

client = MemoryClient(
    pipeline=MemoryPipelineConfig(
        brain=BrainConfig(mode="attention_fast"),
        default_include_trace=True,
    )
)

pack = client.context.build(query="What matters right now?", user_id="u1")
print(pack.working_memory)
```

Use this when you want the system to behave more like a fast attention layer rather than only a broad search layer.

## Asset Processing

Memoripy can process asset-like inputs through a configured asset processor.

The built-in `LocalAssetProcessor` can derive text from supported local text/document inputs and metadata fields.

Example:

```python
from memoripy import LocalAssetProcessor, MemoryClient, MemoryPipelineConfig

client = MemoryClient(
    pipeline=MemoryPipelineConfig(asset_processor=LocalAssetProcessor())
)

client.add(
    items=[
        {
            "modality": "document",
            "metadata": {"text": "My favorite city is Tokyo"},
        }
    ],
    user_id="u1",
)
```

## Next Steps

- [Low-level and compatibility guide](./low-level-and-compatibility.md)
- [Brain mode and maintenance](./brain-mode-and-maintenance.md)
