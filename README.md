# Memoripy

Memoripy is a Python memory framework for LLM applications. It gives an assistant or agent a durable memory layer that can ingest conversations and tool activity, extract useful facts, preserve evidence, build grounded context for later turns, and expose the whole system through a local SDK or HTTP service.

At a high level, Memoripy combines three things:

- assistant-first memory capture for messages, tool calls, tool results, and ingestion items
- durable memory storage with evidence, versions, citations, and history
- retrieval and grounding workflows for search, context-building, and chat completions

It supports both a lower-level memory API and a higher-level v3 assistant workflow. The newer surface also includes an optional `attention_fast` mode that adds working-memory selection, activation tracking, dormancy, and explicit consolidation while keeping the hot path local and deterministic.

## What Memoripy Can Do

- Capture conversations and tool events with `capture(...)`
- Store evidence separately from extracted memories
- Maintain semantic memory for facts, preferences, profile attributes, and relations
- Maintain episodic memory for recent or salient interactions
- Track version history, contradictions, and supporting evidence
- Build sectioned context packs with citations using `context.build(...)`
- Ground `chat.completions.create(...)` responses with memory
- Expose ranked search, trace output, and low-level memory control
- Run in-memory, file-backed, or SQL-backed storage
- Serve the same behavior over HTTP with `MemoryService`, `serve_http()`, and `create_fastapi_app()`

## Install

Base package:

```bash
pip install memoripy
```

Optional extras:

```bash
pip install "memoripy[service]"
pip install "memoripy[postgres]"
pip install "memoripy[dynamo]"
pip install "memoripy[dev]"
```

Installed extras map directly to the package metadata in [setup.py](./setup.py):

- `service`: FastAPI and Uvicorn
- `postgres`: SQLAlchemy, Alembic, Psycopg, and pgvector
- `dynamo`: PynamoDB and python-dotenv
- `dev`: pytest and ruff

## 2-Minute Quickstart

```python
from memoripy import MemoryClient

client = MemoryClient.from_path("./.memoripy")

client.capture(
    messages=[
        {"role": "user", "content": "My name is Khazar"},
        {"role": "assistant", "content": "Nice to meet you, Khazar."},
    ],
    events=[
        {
            "event_type": "tool_result",
            "name": "calendar.lookup",
            "content": "Dinner with Mert is tomorrow at 7 PM",
        }
    ],
    user_id="khazar",
    agent_id="jarvis",
    run_id="session-1",
    idempotency_key="intro-1",
)

pack = client.context.build(
    query="What do you remember about me and what is on my calendar?",
    user_id="khazar",
    agent_id="jarvis",
    run_id="session-1",
)

print(pack.profile)
print(pack.tool_observations)
```

What happened here:

1. `capture(...)` stored raw evidence for the conversation and tool result.
2. Memoripy extracted durable memories from that evidence.
3. `context.build(...)` assembled a structured `ContextPack` you can use directly or pass into chat grounding.

## Choose Your Mode

### `v2`: Low-Level Memory Control

Use the lower-level API when you want direct memory CRUD and search primitives:

```python
client.add(text="I live in Istanbul", user_id="u1")
results = client.search(query="where do i live", user_id="u1")
history = client.history(memory_id=results["results"][0]["memory"]["record_id"])
```

Choose this when you want explicit control over stored memory rather than assistant-turn workflows.

### `v3`: Assistant-First Memory

Use `capture(...)`, `context.build(...)`, and `chat.completions.create(..., memory_strategy="v3")` when the unit of work is an assistant interaction:

```python
client.capture(
    messages=[{"role": "user", "content": "My favorite city is Tokyo"}],
    user_id="u1",
    agent_id="jarvis",
)

pack = client.context.build(query="What city do I like?", user_id="u1", agent_id="jarvis")
print(pack.preferences[0]["summary"])
```

### `attention_fast`: Brain-Like Fast Mode

Use `BrainConfig(mode="attention_fast")` when you want activation-aware ranking, working memory, dormancy, and explicit consolidation:

```python
from memoripy import BrainConfig, MemoryClient, MemoryPipelineConfig

client = MemoryClient(
    pipeline=MemoryPipelineConfig(
        brain=BrainConfig(mode="attention_fast"),
        default_include_trace=True,
    )
)

pack = client.context.build(query="What do you remember about me?", user_id="u1")
print(pack.working_memory)
```

This mode stays local-first. It does not require LLM calls in the hot path.

## Documentation Map

Start here depending on what you need:

- [Detailed docs home](./docs/index.md)
- [Concepts and mental model](./docs/concepts.md)
- [Getting started](./docs/getting-started.md)
- [Assistant-first memory guide](./docs/assistant-memory.md)
- [Low-level and compatibility guide](./docs/low-level-and-compatibility.md)
- [Brain mode and maintenance](./docs/brain-mode-and-maintenance.md)
- [Service and storage](./docs/service-and-storage.md)
- [API reference](./docs/api-reference.md)
- [Architecture appendix](./docs/architecture.md)
- [Troubleshooting and FAQ](./docs/faq.md)

Runnable examples:

- [Examples overview](./examples/README.md)
- [v3 basic example](./examples/v3_basic.py)
- [v2 basic example](./examples/v2_basic.py)
- [Postgres example](./examples/postgres_example.py)
- [Benchmark example](./examples/benchmark_eval.py)

## Core Public Surface

Primary SDK:

- `MemoryClient`
- `AsyncMemoryClient`
- `capture(...)`
- `context.build(...)`
- `chat.completions.create(...)`
- `maintenance.consolidate(...)`
- `add/search/get/get_all/update/delete/delete_all/history/export/import`

Configuration and types:

- `MemoryPipelineConfig`
- `BrainConfig`
- `SearchFilters`
- `ContextPack`
- `MemoryState`

Repositories and service:

- `InMemoryRepository`
- `FileMemoryRepository`
- `PostgresRepository`
- `MemoryService`
- `serve_http()`
- `create_fastapi_app()`

## Benchmarks

Memoripy ships a reproducible benchmark harness:

```bash
python3 -m benchmarks.runner --target memoripy
python3 -m benchmarks.runner --target memoripy --json
python3 -m benchmarks.runner --latency --json
```

See [benchmarks/README.md](./benchmarks/README.md) for details.

## Tests

Run the built-in suite from the repo root:

```bash
python3 -m unittest discover -s tests -v
```

## Current Limitations

- The built-in Mem0 benchmark target is a placeholder unless you run the shared scenarios in an external Mem0 environment.
- Background consolidation is explicit and user-triggered through `maintenance.consolidate(...)`; there is no built-in scheduler.
- `PostgresRepository` requires the optional `postgres` extras.
- The Dynamo integration still exists, but the actively maintained durable backend is `PostgresRepository`.

## License

Memoripy is released under the [Apache License 2.0](./LICENSE).
