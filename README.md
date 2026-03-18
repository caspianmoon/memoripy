# Memoripy

Memoripy is an assistant-first memory framework for LLM applications.

The v3 layer adds Jarvis-style contextual recall on top of the existing versioned storage core:

- Immutable evidence for messages, tool calls, tool results, and assistant actions
- Durable semantic memory for facts, preferences, profile attributes, and relations
- Episodic memory for recent, high-salience interactions and tool observations
- Hierarchical scope recall across run, assistant, user, and broader assistant context
- Context packs for live assistant turns with citations and ranking breakdowns
- Backward-compatible v2 APIs for low-level add/search/history workflows

## What Ships In This Repo

- `MemoryClient` and `AsyncMemoryClient`
- `capture(...)` for assistant-first ingestion
- `context.build(...)` for sectioned live recall
- `add/search/get/history/export/import` for lower-level control
- `chat.completions.create(..., memory_strategy="v3")`
- `MemoryService`, `serve_http()`, and `create_fastapi_app()`
- In-memory and file-backed repositories
- Export/import plus schema migration from older snapshots

## Install

Base runtime:

```bash
pip install memoripy
```

Optional extras:

```bash
pip install "memoripy[service]"
pip install "memoripy[dynamo]"
pip install "memoripy[postgres]"
```

## Quickstart

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

memory_pack = client.context.build(
    query="What do you remember about me and what is on my calendar?",
    user_id="khazar",
    agent_id="jarvis",
    run_id="session-1",
)

print(memory_pack.profile)
print(memory_pack.tool_observations)
```

## API Surface

### Assistant-First SDK

```python
from memoripy import MemoryClient

client = MemoryClient()

client.capture(
    messages=[{"role": "user", "content": "My favorite city is Tokyo"}],
    user_id="u1",
    agent_id="jarvis",
)

pack = client.context.build(query="What city do I like?", user_id="u1", agent_id="jarvis")
print(pack.preferences[0]["summary"])
```

### V3 Chat Grounding

```python
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "What do you remember about me?"}],
    user_id="u1",
    agent_id="jarvis",
    memory_strategy="v3",
    include_memory_pack=True,
)

print(response["choices"][0]["message"]["content"])
print(response["memory_pack"]["profile"])
```

### Service Layer

```python
from memoripy import MemoryClient, serve_http

server = serve_http(port=8000, client=MemoryClient.from_path("./.memoripy"))
server.serve_forever()
```

Available endpoints:

- `POST /v1/memories`
- `GET /v1/memories`
- `GET /v1/memories/{id}`
- `PATCH /v1/memories/{id}`
- `DELETE /v1/memories/{id}`
- `GET /v1/memories/{id}/history`
- `POST /v1/search`
- `POST /v1/export`
- `POST /v1/import`
- `POST /v1/chat/completions`
- `POST /v3/capture`
- `POST /v3/context`

## Reliability Model

- Mutating operations support idempotency keys.
- Writes go through immutable evidence plus versioned memory records.
- Semantic memory and episodic memory share the same durable version history.
- Search and context assembly expose provenance and ranking breakdowns.
- Older exported snapshots load into schema version 3 automatically.

## Compatibility

The v2 APIs remain available for low-level workflows and existing integrations:

```python
client.add(text="I live in Istanbul", user_id="u1")
client.search(query="where do i live", user_id="u1")
client.history(memory_id="memory_...")
client.export()
```

The legacy `MemoryManager` shim also still works:

```python
from memoripy import JSONStorage, MemoryManager

manager = MemoryManager(storage=JSONStorage("memory.json"))
manager.add_interaction("My name is Khazar", "Nice to meet you")
print(manager.retrieve_relevant_interactions("name"))
```

## Tests

Run the built-in suite from the repo root:

```bash
python3 -m unittest discover -s tests -v
```
