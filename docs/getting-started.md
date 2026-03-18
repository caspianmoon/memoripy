# Getting Started

This guide takes you from install to a working local example.

## 1. Install Memoripy

Base install:

```bash
pip install memoripy
```

Optional extras:

```bash
pip install "memoripy[service]"
pip install "memoripy[postgres]"
pip install "memoripy[dynamo]"
```

Use the extras only when you need those integrations.

## 2. Create a Client

The simplest client stores everything in memory:

```python
from memoripy import MemoryClient

client = MemoryClient()
```

For local persistence, use a file-backed repository through `from_path(...)`:

```python
from memoripy import MemoryClient

client = MemoryClient.from_path("./.memoripy")
```

This is the best starting point for local development.

## 3. Capture an Interaction

Use `capture(...)` when you are storing an assistant turn or a tool-driven interaction.

```python
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
```

Important parameters:

- `messages`: chat messages to ingest
- `events`: tool calls, tool results, or assistant actions
- `user_id`, `agent_id`, `run_id`: scope controls
- `idempotency_key`: prevents duplicate writes when the same operation is retried

## 4. Search Memory

Use `search(...)` when you want a ranked flat result set.

```python
result = client.search(
    query="What is my name?",
    user_id="khazar",
    agent_id="jarvis",
    include_trace=True,
)

print(result["results"][0]["memory"]["summary"])
print(result["trace"]["ranking"]["results"][0]["rank_breakdown"])
```

The result payload includes:

- `query`
- `filters`
- `results`
- `projection_status`
- optional `trace`

Each result entry includes:

- `memory`
- `score`
- `rank_breakdown`
- `evidence`

## 5. Build a Context Pack

Use `context.build(...)` when you want a structured memory bundle for grounding.

```python
pack = client.context.build(
    query="What do you remember about me and what is on my calendar?",
    user_id="khazar",
    agent_id="jarvis",
    run_id="session-1",
    include_trace=True,
)

print(pack.profile)
print(pack.tool_observations)
print(pack.citations)
```

The `ContextPack` is usually a better input for prompts than raw search results.

It can include:

- `working_memory`
- `profile`
- `preferences`
- `relationships`
- `recent_episodes`
- `tool_observations`
- `citations`
- `debug`
- `trace`

## 6. Ground Chat Completions

Use the built-in chat surface when you want Memoripy to perform memory retrieval and prompt assembly before calling the chat model.

```python
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "What do you remember about me?"}],
    user_id="khazar",
    agent_id="jarvis",
    run_id="session-1",
    memory_strategy="v3",
    include_memory_pack=True,
    include_trace=True,
)

print(response["choices"][0]["message"]["content"])
print(response["memory_pack"]["profile"])
```

Useful options:

- `memory_strategy="v3"`: use the assistant-first memory workflow
- `include_memory_pack=True`: include the structured pack in the response
- `include_trace=True`: expose retrieval and grounding trace
- `context_policy`: choose how compact or verbose the memory grounding should be

## 7. Turn on Brain Mode

If you want working memory, activation, dormancy, and explicit consolidation, configure `attention_fast`.

```python
from memoripy import BrainConfig, MemoryClient, MemoryPipelineConfig

client = MemoryClient(
    pipeline=MemoryPipelineConfig(
        brain=BrainConfig(mode="attention_fast"),
        default_include_trace=True,
    )
)
```

Now `context.build(...)` can include `working_memory`, and `search(...)` trace can include activation details.

## 8. Persist Beyond a Local Folder

For Postgres-backed persistence:

```python
from memoripy import MemoryClient, PostgresRepository

client = MemoryClient(
    repository=PostgresRepository("postgresql://postgres:postgres@localhost:5432/memoripy")
)
```

Install the `postgres` extras first.

## 9. Next Steps

After you have a local example working, continue with:

- [Assistant-first memory guide](./assistant-memory.md)
- [Brain mode and maintenance](./brain-mode-and-maintenance.md)
- [Service and storage](./service-and-storage.md)
