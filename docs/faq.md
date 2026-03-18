# Troubleshooting and FAQ

## Why didn't a message become memory?

Common reasons:

- the message produced evidence but no durable semantic candidate
- the content was too weak or too low-signal for promotion
- the resulting candidate stayed episodic or pending rather than semantic
- the extractor did not recognize the pattern

What to do:

- inspect the write result from `capture(...)` or `add(...)`
- run `search(..., include_trace=True)`
- inspect `context.build(..., include_trace=True)`
- consider whether a custom extractor or pipeline config is needed

## When should I use `capture(...)` vs `add(...)`?

Use `capture(...)` for assistant-turn workflows with messages and tool events.

Use `add(...)` for lower-level direct ingestion when you want simpler memory writes.

If you are building a conversational assistant, default to `capture(...)`.

## Why is a memory in trace but not in grounding?

Because ranking and grounding are different steps.

A memory may rank but still be omitted due to:

- token budget
- duplicate suppression
- section limits
- intent prioritization
- dormancy behavior in `attention_fast`

Trace is the right place to inspect this:

- `trace["ranking"]`
- `trace["grounding"]`

## Why is a memory dormant?

In `attention_fast`, a memory can move into `MemoryState.DORMANT` when its activation falls low enough.

Dormant means:

- still stored
- still recoverable
- less likely to appear by default

It can reactivate when the query is a strong enough direct cue.

## How do I persist to disk?

Use:

```python
client = MemoryClient.from_path("./.memoripy")
```

This uses the file-backed repository.

## How do I persist to Postgres?

Install the extras first:

```bash
pip install "memoripy[postgres]"
```

Then configure:

```python
from memoripy import MemoryClient, PostgresRepository

client = MemoryClient(
    repository=PostgresRepository("postgresql://postgres:postgres@localhost:5432/memoripy")
)
```

## How do I expose Memoripy over HTTP?

Use the built-in HTTP server:

```python
from memoripy import MemoryClient, serve_http

server = serve_http(port=8000, client=MemoryClient.from_path("./.memoripy"))
server.serve_forever()
```

Or use FastAPI:

```python
from memoripy import MemoryClient, create_fastapi_app

app = create_fastapi_app(client=MemoryClient.from_path("./.memoripy"))
```

## Why should I enable `include_trace=True`?

Enable it when you need to debug:

- ranking
- grounding choices
- reconciliation reasoning
- activation and working-memory behavior
- consolidation metadata

Leave it off by default if you only need the main user-facing result.

## What is the easiest setup for local development?

Use:

```python
client = MemoryClient.from_path("./.memoripy")
```

That gives you persistence without any external service.

## Is Dynamo the recommended durable backend?

No. Dynamo support still exists, but the actively maintained durable backend in this repo is `PostgresRepository`.

## Where should I look for runnable examples?

Start with:

- [examples/v3_basic.py](../examples/v3_basic.py)
- [examples/v2_basic.py](../examples/v2_basic.py)
- [examples/postgres_example.py](../examples/postgres_example.py)
- [examples/README.md](../examples/README.md)
