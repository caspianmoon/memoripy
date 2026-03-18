# Service and Storage

Memoripy supports local SDK usage, pluggable storage backends, and an HTTP service layer.

## Repository Options

The main repository choices are:

- `InMemoryRepository`
- `FileMemoryRepository`
- `PostgresRepository`

### `InMemoryRepository`

This is the default when you construct `MemoryClient()` with no explicit repository.

Use it for:

- tests
- throwaway experiments
- short-lived local processes

### `FileMemoryRepository`

This is what `MemoryClient.from_path(...)` gives you.

```python
from memoripy import MemoryClient

client = MemoryClient.from_path("./.memoripy")
```

Use it for:

- local development
- small applications
- deterministic local persistence without setting up a database

### `PostgresRepository`

Use this when you want a SQL-backed durable repository.

```python
from memoripy import MemoryClient, PostgresRepository

client = MemoryClient(
    repository=PostgresRepository("postgresql://postgres:postgres@localhost:5432/memoripy")
)
```

Requirements:

- install `memoripy[postgres]`
- make sure your environment has a reachable Postgres instance

## Dynamo

The repo still contains Dynamo support and examples, but the actively maintained durable backend is `PostgresRepository`.

Use Dynamo only when you specifically need compatibility with an older or existing integration.

Related files:

- [examples/dynamo/README.md](../examples/dynamo/README.md)
- [examples/dynamo/dynamo_example.py](../examples/dynamo/dynamo_example.py)

## SDK and Service

Memoripy can be used in-process through the SDK or exposed as an HTTP service.

## `MemoryService`

`MemoryService` is the request-handling layer used by both the lightweight HTTP server and the FastAPI app.

Example:

```python
from memoripy import MemoryService

service = MemoryService()
status, payload = service.handle_request(
    method="POST",
    path="/v1/search",
    payload={"query": "name", "user_id": "u1"},
)
```

This is useful for testing and embedding the service behavior without a full server process.

## `serve_http()`

`serve_http()` starts a `ThreadingHTTPServer`.

```python
from memoripy import MemoryClient, serve_http

server = serve_http(port=8000, client=MemoryClient.from_path("./.memoripy"))
server.serve_forever()
```

Use it when you want a minimal built-in HTTP server with no extra framework setup.

## `create_fastapi_app()`

`create_fastapi_app()` returns a FastAPI application.

```python
from memoripy import MemoryClient, create_fastapi_app

app = create_fastapi_app(client=MemoryClient.from_path("./.memoripy"))
```

Requirements:

- install `memoripy[service]`

Use this when you want to run Memoripy inside an existing FastAPI stack or under Uvicorn.

## HTTP Routes

These are the routes currently implemented in `memoripy/service.py`.

### Memory Routes

- `POST /v1/memories`
- `GET /v1/memories`
- `GET /v1/memories/{memory_id}`
- `PATCH /v1/memories/{memory_id}`
- `DELETE /v1/memories/{memory_id}`
- `DELETE /v1/memories`
- `GET /v1/memories/{memory_id}/history`

### Retrieval and Snapshot Routes

- `POST /v1/search`
- `POST /v1/export`
- `POST /v1/import`

### Chat and v3 Routes

- `POST /v1/chat/completions`
- `POST /v3/capture`
- `POST /v3/context`
- `POST /v3/maintenance/consolidate`

## Recommended Storage Choices

Start with:

- `MemoryClient.from_path(...)` for local development

Move to:

- `PostgresRepository` for a production-grade durable backend in this repo

Stay in memory only when persistence is not required.

## Related Guides

- [Getting started](./getting-started.md)
- [API reference](./api-reference.md)
