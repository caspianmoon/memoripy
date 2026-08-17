from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .engine import MemoryEngine
from .pipeline import MemoryPipelineConfig
from .repository import BaseRepository, FileMemoryRepository, InMemoryRepository


class _ChatCompletions:
    def __init__(self, client: "MemoryClient"):
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._client._engine.chat_completion(**kwargs)


class _ChatNamespace:
    def __init__(self, client: "MemoryClient"):
        self.completions = _ChatCompletions(client)


class _MaintenanceNamespace:
    def __init__(self, client: "MemoryClient"):
        self._client = client

    def consolidate(self, **kwargs: Any) -> dict[str, Any]:
        return self._client._engine.consolidate(**kwargs)


class _ContextNamespace:
    def __init__(self, client: "MemoryClient"):
        self._client = client

    def build(self, **kwargs: Any):
        return self._client._engine.build_context(**kwargs)


class MemoryClient:
    def __init__(
        self,
        repository: BaseRepository | None = None,
        *,
        chat_model: Any | None = None,
        embedding_model: Any | None = None,
        extractor: Any | None = None,
        pipeline: MemoryPipelineConfig | None = None,
    ):
        self._engine = MemoryEngine(
            repository=repository or InMemoryRepository(),
            chat_model=chat_model,
            embedding_model=embedding_model,
            extractor=extractor,
            pipeline=pipeline,
        )
        self.chat = _ChatNamespace(self)
        self.context = _ContextNamespace(self)
        self.maintenance = _MaintenanceNamespace(self)

    @classmethod
    def from_path(
        cls,
        root_path: str | Path,
        *,
        chat_model: Any | None = None,
        embedding_model: Any | None = None,
        extractor: Any | None = None,
        pipeline: MemoryPipelineConfig | None = None,
    ) -> "MemoryClient":
        return cls(
            repository=FileMemoryRepository(root_path),
            chat_model=chat_model,
            embedding_model=embedding_model,
            extractor=extractor,
            pipeline=pipeline,
        )

    @property
    def repository(self) -> BaseRepository:
        return self._engine.repository

    def add(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.add(**kwargs)

    def capture(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.capture(**kwargs)

    def write(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.write(**kwargs)

    def search(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.search(**kwargs)

    def recall(self, **kwargs: Any):
        return self._engine.build_context(**kwargs)

    def get(self, *, memory_id: str) -> dict[str, Any]:
        return self._engine.get(memory_id)

    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.get_all(**kwargs)

    def update(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.update(**kwargs)

    def correct(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.correct(**kwargs)

    def delete(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.delete(**kwargs)

    def forget(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.forget(**kwargs)

    def delete_all(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.delete_all(**kwargs)

    def history(self, *, memory_id: str) -> dict[str, Any]:
        return self._engine.history(memory_id)

    def explain(self, *, memory_id: str) -> dict[str, Any]:
        return self._engine.explain(memory_id=memory_id)

    def audit(self):
        return self._engine.audit()

    def feedback(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.feedback(**kwargs)

    def export(self) -> dict[str, Any]:
        return self._engine.export()

    def import_(self, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._engine.import_snapshot(payload, **kwargs)

    def recover(self) -> dict[str, Any]:
        return self.repository.recover_from_backup()


setattr(MemoryClient, "import", MemoryClient.import_)


class Memory:
    """Small v4 facade for the common capture, recall, and correction workflow."""

    def __init__(self, path: str | Path | None = None, **kwargs: Any):
        self.client = MemoryClient.from_path(path, **kwargs) if path is not None else MemoryClient(**kwargs)

    def capture(self, text: str | None = None, **kwargs: Any) -> dict[str, Any]:
        if text is not None and not any(key in kwargs for key in ("messages", "items", "events")):
            kwargs["messages"] = [{"role": "user", "content": text}]
        return self.client.capture(**kwargs)

    def recall(self, query: str, **kwargs: Any):
        return self.client.recall(query=query, **kwargs)

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return self.client.search(query=query, **kwargs)

    def write(self, *, key: str, value: str, **kwargs: Any) -> dict[str, Any]:
        return self.client.write(key=key, value=value, **kwargs)

    def explain(self, memory_id: str) -> dict[str, Any]:
        return self.client.explain(memory_id=memory_id)

    def correct(self, memory_id: str, value: str, **kwargs: Any) -> dict[str, Any]:
        return self.client.correct(memory_id=memory_id, value=value, **kwargs)

    def forget(self, memory_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.client.forget(memory_id=memory_id, **kwargs)

    def audit(self):
        return self.client.audit()


class _AsyncChatCompletions:
    def __init__(self, client: "AsyncMemoryClient"):
        self._client = client

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._client._sync.chat.completions.create, **kwargs)


class _AsyncChatNamespace:
    def __init__(self, client: "AsyncMemoryClient"):
        self.completions = _AsyncChatCompletions(client)


class _AsyncMaintenanceNamespace:
    def __init__(self, client: "AsyncMemoryClient"):
        self._client = client

    async def consolidate(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._client._sync.maintenance.consolidate, **kwargs)


class _AsyncContextNamespace:
    def __init__(self, client: "AsyncMemoryClient"):
        self._client = client

    async def build(self, **kwargs: Any):
        return await asyncio.to_thread(self._client._sync.context.build, **kwargs)


class AsyncMemoryClient:
    def __init__(self, sync_client: MemoryClient | None = None, **kwargs: Any):
        self._sync = sync_client or MemoryClient(**kwargs)
        self.chat = _AsyncChatNamespace(self)
        self.context = _AsyncContextNamespace(self)
        self.maintenance = _AsyncMaintenanceNamespace(self)

    @classmethod
    def from_path(cls, root_path: str | Path, **kwargs: Any) -> "AsyncMemoryClient":
        return cls(sync_client=MemoryClient.from_path(root_path, **kwargs))

    @property
    def repository(self) -> BaseRepository:
        return self._sync.repository

    async def add(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.add, **kwargs)

    async def capture(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.capture, **kwargs)

    async def write(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.write, **kwargs)

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.search, **kwargs)

    async def recall(self, **kwargs: Any):
        return await asyncio.to_thread(self._sync.recall, **kwargs)

    async def get(self, *, memory_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get, memory_id=memory_id)

    async def get_all(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_all, **kwargs)

    async def update(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.update, **kwargs)

    async def correct(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.correct, **kwargs)

    async def delete(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.delete, **kwargs)

    async def forget(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.forget, **kwargs)

    async def delete_all(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.delete_all, **kwargs)

    async def history(self, *, memory_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.history, memory_id=memory_id)

    async def explain(self, *, memory_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.explain, memory_id=memory_id)

    async def audit(self):
        return await asyncio.to_thread(self._sync.audit)

    async def feedback(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.feedback, **kwargs)

    async def export(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.export)

    async def import_(self, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.import_, payload, **kwargs)


setattr(AsyncMemoryClient, "import", AsyncMemoryClient.import_)
