from __future__ import annotations

import asyncio
from typing import Any

from .engine import MemoryEngine
from .repository import BaseRepository, FileMemoryRepository, InMemoryRepository


class _ChatCompletions:
    def __init__(self, client: "MemoryClient"):
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._client._engine.chat_completion(**kwargs)


class _ChatNamespace:
    def __init__(self, client: "MemoryClient"):
        self.completions = _ChatCompletions(client)


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
    ):
        self._engine = MemoryEngine(
            repository=repository or InMemoryRepository(),
            chat_model=chat_model,
            embedding_model=embedding_model,
            extractor=extractor,
        )
        self.chat = _ChatNamespace(self)
        self.context = _ContextNamespace(self)

    @classmethod
    def from_path(
        cls,
        root_path: str,
        *,
        chat_model: Any | None = None,
        embedding_model: Any | None = None,
        extractor: Any | None = None,
    ) -> "MemoryClient":
        return cls(
            repository=FileMemoryRepository(root_path),
            chat_model=chat_model,
            embedding_model=embedding_model,
            extractor=extractor,
        )

    @property
    def repository(self) -> BaseRepository:
        return self._engine.repository

    def add(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.add(**kwargs)

    def capture(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.capture(**kwargs)

    def search(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.search(**kwargs)

    def get(self, *, memory_id: str) -> dict[str, Any]:
        return self._engine.get(memory_id)

    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.get_all(**kwargs)

    def update(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.update(**kwargs)

    def delete(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.delete(**kwargs)

    def delete_all(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.delete_all(**kwargs)

    def history(self, *, memory_id: str) -> dict[str, Any]:
        return self._engine.history(memory_id)

    def export(self) -> dict[str, Any]:
        return self._engine.export()

    def import_(self, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._engine.import_snapshot(payload, **kwargs)


setattr(MemoryClient, "import", MemoryClient.import_)


class _AsyncChatCompletions:
    def __init__(self, client: "AsyncMemoryClient"):
        self._client = client

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._client._sync.chat.completions.create, **kwargs)


class _AsyncChatNamespace:
    def __init__(self, client: "AsyncMemoryClient"):
        self.completions = _AsyncChatCompletions(client)


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

    @classmethod
    def from_path(cls, root_path: str, **kwargs: Any) -> "AsyncMemoryClient":
        return cls(sync_client=MemoryClient.from_path(root_path, **kwargs))

    @property
    def repository(self) -> BaseRepository:
        return self._sync.repository

    async def add(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.add, **kwargs)

    async def capture(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.capture, **kwargs)

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.search, **kwargs)

    async def get(self, *, memory_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get, memory_id=memory_id)

    async def get_all(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_all, **kwargs)

    async def update(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.update, **kwargs)

    async def delete(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.delete, **kwargs)

    async def delete_all(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.delete_all, **kwargs)

    async def history(self, *, memory_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.history, memory_id=memory_id)

    async def export(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.export)

    async def import_(self, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.import_, payload, **kwargs)


setattr(AsyncMemoryClient, "import", AsyncMemoryClient.import_)
