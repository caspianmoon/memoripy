from __future__ import annotations

from .repository import BaseRepository, InMemoryRepository
from .storage import BaseStorage


class InMemoryStorage(BaseStorage):
    def __init__(self):
        self._repository = InMemoryRepository()

    def build_repository(self) -> BaseRepository:
        return self._repository
