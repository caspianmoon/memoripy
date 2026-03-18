from __future__ import annotations

from .repository import BaseRepository, FileMemoryRepository
from .storage import BaseStorage


class JSONStorage(BaseStorage):
    def __init__(self, file_path: str = "interaction_history.json"):
        self.file_path = file_path
        self._repository = FileMemoryRepository(file_path)

    def build_repository(self) -> BaseRepository:
        return self._repository
