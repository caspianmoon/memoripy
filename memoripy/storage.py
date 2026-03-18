from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .repository import BaseRepository


class BaseStorage(ABC):
    @abstractmethod
    def build_repository(self) -> BaseRepository:
        raise NotImplementedError

    def load_history(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        repository = self.build_repository()
        state = repository.load_state()
        short_term: list[dict[str, Any]] = []
        for evidence in sorted(state.evidence.values(), key=lambda item: item.created_at):
            short_term.append(
                {
                    "id": evidence.evidence_id,
                    "prompt": evidence.text if evidence.role == "user" else "",
                    "output": evidence.text if evidence.role == "assistant" else "",
                    "timestamp": evidence.created_at,
                    "concepts": [],
                }
            )
        long_term = [record.to_dict() for record in state.memories.values()]
        long_term.sort(key=lambda item: item["updated_at"], reverse=True)
        return short_term, long_term

    def save_memory_to_history(self, memory_store: Any) -> None:
        if hasattr(memory_store, "export_state"):
            payload = memory_store.export_state()
        elif hasattr(memory_store, "export"):
            payload = memory_store.export()
        elif hasattr(memory_store, "_engine"):
            payload = memory_store._engine.export()
        else:
            raise TypeError("Unsupported memory_store payload")
        repository = self.build_repository()
        from .repository import EngineState

        repository.replace_state(EngineState.from_dict(payload))
