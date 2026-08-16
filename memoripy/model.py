from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingModel(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for text."""

    def initialize_embedding_dimension(self) -> int | None:
        return None


class ChatModel(ABC):
    @abstractmethod
    def invoke(self, messages: list[dict[str, Any]]) -> str:
        """Generate a response for chat messages."""

    def extract_concepts(self, text: str) -> list[str]:
        return []
