from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .client import MemoryClient
from .implemented_models import SimpleKeywordEmbeddingModel
from .in_memory_storage import InMemoryStorage
from .storage import BaseStorage
from .utils import hashed_embedding, unique_tokens


@dataclass
class ConceptExtractionResponse:
    concepts: list[str] = field(default_factory=list)


class MemoryManager:
    """Backward-compatible wrapper over the lower-level memory API."""

    def __init__(self, chat_model: Any | None = None, embedding_model: Any | None = None, storage: BaseStorage | None = None):
        self.chat_model = chat_model
        self.embedding_model = embedding_model or SimpleKeywordEmbeddingModel()
        self.storage = storage or InMemoryStorage()
        self.client = MemoryClient(
            repository=self.storage.build_repository(),
            chat_model=chat_model,
            embedding_model=self.embedding_model,
        )
        self.dimension = self.embedding_model.initialize_embedding_dimension() or 128

    def standardize_embedding(self, embedding: list[float]) -> list[float]:
        if len(embedding) == self.dimension:
            return embedding
        if len(embedding) < self.dimension:
            return embedding + ([0.0] * (self.dimension - len(embedding)))
        return embedding[: self.dimension]

    def load_history(self):
        return self.storage.load_history()

    def save_memory_to_history(self):
        self.storage.save_memory_to_history(self.client)

    def add_interaction(self, prompt: str, output: str, embedding: list[float] | None = None, concepts: list[str] | None = None):
        del embedding
        result = self.client.add(
            messages=[
                {"role": "user", "content": prompt, "metadata": {"concepts": concepts or []}},
                {"role": "assistant", "content": output, "metadata": {"concepts": concepts or []}},
            ]
        )
        self.save_memory_to_history()
        return result

    def get_embedding(self, text: str) -> list[float]:
        if self.embedding_model is None:
            return hashed_embedding(text, dimensions=self.dimension)
        return self.standardize_embedding(self.embedding_model.get_embedding(text))

    def extract_concepts(self, text: str) -> list[str]:
        if self.chat_model is not None and hasattr(self.chat_model, "extract_concepts"):
            return list(self.chat_model.extract_concepts(text))
        return unique_tokens(text)[:10]

    def initialize_memory(self):
        return self.load_history()

    def retrieve_relevant_interactions(self, query: str, similarity_threshold: float = 40, exclude_last_n: int = 0) -> list[dict[str, Any]]:
        limit = max(5, exclude_last_n + 5)
        results = self.client.search(query=query, limit=limit)
        interactions: list[dict[str, Any]] = []
        for entry in results["results"][exclude_last_n:]:
            interactions.append(
                {
                    "id": entry["memory"]["record_id"],
                    "prompt": entry["memory"]["summary"],
                    "output": entry["memory"]["value"],
                    "total_score": entry["score"] * 100,
                    "timestamp": entry["memory"]["updated_at"],
                    "access_count": entry["memory"]["access_count"],
                    "concepts": entry["memory"]["entity_names"],
                    "decay_factor": max(0.1, min(entry["score"], 1.0)),
                }
            )
        return [item for item in interactions if item["total_score"] >= similarity_threshold]

    def generate_response(self, prompt: str, last_interactions: list, retrievals: list, context_window: int = 3) -> str:
        messages: list[dict[str, Any]] = []
        for interaction in last_interactions[-context_window:]:
            if interaction.get("prompt"):
                messages.append({"role": "user", "content": interaction["prompt"]})
            if interaction.get("output"):
                messages.append({"role": "assistant", "content": interaction["output"]})
        for retrieval in retrievals[:context_window]:
            messages.append(
                {
                    "role": "system",
                    "content": f"Relevant memory: {retrieval.get('prompt', '')} -> {retrieval.get('output', '')}",
                }
            )
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(messages=messages)
        return response["choices"][0]["message"]["content"]
