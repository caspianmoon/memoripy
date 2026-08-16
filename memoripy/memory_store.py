from __future__ import annotations

from typing import Any

from .client import MemoryClient
from .repository import BaseRepository, InMemoryRepository


class MemoryStore:
    """Thin v2-compatible facade retained for legacy imports."""

    def __init__(self, dimension: int = 128, repository: BaseRepository | None = None):
        self.dimension = dimension
        self.client = MemoryClient(repository=repository or InMemoryRepository())

    def add_interaction(self, interaction: dict[str, Any]):
        prompt = interaction.get("prompt", "")
        output = interaction.get("output", "")
        return self.client.add(
            messages=[
                {"role": "user", "content": prompt, "metadata": {"legacy_interaction_id": interaction.get("id")}},
                {"role": "assistant", "content": output, "metadata": {"legacy_interaction_id": interaction.get("id")}},
            ]
        )

    def retrieve(self, query_embedding, query_concepts, similarity_threshold: float = 0.0, exclude_last_n: int = 0):
        del query_embedding
        results = self.client.search(query=" ".join(query_concepts), limit=10)
        return [
            {
                "id": entry["memory"]["record_id"],
                "prompt": entry["memory"]["summary"],
                "output": entry["memory"]["value"],
                "total_score": entry["score"] * 100,
            }
            for entry in results["results"][exclude_last_n:]
            if entry["score"] * 100 >= similarity_threshold
        ]
