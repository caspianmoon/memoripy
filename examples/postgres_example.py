from __future__ import annotations

from memoripy import MemoryClient, PostgresRepository


def main() -> None:
    repository = PostgresRepository("postgresql://postgres:postgres@localhost:5432/memoripy")
    client = MemoryClient(repository=repository)

    client.capture(
        messages=[{"role": "user", "content": "My favorite city is Tokyo"}],
        user_id="postgres-user",
        agent_id="jarvis",
        idempotency_key="favorite-city",
    )

    result = client.search(
        query="favorite city",
        user_id="postgres-user",
        agent_id="jarvis",
        include_trace=True,
    )
    print(result["results"][0]["memory"]["summary"])
    print(result["trace"]["pipeline"])


if __name__ == "__main__":
    main()
