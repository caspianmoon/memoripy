from memoripy import MemoryClient


def main():
    client = MemoryClient.from_path("./.memoripy-example")

    client.add(
        messages=[
            {"role": "user", "content": "My favorite city is Tokyo"},
            {"role": "assistant", "content": "Tokyo is a great city."},
        ],
        user_id="example-user",
        idempotency_key="favorite-city",
    )

    results = client.search(query="favorite city", user_id="example-user")
    print(results["results"][0]["memory"]["summary"])

    reply = client.chat.completions.create(
        messages=[{"role": "user", "content": "What city do I like?"}],
        user_id="example-user",
    )
    print(reply["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
