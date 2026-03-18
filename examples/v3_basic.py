from memoripy import MemoryClient


def main():
    client = MemoryClient.from_path("./.memoripy-v3-example")

    client.capture(
        messages=[
            {"role": "user", "content": "My favorite city is Tokyo"},
            {"role": "assistant", "content": "Tokyo is a great city."},
        ],
        events=[
            {
                "event_type": "tool_result",
                "name": "calendar.lookup",
                "content": "Dinner with Ayse is tomorrow at 7 PM",
            }
        ],
        user_id="example-user",
        agent_id="jarvis",
        run_id="session-1",
        idempotency_key="favorite-city",
    )

    pack = client.context.build(
        query="What city do I like and what is on my calendar?",
        user_id="example-user",
        agent_id="jarvis",
        run_id="session-1",
    )
    print(pack.preferences[0]["summary"])
    print(pack.tool_observations[0]["summary"])

    reply = client.chat.completions.create(
        messages=[{"role": "user", "content": "What do you remember about me?"}],
        user_id="example-user",
        agent_id="jarvis",
        run_id="session-1",
        memory_strategy="v3",
        include_memory_pack=True,
    )
    print(reply["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
