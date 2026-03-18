from __future__ import annotations

from .client import MemoryClient


def main():
    client = MemoryClient.from_path("./.memoripy-demo")
    client.capture(
        messages=[
            {"role": "user", "content": "My name is Khazar"},
            {"role": "assistant", "content": "Nice to meet you, Khazar."},
        ],
        events=[
            {
                "event_type": "tool_result",
                "name": "calendar.lookup",
                "content": "Lunch with Ada is on Friday at noon.",
            }
        ],
        user_id="demo-user",
        agent_id="jarvis",
        run_id="demo-run",
        idempotency_key="demo-intro",
    )

    memory_pack = client.context.build(
        query="what is my name and what is on my calendar",
        user_id="demo-user",
        agent_id="jarvis",
        run_id="demo-run",
    )
    print("Context pack:")
    print("- profile:", [item["summary"] for item in memory_pack.profile])
    print("- tool observations:", [item["summary"] for item in memory_pack.tool_observations])

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "What do you remember about me?"}],
        user_id="demo-user",
        agent_id="jarvis",
        run_id="demo-run",
        memory_strategy="v3",
        include_memory_pack=True,
    )
    print("\nChat completion:")
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
