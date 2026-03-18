from __future__ import annotations

from pathlib import Path
from pprint import pprint
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memoripy import MemoryClient


STORE_DIR = Path(".memoripy-manual-demo")
USER_ID = "demo-user"
AGENT_ID = "jarvis"
RUN_ID = "session-1"


def section(title: str) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")


# %%
section("Create a file-backed client")
client = MemoryClient.from_path(STORE_DIR)
print(f"Store directory: {STORE_DIR.resolve()}")
print("Tip: delete that directory later if you want a clean rerun.")


# %%
section("Capture a short conversation and a tool result")
capture_result = client.capture(
    messages=[
        {"role": "user", "content": "My name is Khazar"},
        {"role": "user", "content": "I live in Istanbul"},
        {"role": "user", "content": "My favorite city is Tokyo"},
        {"role": "assistant", "content": "I will remember that."},
    ],
    events=[
        {
            "event_type": "tool_result",
            "name": "calendar.lookup",
            "content": "Dinner with Mert is tomorrow at 7 PM",
        }
    ],
    user_id=USER_ID,
    agent_id=AGENT_ID,
    run_id=RUN_ID,
    idempotency_key="manual-intro-1",
)

pprint(
    {
        "strategy": capture_result["strategy"],
        "semantic_memory_ids": capture_result["semantic_memory_ids"],
        "episodic_memory_ids": capture_result["episodic_memory_ids"],
        "projection_status": capture_result["projection_status"],
    }
)


# %%
section("Build a context pack")
pack = client.context.build(
    query="What do you remember about me and what is on my calendar?",
    user_id=USER_ID,
    agent_id=AGENT_ID,
    run_id=RUN_ID,
    include_debug=True,
)

pprint(
    {
        "intent": pack.intent,
        "profile": pack.profile,
        "preferences": pack.preferences,
        "tool_observations": pack.tool_observations,
        "recent_episodes": pack.recent_episodes,
        "citations": pack.citations,
        "debug": pack.debug,
    }
)


# %%
section("Inspect raw search results")
search_result = client.search(
    query="favorite city",
    user_id=USER_ID,
    agent_id=AGENT_ID,
    run_id=RUN_ID,
)

pprint(search_result)


# %%
section("Ground a chat completion with memory_strategy='v3'")
reply = client.chat.completions.create(
    messages=[{"role": "user", "content": "What do you remember about me?"}],
    user_id=USER_ID,
    agent_id=AGENT_ID,
    run_id=RUN_ID,
    memory_strategy="v3",
    include_memory_pack=True,
)

print(reply["choices"][0]["message"]["content"])
print("\nMemory hits returned with the reply:", len(reply["memory"]["results"]))
print("Memory pack sections:", list(reply["memory_pack"].keys()))
print(
    "\nNote: if you do not pass a real chat model, Memoripy still builds the grounding, "
    "but the assistant text is a built-in placeholder response."
)


# %%
section("Update a fact and see the newer value win")
client.capture(
    messages=[{"role": "user", "content": "I moved to Berlin last month"}],
    user_id=USER_ID,
    agent_id=AGENT_ID,
    run_id=RUN_ID,
    idempotency_key="manual-move-1",
)

updated_pack = client.context.build(
    query="Where do I live?",
    user_id=USER_ID,
    agent_id=AGENT_ID,
    run_id=RUN_ID,
)

pprint(updated_pack.profile)


# %%
section("Inspect everything currently stored")
all_memories = client.get_all(user_id=USER_ID, agent_id=AGENT_ID, run_id=RUN_ID)
snapshot = client.export()

print("Stored memory count:", len(all_memories["results"]))
print("Evidence count:", len(snapshot["evidence"]))
print("Version count:", len(snapshot["versions"]))
print("Top-level snapshot keys:", list(snapshot.keys()))

for index, item in enumerate(all_memories["results"], start=1):
    memory = item["memory"]
    print(
        f"{index}. kind={memory['kind']} key={memory['key']} "
        f"value={memory['value']} state={memory['state']}"
    )
