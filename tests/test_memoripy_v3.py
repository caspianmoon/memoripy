from __future__ import annotations

import unittest

from memoripy import MemoryClient, MemoryService


class RecordingChatModel:
    def __init__(self):
        self.calls: list[list[dict[str, str]]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return "ok"


class MemoripyV3Tests(unittest.TestCase):
    def test_capture_builds_context_pack_with_sections_and_citations(self):
        client = MemoryClient()
        result = client.capture(
            messages=[
                {"role": "user", "content": "My name is Khazar"},
                {"role": "assistant", "content": "I will remember that."},
            ],
            user_id="user-1",
            agent_id="jarvis",
            run_id="run-1",
            idempotency_key="capture-name",
        )

        self.assertTrue(result["semantic_memory_ids"])

        memory_pack = client.context.build(
            query="What is my name?",
            user_id="user-1",
            agent_id="jarvis",
            run_id="run-1",
        )
        self.assertTrue(memory_pack.profile)
        self.assertEqual(memory_pack.profile[0]["value"], "Khazar")
        self.assertTrue(memory_pack.citations)

    def test_tool_results_land_in_tool_observations(self):
        client = MemoryClient()
        client.capture(
            messages=[{"role": "user", "content": "Please check tomorrow's weather"}],
            events=[
                {
                    "event_type": "tool_result",
                    "name": "weather.lookup",
                    "content": "Tomorrow in Istanbul it will be sunny and 21 C",
                }
            ],
            user_id="user-1",
            agent_id="jarvis",
            run_id="run-1",
        )

        memory_pack = client.context.build(
            query="What's the weather tomorrow?",
            user_id="user-1",
            agent_id="jarvis",
            run_id="run-1",
        )
        self.assertTrue(memory_pack.tool_observations)
        self.assertIn("Istanbul", memory_pack.tool_observations[0]["value"])

    def test_smalltalk_stays_as_evidence_without_durable_semantic_memory(self):
        client = MemoryClient()
        client.capture(
            messages=[{"role": "user", "content": "ok thanks"}],
            user_id="user-1",
            agent_id="jarvis",
        )

        snapshot = client.export()
        self.assertEqual(snapshot["schema_version"], 4)
        self.assertEqual(len(snapshot["evidence"]), 1)
        self.assertFalse(snapshot["memories"])

    def test_scope_hierarchy_prefers_run_specific_context(self):
        client = MemoryClient()
        client.add(text="I live in Paris", user_id="user-1")
        client.add(text="I live in Berlin", user_id="user-1", agent_id="jarvis", run_id="trip-1")

        scoped_pack = client.context.build(
            query="Where do I live?",
            user_id="user-1",
            agent_id="jarvis",
            run_id="trip-1",
        )
        self.assertEqual(scoped_pack.profile[0]["value"], "Berlin")

        broader_pack = client.context.build(query="Where do I live?", user_id="user-1")
        self.assertEqual(broader_pack.profile[0]["value"], "Paris")

    def test_context_pack_suppresses_superseded_values(self):
        client = MemoryClient()
        client.capture(messages=[{"role": "user", "content": "I live in Paris"}], user_id="user-1")
        client.capture(messages=[{"role": "user", "content": "I live in Berlin"}], user_id="user-1")

        memory_pack = client.context.build(query="Where do I live?", user_id="user-1")
        values = [item["value"] for item in memory_pack.profile]
        self.assertIn("Berlin", values[0])
        self.assertNotIn("Paris", values)

    def test_chat_completions_v3_returns_memory_pack(self):
        client = MemoryClient()
        client.capture(
            messages=[{"role": "user", "content": "My favorite city is Tokyo"}],
            user_id="user-1",
            agent_id="jarvis",
        )

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "What city do I like?"}],
            user_id="user-1",
            agent_id="jarvis",
            memory_strategy="v3",
            include_memory_pack=True,
        )

        self.assertEqual(response["object"], "chat.completion")
        self.assertIn("memory_pack", response)
        self.assertTrue(response["memory_pack"]["preferences"])

    def test_compound_statement_extracts_clean_semantic_values(self):
        client = MemoryClient()
        client.capture(
            messages=[
                {
                    "role": "user",
                    "content": "My name is Khazar and I live in Istanbul and my favorite city is Tokyo",
                }
            ],
            user_id="user-1",
            agent_id="jarvis",
        )

        memory_pack = client.context.build(
            query="What do you know about me?",
            user_id="user-1",
            agent_id="jarvis",
        )

        profile = {item["key"]: item["value"] for item in memory_pack.profile}
        preferences = {item["key"]: item["value"] for item in memory_pack.preferences}
        self.assertEqual(memory_pack.intent, "general")
        self.assertEqual(profile["name"], "Khazar")
        self.assertEqual(profile["location"], "Istanbul")
        self.assertEqual(preferences["favorite_city"], "Tokyo")

    def test_compact_context_dedupes_episode_and_exposes_debug_stats(self):
        client = MemoryClient()
        client.capture(
            messages=[
                {
                    "role": "user",
                    "content": "My name is Khazar and I live in Istanbul and my favorite city is Tokyo",
                }
            ],
            user_id="user-1",
            agent_id="jarvis",
        )

        memory_pack = client.context.build(
            query="What do you know about me?",
            user_id="user-1",
            agent_id="jarvis",
            include_debug=True,
        )

        self.assertFalse(memory_pack.recent_episodes)
        for key in (
            "prompt_tokens_estimate",
            "selected_count",
            "dropped_duplicate_count",
            "dropped_budget_count",
            "omitted_memory_ids",
            "grounding_preview",
        ):
            self.assertIn(key, memory_pack.debug)
        self.assertTrue(memory_pack.debug["omitted_memory_ids"])

    def test_explicit_history_query_keeps_recent_episode(self):
        client = MemoryClient()
        client.capture(
            messages=[
                {
                    "role": "user",
                    "content": "My name is Khazar and I live in Istanbul and my favorite city is Tokyo",
                }
            ],
            user_id="user-1",
            agent_id="jarvis",
        )

        memory_pack = client.context.build(
            query="What happened earlier?",
            user_id="user-1",
            agent_id="jarvis",
        )

        self.assertEqual(memory_pack.intent, "episodic")
        self.assertTrue(memory_pack.recent_episodes)

    def test_compact_chat_prompt_omits_ids_and_citations_and_is_shorter_than_verbose(self):
        recorder = RecordingChatModel()
        client = MemoryClient(chat_model=recorder)
        client.capture(
            messages=[
                {
                    "role": "user",
                    "content": "My name is Khazar and I live in Istanbul and my favorite city is Tokyo",
                }
            ],
            user_id="user-1",
            agent_id="jarvis",
        )

        client.chat.completions.create(
            messages=[{"role": "user", "content": "What do you know about me?"}],
            user_id="user-1",
            agent_id="jarvis",
            memory_strategy="v3",
            context_policy="compact",
        )
        compact_prompt = recorder.calls[-1][0]["content"]

        client.chat.completions.create(
            messages=[{"role": "user", "content": "What do you know about me?"}],
            user_id="user-1",
            agent_id="jarvis",
            memory_strategy="v3",
            context_policy="verbose",
        )
        verbose_prompt = recorder.calls[-1][0]["content"]

        self.assertNotIn("memory_id=", compact_prompt)
        self.assertNotIn("Citations:", compact_prompt)
        self.assertIn("memory_id=", verbose_prompt)
        self.assertIn("Citations:", verbose_prompt)
        self.assertLess(len(compact_prompt.split()), len(verbose_prompt.split()))

    def test_chat_v3_ranks_once_and_returns_memory_and_memory_pack(self):
        client = MemoryClient()
        client.capture(messages=[{"role": "user", "content": "My favorite city is Tokyo"}], user_id="user-1", agent_id="jarvis")

        original_rank = client._engine._rank
        rank_calls = {"count": 0}

        def counting_rank(*args, **kwargs):
            rank_calls["count"] += 1
            return original_rank(*args, **kwargs)

        client._engine._rank = counting_rank
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "What do you know about me?"}],
                user_id="user-1",
                agent_id="jarvis",
                memory_strategy="v3",
                include_memory_pack=True,
            )
        finally:
            client._engine._rank = original_rank

        self.assertEqual(rank_calls["count"], 1)
        self.assertTrue(response["memory"]["results"])
        self.assertIn("memory_pack", response)

    def test_service_v3_routes(self):
        service = MemoryService()
        status, created = service.handle_request(
            method="POST",
            path="/v3/capture",
            payload={
                "messages": [{"role": "user", "content": "My name is Route Runner"}],
                "user_id": "user-1",
                "agent_id": "jarvis",
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(created["memory_ids"])

        status, context = service.handle_request(
            method="POST",
            path="/v3/context",
            payload={
                "query": "What is my name?",
                "user_id": "user-1",
                "agent_id": "jarvis",
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(context["profile"])

    def test_import_of_v2_snapshot_migrates_to_schema_four(self):
        source = MemoryClient()
        source.add(text="I work at OpenAI", user_id="user-1")
        snapshot = source.export()
        snapshot["schema_version"] = 2
        for payload in snapshot["evidence"].values():
            payload.pop("event_type", None)
            payload.pop("name", None)
            payload.pop("attributes", None)
            payload.pop("occurred_at", None)
            payload.pop("source_type", None)
        for payload in snapshot["versions"].values():
            payload.pop("salience", None)
            payload.pop("source_type", None)
            payload.pop("layer", None)
            payload.pop("citation_evidence_ids", None)
            payload.pop("contradicted_by", None)
        for payload in snapshot["memories"].values():
            payload.pop("salience", None)
            payload.pop("source_type", None)
            payload.pop("layer", None)
            payload.pop("confirmation_count", None)
            payload.pop("last_confirmed_at", None)
            payload.pop("contradicted_by", None)
            payload.pop("citation_evidence_ids", None)

        destination = MemoryClient()
        imported = destination.import_(snapshot, mode="replace")
        self.assertEqual(imported["schema_version"], 4)

        results = destination.search(query="where do i work", user_id="user-1")
        self.assertIn("OpenAI", results["results"][0]["memory"]["value"])


if __name__ == "__main__":
    unittest.main()
