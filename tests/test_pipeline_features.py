from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoripy import (
    BrainConfig,
    KeywordBoostReranker,
    LocalAssetProcessor,
    MemoryClient,
    MemoryState,
    MemoryService,
    MemoryPipelineConfig,
    PostgresRepository,
    RerankOutcome,
)
from memoripy.extractors import DefaultMemoryExtractor, MemoryCandidate
from memoripy.types import MemoryKind, MemoryLayer


class NoopExtractor:
    def extract_semantic(self, evidence):
        return []

    def build_episode_candidate(self, evidence):
        return None

    def extract(self, evidence):
        return []


class FixedBoostReranker(KeywordBoostReranker):
    def rerank(self, *, query, candidates, state, search_filters, intent):
        del query, state, search_filters, intent
        outcomes: dict[str, RerankOutcome] = {}
        for _, record, _ in candidates:
            boost = 100.0 if record.key == "location" else 0.0
            outcomes[record.record_id] = RerankOutcome(score=boost, details={"forced": boost})
        return outcomes


class RepeatedFactExtractor:
    def extract_semantic(self, evidence):
        text = evidence.text.lower()
        if "atlas" not in text:
            return []
        return [
            MemoryCandidate(
                kind=MemoryKind.FACT.value,
                key="project",
                value="Atlas",
                summary="Project: Atlas",
                confidence=0.9,
                metadata={"topic": "project"},
                tags=["project"],
                salience=0.6,
                source_type=evidence.source_type,
            )
        ]

    def build_episode_candidate(self, evidence):
        return MemoryCandidate(
            kind=MemoryKind.EPISODIC_SUMMARY.value,
            key=f"episode_project_{abs(hash(evidence.text))}",
            value=evidence.text,
            summary=f"Episode: {evidence.text}",
            confidence=0.8,
            state=MemoryState.ACTIVE.value,
            metadata={"source": "episodic"},
            tags=["episodic", "project"],
            layer=MemoryLayer.EPISODIC.value,
            salience=0.7,
            source_type=evidence.source_type,
        )

    def extract(self, evidence):
        return self.extract_semantic(evidence)


class ConflictingProfileExtractor:
    def extract_semantic(self, evidence):
        text = evidence.text.lower()
        if "alice" in text:
            value = "Alice"
        elif "bob" in text:
            value = "Bob"
        else:
            return []
        return [
            MemoryCandidate(
                kind=MemoryKind.PROFILE_ATTRIBUTE.value,
                key="name",
                value=value,
                summary=f"Name: {value}",
                confidence=0.9,
                metadata={"topic": "name"},
                tags=["name"],
                salience=0.6,
                source_type=evidence.source_type,
            )
        ]

    def build_episode_candidate(self, evidence):
        return MemoryCandidate(
            kind=MemoryKind.EPISODIC_SUMMARY.value,
            key=f"episode_name_{abs(hash(evidence.text))}",
            value=evidence.text,
            summary=f"Episode: {evidence.text}",
            confidence=0.8,
            state=MemoryState.ACTIVE.value,
            metadata={"source": "episodic"},
            tags=["episodic", "name"],
            layer=MemoryLayer.EPISODIC.value,
            salience=0.7,
            source_type=evidence.source_type,
        )

    def extract(self, evidence):
        return self.extract_semantic(evidence)


class PipelineFeatureTests(unittest.TestCase):
    def test_search_include_trace_exposes_pipeline_and_reasoning(self):
        client = MemoryClient()
        client.capture(messages=[{"role": "user", "content": "My name is Tracey"}], user_id="user-1", agent_id="jarvis")

        result = client.search(query="name", user_id="user-1", agent_id="jarvis", include_trace=True)

        self.assertIn("trace", result)
        self.assertEqual(result["trace"]["query"], "name")
        self.assertIn("pipeline", result["trace"])
        self.assertTrue(result["trace"]["ranking"]["results"])
        self.assertIn("latest_action", result["trace"]["ranking"]["results"][0])
        self.assertIn("reasoning_trace", result["trace"]["ranking"]["results"][0])

    def test_context_include_trace_reports_grounding_selection(self):
        client = MemoryClient()
        client.capture(
            messages=[
                {"role": "user", "content": "My name is Khazar and I live in Istanbul and my favorite city is Tokyo"}
            ],
            user_id="user-1",
            agent_id="jarvis",
        )

        pack = client.context.build(
            query="What do you know about me?",
            user_id="user-1",
            agent_id="jarvis",
            include_trace=True,
        )

        self.assertIn("ranking", pack.trace)
        self.assertIn("grounding", pack.trace)
        self.assertTrue(pack.trace["grounding"]["selected_memory_ids"])
        self.assertIn("section_counts", pack.trace["grounding"])

    def test_pipeline_argument_wins_over_legacy_extractor_argument(self):
        client = MemoryClient(
            extractor=NoopExtractor(),
            pipeline=MemoryPipelineConfig(extractor=DefaultMemoryExtractor()),
        )

        client.capture(messages=[{"role": "user", "content": "My name is Priority Piper"}], user_id="user-1")
        result = client.search(query="name", user_id="user-1")

        self.assertEqual(result["results"][0]["memory"]["value"], "Priority Piper")

    def test_reranker_stage_can_reorder_results(self):
        client = MemoryClient(pipeline=MemoryPipelineConfig(reranker=FixedBoostReranker(), default_include_trace=True))
        client.capture(messages=[{"role": "user", "content": "My name is Alice"}], user_id="user-1")
        client.capture(messages=[{"role": "user", "content": "I live in Paris"}], user_id="user-1")

        result = client.search(query="tell me something", user_id="user-1", limit=2)

        self.assertEqual(result["results"][0]["memory"]["key"], "location")
        self.assertGreater(result["trace"]["ranking"]["results"][0]["rank_breakdown"]["reranker"], 0.0)

    def test_local_asset_processor_reads_document_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = Path(tmpdir) / "notes.txt"
            asset_path.write_text("My favorite city is Tokyo", encoding="utf-8")

            client = MemoryClient(pipeline=MemoryPipelineConfig(asset_processor=LocalAssetProcessor()))
            client.add(items=[{"modality": "document", "asset_ref": str(asset_path)}], user_id="user-1")
            result = client.search(query="favorite city", user_id="user-1")

            self.assertEqual(result["results"][0]["memory"]["value"], "Tokyo")

    def test_preference_contradiction_supersedes_existing_record(self):
        client = MemoryClient()
        client.capture(messages=[{"role": "user", "content": "I like pizza"}], user_id="user-1")
        client.capture(messages=[{"role": "user", "content": "I don't like pizza"}], user_id="user-1")

        result = client.search(query="pizza", user_id="user-1", include_trace=True)
        record = result["results"][0]["memory"]
        history = client.history(memory_id=record["record_id"])

        self.assertEqual(record["metadata"]["sentiment"], "negative")
        self.assertEqual(len(history["history"]), 2)
        self.assertIn(history["history"][-1]["action"], ("SUPERSEDE", "UPDATE"))
        self.assertTrue(result["trace"]["ranking"]["results"][0]["reasoning_trace"])

    def test_postgres_repository_requires_optional_dependency(self):
        with self.assertRaises(RuntimeError):
            PostgresRepository("postgresql://localhost/memoripy")

    def test_service_route_can_return_trace(self):
        service = MemoryService()
        service.handle_request(
            method="POST",
            path="/v1/memories",
            payload={"text": "My name is Service Trace", "user_id": "user-1"},
        )

        status, response = service.handle_request(
            method="POST",
            path="/v1/search",
            payload={"query": "name", "user_id": "user-1", "include_trace": True},
        )

        self.assertEqual(status, 200)
        self.assertIn("trace", response)
        self.assertTrue(response["trace"]["ranking"]["results"])

    def test_attention_fast_search_trace_exposes_activation_fields(self):
        client = MemoryClient(pipeline=MemoryPipelineConfig(brain=BrainConfig(mode="attention_fast")))
        capture = client.capture(messages=[{"role": "user", "content": "My name is Nora"}], user_id="user-1")

        snapshot = client.export()
        record_id = capture["semantic_memory_ids"][0]
        self.assertEqual(snapshot["projections"]["activation"][record_id]["retrieval_count"], 0)

        result = client.search(query="Nora", user_id="user-1", include_trace=True)

        self.assertIn("activation", result["trace"])
        self.assertGreater(result["trace"]["activation"]["results"][0]["activation_score"], 0.0)
        self.assertGreater(result["trace"]["ranking"]["results"][0]["rank_breakdown"]["exact_cue"], 0.85)
        updated = client.export()
        self.assertEqual(updated["projections"]["activation"][record_id]["retrieval_count"], 1)

    def test_attention_fast_context_returns_working_memory(self):
        client = MemoryClient(pipeline=MemoryPipelineConfig(brain=BrainConfig(mode="attention_fast")))
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

        pack = client.context.build(
            query="What do you know about me?",
            user_id="user-1",
            agent_id="jarvis",
            include_trace=True,
        )

        self.assertTrue(pack.working_memory)
        self.assertIn("working_memory", pack.trace)
        self.assertTrue(pack.trace["working_memory"]["selected_memory_ids"])

    def test_dormant_memory_reactivates_on_direct_cue(self):
        client = MemoryClient(
            pipeline=MemoryPipelineConfig(
                brain=BrainConfig(mode="attention_fast", consolidation_window_hours=1, dormancy_threshold=0.4)
            )
        )
        capture = client.capture(messages=[{"role": "user", "content": "My name is Nora"}], user_id="user-1")
        record_id = capture["semantic_memory_ids"][0]

        snapshot = client.export()
        snapshot["memories"][record_id]["updated_at"] = "2000-01-01T00:00:00+00:00"
        snapshot["memories"][record_id]["last_confirmed_at"] = "2000-01-01T00:00:00+00:00"
        snapshot["memories"][record_id]["salience"] = 0.0
        snapshot["projections"]["activation"][record_id]["last_activated_at"] = "2000-01-01T00:00:00+00:00"
        client.import_(snapshot, mode="replace")

        maintenance = client.maintenance.consolidate(user_id="user-1", limit=0, budget_ms=5)
        self.assertEqual(maintenance["dormancy_transitions"][0]["type"], "memory_dormant")

        result = client.search(query="Nora", user_id="user-1")
        self.assertEqual(result["results"][0]["memory"]["state"], MemoryState.ACTIVE.value)
        self.assertEqual(client.export()["memories"][record_id]["state"], MemoryState.ACTIVE.value)

    def test_consolidation_promotes_repeated_episode_into_semantic_memory(self):
        client = MemoryClient(
            pipeline=MemoryPipelineConfig(
                extractor=RepeatedFactExtractor(),
                semantic_promotion_threshold=0.95,
                brain=BrainConfig(mode="attention_fast"),
            )
        )
        client.capture(messages=[{"role": "user", "content": "I mentioned Atlas in passing"}], user_id="user-1")
        client.capture(messages=[{"role": "user", "content": "Atlas matters for the roadmap"}], user_id="user-1")

        pre = client.search(query="Atlas", user_id="user-1")
        self.assertTrue(all(item["memory"]["layer"] == "episodic" for item in pre["results"]))

        maintenance = client.maintenance.consolidate(user_id="user-1")
        self.assertTrue(maintenance["promotions"])

        post = client.search(query="Atlas", user_id="user-1")
        semantic_results = [item for item in post["results"] if item["memory"]["layer"] == "semantic"]
        self.assertTrue(semantic_results)
        self.assertEqual(semantic_results[0]["memory"]["key"], "project")

    def test_consolidation_skips_conflicting_profile_values(self):
        client = MemoryClient(
            pipeline=MemoryPipelineConfig(
                extractor=ConflictingProfileExtractor(),
                semantic_promotion_threshold=0.95,
                brain=BrainConfig(mode="attention_fast"),
            )
        )
        client.capture(messages=[{"role": "user", "content": "Call me Alice"}], user_id="user-1")
        client.capture(messages=[{"role": "user", "content": "Actually call me Bob"}], user_id="user-1")

        maintenance = client.maintenance.consolidate(user_id="user-1")

        self.assertTrue(maintenance["skipped"])
        self.assertEqual(maintenance["skipped"][0]["reason"], "profile_value_conflict")

    def test_classic_mode_keeps_working_memory_empty(self):
        client = MemoryClient()
        client.capture(messages=[{"role": "user", "content": "My name is Classic Casey"}], user_id="user-1")

        pack = client.context.build(query="What do you know about me?", user_id="user-1", include_trace=True)

        self.assertFalse(pack.working_memory)
        self.assertFalse(pack.trace["working_memory"]["items"])

    def test_chat_trace_surfaces_working_memory_and_activation(self):
        client = MemoryClient(pipeline=MemoryPipelineConfig(brain=BrainConfig(mode="attention_fast")))
        client.capture(messages=[{"role": "user", "content": "My favorite city is Tokyo"}], user_id="user-1")

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "What city do I like?"}],
            user_id="user-1",
            memory_strategy="v3",
            include_memory_pack=True,
            include_trace=True,
        )

        self.assertIn("activation", response["trace"])
        self.assertIn("working_memory", response["trace"])
        self.assertTrue(response["memory_pack"]["working_memory"])

    def test_service_consolidate_route_is_idempotent_and_scoped(self):
        client = MemoryClient(
            pipeline=MemoryPipelineConfig(
                extractor=RepeatedFactExtractor(),
                semantic_promotion_threshold=0.95,
                brain=BrainConfig(mode="attention_fast"),
            )
        )
        service = MemoryService(client=client)
        service.handle_request(
            method="POST",
            path="/v3/capture",
            payload={"messages": [{"role": "user", "content": "Atlas note one"}], "user_id": "user-1"},
        )
        service.handle_request(
            method="POST",
            path="/v3/capture",
            payload={"messages": [{"role": "user", "content": "Atlas note two"}], "user_id": "user-1"},
        )
        service.handle_request(
            method="POST",
            path="/v3/capture",
            payload={"messages": [{"role": "user", "content": "Atlas note other user"}], "user_id": "user-2"},
        )

        status, first = service.handle_request(
            method="POST",
            path="/v3/maintenance/consolidate",
            payload={"user_id": "user-1", "idempotency_key": "maint-1"},
        )
        _, second = service.handle_request(
            method="POST",
            path="/v3/maintenance/consolidate",
            payload={"user_id": "user-1", "idempotency_key": "maint-1"},
        )
        other_user = client.search(query="Atlas", user_id="user-2")

        self.assertEqual(status, 200)
        self.assertEqual(first, second)
        self.assertTrue(first["promotions"])
        self.assertTrue(all(item["memory"]["layer"] == "episodic" for item in other_user["results"]))


if __name__ == "__main__":
    unittest.main()
