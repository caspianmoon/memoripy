from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoripy import (
    AdmissionDecision,
    BrainConfig,
    KeywordBoostReranker,
    LocalAssetProcessor,
    MemoryAction,
    MemoryClient,
    MemoryPipelineConfig,
    MemoryService,
    MemoryState,
    RerankOutcome,
)
from memoripy.extractors import DefaultMemoryExtractor, MemoryCandidate
from memoripy.types import Durability, MemoryKind, MemoryLayer, TrustLevel


class NoopExtractor:
    def extract_semantic(self, evidence):
        return []

    def build_episode_candidate(self, evidence):
        return None

    def extract(self, evidence):
        return []


class LocationBoostReranker(KeywordBoostReranker):
    def rerank(self, *, query, candidates, state, search_filters, intent):
        del query, state, search_filters, intent
        return {
            record.record_id: RerankOutcome(score=100.0 if record.key == "location" else 0.0)
            for _, record, _ in candidates
        }


class QuarantineAllPolicy:
    def evaluate(self, *, candidate, evidence, state):
        del evidence, state
        return AdmissionDecision(
            action=MemoryAction.QUARANTINE.value,
            state=MemoryState.QUARANTINED.value,
            reason_codes=["TEST_POLICY"],
            confidence=candidate.confidence,
            trust_level=TrustLevel.QUARANTINED.value,
            durability=Durability.EPHEMERAL.value,
        )


class EpisodeThenFactExtractor:
    def extract_semantic(self, evidence):
        if "atlas" not in evidence.text.casefold():
            return []
        return [
            MemoryCandidate(
                kind=MemoryKind.FACT.value,
                key="project",
                value="Atlas",
                summary="Project: Atlas",
                confidence=0.9,
                metadata={"extractor": "test", "topic": "project"},
                tags=["project"],
                layer=MemoryLayer.SEMANTIC.value,
                salience=0.75,
                evidence_spans=list(evidence.evidence_spans),
            )
        ]

    def build_episode_candidate(self, evidence):
        return MemoryCandidate(
            kind=MemoryKind.EPISODIC_SUMMARY.value,
            key=f"episode_{evidence.evidence_id}",
            value=evidence.text,
            summary=f"Episode: {evidence.text}",
            confidence=0.9,
            metadata={"extractor": "test"},
            tags=["episodic", "project"],
            layer=MemoryLayer.EPISODIC.value,
            salience=0.75,
            evidence_spans=list(evidence.evidence_spans),
        )

    def extract(self, evidence):
        episode = self.build_episode_candidate(evidence)
        return [episode] if episode is not None else []


class PipelineFeatureTests(unittest.TestCase):
    def test_pipeline_extractor_wins_over_legacy_argument(self):
        client = MemoryClient(
            extractor=NoopExtractor(),
            pipeline=MemoryPipelineConfig(extractor=DefaultMemoryExtractor()),
        )
        client.capture(messages=[{"role": "user", "content": "My name is Priority Piper"}], user_id="u1")
        result = client.search(query="name", user_id="u1", track_usage=False)
        self.assertEqual(result["results"][0]["memory"]["value"], "Priority Piper")

    def test_custom_admission_policy_quarantines_candidate(self):
        client = MemoryClient(pipeline=MemoryPipelineConfig(admission_policy=QuarantineAllPolicy()))
        result = client.capture(messages=[{"role": "user", "content": "My name is Alice"}], user_id="u1")
        self.assertTrue(result["quarantined"])
        recalled = client.search(query="name", user_id="u1", track_usage=False)
        self.assertFalse(recalled["results"])

    def test_reranker_can_reorder_results(self):
        client = MemoryClient(pipeline=MemoryPipelineConfig(reranker=LocationBoostReranker()))
        client.write(kind="profile_attribute", key="name", value="Alice", user_id="u1")
        client.write(kind="profile_attribute", key="location", value="Paris", user_id="u1")
        result = client.search(query="tell me something", user_id="u1", limit=2, track_usage=False)
        self.assertEqual(result["results"][0]["memory"]["key"], "location")
        self.assertEqual(result["results"][0]["rank_breakdown"]["reranker"], 100.0)

    def test_local_asset_processor_reads_text_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.txt"
            path.write_text("My favorite city is Tokyo", encoding="utf-8")
            client = MemoryClient(pipeline=MemoryPipelineConfig(asset_processor=LocalAssetProcessor()))
            client.add(items=[{"modality": "document", "asset_ref": str(path)}], user_id="u1")
            result = client.search(query="favorite city", user_id="u1", track_usage=False)
            self.assertEqual(result["results"][0]["memory"]["value"], "Tokyo")

    def test_trace_exposes_retrieval_lanes_and_receipt(self):
        client = MemoryClient()
        client.capture(messages=[{"role": "user", "content": "My name is Tracey"}], user_id="u1")
        result = client.search(query="name", user_id="u1", include_trace=True, track_usage=False)
        self.assertIn("retrieval", result["trace"])
        self.assertTrue(result["trace"]["ranking"])
        self.assertIn("receipt", result["trace"]["ranking"][0])
        self.assertTrue(result["results"][0]["receipt"]["retrieval_lanes"])

    def test_exact_scope_does_not_leak_into_broader_query(self):
        client = MemoryClient()
        client.add(text="I live in Paris", user_id="u1")
        client.add(text="I live in Berlin", user_id="u1", agent_id="a1", run_id="r1")
        broad = client.search(query="where do I live", user_id="u1", track_usage=False)
        narrow = client.search(query="where do I live", user_id="u1", agent_id="a1", run_id="r1", track_usage=False)
        self.assertEqual(broad["results"][0]["memory"]["value"], "Paris")
        self.assertEqual(narrow["results"][0]["memory"]["value"], "Berlin")

    def test_service_api_key(self):
        service = MemoryService(api_key="secret")
        status, _ = service.handle_request(method="GET", path="/v4/health")
        self.assertEqual(status, 401)
        status, payload = service.handle_request(
            method="GET",
            path="/v4/health",
            headers={"Authorization": "Bearer secret"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["version"], "4.0")

    def test_consolidation_requires_independent_support(self):
        client = MemoryClient(
            pipeline=MemoryPipelineConfig(
                extractor=EpisodeThenFactExtractor(),
                brain=BrainConfig(mode="attention_fast", consolidation_min_support=2),
            )
        )
        client.capture(messages=[{"role": "user", "content": "Atlas appeared once"}], user_id="u1")
        first = client.maintenance.consolidate(user_id="u1")
        self.assertFalse(first["promotions"])
        client.capture(messages=[{"role": "user", "content": "Atlas appeared again"}], user_id="u1")
        second = client.maintenance.consolidate(user_id="u1")
        self.assertTrue(second["promotions"])
        result = client.search(query="Atlas", user_id="u1", track_usage=False)
        self.assertTrue(any(item["memory"]["layer"] == "semantic" for item in result["results"]))

    def test_explicit_policy_is_authoritative_and_pinned(self):
        client = MemoryClient()
        created = client.write(
            kind="policy",
            key="refunds",
            value="Refunds older than 30 days require approval.",
            organization_id="org-1",
            durability="pinned",
        )
        memory_id = created["semantic_memory_ids"][0]
        record = client.get(memory_id=memory_id)["memory"]
        self.assertEqual(record["trust_level"], "authoritative")
        self.assertEqual(record["durability"], "pinned")
        result = client.search(query="refund approval", organization_id="org-1", track_usage=False)
        self.assertEqual(result["results"][0]["memory"]["kind"], "policy")

    def test_attention_feedback_changes_utility_not_retrieval_count(self):
        client = MemoryClient(pipeline=MemoryPipelineConfig(brain=BrainConfig(mode="attention_fast")))
        created = client.capture(messages=[{"role": "user", "content": "My name is Nora"}], user_id="u1")
        memory_id = created["semantic_memory_ids"][0]
        client.search(query="Nora", user_id="u1")
        before = client.get(memory_id=memory_id)["memory"]
        client.feedback(memory_id=memory_id, outcome="success")
        after = client.get(memory_id=memory_id)["memory"]
        self.assertEqual(after["retrieval_count"], before["retrieval_count"])
        self.assertEqual(after["associated_success_count"], before["associated_success_count"] + 1)


if __name__ == "__main__":
    unittest.main()
