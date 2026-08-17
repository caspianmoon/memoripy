from __future__ import annotations

import unittest
from datetime import datetime, timezone

from memoripy import MemoryClient, MemoryPipelineConfig
from memoripy.assisted import AssistedExtractionConfig, AssistedMemoryExtractor, StructuredExtractionError
from memoripy.temporal import infer_temporal_bounds


class FakeStructuredModel:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.payload


class AssistedAndTemporalTests(unittest.TestCase):
    def test_last_month_changes_valid_from(self):
        reference = "2026-08-17T12:00:00Z"
        bounds = infer_temporal_bounds("I moved to Istanbul last month", reference)
        self.assertEqual(bounds.valid_from, "2026-07-17T12:00:00Z")
        self.assertEqual(bounds.source, "relative_last_month")

    def test_explicit_range_has_valid_to(self):
        bounds = infer_temporal_bounds("I lived in Paris from 2024-01-01 to 2025-01-01", "2026-08-17T00:00:00Z")
        self.assertEqual(bounds.valid_from, "2024-01-01T00:00:00Z")
        self.assertEqual(bounds.valid_to, "2025-01-02T00:00:00Z")

    def test_assisted_extractor_accepts_grounded_candidate(self):
        model = FakeStructuredModel(
            {
                "memories": [
                    {
                        "kind": "constraint",
                        "key": "no_morning_meetings",
                        "value": "do not schedule meetings before noon",
                        "summary": "Constraint: no meetings before noon",
                        "confidence": 0.96,
                        "durability": "pinned",
                        "quote": "do not schedule meetings before noon",
                    }
                ]
            }
        )
        client = MemoryClient(
            pipeline=MemoryPipelineConfig(
                extractor=AssistedMemoryExtractor(
                    model,
                    config=AssistedExtractionConfig(include_deterministic_fallback=False),
                )
            )
        )
        result = client.capture(
            messages=[{"role": "user", "content": "Please do not schedule meetings before noon."}],
            user_id="u1",
        )
        self.assertTrue(result["memory_ids"])
        recalled = client.search(query="morning meetings", user_id="u1")
        self.assertEqual(recalled["results"][0]["memory"]["kind"], "constraint")
        self.assertEqual(recalled["results"][0]["memory"]["metadata"]["extractor"], "assisted")

    def test_assisted_extractor_rejects_ungrounded_candidate(self):
        model = FakeStructuredModel(
            {"memories": [{"kind": "fact", "value": "User owns a yacht", "confidence": 0.99, "quote": "owns a yacht"}]}
        )
        client = MemoryClient(
            pipeline=MemoryPipelineConfig(
                extractor=AssistedMemoryExtractor(
                    model,
                    config=AssistedExtractionConfig(include_deterministic_fallback=False),
                )
            )
        )
        result = client.capture(messages=[{"role": "user", "content": "I like tea."}], user_id="u1")
        self.assertFalse(result["semantic_memory_ids"])

    def test_assisted_extractor_parses_json_fence(self):
        model = FakeStructuredModel(
            '```json\n{"memories":[{"kind":"preference","value":"black coffee","confidence":0.9,"quote":"black coffee"}]}\n```'
        )
        client = MemoryClient(
            pipeline=MemoryPipelineConfig(
                extractor=AssistedMemoryExtractor(
                    model,
                    config=AssistedExtractionConfig(include_deterministic_fallback=False),
                )
            )
        )
        client.capture(messages=[{"role": "user", "content": "I prefer black coffee."}], user_id="u1")
        result = client.search(query="coffee", user_id="u1")
        self.assertIn("black coffee", result["results"][0]["memory"]["value"])

    def test_assisted_extractor_raises_for_invalid_json(self):
        extractor = AssistedMemoryExtractor(FakeStructuredModel("not-json"))
        client = MemoryClient(pipeline=MemoryPipelineConfig(extractor=extractor))
        with self.assertRaises(StructuredExtractionError):
            client.capture(messages=[{"role": "user", "content": "My name is Alice"}], user_id="u1")

    def test_default_extractor_applies_relative_temporal_bounds(self):
        client = MemoryClient()
        result = client.capture(
            messages=[{"role": "user", "content": "I moved to Istanbul last month", "timestamp": "2026-08-17T12:00:00Z"}],
            user_id="u1",
        )
        memory = client.get(memory_id=result["semantic_memory_ids"][0])["memory"]
        self.assertEqual(memory["valid_from"], "2026-07-17T12:00:00Z")

    def test_relative_day_ago(self):
        bounds = infer_temporal_bounds("It happened 3 days ago", datetime(2026, 8, 17, tzinfo=timezone.utc))
        self.assertEqual(bounds.valid_from, "2026-08-14T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
