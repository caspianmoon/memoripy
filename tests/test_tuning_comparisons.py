from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoripy.comparisons import (
    AdapterAvailability,
    HindsightComparisonAdapter,
    MemoripyComparisonAdapter,
    Mem0ComparisonAdapter,
    run_comparison,
)
from memoripy.evals import BUILTIN_CONTRACTS
from memoripy.pipeline import RetrievalConfig
from memoripy.tuning import load_retrieval_profile, save_retrieval_profile, tune_retrieval


class FakeMem0:
    def __init__(self):
        self.values = {}

    def add(self, messages, user_id, **kwargs):
        content = messages[0]["content"] if isinstance(messages, list) else str(messages)
        self.values.setdefault(user_id, []).append(content)

    def search(self, query, user_id, limit=5):
        return {"results": [{"memory": value} for value in self.values.get(user_id, [])[:limit]]}


class FakeHindsight:
    def __init__(self):
        self.values = {}

    def retain(self, bank_id, content, **kwargs):
        self.values.setdefault(bank_id, []).append(content)

    def recall(self, bank_id, query, **kwargs):
        return {"results": [{"content": value} for value in self.values.get(bank_id, [])]}


class UnavailableAdapter:
    name = "unavailable"
    def availability(self): return AdapterAvailability(False, "missing service")
    def retain(self, **kwargs): raise AssertionError
    def recall(self, **kwargs): raise AssertionError
    def close(self): return None


class TuningAndComparisonTests(unittest.TestCase):
    def test_tuner_selects_a_profile(self):
        result = tune_retrieval(BUILTIN_CONTRACTS[:2])
        self.assertGreaterEqual(result.selected.score_ratio, 0.5)
        self.assertEqual(result.contract_count, 2)
        self.assertTrue(result.candidates)

    def test_profile_roundtrip(self):
        result = tune_retrieval(BUILTIN_CONTRACTS[:1], candidates={"only": RetrievalConfig()})
        with tempfile.TemporaryDirectory() as tmp:
            path = save_retrieval_profile(result.selected, Path(tmp) / "profile.json")
            loaded = load_retrieval_profile(path)
        self.assertEqual(loaded.name, "only")
        self.assertEqual(loaded.retrieval.describe(), RetrievalConfig().describe())

    def test_unavailable_adapter_has_no_fake_score(self):
        summary = run_comparison(BUILTIN_CONTRACTS[:1], [UnavailableAdapter()])
        result = summary["results"][0]
        self.assertFalse(result["availability"]["available"])
        self.assertIsNone(result["score_ratio"])

    def test_memoripy_comparison_runs(self):
        summary = run_comparison(BUILTIN_CONTRACTS[:1], [MemoripyComparisonAdapter()])
        self.assertEqual(summary["results"][0]["score_ratio"], 1.0)

    def test_mem0_adapter_matches_current_shape(self):
        adapter = Mem0ComparisonAdapter(memory=FakeMem0())
        adapter.retain(scope="u1", content="Alice likes tea")
        self.assertIn("Alice likes tea", adapter.recall(scope="u1", query="tea"))

    def test_hindsight_adapter_matches_current_shape(self):
        adapter = HindsightComparisonAdapter(client=FakeHindsight())
        adapter.retain(scope="bank", content="Alice works at Google")
        self.assertIn("Alice works at Google", adapter.recall(scope="bank", query="work"))


if __name__ == "__main__":
    unittest.main()
