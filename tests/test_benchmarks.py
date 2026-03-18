from __future__ import annotations

import unittest

from benchmarks.runner import MemoripyBenchmarkAdapter, run_benchmarks


class BenchmarkHarnessTests(unittest.TestCase):
    def test_memoripy_benchmark_harness_runs_cleanly(self):
        summary = run_benchmarks(MemoripyBenchmarkAdapter())

        self.assertEqual(summary["adapter"], "memoripy")
        self.assertEqual(summary["scenario_count"], 8)
        self.assertEqual(summary["earned_score"], summary["max_score"])
        self.assertTrue(all(result["passed"] for result in summary["results"]))


if __name__ == "__main__":
    unittest.main()
