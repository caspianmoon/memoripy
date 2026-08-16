from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoripy.evals import BUILTIN_CONTRACTS, load_contracts, run_contracts


class MemoryContractTests(unittest.TestCase):
    def test_builtin_contracts_pass(self):
        summary = run_contracts()
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["passed_count"], len(BUILTIN_CONTRACTS))

    def test_json_contract_file(self):
        payload = {
            "contracts": [
                {
                    "name": "simple_name",
                    "events": [
                        {
                            "messages": [{"role": "user", "content": "My name is Alice"}],
                            "user_id": "u1",
                        }
                    ],
                    "queries": [
                        {
                            "query": "What is my name?",
                            "user_id": "u1",
                            "expect_contains": ["Alice"],
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "contracts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            contracts = load_contracts(path)
            summary = run_contracts(contracts)
        self.assertEqual(summary["passed_count"], 1)
        self.assertEqual(summary["failed_count"], 0)

    def test_failing_contract_reports_reason(self):
        contract = BUILTIN_CONTRACTS[0]
        bad = type(contract)(
            name="bad_expectation",
            description="A deliberately failing assertion.",
            events=contract.events,
            queries=[
                {
                    "query": "Where do I live now?",
                    "user_id": "u1",
                    "expect_contains": ["Moon"],
                }
            ],
        )
        summary = run_contracts([bad])
        self.assertEqual(summary["failed_count"], 1)
        self.assertIn("missing:Moon", summary["results"][0]["failures"][0])


if __name__ == "__main__":
    unittest.main()
