from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoripy import MemoryClient
from memoripy.mcp_server import MCPAccessPolicy, MemoripyMCPTools, load_token_file, run_mcp_server


class MCPToolsTests(unittest.TestCase):
    def test_capture_recall_and_explain(self):
        tools = MemoripyMCPTools(MemoryClient())
        created = tools.capture("My favorite city is Tokyo", user_id="u1")
        recalled = tools.recall("favorite city", user_id="u1")
        self.assertTrue(recalled["results"])
        memory_id = created["semantic_memory_ids"][0]
        explained = tools.explain(memory_id)
        self.assertEqual(explained["memory"]["record_id"], memory_id)
        self.assertTrue(explained["evidence"])

    def test_read_only_policy_blocks_mutations(self):
        tools = MemoripyMCPTools(MemoryClient(), policy=MCPAccessPolicy(read_only=True))
        with self.assertRaises(PermissionError):
            tools.capture("My name is Alice")

    def test_scope_defaults_are_applied(self):
        tools = MemoripyMCPTools(
            MemoryClient(), policy=MCPAccessPolicy(scope_defaults={"organization_id": "tenant-a"})
        )
        result = tools.capture("My name is Alice", user_id="u1")
        memory = tools.client.get(memory_id=result["semantic_memory_ids"][0])["memory"]
        self.assertEqual(memory["scope"]["organization_id"], "tenant-a")

    def test_correct_and_forget(self):
        tools = MemoripyMCPTools(MemoryClient())
        result = tools.capture("I live in Paris", user_id="u1")
        memory_id = result["semantic_memory_ids"][0]
        tools.correct(memory_id, "Istanbul")
        self.assertEqual(tools.explain(memory_id)["memory"]["value"], "Istanbul")
        tools.forget(memory_id)
        self.assertEqual(tools.explain(memory_id)["memory"]["state"], "deleted")

    def test_network_transport_requires_token_file_before_importing_mcp(self):
        with self.assertRaisesRegex(ValueError, "require --token-file"):
            run_mcp_server(path="unused", transport="streamable-http")

    def test_token_file_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.json"
            path.write_text(json.dumps({"tokens": {"secret": {"subject": "u1", "scopes": ["memoripy:read"]}}}))
            payload = load_token_file(path)
        self.assertEqual(payload["secret"]["subject"], "u1")
        self.assertEqual(payload["secret"]["scopes"], ["memoripy:read"])


if __name__ == "__main__":
    unittest.main()
