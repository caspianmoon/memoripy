from __future__ import annotations

import asyncio
import unittest

from memoripy import MemoryClient
from memoripy.mcp_server import build_mcp_server

try:
    from mcp import Client
except ImportError:
    Client = None


@unittest.skipIf(Client is None, "mcp optional dependency is not installed")
class MCPIntegrationTests(unittest.TestCase):
    def test_in_process_server_lists_and_calls_tools(self):
        async def run():
            server = build_mcp_server(client=MemoryClient())
            async with Client(server) as client:
                tools = await client.list_tools()
                names = {tool.name for tool in tools.tools}
                self.assertIn("memoripy_capture", names)
                self.assertIn("memoripy_recall", names)
                result = await client.call_tool(
                    "memoripy_capture",
                    {"text": "My favorite city is Tokyo", "user_id": "u1"},
                )
                self.assertFalse(result.is_error)
                recall = await client.call_tool(
                    "memoripy_recall",
                    {"query": "favorite city", "user_id": "u1"},
                )
                self.assertFalse(recall.is_error)
                rendered = str(recall.structured_content or recall.content)
                self.assertIn("Tokyo", rendered)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
