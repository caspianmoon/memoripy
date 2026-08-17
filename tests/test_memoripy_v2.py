from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from memoripy import AsyncMemoryClient, JSONStorage, MemoryClient, MemoryManager, MemoryService


class MemoripyV2Tests(unittest.TestCase):
    def test_add_search_and_history(self):
        client = MemoryClient()
        add_result = client.add(text="My name is Khazar", user_id="user-1", idempotency_key="name-1")
        self.assertEqual(len(add_result["memory_ids"]), 1)

        search_result = client.search(query="What is my name?", user_id="user-1")
        self.assertEqual(search_result["results"][0]["memory"]["summary"], "Name: Khazar")

        memory_id = search_result["results"][0]["memory"]["record_id"]
        history = client.history(memory_id=memory_id)
        self.assertEqual(len(history["history"]), 1)

    def test_idempotent_add_does_not_duplicate_memories(self):
        client = MemoryClient()
        first = client.add(text="I live in Istanbul", user_id="user-1", idempotency_key="loc-1")
        second = client.add(text="I live in Istanbul", user_id="user-1", idempotency_key="loc-1")
        self.assertEqual(first, second)

        all_memories = client.get_all(user_id="user-1")
        self.assertEqual(len(all_memories["results"]), 1)

    def test_supersede_and_current_value(self):
        client = MemoryClient()
        client.add(text="I live in Paris", user_id="user-1")
        client.add(text="I live in Berlin", user_id="user-1")

        results = client.search(query="Where do I live?", user_id="user-1")
        memory = results["results"][0]["memory"]
        self.assertEqual(memory["value"], "Berlin")

        history = client.history(memory_id=memory["record_id"])
        self.assertEqual(len(history["history"]), 2)
        self.assertEqual(history["history"][0]["state"], "superseded")
        self.assertEqual(history["history"][1]["state"], "active")

    def test_scope_isolation(self):
        client = MemoryClient()
        client.add(text="My favorite color is red", user_id="alice")
        client.add(text="My favorite color is blue", user_id="bob")

        alice_results = client.search(query="favorite color", user_id="alice")
        bob_results = client.search(query="favorite color", user_id="bob")

        self.assertIn("red", alice_results["results"][0]["memory"]["value"].lower())
        self.assertIn("blue", bob_results["results"][0]["memory"]["value"].lower())

    def test_multimodal_caption_ingestion(self):
        client = MemoryClient()
        client.add(
            items=[
                {
                    "modality": "image",
                    "metadata": {"caption": "My favorite city is Tokyo"},
                }
            ],
            user_id="user-1",
        )
        results = client.search(query="favorite city", user_id="user-1")
        self.assertIn("Tokyo", results["results"][0]["memory"]["value"])

    def test_export_import_roundtrip(self):
        source = MemoryClient()
        source.add(text="I work at OpenAI", user_id="user-1")
        snapshot = source.export()

        destination = MemoryClient()
        imported = destination.import_(snapshot, mode="replace")
        self.assertEqual(imported["status"], "ok")
        results = destination.search(query="where do i work", user_id="user-1")
        self.assertIn("OpenAI", results["results"][0]["memory"]["value"])

    def test_async_client_parity(self):
        async def runner():
            client = AsyncMemoryClient()
            await client.add(text="My name is Async Alice", user_id="user-1")
            results = await client.search(query="name", user_id="user-1")
            return results["results"][0]["memory"]["value"]

        self.assertEqual(asyncio.run(runner()), "Async Alice")

    def test_memory_manager_compatibility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "legacy.json")
            manager = MemoryManager(storage=JSONStorage(file_path=path))
            manager.add_interaction("My name is Legacy Lisa", "Nice to meet you")
            retrievals = manager.retrieve_relevant_interactions("name", similarity_threshold=0)
            self.assertTrue(retrievals)
            self.assertIn("Legacy Lisa", retrievals[0]["output"] or retrievals[0]["prompt"])

    def test_service_routes(self):
        service = MemoryService()
        status, created = service.handle_request(
            method="POST",
            path="/v1/memories",
            payload={"text": "I am from Ankara", "user_id": "user-1"},
        )
        self.assertEqual(status, 200)
        memory_id = created["memory_ids"][0]

        status, fetched = service.handle_request(method="GET", path=f"/v1/memories/{memory_id}")
        self.assertEqual(status, 200)
        self.assertIn("Ankara", fetched["memory"]["value"])

    def test_chat_completions_shape(self):
        client = MemoryClient()
        client.add(text="My name is Khazar", user_id="user-1")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "What is my name?"}],
            user_id="user-1",
        )
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
