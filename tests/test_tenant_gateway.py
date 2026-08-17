from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoripy.gateway import TenantMemoryGateway
from memoripy.inspector import InspectorService, inspector_html
from memoripy.tenant import ADMIN_SCOPE, READ_SCOPE, WRITE_SCOPE, TenantRegistry, TenantStoreManager


class TenantGatewayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.registry = TenantRegistry(root / "registry.json")
        self.stores = TenantStoreManager(root / "stores")
        self.gateway = TenantMemoryGateway(stores=self.stores, registry=self.registry)

    def tearDown(self):
        self.tmp.cleanup()

    def _headers(self, token: str):
        return {"Authorization": f"Bearer {token}"}

    def test_key_file_contains_digest_not_plaintext(self):
        token, record = self.registry.create_key(tenant_id="tenant-a")
        payload = json.loads(Path(self.registry.path).read_text())
        rendered = json.dumps(payload)
        self.assertNotIn(token, rendered)
        self.assertIn(record.key_id, rendered)

    def test_two_tenants_are_isolated(self):
        token_a, _ = self.registry.create_key(tenant_id="tenant-a")
        token_b, _ = self.registry.create_key(tenant_id="tenant-b")
        status, _, _ = self.gateway.handle_request(
            method="POST",
            path="/v4/capture",
            payload={"messages": [{"role": "user", "content": "My favorite city is Tokyo"}], "user_id": "u1"},
            headers=self._headers(token_a),
        )
        self.assertEqual(status, 200)
        status, _, result = self.gateway.handle_request(
            method="POST",
            path="/v4/recall",
            payload={"query": "favorite city", "user_id": "u1"},
            headers=self._headers(token_b),
        )
        self.assertEqual(status, 200)
        self.assertFalse(result["results"])
        self.assertNotEqual(self.stores.tenant_path("tenant-a"), self.stores.tenant_path("tenant-b"))

    def test_read_only_key_cannot_write(self):
        token, _ = self.registry.create_key(tenant_id="tenant-a", scopes=[READ_SCOPE])
        status, _, result = self.gateway.handle_request(
            method="POST",
            path="/v4/capture",
            payload={"messages": [{"role": "user", "content": "My name is Alice"}]},
            headers=self._headers(token),
        )
        self.assertEqual(status, 401)
        self.assertEqual(result["required_scope"], WRITE_SCOPE)

    def test_tenant_scope_conflict_is_rejected(self):
        token, _ = self.registry.create_key(tenant_id="tenant-a")
        status, _, result = self.gateway.handle_request(
            method="POST",
            path="/v4/recall",
            payload={"query": "anything", "organization_id": "tenant-b"},
            headers=self._headers(token),
        )
        self.assertEqual(status, 403)
        self.assertEqual(result["error"], "tenant_scope_conflict")

    def test_revoked_key_stops_working(self):
        token, record = self.registry.create_key(tenant_id="tenant-a")
        self.assertIsNotNone(self.registry.authenticate(token))
        self.assertTrue(self.registry.revoke(record.key_id))
        self.assertIsNone(self.registry.authenticate(token))

    def test_admin_can_list_keys(self):
        token, _ = self.registry.create_key(tenant_id="tenant-a", scopes=[ADMIN_SCOPE])
        status, _, result = self.gateway.handle_request(
            method="GET", path="/v4/admin/keys", headers=self._headers(token)
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(result["keys"]), 1)
        self.assertNotIn("digest", result["keys"][0])

    def test_inspector_page_contains_authenticated_calls(self):
        page = inspector_html()
        self.assertIn("Memoripy Inspector", page)
        self.assertIn("Authorization", page)
        self.assertIn("/memories/", page)
        self.assertIn("Run memory audit", page)

    def test_single_store_inspector_serves_html_and_api(self):
        client = self.stores.client("tenant-a")
        service = InspectorService(client, api_key="secret")
        status, content_type, body = service.handle(
            method="GET", path="/inspector", payload={}, query={}, headers={}
        )
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith("text/html"))
        self.assertIn("Memoripy Inspector", body)
        status, _, body = service.handle(
            method="GET", path="/v4/audit", payload={}, query={}, headers={"Authorization": "Bearer secret"}
        )
        self.assertEqual(status, 200)
        self.assertIn("schema_version", json.loads(body))


if __name__ == "__main__":
    unittest.main()
