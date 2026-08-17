from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .inspector import inspector_html
from .service import MemoryService
from .tenant import ADMIN_SCOPE, READ_SCOPE, WRITE_SCOPE, TenantPrincipal, TenantRegistry, TenantStoreManager, bearer_token


class TenantMemoryGateway:
    """Authenticated, physically isolated multi-tenant HTTP gateway."""

    def __init__(self, *, stores: TenantStoreManager, registry: TenantRegistry) -> None:
        self.stores = stores
        self.registry = registry

    def handle_request(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, list[str]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str, Any]:
        payload = dict(payload or {})
        query = {key: list(value) for key, value in (query or {}).items()}
        headers = headers or {}
        route = path.rstrip("/") or "/"
        if method == "GET" and route in ("/", "/inspector"):
            return HTTPStatus.OK, "text/html; charset=utf-8", inspector_html(title="Memoripy Tenant Inspector")
        if method == "GET" and route in ("/health", "/v4/health"):
            return HTTPStatus.OK, "application/json; charset=utf-8", {"status": "ok", "service": "memoripy-gateway", "version": "0.4.0"}

        required = self._required_scope(method, route)
        principal = self.registry.authenticate(bearer_token(headers), required_scope=required)
        if principal is None:
            return HTTPStatus.UNAUTHORIZED, "application/json; charset=utf-8", {"error": "unauthorized", "required_scope": required}

        if route == "/v4/admin/keys" and method == "GET":
            return HTTPStatus.OK, "application/json; charset=utf-8", {"keys": self.registry.list_keys(tenant_id=principal.tenant_id)}
        if route.startswith("/v4/admin/keys/") and route.endswith("/revoke") and method == "POST":
            key_id = route.split("/")[-2]
            return HTTPStatus.OK, "application/json; charset=utf-8", {"revoked": self.registry.revoke(key_id), "key_id": key_id}

        conflict = self._scope_conflict(principal, payload, query)
        if conflict:
            return HTTPStatus.FORBIDDEN, "application/json; charset=utf-8", {"error": "tenant_scope_conflict", "detail": conflict}
        self._inject_scope(principal, payload, query)
        service = MemoryService(client=self.stores.client(principal.tenant_id))
        status, response = service.handle_request(method=method, path=path, payload=payload, query=query, headers={})
        return status, "application/json; charset=utf-8", response

    def _required_scope(self, method: str, route: str) -> str:
        if route.startswith("/v4/admin/"):
            return ADMIN_SCOPE
        if method in {"GET", "HEAD"}:
            return READ_SCOPE
        if method == "POST" and route in {"/v4/recall", "/v4/context", "/v4/audit", "/v4/chat/completions"}:
            return READ_SCOPE
        return WRITE_SCOPE

    def _scope_conflict(self, principal: TenantPrincipal, payload: dict[str, Any], query: dict[str, list[str]]) -> str | None:
        supplied = payload.get("organization_id")
        if supplied is not None and str(supplied) != principal.tenant_id:
            return "payload organization_id does not match authenticated tenant"
        query_values = query.get("organization_id") or []
        if query_values and any(str(value) != principal.tenant_id for value in query_values):
            return "query organization_id does not match authenticated tenant"
        filters = payload.get("filters")
        if isinstance(filters, dict):
            scope = filters.get("scope")
            if isinstance(scope, dict) and scope.get("organization_id") not in (None, principal.tenant_id):
                return "filter scope organization_id does not match authenticated tenant"
        return None

    def _inject_scope(self, principal: TenantPrincipal, payload: dict[str, Any], query: dict[str, list[str]]) -> None:
        payload["organization_id"] = principal.tenant_id
        query["organization_id"] = [principal.tenant_id]
        filters = payload.get("filters")
        if isinstance(filters, dict):
            scope = filters.setdefault("scope", {})
            if isinstance(scope, dict):
                scope["organization_id"] = principal.tenant_id


def serve_gateway(
    *,
    stores: TenantStoreManager,
    registry: TenantRegistry,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> ThreadingHTTPServer:
    gateway = TenantMemoryGateway(stores=stores, registry=registry)

    class Handler(BaseHTTPRequestHandler):
        server_version = "MemoripyGateway/0.4.0"
        def do_GET(self) -> None: self._dispatch("GET")
        def do_POST(self) -> None: self._dispatch("POST")
        def do_PATCH(self) -> None: self._dispatch("PATCH")
        def do_DELETE(self) -> None: self._dispatch("DELETE")
        def log_message(self, format: str, *args: Any) -> None: return

        def _dispatch(self, method: str) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                payload = {}
            status, content_type, response = gateway.handle_request(
                method=method,
                path=parsed.path,
                payload=payload,
                query=parse_qs(parsed.query),
                headers={key: value for key, value in self.headers.items()},
            )
            if isinstance(response, str) and content_type.startswith("text/html"):
                body = response.encode("utf-8")
            else:
                body = json.dumps(response, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)
