from __future__ import annotations

import hmac
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .client import MemoryClient


class MemoryService:
    """Local-first HTTP wrapper. Configure api_key before exposing it beyond localhost."""

    def __init__(self, client: MemoryClient | None = None, *, api_key: str | None = None):
        self.client = client or MemoryClient()
        self.api_key = api_key

    def handle_request(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, list[str]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        payload = payload or {}
        query = query or {}
        headers = headers or {}
        route = path.rstrip("/") or "/"
        if self.api_key and not self._authorized(headers):
            return HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"}
        try:
            if method == "GET" and route in ("/", "/health", "/v4/health"):
                return HTTPStatus.OK, {
                    "status": "ok",
                    "service": "memoripy",
                    "version": "4.0",
                    "warning": "local-development server unless api_key and a production proxy are configured",
                }
            if method == "POST" and route in ("/v1/memories", "/v4/memories"):
                return HTTPStatus.OK, self.client.add(**payload)
            if method == "POST" and route in ("/v1/search", "/v4/recall"):
                return HTTPStatus.OK, self.client.search(**payload)
            if method == "POST" and route in ("/v3/capture", "/v4/capture"):
                return HTTPStatus.OK, self.client.capture(**payload)
            if method == "POST" and route in ("/v3/context", "/v4/context"):
                return HTTPStatus.OK, self._jsonable(self.client.context.build(**payload))
            if method == "POST" and route == "/v4/write":
                return HTTPStatus.OK, self.client.write(**payload)
            if method == "POST" and route in ("/v3/maintenance/consolidate", "/v4/maintenance/consolidate"):
                return HTTPStatus.OK, self.client.maintenance.consolidate(**payload)
            if method == "GET" and route == "/v4/audit":
                return HTTPStatus.OK, self.client.audit().to_dict()
            if method == "POST" and route == "/v4/audit":
                return HTTPStatus.OK, self.client.audit().to_dict()
            if method == "POST" and route in ("/v1/export", "/v4/export"):
                return HTTPStatus.OK, self.client.export()
            if method == "POST" and route in ("/v1/import", "/v4/import"):
                snapshot = payload.get("snapshot") or payload
                mode = payload.get("mode", "merge")
                return HTTPStatus.OK, self.client.import_(
                    snapshot,
                    mode=mode,
                    idempotency_key=payload.get("idempotency_key"),
                )
            if method == "POST" and route in ("/v1/chat/completions", "/v4/chat/completions"):
                return HTTPStatus.OK, self.client.chat.completions.create(**payload)
            if method == "DELETE" and route in ("/v1/memories", "/v4/memories"):
                delete_payload = dict(payload)
                for key in ("user_id", "agent_id", "run_id", "project_id", "organization_id", "namespace"):
                    delete_payload[key] = delete_payload.get(key) or self._first(query, key)
                return HTTPStatus.OK, self.client.delete_all(**delete_payload)
            if method == "GET" and route in ("/v1/memories", "/v4/memories"):
                return HTTPStatus.OK, self.client.get_all(
                    user_id=self._first(query, "user_id"),
                    agent_id=self._first(query, "agent_id"),
                    run_id=self._first(query, "run_id"),
                    project_id=self._first(query, "project_id"),
                    organization_id=self._first(query, "organization_id"),
                    namespace=self._first(query, "namespace"),
                )

            memory_prefix = "/v4/memories/" if route.startswith("/v4/memories/") else "/v1/memories/"
            if route.startswith(memory_prefix):
                suffix = route[len(memory_prefix) :]
                if suffix.endswith("/history"):
                    memory_id = suffix[: -len("/history")]
                    if method != "GET":
                        return HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed"}
                    return HTTPStatus.OK, self.client.history(memory_id=memory_id)
                if suffix.endswith("/explain"):
                    memory_id = suffix[: -len("/explain")]
                    if method != "GET":
                        return HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed"}
                    return HTTPStatus.OK, self.client.explain(memory_id=memory_id)
                if suffix.endswith("/correct"):
                    memory_id = suffix[: -len("/correct")]
                    if method != "POST":
                        return HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed"}
                    return HTTPStatus.OK, self.client.correct(memory_id=memory_id, **payload)
                if suffix.endswith("/feedback"):
                    memory_id = suffix[: -len("/feedback")]
                    if method != "POST":
                        return HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed"}
                    return HTTPStatus.OK, self.client.feedback(memory_id=memory_id, **payload)

                memory_id = suffix
                if method == "GET":
                    return HTTPStatus.OK, self.client.get(memory_id=memory_id)
                if method == "PATCH":
                    return HTTPStatus.OK, self.client.update(
                        memory_id=memory_id,
                        data=payload.get("data", payload),
                        idempotency_key=payload.get("idempotency_key"),
                    )
                if method == "DELETE":
                    return HTTPStatus.OK, self.client.delete(
                        memory_id=memory_id,
                        idempotency_key=payload.get("idempotency_key"),
                    )

            return HTTPStatus.NOT_FOUND, {"error": "not_found", "path": route}
        except KeyError as exc:
            return HTTPStatus.NOT_FOUND, {"error": "not_found", "detail": str(exc)}
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, {"error": "bad_request", "detail": str(exc)}
        except Exception as exc:
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": "internal_error",
                "detail": str(exc),
            }

    def _authorized(self, headers: dict[str, str]) -> bool:
        supplied = headers.get("authorization") or headers.get("Authorization") or ""
        if supplied.lower().startswith("bearer "):
            supplied = supplied[7:]
        return hmac.compare_digest(supplied, self.api_key or "")

    @staticmethod
    def _first(query: dict[str, list[str]], key: str) -> str | None:
        values = query.get(key) or []
        return values[0] if values else None

    @staticmethod
    def _jsonable(payload: Any) -> Any:
        return payload.to_dict() if hasattr(payload, "to_dict") else payload


def make_http_handler(service: MemoryService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MemoripyHTTP/4.0"

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_PATCH(self) -> None:
            self._dispatch("PATCH")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _dispatch(self, method: str) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                payload = {}
            status, response = service.handle_request(
                method=method,
                path=parsed.path,
                payload=payload,
                query=parse_qs(parsed.query),
                headers={key: value for key, value in self.headers.items()},
            )
            body = json.dumps(response, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve_http(
    host: str = "127.0.0.1",
    port: int = 8000,
    client: MemoryClient | None = None,
    *,
    api_key: str | None = None,
) -> ThreadingHTTPServer:
    service = MemoryService(client=client, api_key=api_key)
    return ThreadingHTTPServer((host, port), make_http_handler(service))


def create_fastapi_app(client: MemoryClient | None = None, *, api_key: str | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:
        raise RuntimeError("FastAPI is not installed. Install memoripy with the service extra.") from exc

    service = MemoryService(client=client, api_key=api_key)
    app = FastAPI(title="Memoripy", version="4.0.0")

    async def dispatch(request: Request, path: str, method: str, payload: dict[str, Any] | None = None):
        status, response = service.handle_request(
            method=method,
            path=path,
            payload=payload,
            query={key: request.query_params.getlist(key) for key in request.query_params.keys()},
            headers={key: value for key, value in request.headers.items()},
        )
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.get("/v4/health")
    async def health(request: Request):
        return await dispatch(request, "/v4/health", "GET")

    @app.post("/v4/capture")
    async def capture(request: Request, payload: dict[str, Any]):
        return await dispatch(request, "/v4/capture", "POST", payload)

    @app.post("/v4/write")
    async def write(request: Request, payload: dict[str, Any]):
        return await dispatch(request, "/v4/write", "POST", payload)

    @app.post("/v4/recall")
    async def recall(request: Request, payload: dict[str, Any]):
        return await dispatch(request, "/v4/recall", "POST", payload)

    @app.post("/v4/context")
    async def context(request: Request, payload: dict[str, Any]):
        return await dispatch(request, "/v4/context", "POST", payload)

    @app.get("/v4/audit")
    async def audit(request: Request):
        return await dispatch(request, "/v4/audit", "GET")

    @app.get("/v4/memories")
    async def list_memories(request: Request):
        return await dispatch(request, "/v4/memories", "GET")

    @app.get("/v4/memories/{memory_id}")
    async def get_memory(request: Request, memory_id: str):
        return await dispatch(request, f"/v4/memories/{memory_id}", "GET")

    @app.get("/v4/memories/{memory_id}/explain")
    async def explain(request: Request, memory_id: str):
        return await dispatch(request, f"/v4/memories/{memory_id}/explain", "GET")

    @app.post("/v4/memories/{memory_id}/correct")
    async def correct(request: Request, memory_id: str, payload: dict[str, Any]):
        return await dispatch(request, f"/v4/memories/{memory_id}/correct", "POST", payload)

    @app.delete("/v4/memories/{memory_id}")
    async def delete(request: Request, memory_id: str):
        return await dispatch(request, f"/v4/memories/{memory_id}", "DELETE")

    @app.post("/v4/maintenance/consolidate")
    async def consolidate(request: Request, payload: dict[str, Any]):
        return await dispatch(request, "/v4/maintenance/consolidate", "POST", payload)

    @app.post("/v4/chat/completions")
    async def chat(request: Request, payload: dict[str, Any]):
        return await dispatch(request, "/v4/chat/completions", "POST", payload)

    return app
