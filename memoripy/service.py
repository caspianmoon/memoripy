from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .client import MemoryClient


class MemoryService:
    def __init__(self, client: MemoryClient | None = None):
        self.client = client or MemoryClient()

    def handle_request(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, list[str]] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        payload = payload or {}
        query = query or {}
        route = path.rstrip("/") or "/"
        try:
            if method == "POST" and route == "/v1/memories":
                return HTTPStatus.OK, self.client.add(**payload)
            if method == "POST" and route == "/v1/search":
                return HTTPStatus.OK, self.client.search(**payload)
            if method == "POST" and route == "/v3/capture":
                return HTTPStatus.OK, self.client.capture(**payload)
            if method == "POST" and route == "/v3/context":
                return HTTPStatus.OK, self._jsonable(self.client.context.build(**payload))
            if method == "POST" and route == "/v1/export":
                return HTTPStatus.OK, self.client.export()
            if method == "POST" and route == "/v1/import":
                snapshot = payload.get("snapshot") or payload
                mode = payload.get("mode", "merge")
                return HTTPStatus.OK, self.client.import_(snapshot, mode=mode, idempotency_key=payload.get("idempotency_key"))
            if method == "POST" and route == "/v1/chat/completions":
                return HTTPStatus.OK, self.client.chat.completions.create(**payload)
            if method == "DELETE" and route == "/v1/memories":
                delete_payload = dict(payload)
                delete_payload["user_id"] = delete_payload.get("user_id") or self._first(query, "user_id")
                delete_payload["agent_id"] = delete_payload.get("agent_id") or self._first(query, "agent_id")
                delete_payload["run_id"] = delete_payload.get("run_id") or self._first(query, "run_id")
                return HTTPStatus.OK, self.client.delete_all(**delete_payload)
            if method == "GET" and route == "/v1/memories":
                return HTTPStatus.OK, self.client.get_all(
                    user_id=self._first(query, "user_id"),
                    agent_id=self._first(query, "agent_id"),
                    run_id=self._first(query, "run_id"),
                )

            if route.startswith("/v1/memories/"):
                suffix = route[len("/v1/memories/") :]
                if suffix.endswith("/history"):
                    memory_id = suffix[: -len("/history")]
                    if method != "GET":
                        return HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed"}
                    return HTTPStatus.OK, self.client.history(memory_id=memory_id)

                memory_id = suffix
                if method == "GET":
                    return HTTPStatus.OK, self.client.get(memory_id=memory_id)
                if method == "PATCH":
                    return HTTPStatus.OK, self.client.update(memory_id=memory_id, data=payload.get("data", payload), idempotency_key=payload.get("idempotency_key"))
                if method == "DELETE":
                    return HTTPStatus.OK, self.client.delete(memory_id=memory_id, idempotency_key=payload.get("idempotency_key"))

            return HTTPStatus.NOT_FOUND, {"error": "not_found", "path": route}
        except KeyError as exc:
            return HTTPStatus.NOT_FOUND, {"error": "not_found", "detail": str(exc)}
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, {"error": "bad_request", "detail": str(exc)}

    @staticmethod
    def _first(query: dict[str, list[str]], key: str) -> str | None:
        values = query.get(key) or []
        return values[0] if values else None

    @staticmethod
    def _jsonable(payload: Any) -> Any:
        if hasattr(payload, "to_dict"):
            return payload.to_dict()
        return payload


def make_http_handler(service: MemoryService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MemoripyHTTP/2.0"

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
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            status, response = service.handle_request(
                method=method,
                path=parsed.path,
                payload=payload,
                query=parse_qs(parsed.query),
            )
            body = json.dumps(response).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve_http(host: str = "127.0.0.1", port: int = 8000, client: MemoryClient | None = None) -> ThreadingHTTPServer:
    service = MemoryService(client=client)
    server = ThreadingHTTPServer((host, port), make_http_handler(service))
    return server


def create_fastapi_app(client: MemoryClient | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:
        raise RuntimeError("FastAPI is not installed. Install memoripy with the service extras.") from exc

    service = MemoryService(client=client)
    app = FastAPI(title="Memoripy", version="2.0")

    @app.post("/v1/memories")
    async def add_memory(payload: dict[str, Any]) -> dict[str, Any]:
        status, response = service.handle_request(method="POST", path="/v1/memories", payload=payload)
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.post("/v1/search")
    async def search(payload: dict[str, Any]) -> dict[str, Any]:
        status, response = service.handle_request(method="POST", path="/v1/search", payload=payload)
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.post("/v3/capture")
    async def capture(payload: dict[str, Any]) -> dict[str, Any]:
        status, response = service.handle_request(method="POST", path="/v3/capture", payload=payload)
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.post("/v3/context")
    async def context(payload: dict[str, Any]) -> dict[str, Any]:
        status, response = service.handle_request(method="POST", path="/v3/context", payload=payload)
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.get("/v1/memories")
    async def list_memories(request: Request) -> dict[str, Any]:
        status, response = service.handle_request(
            method="GET",
            path="/v1/memories",
            query={key: request.query_params.getlist(key) for key in request.query_params.keys()},
        )
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.get("/v1/memories/{memory_id}")
    async def get_memory(memory_id: str) -> dict[str, Any]:
        status, response = service.handle_request(method="GET", path=f"/v1/memories/{memory_id}")
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.patch("/v1/memories/{memory_id}")
    async def update_memory(memory_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        status, response = service.handle_request(method="PATCH", path=f"/v1/memories/{memory_id}", payload=payload)
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.delete("/v1/memories/{memory_id}")
    async def delete_memory(memory_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        status, response = service.handle_request(method="DELETE", path=f"/v1/memories/{memory_id}", payload=payload or {})
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.get("/v1/memories/{memory_id}/history")
    async def memory_history(memory_id: str) -> dict[str, Any]:
        status, response = service.handle_request(method="GET", path=f"/v1/memories/{memory_id}/history")
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.post("/v1/export")
    async def export_memories() -> dict[str, Any]:
        status, response = service.handle_request(method="POST", path="/v1/export")
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.post("/v1/import")
    async def import_memories(payload: dict[str, Any]) -> dict[str, Any]:
        status, response = service.handle_request(method="POST", path="/v1/import", payload=payload)
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        status, response = service.handle_request(method="POST", path="/v1/chat/completions", payload=payload)
        if status >= 400:
            raise HTTPException(status_code=status, detail=response)
        return response

    return app
