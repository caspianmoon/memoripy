from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import MemoryClient


@dataclass
class MCPAccessPolicy:
    read_only: bool = False
    scope_defaults: dict[str, str] = field(default_factory=dict)


class MemoripyMCPTools:
    """Transport-independent implementation behind the official MCP tools."""

    def __init__(self, client: MemoryClient, *, policy: MCPAccessPolicy | None = None) -> None:
        self.client = client
        self.policy = policy or MCPAccessPolicy()

    def capture(
        self,
        text: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        project_id: str | None = None,
        organization_id: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        self._require_write()
        return self.client.capture(
            messages=[{"role": "user", "content": text}],
            **self._scope(
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                project_id=project_id,
                organization_id=organization_id,
                namespace=namespace,
            ),
        )

    def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        project_id: str | None = None,
        organization_id: str | None = None,
        namespace: str | None = None,
        limit: int = 5,
        include_historical: bool = False,
    ) -> dict[str, Any]:
        return self.client.search(
            query=query,
            limit=limit,
            include_historical=include_historical,
            include_trace=True,
            **self._scope(
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                project_id=project_id,
                organization_id=organization_id,
                namespace=namespace,
            ),
        )

    def explain(self, memory_id: str) -> dict[str, Any]:
        return self.client.explain(memory_id=memory_id)

    def correct(self, memory_id: str, value: str, reason: str = "Corrected through MCP") -> dict[str, Any]:
        self._require_write()
        return self.client.correct(memory_id=memory_id, value=value, reason=reason)

    def forget(self, memory_id: str) -> dict[str, Any]:
        self._require_write()
        return self.client.forget(memory_id=memory_id)

    def audit(self) -> dict[str, Any]:
        return self.client.audit().to_dict()

    def list_memories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str | None = None,
        organization_id: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        return self.client.get_all(
            **self._scope(
                user_id=user_id,
                agent_id=agent_id,
                project_id=project_id,
                organization_id=organization_id,
                namespace=namespace,
            )
        )

    def _require_write(self) -> None:
        if self.policy.read_only:
            raise PermissionError("This Memoripy MCP server is read-only")

    def _scope(self, **values: str | None) -> dict[str, str | None]:
        scope: dict[str, str | None] = {key: value for key, value in values.items() if value is not None}
        # Configured defaults are policy constraints, not user-overridable suggestions.
        scope.update(self.policy.scope_defaults)
        return scope


def build_mcp_server(
    *,
    client: MemoryClient | None = None,
    path: str | Path | None = None,
    read_only: bool = False,
    scope_defaults: dict[str, str] | None = None,
    tokens: dict[str, dict[str, Any]] | None = None,
    issuer_url: str = "https://memoripy.local",
    resource_server_url: str = "http://127.0.0.1:8000/mcp",
):
    """Build the official MCP v2 server.

    ``mcp`` remains an optional dependency. Stdio can run without bearer
    tokens. Network transports should pass ``tokens`` so the MCP SDK exposes
    protected-resource metadata and validates bearer tokens.
    """

    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise RuntimeError('Install the MCP extra with: pip install "memoripy[mcp]"') from exc

    resolved_client = client or (MemoryClient.from_path(path) if path is not None else MemoryClient())
    tools = MemoripyMCPTools(
        resolved_client,
        policy=MCPAccessPolicy(read_only=read_only, scope_defaults=dict(scope_defaults or {})),
    )
    kwargs: dict[str, Any] = {}
    if tokens:
        from mcp.server.auth.settings import AuthSettings

        kwargs["auth"] = AuthSettings(
            issuer_url=issuer_url,
            resource_server_url=resource_server_url,
            required_scopes=["memoripy:read"],
        )
        kwargs["token_verifier"] = _static_token_verifier(tokens)

    server = MCPServer(
        "memoripy",
        title="Memoripy",
        description="Evidence-first, temporal memory for AI agents with receipts.",
        instructions=(
            "Use memoripy_recall before asking the user to repeat durable facts. "
            "Use memoripy_capture only for explicit user statements or trusted observations. "
            "Use memoripy_explain when provenance matters."
        ),
        version="0.4.0",
        **kwargs,
    )

    def require_scope(scope: str) -> None:
        if not tokens:
            return
        from mcp.server.auth.middleware.auth_context import get_access_token

        access_token = get_access_token()
        scopes = set(access_token.scopes if access_token is not None else [])
        if access_token is None or (scope not in scopes and "memoripy:admin" not in scopes):
            raise PermissionError(f"MCP token lacks required scope: {scope}")

    def authenticated_scope(
        user_id: str | None,
        organization_id: str | None,
    ) -> tuple[str | None, str | None]:
        if not tokens:
            return user_id, organization_id
        from mcp.server.auth.middleware.auth_context import get_access_token

        access_token = get_access_token()
        if access_token is None:
            return user_id, organization_id
        claims = dict(access_token.claims or {})
        locked_user = access_token.subject or user_id
        locked_tenant = claims.get("tenant_id") or organization_id
        return locked_user, str(locked_tenant) if locked_tenant is not None else None

    @server.tool(name="memoripy_capture", description="Capture an explicit user statement as evidence-first memory.")
    def capture(
        text: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        project_id: str | None = None,
        organization_id: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        require_scope("memoripy:write")
        user_id, organization_id = authenticated_scope(user_id, organization_id)
        return tools.capture(
            text,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            project_id=project_id,
            organization_id=organization_id,
            namespace=namespace,
        )

    @server.tool(name="memoripy_recall", description="Recall relevant memory with evidence and retrieval receipts.")
    def recall(
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        project_id: str | None = None,
        organization_id: str | None = None,
        namespace: str | None = None,
        limit: int = 5,
        include_historical: bool = False,
    ) -> dict[str, Any]:
        require_scope("memoripy:read")
        user_id, organization_id = authenticated_scope(user_id, organization_id)
        return tools.recall(
            query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            project_id=project_id,
            organization_id=organization_id,
            namespace=namespace,
            limit=limit,
            include_historical=include_historical,
        )

    @server.tool(name="memoripy_explain", description="Explain a memory's evidence, trust, temporal validity, and version history.")
    def explain(memory_id: str) -> dict[str, Any]:
        require_scope("memoripy:read")
        return tools.explain(memory_id)

    @server.tool(name="memoripy_correct", description="Correct a memory while preserving its prior version and evidence trail.")
    def correct(memory_id: str, value: str, reason: str = "Corrected through MCP") -> dict[str, Any]:
        require_scope("memoripy:write")
        return tools.correct(memory_id, value, reason)

    @server.tool(name="memoripy_forget", description="Version-delete a memory without silently destroying its audit trail.")
    def forget(memory_id: str) -> dict[str, Any]:
        require_scope("memoripy:write")
        return tools.forget(memory_id)

    @server.tool(name="memoripy_audit", description="Audit the memory store for pollution, conflicts, and provenance gaps.")
    def audit() -> dict[str, Any]:
        require_scope("memoripy:read")
        return tools.audit()

    @server.tool(name="memoripy_list", description="List scoped memories for inspection.")
    def list_memories(
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str | None = None,
        organization_id: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        require_scope("memoripy:read")
        user_id, organization_id = authenticated_scope(user_id, organization_id)
        return tools.list_memories(
            user_id=user_id,
            agent_id=agent_id,
            project_id=project_id,
            organization_id=organization_id,
            namespace=namespace,
        )

    return server


def run_mcp_server(
    *,
    path: str | Path,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    read_only: bool = False,
    token_file: str | Path | None = None,
    scope_defaults: dict[str, str] | None = None,
) -> None:
    tokens = load_token_file(token_file) if token_file else None
    if transport != "stdio" and not tokens:
        raise ValueError("Network MCP transports require --token-file; unauthenticated network memory is refused")
    server = build_mcp_server(
        path=path,
        read_only=read_only,
        scope_defaults=scope_defaults,
        tokens=tokens,
        resource_server_url=f"http://{host}:{port}/mcp",
    )
    kwargs = {"host": host, "port": port} if transport != "stdio" else {}
    server.run(transport=transport, **kwargs)


def load_token_file(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "tokens" in payload:
        payload = payload["tokens"]
    if not isinstance(payload, dict):
        raise ValueError("MCP token file must contain a token-to-principal object")
    output: dict[str, dict[str, Any]] = {}
    for token, record in payload.items():
        if not isinstance(record, dict):
            record = {"subject": str(record)}
        output[str(token)] = {
            "client_id": str(record.get("client_id") or record.get("subject") or "memoripy-client"),
            "subject": record.get("subject"),
            "tenant_id": record.get("tenant_id"),
            "scopes": list(record.get("scopes") or ["memoripy:read", "memoripy:write"]),
            "expires_at": record.get("expires_at"),
        }
    return output


def _static_token_verifier(tokens: dict[str, dict[str, Any]]):
    from mcp.server.auth.provider import AccessToken

    class StaticTokenVerifier:
        async def verify_token(self, token: str):
            record = tokens.get(token)
            if record is None:
                return None
            return AccessToken(
                token=token,
                client_id=str(record.get("client_id") or "memoripy-client"),
                scopes=list(record.get("scopes") or ["memoripy:read"]),
                expires_at=record.get("expires_at"),
                subject=record.get("subject"),
                claims={
                    "iss": "memoripy-static-token-file",
                    **({"tenant_id": record.get("tenant_id")} if record.get("tenant_id") else {}),
                },
            )

    return StaticTokenVerifier()
