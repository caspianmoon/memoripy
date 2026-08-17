from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import MemoryClient
from .pipeline import MemoryPipelineConfig


READ_SCOPE = "memoripy:read"
WRITE_SCOPE = "memoripy:write"
ADMIN_SCOPE = "memoripy:admin"
ALL_SCOPES = {READ_SCOPE, WRITE_SCOPE, ADMIN_SCOPE}


@dataclass(frozen=True)
class TenantPrincipal:
    tenant_id: str
    key_id: str
    scopes: frozenset[str]

    def allows(self, scope: str) -> bool:
        return ADMIN_SCOPE in self.scopes or scope in self.scopes


@dataclass
class ApiKeyRecord:
    key_id: str
    tenant_id: str
    digest: str
    scopes: list[str] = field(default_factory=lambda: [READ_SCOPE, WRITE_SCOPE])
    created_at: str = field(default_factory=lambda: _now())
    expires_at: str | None = None
    disabled: bool = False
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ApiKeyRecord":
        return cls(
            key_id=str(payload["key_id"]),
            tenant_id=str(payload["tenant_id"]),
            digest=str(payload["digest"]),
            scopes=[str(item) for item in payload.get("scopes") or [READ_SCOPE]],
            created_at=str(payload.get("created_at") or _now()),
            expires_at=payload.get("expires_at"),
            disabled=bool(payload.get("disabled", False)),
            label=payload.get("label"),
        )


class TenantRegistry:
    """Hashed API-key registry for the hosted gateway.

    Plaintext tokens are returned only when created. Registry files contain a
    SHA-256 digest, never the reusable bearer token.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._records: dict[str, ApiKeyRecord] = {}
        if self.path and self.path.exists():
            self._load()

    def create_key(
        self,
        *,
        tenant_id: str,
        scopes: list[str] | None = None,
        label: str | None = None,
        expires_at: str | None = None,
    ) -> tuple[str, ApiKeyRecord]:
        tenant_id = _validate_tenant_id(tenant_id)
        resolved_scopes = list(dict.fromkeys(scopes or [READ_SCOPE, WRITE_SCOPE]))
        unknown = set(resolved_scopes).difference(ALL_SCOPES)
        if unknown:
            raise ValueError(f"Unknown scopes: {sorted(unknown)}")
        key_id = secrets.token_hex(8)
        token = f"mry_{key_id}_{secrets.token_urlsafe(32)}"
        record = ApiKeyRecord(
            key_id=key_id,
            tenant_id=tenant_id,
            digest=_digest(token),
            scopes=resolved_scopes,
            expires_at=expires_at,
            label=label,
        )
        with self._lock:
            self._records[key_id] = record
            self._save()
        return token, record

    def authenticate(self, token: str | None, *, required_scope: str = READ_SCOPE) -> TenantPrincipal | None:
        if not token:
            return None
        key_id = _key_id(token)
        if not key_id:
            return None
        with self._lock:
            record = self._records.get(key_id)
            if record is None or record.disabled or _expired(record.expires_at):
                return None
            if not hmac.compare_digest(record.digest, _digest(token)):
                return None
            principal = TenantPrincipal(record.tenant_id, record.key_id, frozenset(record.scopes))
            return principal if principal.allows(required_scope) else None

    def revoke(self, key_id: str) -> bool:
        with self._lock:
            record = self._records.get(key_id)
            if record is None:
                return False
            record.disabled = True
            self._save()
            return True

    def list_keys(self, *, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._records.values())
        if tenant_id is not None:
            records = [record for record in records if record.tenant_id == tenant_id]
        output = []
        for record in sorted(records, key=lambda item: item.created_at):
            payload = record.to_dict()
            payload.pop("digest", None)
            output.append(payload)
        return output

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"version": 1, "keys": [record.to_dict() for record in self._records.values()]}

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        records = payload.get("keys", payload if isinstance(payload, list) else [])
        self._records = {record.key_id: record for record in map(ApiKeyRecord.from_dict, records)}

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)


class TenantStoreManager:
    """Creates physically isolated file stores per tenant."""

    def __init__(self, root: str | Path, *, pipeline: MemoryPipelineConfig | None = None) -> None:
        self.root = Path(root)
        self.pipeline = pipeline
        self.tenants_dir = self.root / "tenants"
        self.tenants_dir.mkdir(parents=True, exist_ok=True)
        self._clients: dict[str, MemoryClient] = {}
        self._lock = threading.RLock()

    def client(self, tenant_id: str) -> MemoryClient:
        tenant_id = _validate_tenant_id(tenant_id)
        with self._lock:
            if tenant_id not in self._clients:
                store_path = self.tenant_path(tenant_id)
                store_path.mkdir(parents=True, exist_ok=True)
                metadata_path = store_path / "tenant.json"
                if not metadata_path.exists():
                    metadata_path.write_text(
                        json.dumps({"tenant_id": tenant_id, "created_at": _now()}, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                self._clients[tenant_id] = MemoryClient.from_path(store_path / "memory", pipeline=self.pipeline)
            return self._clients[tenant_id]

    def tenant_path(self, tenant_id: str) -> Path:
        digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:24]
        return self.tenants_dir / digest


def bearer_token(headers: dict[str, str]) -> str | None:
    supplied = headers.get("authorization") or headers.get("Authorization") or ""
    if supplied.lower().startswith("bearer "):
        return supplied[7:].strip()
    return None


def _validate_tenant_id(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > 255:
        raise ValueError("tenant_id must contain between 1 and 255 characters")
    return normalized


def _key_id(token: str) -> str | None:
    parts = token.split("_", 2)
    if len(parts) != 3 or parts[0] != "mry" or not parts[1]:
        return None
    return parts[1]


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _expired(value: str | None) -> bool:
    if not value:
        return False
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed <= datetime.now(timezone.utc)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
