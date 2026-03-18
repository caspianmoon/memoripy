from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .types import EvidenceItem, MemoryRecord, MemoryVersion, ProjectionStatus, RelationEdge
from .utils import append_jsonl, atomic_write_json, deep_copy_json, file_lock, utc_now


@dataclass
class EngineState:
    schema_version: int = 3
    evidence: dict[str, EvidenceItem] = field(default_factory=dict)
    memories: dict[str, MemoryRecord] = field(default_factory=dict)
    versions: dict[str, MemoryVersion] = field(default_factory=dict)
    relations: dict[str, RelationEdge] = field(default_factory=dict)
    idempotency: dict[str, dict[str, Any]] = field(default_factory=dict)
    lookup: dict[str, str] = field(default_factory=dict)
    projections: dict[str, Any] = field(
        default_factory=lambda: {
            "lexical": {},
            "graph": {},
            "status": ProjectionStatus().to_dict(),
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence": {key: item.to_dict() for key, item in self.evidence.items()},
            "memories": {key: item.to_dict() for key, item in self.memories.items()},
            "versions": {key: item.to_dict() for key, item in self.versions.items()},
            "relations": {key: item.to_dict() for key, item in self.relations.items()},
            "idempotency": deep_copy_json(self.idempotency),
            "lookup": dict(self.lookup),
            "projections": deep_copy_json(self.projections),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EngineState":
        payload = payload or {}
        schema_version = max(int(payload.get("schema_version", 2)), 3)
        return cls(
            schema_version=schema_version,
            evidence={
                key: EvidenceItem.from_dict(value)
                for key, value in (payload.get("evidence") or {}).items()
            },
            memories={
                key: MemoryRecord.from_dict(value)
                for key, value in (payload.get("memories") or {}).items()
            },
            versions={
                key: MemoryVersion.from_dict(value)
                for key, value in (payload.get("versions") or {}).items()
            },
            relations={
                key: RelationEdge.from_dict(value)
                for key, value in (payload.get("relations") or {}).items()
            },
            idempotency=deep_copy_json(payload.get("idempotency") or {}),
            lookup=dict(payload.get("lookup") or {}),
            projections=deep_copy_json(
                payload.get("projections")
                or {"lexical": {}, "graph": {}, "status": ProjectionStatus().to_dict()}
            ),
        )


class BaseRepository:
    def load_state(self) -> EngineState:
        raise NotImplementedError

    def transaction(
        self,
        operation_name: str,
        idempotency_key: str | None,
        operation: Callable[[EngineState], tuple[Any, list[dict[str, Any]]]],
    ) -> Any:
        raise NotImplementedError

    def replace_state(
        self,
        state: EngineState,
        operation_name: str = "replace_state",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def export_state(self) -> dict[str, Any]:
        return self.load_state().to_dict()


class InMemoryRepository(BaseRepository):
    def __init__(self, initial_state: EngineState | None = None):
        self._lock = threading.RLock()
        self._state = initial_state or EngineState()

    def load_state(self) -> EngineState:
        with self._lock:
            return EngineState.from_dict(self._state.to_dict())

    def transaction(
        self,
        operation_name: str,
        idempotency_key: str | None,
        operation: Callable[[EngineState], tuple[Any, list[dict[str, Any]]]],
    ) -> Any:
        with self._lock:
            state = self._state
            key = f"{operation_name}:{idempotency_key}" if idempotency_key else None
            if key and key in state.idempotency:
                return deep_copy_json(state.idempotency[key]["result"])
            result, events = operation(state)
            if key:
                state.idempotency[key] = {"result": deep_copy_json(result), "events": deep_copy_json(events)}
            return deep_copy_json(result)

    def replace_state(
        self,
        state: EngineState,
        operation_name: str = "replace_state",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._state = EngineState.from_dict(state.to_dict())
            result = {
                "status": "ok",
                "operation": operation_name,
                "schema_version": self._state.schema_version,
                "replaced_at": utc_now(),
            }
            if idempotency_key:
                key = f"{operation_name}:{idempotency_key}"
                self._state.idempotency[key] = {"result": deep_copy_json(result), "events": []}
            return result


class FileMemoryRepository(BaseRepository):
    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path)
        if self.root_path.suffix:
            self.state_path = self.root_path
            self.base_dir = self.root_path.parent
            self.assets_dir = self.base_dir / f"{self.root_path.stem}_assets"
            self.events_path = self.base_dir / f"{self.root_path.stem}.events.jsonl"
            self.lock_path = self.base_dir / f"{self.root_path.stem}.lock"
        else:
            self.base_dir = self.root_path
            self.state_path = self.base_dir / "state.json"
            self.assets_dir = self.base_dir / "assets"
            self.events_path = self.base_dir / "events.jsonl"
            self.lock_path = self.base_dir / "state.lock"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def _read_state_unlocked(self) -> EngineState:
        if not self.state_path.exists():
            return EngineState()
        import json

        with self.state_path.open("r", encoding="utf-8") as handle:
            try:
                return EngineState.from_dict(json.load(handle))
            except json.JSONDecodeError:
                return EngineState()

    def load_state(self) -> EngineState:
        with file_lock(self.lock_path):
            return self._read_state_unlocked()

    def transaction(
        self,
        operation_name: str,
        idempotency_key: str | None,
        operation: Callable[[EngineState], tuple[Any, list[dict[str, Any]]]],
    ) -> Any:
        with file_lock(self.lock_path):
            state = self._read_state_unlocked()
            key = f"{operation_name}:{idempotency_key}" if idempotency_key else None
            if key and key in state.idempotency:
                return deep_copy_json(state.idempotency[key]["result"])

            result, events = operation(state)
            timestamp = utc_now()
            for event in events:
                event_payload = {
                    "timestamp": timestamp,
                    "operation": operation_name,
                    "idempotency_key": idempotency_key,
                    **deep_copy_json(event),
                }
                append_jsonl(self.events_path, event_payload)

            if key:
                state.idempotency[key] = {"result": deep_copy_json(result), "events": deep_copy_json(events)}
            atomic_write_json(self.state_path, state.to_dict())
            return deep_copy_json(result)

    def replace_state(
        self,
        state: EngineState,
        operation_name: str = "replace_state",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        with file_lock(self.lock_path):
            key = f"{operation_name}:{idempotency_key}" if idempotency_key else None
            current_state = self._read_state_unlocked()
            if key and key in current_state.idempotency:
                return deep_copy_json(current_state.idempotency[key]["result"])

            result = {
                "status": "ok",
                "operation": operation_name,
                "schema_version": state.schema_version,
                "replaced_at": utc_now(),
            }
            if key:
                state.idempotency[key] = {"result": deep_copy_json(result), "events": []}
            append_jsonl(
                self.events_path,
                {
                    "timestamp": utc_now(),
                    "operation": operation_name,
                    "idempotency_key": idempotency_key,
                    "type": "state_replaced",
                    "schema_version": state.schema_version,
                },
            )
            atomic_write_json(self.state_path, state.to_dict())
            return result
