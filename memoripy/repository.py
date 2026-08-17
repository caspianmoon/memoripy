from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .types import EvidenceItem, MemoryRecord, MemoryVersion, ProjectionStatus, RelationEdge
from .utils import (
    append_jsonl,
    atomic_copy,
    atomic_write_json,
    deep_copy_json,
    file_lock,
    generate_id,
    payload_checksum,
    read_json_file,
    utc_now,
)


class MemoryRepositoryError(RuntimeError):
    pass


class MemoryCorruptionError(MemoryRepositoryError):
    def __init__(self, message: str, *, state_path: str | None = None, backup_path: str | None = None):
        super().__init__(message)
        self.state_path = state_path
        self.backup_path = backup_path


@dataclass
class EngineState:
    schema_version: int = 4
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
            "activation": {},
            "temporal": {},
            "consolidation": {},
            "status": ProjectionStatus().to_dict(),
        }
    )
    admission_log: list[dict[str, Any]] = field(default_factory=list)

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
            "admission_log": deep_copy_json(self.admission_log),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EngineState":
        payload = payload or {}
        projections = deep_copy_json(payload.get("projections") or {})
        projections.setdefault("lexical", {})
        projections.setdefault("graph", {})
        projections.setdefault("activation", {})
        projections.setdefault("temporal", {})
        projections.setdefault("consolidation", {})
        projections.setdefault("status", ProjectionStatus().to_dict())
        state = cls(
            schema_version=max(int(payload.get("schema_version", 2)), 4),
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
            projections=projections,
            admission_log=list(payload.get("admission_log") or []),
        )
        state.validate()
        return state

    def validate(self) -> None:
        for record_id, record in self.memories.items():
            if record.record_id != record_id:
                raise MemoryCorruptionError(f"Memory record key mismatch for {record_id}")
            if record.current_version_id and record.current_version_id not in self.versions:
                raise MemoryCorruptionError(
                    f"Memory {record_id} references missing current version {record.current_version_id}"
                )
            for version_id in record.version_ids:
                if version_id not in self.versions:
                    raise MemoryCorruptionError(f"Memory {record_id} references missing version {version_id}")
        for version_id, version in self.versions.items():
            if version.version_id != version_id:
                raise MemoryCorruptionError(f"Memory version key mismatch for {version_id}")
            if version.record_id not in self.memories:
                raise MemoryCorruptionError(
                    f"Memory version {version_id} references missing record {version.record_id}"
                )
        for evidence_id, evidence in self.evidence.items():
            if evidence.evidence_id != evidence_id:
                raise MemoryCorruptionError(f"Evidence key mismatch for {evidence_id}")


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

    def recover_from_backup(self) -> dict[str, Any]:
        raise MemoryRepositoryError("This repository does not expose a backup")


class InMemoryRepository(BaseRepository):
    def __init__(self, initial_state: EngineState | None = None):
        self._lock = threading.RLock()
        self._state = EngineState.from_dict((initial_state or EngineState()).to_dict())

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
            key = f"{operation_name}:{idempotency_key}" if idempotency_key else None
            if key and key in self._state.idempotency:
                return deep_copy_json(self._state.idempotency[key]["result"])
            working_state = EngineState.from_dict(self._state.to_dict())
            result, events = operation(working_state)
            if key:
                working_state.idempotency[key] = {
                    "result": deep_copy_json(result),
                    "events": deep_copy_json(events),
                }
            working_state.validate()
            self._state = working_state
            return deep_copy_json(result)

    def replace_state(
        self,
        state: EngineState,
        operation_name: str = "replace_state",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state.validate()
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
        self.root_path = Path(root_path).expanduser()
        is_file_path = (
            (self.root_path.exists() and self.root_path.is_file())
            or (not self.root_path.exists() and self.root_path.suffix.casefold() == ".json")
        )
        if is_file_path:
            self.state_path = self.root_path
            self.base_dir = self.root_path.parent
            self.assets_dir = self.base_dir / f"{self.root_path.stem}_assets"
            self.events_path = self.base_dir / f"{self.root_path.stem}.events.jsonl"
            self.lock_path = self.base_dir / f"{self.root_path.stem}.lock"
            self.backup_path = self.base_dir / f"{self.root_path.name}.bak"
            self.journal_path = self.base_dir / f"{self.root_path.name}.journal"
        else:
            self.base_dir = self.root_path
            self.state_path = self.base_dir / "state.json"
            self.assets_dir = self.base_dir / "assets"
            self.events_path = self.base_dir / "events.jsonl"
            self.lock_path = self.base_dir / "state.lock"
            self.backup_path = self.base_dir / "state.json.bak"
            self.journal_path = self.base_dir / "transaction.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def _read_state_file(self, path: Path) -> EngineState:
        try:
            payload = read_json_file(path)
            return EngineState.from_dict(payload)
        except (json.JSONDecodeError, OSError, ValueError, KeyError, TypeError, MemoryCorruptionError) as exc:
            raise MemoryCorruptionError(
                f"Memoripy refused to replace or ignore corrupt state at {path}: {exc}",
                state_path=str(path),
                backup_path=str(self.backup_path) if self.backup_path.exists() else None,
            ) from exc

    def _read_state_unlocked(self) -> EngineState:
        self._recover_journal_unlocked()
        if not self.state_path.exists():
            return EngineState()
        return self._read_state_file(self.state_path)

    def _event_tx_seen(self, tx_id: str) -> bool:
        if not self.events_path.exists():
            return False
        try:
            with self.events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if f'"tx_id": "{tx_id}"' in line or f'"tx_id":"{tx_id}"' in line:
                        return True
        except OSError:
            return False
        return False

    def _append_events(self, *, tx_id: str, operation_name: str, idempotency_key: str | None, events: list[dict[str, Any]]) -> None:
        if self._event_tx_seen(tx_id):
            return
        timestamp = utc_now()
        for index, event in enumerate(events):
            append_jsonl(
                self.events_path,
                {
                    "tx_id": tx_id,
                    "event_index": index,
                    "timestamp": timestamp,
                    "operation": operation_name,
                    "idempotency_key": idempotency_key,
                    **deep_copy_json(event),
                },
            )

    def _recover_journal_unlocked(self) -> None:
        if not self.journal_path.exists():
            return
        try:
            journal = read_json_file(self.journal_path)
            next_state_payload = journal.get("next_state")
            if not isinstance(next_state_payload, dict):
                raise ValueError("journal does not contain next_state")
            expected = str(journal.get("checksum", ""))
            if payload_checksum(next_state_payload) != expected:
                raise ValueError("journal checksum mismatch")
            recovered = EngineState.from_dict(next_state_payload)
            atomic_write_json(self.state_path, recovered.to_dict())
            self._append_events(
                tx_id=str(journal.get("tx_id", generate_id("tx"))),
                operation_name=str(journal.get("operation", "recovery")),
                idempotency_key=journal.get("idempotency_key"),
                events=list(journal.get("events") or []),
            )
            self.journal_path.unlink(missing_ok=True)
        except Exception as exc:
            raise MemoryCorruptionError(
                f"Memoripy found an incomplete transaction journal that could not be recovered: {exc}",
                state_path=str(self.state_path),
                backup_path=str(self.backup_path) if self.backup_path.exists() else None,
            ) from exc

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

            working_state = EngineState.from_dict(state.to_dict())
            result, events = operation(working_state)
            if key:
                working_state.idempotency[key] = {
                    "result": deep_copy_json(result),
                    "events": deep_copy_json(events),
                }
            working_state.validate()

            tx_id = generate_id("tx")
            journal = {
                "tx_id": tx_id,
                "operation": operation_name,
                "idempotency_key": idempotency_key,
                "created_at": utc_now(),
                "checksum": payload_checksum(working_state.to_dict()),
                "next_state": working_state.to_dict(),
                "events": deep_copy_json(events),
            }
            atomic_write_json(self.journal_path, journal)
            if self.state_path.exists():
                atomic_copy(self.state_path, self.backup_path)
            atomic_write_json(self.state_path, working_state.to_dict())
            if not self.backup_path.exists():
                atomic_copy(self.state_path, self.backup_path)
            self._append_events(
                tx_id=tx_id,
                operation_name=operation_name,
                idempotency_key=idempotency_key,
                events=events,
            )
            self.journal_path.unlink(missing_ok=True)
            return deep_copy_json(result)

    def replace_state(
        self,
        state: EngineState,
        operation_name: str = "replace_state",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        state.validate()

        def operation(_: EngineState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            result = {
                "status": "ok",
                "operation": operation_name,
                "schema_version": state.schema_version,
                "replaced_at": utc_now(),
            }
            return result, [{"type": "state_replaced", "schema_version": state.schema_version}]

        with file_lock(self.lock_path):
            current = self._read_state_unlocked()
            key = f"{operation_name}:{idempotency_key}" if idempotency_key else None
            if key and key in current.idempotency:
                return deep_copy_json(current.idempotency[key]["result"])
            result, events = operation(current)
            replacement = EngineState.from_dict(state.to_dict())
            if key:
                replacement.idempotency[key] = {"result": deep_copy_json(result), "events": events}
            tx_id = generate_id("tx")
            journal = {
                "tx_id": tx_id,
                "operation": operation_name,
                "idempotency_key": idempotency_key,
                "created_at": utc_now(),
                "checksum": payload_checksum(replacement.to_dict()),
                "next_state": replacement.to_dict(),
                "events": events,
            }
            atomic_write_json(self.journal_path, journal)
            if self.state_path.exists():
                atomic_copy(self.state_path, self.backup_path)
            atomic_write_json(self.state_path, replacement.to_dict())
            if not self.backup_path.exists():
                atomic_copy(self.state_path, self.backup_path)
            self._append_events(
                tx_id=tx_id,
                operation_name=operation_name,
                idempotency_key=idempotency_key,
                events=events,
            )
            self.journal_path.unlink(missing_ok=True)
            return result

    def recover_from_backup(self) -> dict[str, Any]:
        with file_lock(self.lock_path):
            if not self.backup_path.exists():
                raise MemoryRepositoryError(f"No backup exists at {self.backup_path}")
            backup_state = self._read_state_file(self.backup_path)
            if self.state_path.exists():
                corrupt_copy = self.base_dir / f"{self.state_path.name}.corrupt-{generate_id('snapshot')}"
                atomic_copy(self.state_path, corrupt_copy)
            atomic_write_json(self.state_path, backup_state.to_dict())
            return {
                "status": "recovered",
                "state_path": str(self.state_path),
                "backup_path": str(self.backup_path),
                "schema_version": backup_state.schema_version,
                "recovered_at": utc_now(),
            }
