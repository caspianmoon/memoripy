from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MemoryAction(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"
    DELETE = "DELETE"
    NONE = "NONE"


class MemoryState(str, Enum):
    ACTIVE = "active"
    PENDING = "pending"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class MemoryKind(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    PROFILE_ATTRIBUTE = "profile_attribute"
    EPISODIC_SUMMARY = "episodic_summary"
    ENTITY = "entity"
    RELATION = "relation"


class MemoryLayer(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class EventType(str, Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ASSISTANT_ACTION = "assistant_action"
    INGESTION = "ingestion"


@dataclass(frozen=True)
class MemoryScope:
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MemoryScope":
        payload = payload or {}
        return cls(
            user_id=payload.get("user_id"),
            agent_id=payload.get("agent_id"),
            run_id=payload.get("run_id"),
        )

    def matches(self, other: "MemoryScope | None") -> bool:
        if other is None:
            return True
        for field_name in ("user_id", "agent_id", "run_id"):
            expected = getattr(other, field_name)
            actual = getattr(self, field_name)
            if expected is not None and expected != actual:
                return False
        return True

    def is_empty(self) -> bool:
        return not any((self.user_id, self.agent_id, self.run_id))


@dataclass
class ProjectionStatus:
    lexical_current: bool = True
    vector_current: bool = False
    graph_current: bool = True
    last_projected_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ProjectionStatus":
        payload = payload or {}
        return cls(
            lexical_current=bool(payload.get("lexical_current", True)),
            vector_current=bool(payload.get("vector_current", False)),
            graph_current=bool(payload.get("graph_current", True)),
            last_projected_at=payload.get("last_projected_at"),
        )


@dataclass
class IngestionItem:
    content: str = ""
    modality: str = Modality.TEXT.value
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    asset_ref: str | None = None
    event_type: str = EventType.INGESTION.value
    name: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    occurred_at: str | None = None
    source_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "IngestionItem":
        payload = payload or {}
        return cls(
            content=payload.get("content", ""),
            modality=payload.get("modality", Modality.TEXT.value),
            role=payload.get("role"),
            metadata=dict(payload.get("metadata") or {}),
            asset_ref=payload.get("asset_ref"),
            event_type=payload.get("event_type", EventType.INGESTION.value),
            name=payload.get("name"),
            attributes=dict(payload.get("attributes") or {}),
            occurred_at=payload.get("occurred_at"),
            source_type=payload.get("source_type"),
        )


@dataclass
class EvidenceItem:
    evidence_id: str
    content_hash: str
    modality: str
    text: str
    role: str | None
    metadata: dict[str, Any]
    asset_ref: str | None
    scope: MemoryScope
    created_at: str
    event_type: str = EventType.INGESTION.value
    name: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    occurred_at: str | None = None
    source_type: str = EventType.INGESTION.value

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.scope.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceItem":
        return cls(
            evidence_id=payload["evidence_id"],
            content_hash=payload["content_hash"],
            modality=payload["modality"],
            text=payload.get("text", ""),
            role=payload.get("role"),
            metadata=dict(payload.get("metadata") or {}),
            asset_ref=payload.get("asset_ref"),
            scope=MemoryScope.from_dict(payload.get("scope")),
            created_at=payload["created_at"],
            event_type=payload.get("event_type", EventType.INGESTION.value),
            name=payload.get("name"),
            attributes=dict(payload.get("attributes") or {}),
            occurred_at=payload.get("occurred_at"),
            source_type=payload.get("source_type", payload.get("event_type", EventType.INGESTION.value)),
        )


@dataclass
class MemoryVersion:
    version_id: str
    record_id: str
    action: str
    state: str
    value: str
    summary: str
    created_at: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    reasoning_trace: list[str] = field(default_factory=list)
    supersedes_version_id: str | None = None
    salience: float = 0.0
    source_type: str = EventType.INGESTION.value
    layer: str = MemoryLayer.SEMANTIC.value
    citation_evidence_ids: list[str] = field(default_factory=list)
    contradicted_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryVersion":
        return cls(
            version_id=payload["version_id"],
            record_id=payload["record_id"],
            action=payload["action"],
            state=payload["state"],
            value=payload.get("value", ""),
            summary=payload.get("summary", ""),
            created_at=payload["created_at"],
            confidence=float(payload.get("confidence", 0.0)),
            evidence_ids=list(payload.get("evidence_ids") or []),
            metadata=dict(payload.get("metadata") or {}),
            reasoning_trace=list(payload.get("reasoning_trace") or []),
            supersedes_version_id=payload.get("supersedes_version_id"),
            salience=float(payload.get("salience", 0.0)),
            source_type=payload.get("source_type", EventType.INGESTION.value),
            layer=payload.get("layer", MemoryLayer.SEMANTIC.value),
            citation_evidence_ids=list(payload.get("citation_evidence_ids") or payload.get("evidence_ids") or []),
            contradicted_by=list(payload.get("contradicted_by") or []),
        )


@dataclass
class MemoryRecord:
    record_id: str
    kind: str
    key: str
    summary: str
    value: str
    state: str
    scope: MemoryScope
    current_version_id: str | None
    version_ids: list[str]
    evidence_ids: list[str]
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed_at: str | None = None
    search_text: str = ""
    embedding: list[float] = field(default_factory=list)
    confidence: float = 0.0
    salience: float = 0.0
    source_type: str = EventType.INGESTION.value
    layer: str = MemoryLayer.SEMANTIC.value
    confirmation_count: int = 1
    last_confirmed_at: str | None = None
    contradicted_by: list[str] = field(default_factory=list)
    citation_evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.scope.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        return cls(
            record_id=payload["record_id"],
            kind=payload["kind"],
            key=payload["key"],
            summary=payload.get("summary", ""),
            value=payload.get("value", ""),
            state=payload["state"],
            scope=MemoryScope.from_dict(payload.get("scope")),
            current_version_id=payload.get("current_version_id"),
            version_ids=list(payload.get("version_ids") or []),
            evidence_ids=list(payload.get("evidence_ids") or []),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            metadata=dict(payload.get("metadata") or {}),
            tags=list(payload.get("tags") or []),
            entity_names=list(payload.get("entity_names") or []),
            access_count=int(payload.get("access_count", 0)),
            last_accessed_at=payload.get("last_accessed_at"),
            search_text=payload.get("search_text", ""),
            embedding=list(payload.get("embedding") or []),
            confidence=float(payload.get("confidence", 0.0)),
            salience=float(payload.get("salience", payload.get("confidence", 0.0))),
            source_type=payload.get("source_type", EventType.INGESTION.value),
            layer=payload.get("layer", MemoryLayer.SEMANTIC.value),
            confirmation_count=int(payload.get("confirmation_count", 1)),
            last_confirmed_at=payload.get("last_confirmed_at", payload.get("updated_at")),
            contradicted_by=list(payload.get("contradicted_by") or []),
            citation_evidence_ids=list(payload.get("citation_evidence_ids") or payload.get("evidence_ids") or []),
        )


@dataclass
class RelationEdge:
    edge_id: str
    source_record_id: str
    target_record_id: str
    relation_type: str
    weight: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RelationEdge":
        return cls(
            edge_id=payload["edge_id"],
            source_record_id=payload["source_record_id"],
            target_record_id=payload["target_record_id"],
            relation_type=payload["relation_type"],
            weight=float(payload.get("weight", 0.0)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class SearchFilters:
    scope: MemoryScope | None = None
    kinds: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    layers: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)
    include_pending: bool = False
    limit: int = 5
    hierarchical_scope: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.scope.to_dict() if self.scope else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SearchFilters":
        payload = payload or {}
        scope_payload = payload.get("scope")
        return cls(
            scope=MemoryScope.from_dict(scope_payload) if scope_payload else None,
            kinds=list(payload.get("kinds") or []),
            states=list(payload.get("states") or []),
            tags=list(payload.get("tags") or []),
            metadata=dict(payload.get("metadata") or {}),
            layers=list(payload.get("layers") or []),
            source_types=list(payload.get("source_types") or []),
            include_pending=bool(payload.get("include_pending", False)),
            limit=int(payload.get("limit", 5)),
            hierarchical_scope=bool(payload.get("hierarchical_scope", True)),
        )


@dataclass
class SearchResult:
    memory: MemoryRecord
    score: float
    rank_breakdown: dict[str, float]
    evidence: list[EvidenceItem]
    projection_status: ProjectionStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "rank_breakdown": dict(self.rank_breakdown),
            "evidence": [item.to_dict() for item in self.evidence],
            "projection_status": self.projection_status.to_dict(),
        }


@dataclass
class ContextPack:
    query: str
    scope: MemoryScope
    intent: str
    profile: list[dict[str, Any]] = field(default_factory=list)
    preferences: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    recent_episodes: list[dict[str, Any]] = field(default_factory=list)
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    projection_status: ProjectionStatus = field(default_factory=ProjectionStatus)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope.to_dict(),
            "intent": self.intent,
            "profile": self.profile,
            "preferences": self.preferences,
            "relationships": self.relationships,
            "recent_episodes": self.recent_episodes,
            "tool_observations": self.tool_observations,
            "citations": self.citations,
            "projection_status": self.projection_status.to_dict(),
            "debug": self.debug,
        }
