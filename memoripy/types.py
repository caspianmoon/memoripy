from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MemoryAction(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"
    MERGE = "MERGE"
    DEFER = "DEFER"
    REJECT = "REJECT"
    QUARANTINE = "QUARANTINE"
    DELETE = "DELETE"
    NONE = "NONE"


class MemoryState(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    PENDING = "pending"
    SUPERSEDED = "superseded"
    QUARANTINED = "quarantined"
    DELETED = "deleted"


class MemoryKind(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    PROFILE_ATTRIBUTE = "profile_attribute"
    EPISODIC_SUMMARY = "episodic_summary"
    ENTITY = "entity"
    RELATION = "relation"
    POLICY = "policy"
    COMMITMENT = "commitment"
    PROCEDURE = "procedure"
    DECISION = "decision"
    BELIEF = "belief"
    ARTIFACT = "artifact"
    STATE = "state"
    CONSTRAINT = "constraint"


class MemoryLayer(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


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
    SYSTEM_INSTRUCTION = "system_instruction"
    EXTERNAL_DOCUMENT = "external_document"
    RETRIEVED_MEMORY = "retrieved_memory"
    GENERATED_SUMMARY = "generated_summary"
    IMPORTED_RECORD = "imported_record"
    EXPLICIT_WRITE = "explicit_write"
    INGESTION = "ingestion"


class SourceType(str, Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_INSTRUCTION = "system_instruction"
    TOOL_RESULT = "tool_result"
    TOOL_CALL = "tool_call"
    ASSISTANT_ACTION = "assistant_action"
    EXTERNAL_DOCUMENT = "external_document"
    RETRIEVED_MEMORY = "retrieved_memory"
    GENERATED_SUMMARY = "generated_summary"
    IMPORTED_RECORD = "imported_record"
    EXPLICIT_APPLICATION_WRITE = "explicit_application_write"
    UNKNOWN = "unknown"


class TrustLevel(str, Enum):
    AUTHORITATIVE = "authoritative"
    USER_STATED = "user_stated"
    OBSERVED = "observed"
    DERIVED = "derived"
    UNTRUSTED_EXTERNAL = "untrusted_external"
    QUARANTINED = "quarantined"


class Durability(str, Enum):
    EPHEMERAL = "ephemeral"
    SESSION = "session"
    DURABLE = "durable"
    PINNED = "pinned"


class AdmissionReason(str, Enum):
    ACCEPTED = "ACCEPTED"
    EXPLICIT_WRITE = "EXPLICIT_WRITE"
    SOURCE_NOT_TRUSTED = "SOURCE_NOT_TRUSTED"
    RETRIEVED_MEMORY_REINGESTION = "RETRIEVED_MEMORY_REINGESTION"
    TRANSIENT_STATE = "TRANSIENT_STATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DUPLICATE = "DUPLICATE"
    SUBJECT_AMBIGUOUS = "SUBJECT_AMBIGUOUS"
    CONTRADICTS_HIGHER_AUTHORITY = "CONTRADICTS_HIGHER_AUTHORITY"
    SENSITIVE_DATA = "SENSITIVE_DATA"
    SYSTEM_PROMPT_ECHO = "SYSTEM_PROMPT_ECHO"
    LOW_DURABILITY = "LOW_DURABILITY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNTRUSTED_INSTRUCTION = "UNTRUSTED_INSTRUCTION"
    ASSISTANT_SELF_REPORT = "ASSISTANT_SELF_REPORT"
    POLICY_REQUIRES_APPROVAL = "POLICY_REQUIRES_APPROVAL"


@dataclass(frozen=True)
class MemoryScope:
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    project_id: str | None = None
    organization_id: str | None = None
    namespace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MemoryScope":
        payload = payload or {}
        return cls(
            user_id=payload.get("user_id"),
            agent_id=payload.get("agent_id"),
            run_id=payload.get("run_id"),
            project_id=payload.get("project_id"),
            organization_id=payload.get("organization_id"),
            namespace=payload.get("namespace"),
        )

    def matches(self, other: "MemoryScope | None") -> bool:
        if other is None:
            return True
        for field_name in (
            "user_id",
            "agent_id",
            "run_id",
            "project_id",
            "organization_id",
            "namespace",
        ):
            expected = getattr(other, field_name)
            actual = getattr(self, field_name)
            if expected is not None and expected != actual:
                return False
        return True

    def is_empty(self) -> bool:
        return not any(asdict(self).values())

    def specificity(self) -> int:
        return sum(value is not None for value in asdict(self).values())


@dataclass
class ProjectionStatus:
    lexical_current: bool = True
    vector_current: bool = False
    graph_current: bool = True
    temporal_current: bool = True
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
            temporal_current=bool(payload.get("temporal_current", True)),
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
    writer_id: str | None = None
    trust_level: str | None = None
    source_uri: str | None = None
    is_retrieved_memory: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "IngestionItem":
        payload = payload or {}
        return cls(
            content=str(payload.get("content", "")),
            modality=str(payload.get("modality", Modality.TEXT.value)),
            role=payload.get("role"),
            metadata=dict(payload.get("metadata") or {}),
            asset_ref=payload.get("asset_ref"),
            event_type=str(payload.get("event_type", EventType.INGESTION.value)),
            name=payload.get("name"),
            attributes=dict(payload.get("attributes") or {}),
            occurred_at=payload.get("occurred_at"),
            source_type=payload.get("source_type"),
            writer_id=payload.get("writer_id"),
            trust_level=payload.get("trust_level"),
            source_uri=payload.get("source_uri"),
            is_retrieved_memory=bool(payload.get("is_retrieved_memory", False)),
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
    source_type: str = SourceType.UNKNOWN.value
    trust_level: str = TrustLevel.DERIVED.value
    writer_id: str | None = None
    source_uri: str | None = None
    evidence_spans: list[dict[str, Any]] = field(default_factory=list)
    is_retrieved_memory: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.scope.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceItem":
        event_type = str(payload.get("event_type", EventType.INGESTION.value))
        role = payload.get("role")
        source_type = payload.get("source_type") or _legacy_source_type(event_type, role)
        trust_level = payload.get("trust_level") or _default_trust(source_type)
        return cls(
            evidence_id=str(payload["evidence_id"]),
            content_hash=str(payload.get("content_hash", "")),
            modality=str(payload.get("modality", Modality.TEXT.value)),
            text=str(payload.get("text", "")),
            role=role,
            metadata=dict(payload.get("metadata") or {}),
            asset_ref=payload.get("asset_ref"),
            scope=MemoryScope.from_dict(payload.get("scope")),
            created_at=str(payload.get("created_at", "")),
            event_type=event_type,
            name=payload.get("name"),
            attributes=dict(payload.get("attributes") or {}),
            occurred_at=payload.get("occurred_at"),
            source_type=str(source_type),
            trust_level=str(trust_level),
            writer_id=payload.get("writer_id"),
            source_uri=payload.get("source_uri"),
            evidence_spans=list(payload.get("evidence_spans") or []),
            is_retrieved_memory=bool(payload.get("is_retrieved_memory", False)),
        )


@dataclass
class AdmissionDecision:
    action: str
    state: str
    reason_codes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    trust_level: str = TrustLevel.DERIVED.value
    durability: str = Durability.DURABLE.value
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AdmissionDecision":
        payload = payload or {}
        return cls(
            action=str(payload.get("action", MemoryAction.REJECT.value)),
            state=str(payload.get("state", MemoryState.PENDING.value)),
            reason_codes=list(payload.get("reason_codes") or []),
            confidence=float(payload.get("confidence", 0.0)),
            trust_level=str(payload.get("trust_level", TrustLevel.DERIVED.value)),
            durability=str(payload.get("durability", Durability.DURABLE.value)),
            metadata=dict(payload.get("metadata") or {}),
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
    superseded_by_version_id: str | None = None
    salience: float = 0.0
    source_type: str = SourceType.UNKNOWN.value
    trust_level: str = TrustLevel.DERIVED.value
    durability: str = Durability.DURABLE.value
    layer: str = MemoryLayer.SEMANTIC.value
    kind: str = MemoryKind.FACT.value
    subject: str | None = None
    observed_at: str | None = None
    recorded_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    citation_evidence_ids: list[str] = field(default_factory=list)
    contradicted_by: list[str] = field(default_factory=list)
    admission_reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryVersion":
        created_at = str(payload.get("created_at", ""))
        return cls(
            version_id=str(payload["version_id"]),
            record_id=str(payload["record_id"]),
            action=str(payload.get("action", MemoryAction.ADD.value)),
            state=str(payload.get("state", MemoryState.ACTIVE.value)),
            value=str(payload.get("value", "")),
            summary=str(payload.get("summary", "")),
            created_at=created_at,
            confidence=float(payload.get("confidence", 0.0)),
            evidence_ids=list(payload.get("evidence_ids") or []),
            metadata=dict(payload.get("metadata") or {}),
            reasoning_trace=list(payload.get("reasoning_trace") or []),
            supersedes_version_id=payload.get("supersedes_version_id"),
            superseded_by_version_id=payload.get("superseded_by_version_id"),
            salience=float(payload.get("salience", 0.0)),
            source_type=str(payload.get("source_type", SourceType.UNKNOWN.value)),
            trust_level=str(payload.get("trust_level", TrustLevel.DERIVED.value)),
            durability=str(payload.get("durability", Durability.DURABLE.value)),
            layer=str(payload.get("layer", MemoryLayer.SEMANTIC.value)),
            kind=str(payload.get("kind", MemoryKind.FACT.value)),
            subject=payload.get("subject"),
            observed_at=payload.get("observed_at") or created_at,
            recorded_at=payload.get("recorded_at") or created_at,
            valid_from=payload.get("valid_from") or payload.get("observed_at") or created_at,
            valid_to=payload.get("valid_to"),
            citation_evidence_ids=list(payload.get("citation_evidence_ids") or payload.get("evidence_ids") or []),
            contradicted_by=list(payload.get("contradicted_by") or []),
            admission_reason_codes=list(payload.get("admission_reason_codes") or []),
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
    retrieval_count: int = 0
    included_in_context_count: int = 0
    used_in_answer_count: int = 0
    confirmed_by_user_count: int = 0
    associated_success_count: int = 0
    corrected_count: int = 0
    rejected_count: int = 0
    caused_failure_count: int = 0
    last_accessed_at: str | None = None
    search_text: str = ""
    embedding: list[float] = field(default_factory=list)
    confidence: float = 0.0
    salience: float = 0.0
    source_type: str = SourceType.UNKNOWN.value
    trust_level: str = TrustLevel.DERIVED.value
    durability: str = Durability.DURABLE.value
    layer: str = MemoryLayer.SEMANTIC.value
    subject: str | None = None
    observed_at: str | None = None
    recorded_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    confirmation_count: int = 1
    last_confirmed_at: str | None = None
    contradicted_by: list[str] = field(default_factory=list)
    citation_evidence_ids: list[str] = field(default_factory=list)
    admission_reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.scope.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        created_at = str(payload.get("created_at", ""))
        updated_at = str(payload.get("updated_at", created_at))
        return cls(
            record_id=str(payload["record_id"]),
            kind=str(payload.get("kind", MemoryKind.FACT.value)),
            key=str(payload.get("key", "")),
            summary=str(payload.get("summary", "")),
            value=str(payload.get("value", "")),
            state=str(payload.get("state", MemoryState.ACTIVE.value)),
            scope=MemoryScope.from_dict(payload.get("scope")),
            current_version_id=payload.get("current_version_id"),
            version_ids=list(payload.get("version_ids") or []),
            evidence_ids=list(payload.get("evidence_ids") or []),
            created_at=created_at,
            updated_at=updated_at,
            metadata=dict(payload.get("metadata") or {}),
            tags=list(payload.get("tags") or []),
            entity_names=list(payload.get("entity_names") or []),
            access_count=int(payload.get("access_count", 0)),
            retrieval_count=int(payload.get("retrieval_count", payload.get("access_count", 0))),
            included_in_context_count=int(payload.get("included_in_context_count", 0)),
            used_in_answer_count=int(payload.get("used_in_answer_count", 0)),
            confirmed_by_user_count=int(payload.get("confirmed_by_user_count", 0)),
            associated_success_count=int(payload.get("associated_success_count", 0)),
            corrected_count=int(payload.get("corrected_count", 0)),
            rejected_count=int(payload.get("rejected_count", 0)),
            caused_failure_count=int(payload.get("caused_failure_count", 0)),
            last_accessed_at=payload.get("last_accessed_at"),
            search_text=str(payload.get("search_text", "")),
            embedding=[float(value) for value in (payload.get("embedding") or [])],
            confidence=float(payload.get("confidence", 0.0)),
            salience=float(payload.get("salience", payload.get("confidence", 0.0))),
            source_type=str(payload.get("source_type", SourceType.UNKNOWN.value)),
            trust_level=str(payload.get("trust_level", TrustLevel.DERIVED.value)),
            durability=str(payload.get("durability", Durability.DURABLE.value)),
            layer=str(payload.get("layer", MemoryLayer.SEMANTIC.value)),
            subject=payload.get("subject"),
            observed_at=payload.get("observed_at") or created_at,
            recorded_at=payload.get("recorded_at") or created_at,
            valid_from=payload.get("valid_from") or payload.get("observed_at") or created_at,
            valid_to=payload.get("valid_to"),
            confirmation_count=int(payload.get("confirmation_count", 1)),
            last_confirmed_at=payload.get("last_confirmed_at", updated_at),
            contradicted_by=list(payload.get("contradicted_by") or []),
            citation_evidence_ids=list(payload.get("citation_evidence_ids") or payload.get("evidence_ids") or []),
            admission_reason_codes=list(payload.get("admission_reason_codes") or []),
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
            edge_id=str(payload["edge_id"]),
            source_record_id=str(payload["source_record_id"]),
            target_record_id=str(payload["target_record_id"]),
            relation_type=str(payload.get("relation_type", "related")),
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
    trust_levels: list[str] = field(default_factory=list)
    durabilities: list[str] = field(default_factory=list)
    include_pending: bool = False
    include_historical: bool = False
    include_quarantined: bool = False
    as_of: str | None = None
    limit: int = 5
    hierarchical_scope: bool = True
    adaptive_scope: bool = True
    minimum_scope_results: int = 3
    track_usage: bool = True

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
            trust_levels=list(payload.get("trust_levels") or []),
            durabilities=list(payload.get("durabilities") or []),
            include_pending=bool(payload.get("include_pending", False)),
            include_historical=bool(payload.get("include_historical", False)),
            include_quarantined=bool(payload.get("include_quarantined", False)),
            as_of=payload.get("as_of"),
            limit=int(payload.get("limit", 5)),
            hierarchical_scope=bool(payload.get("hierarchical_scope", True)),
            adaptive_scope=bool(payload.get("adaptive_scope", True)),
            minimum_scope_results=int(payload.get("minimum_scope_results", 3)),
            track_usage=bool(payload.get("track_usage", True)),
        )


@dataclass
class RetrievalReceipt:
    memory_id: str
    included: bool
    retrieval_lanes: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_score: float = 0.0
    scope_tier: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    excluded_conflicts: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    current_at_query_time: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RetrievalReceipt":
        payload = payload or {}
        return cls(
            memory_id=str(payload.get("memory_id", "")),
            included=bool(payload.get("included", False)),
            retrieval_lanes=dict(payload.get("retrieval_lanes") or {}),
            final_score=float(payload.get("final_score", 0.0)),
            scope_tier=payload.get("scope_tier"),
            reason_codes=list(payload.get("reason_codes") or []),
            excluded_conflicts=list(payload.get("excluded_conflicts") or []),
            evidence_ids=list(payload.get("evidence_ids") or []),
            current_at_query_time=bool(payload.get("current_at_query_time", True)),
        )


@dataclass
class SearchResult:
    memory: MemoryRecord
    score: float
    rank_breakdown: dict[str, float]
    evidence: list[EvidenceItem]
    projection_status: ProjectionStatus
    receipt: RetrievalReceipt | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "rank_breakdown": dict(self.rank_breakdown),
            "evidence": [item.to_dict() for item in self.evidence],
            "projection_status": self.projection_status.to_dict(),
            "receipt": self.receipt.to_dict() if self.receipt else None,
        }


@dataclass
class ContextPack:
    query: str
    scope: MemoryScope
    intent: str
    working_memory: list[dict[str, Any]] = field(default_factory=list)
    profile: list[dict[str, Any]] = field(default_factory=list)
    preferences: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    policies: list[dict[str, Any]] = field(default_factory=list)
    commitments: list[dict[str, Any]] = field(default_factory=list)
    procedures: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    recent_episodes: list[dict[str, Any]] = field(default_factory=list)
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    projection_status: ProjectionStatus = field(default_factory=ProjectionStatus)
    abstained: bool = False
    abstention_reason: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope.to_dict(),
            "intent": self.intent,
            "working_memory": self.working_memory,
            "profile": self.profile,
            "preferences": self.preferences,
            "relationships": self.relationships,
            "policies": self.policies,
            "commitments": self.commitments,
            "procedures": self.procedures,
            "decisions": self.decisions,
            "constraints": self.constraints,
            "recent_episodes": self.recent_episodes,
            "tool_observations": self.tool_observations,
            "citations": self.citations,
            "receipts": self.receipts,
            "conflicts": self.conflicts,
            "projection_status": self.projection_status.to_dict(),
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "debug": self.debug,
            "trace": self.trace,
        }


@dataclass
class AuditFinding:
    code: str
    severity: str
    message: str
    memory_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    suggested_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    schema_version: int
    generated_at: str
    memory_count: int
    evidence_count: int
    finding_count: int
    findings: list[AuditFinding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "memory_count": self.memory_count,
            "evidence_count": self.evidence_count,
            "finding_count": self.finding_count,
            "findings": [item.to_dict() for item in self.findings],
            "metrics": dict(self.metrics),
        }


def _legacy_source_type(event_type: str, role: str | None) -> str:
    if event_type == EventType.MESSAGE.value:
        return SourceType.ASSISTANT_MESSAGE.value if role == "assistant" else SourceType.USER_MESSAGE.value
    mapping = {
        EventType.TOOL_CALL.value: SourceType.TOOL_CALL.value,
        EventType.TOOL_RESULT.value: SourceType.TOOL_RESULT.value,
        EventType.ASSISTANT_ACTION.value: SourceType.ASSISTANT_ACTION.value,
        EventType.SYSTEM_INSTRUCTION.value: SourceType.SYSTEM_INSTRUCTION.value,
        EventType.EXTERNAL_DOCUMENT.value: SourceType.EXTERNAL_DOCUMENT.value,
        EventType.RETRIEVED_MEMORY.value: SourceType.RETRIEVED_MEMORY.value,
        EventType.GENERATED_SUMMARY.value: SourceType.GENERATED_SUMMARY.value,
        EventType.IMPORTED_RECORD.value: SourceType.IMPORTED_RECORD.value,
        EventType.EXPLICIT_WRITE.value: SourceType.EXPLICIT_APPLICATION_WRITE.value,
    }
    return mapping.get(event_type, SourceType.UNKNOWN.value)


def _default_trust(source_type: str) -> str:
    mapping = {
        SourceType.EXPLICIT_APPLICATION_WRITE.value: TrustLevel.AUTHORITATIVE.value,
        SourceType.SYSTEM_INSTRUCTION.value: TrustLevel.AUTHORITATIVE.value,
        SourceType.USER_MESSAGE.value: TrustLevel.USER_STATED.value,
        SourceType.TOOL_RESULT.value: TrustLevel.OBSERVED.value,
        SourceType.TOOL_CALL.value: TrustLevel.OBSERVED.value,
        SourceType.ASSISTANT_ACTION.value: TrustLevel.OBSERVED.value,
        SourceType.EXTERNAL_DOCUMENT.value: TrustLevel.UNTRUSTED_EXTERNAL.value,
        SourceType.RETRIEVED_MEMORY.value: TrustLevel.QUARANTINED.value,
        SourceType.GENERATED_SUMMARY.value: TrustLevel.DERIVED.value,
        SourceType.IMPORTED_RECORD.value: TrustLevel.DERIVED.value,
        SourceType.ASSISTANT_MESSAGE.value: TrustLevel.DERIVED.value,
    }
    return mapping.get(source_type, TrustLevel.DERIVED.value)
