from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .extractors import MemoryCandidate
from .types import EventType, EvidenceItem, IngestionItem, MemoryAction, MemoryKind, MemoryLayer, MemoryScope, MemoryState, SearchFilters
from .utils import normalize_text, tokenize


class SemanticExtractor(Protocol):
    def extract_semantic(self, evidence: EvidenceItem) -> list[MemoryCandidate]:
        ...

    def build_episode_candidate(self, evidence: EvidenceItem) -> MemoryCandidate | None:
        ...

    def extract(self, evidence: EvidenceItem) -> list[MemoryCandidate]:
        ...


@dataclass
class ReconciliationDecision:
    action: str
    state: str
    matched_record_id: str | None = None
    reasoning_trace: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryReconciler(Protocol):
    def reconcile(
        self,
        *,
        state: Any,
        scope: MemoryScope,
        candidate: MemoryCandidate,
        existing_record: Any | None,
        explicit_action: str | None = None,
    ) -> ReconciliationDecision:
        ...


@dataclass
class RerankOutcome:
    score: float
    details: dict[str, Any] = field(default_factory=dict)


class Reranker(Protocol):
    def rerank(
        self,
        *,
        query: str,
        candidates: list[tuple[float, Any, dict[str, float]]],
        state: Any,
        search_filters: SearchFilters,
        intent: str,
    ) -> dict[str, RerankOutcome]:
        ...


class AssetProcessor(Protocol):
    def process(self, item: IngestionItem) -> list[IngestionItem]:
        ...


@dataclass
class BrainConfig:
    mode: str = "classic"
    working_memory_size: int = 6
    attention_decay_half_life_hours: float = 72.0
    dormancy_threshold: float = 0.18
    activation_spread: float = 0.08
    fast_path_candidate_limit: int = 64
    consolidation_window_hours: int = 168
    consolidation_min_support: int = 2

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "working_memory_size": self.working_memory_size,
            "attention_decay_half_life_hours": self.attention_decay_half_life_hours,
            "dormancy_threshold": self.dormancy_threshold,
            "activation_spread": self.activation_spread,
            "fast_path_candidate_limit": self.fast_path_candidate_limit,
            "consolidation_window_hours": self.consolidation_window_hours,
            "consolidation_min_support": self.consolidation_min_support,
        }


@dataclass
class MemoryPipelineConfig:
    extractor: SemanticExtractor | Any | None = None
    reconciler: MemoryReconciler | None = None
    reranker: Reranker | None = None
    asset_processor: AssetProcessor | None = None
    brain: BrainConfig = field(default_factory=BrainConfig)
    semantic_promotion_threshold: float = 0.72
    pending_confidence_threshold: float = 0.75
    default_include_trace: bool = False
    max_trace_results: int = 10

    def describe(self) -> dict[str, Any]:
        return {
            "extractor": _component_name(self.extractor),
            "reconciler": _component_name(self.reconciler),
            "reranker": _component_name(self.reranker),
            "asset_processor": _component_name(self.asset_processor),
            "brain": self.brain.describe(),
            "semantic_promotion_threshold": self.semantic_promotion_threshold,
            "pending_confidence_threshold": self.pending_confidence_threshold,
            "default_include_trace": self.default_include_trace,
            "max_trace_results": self.max_trace_results,
        }


class DefaultMemoryReconciler:
    PROFILE_KEYS = {"name", "age", "location", "employer", "occupation"}

    def __init__(self, pending_confidence_threshold: float = 0.75):
        self.pending_confidence_threshold = pending_confidence_threshold

    def reconcile(
        self,
        *,
        state: Any,
        scope: MemoryScope,
        candidate: MemoryCandidate,
        existing_record: Any | None,
        explicit_action: str | None = None,
    ) -> ReconciliationDecision:
        matched = existing_record or self._find_related_record(state=state, scope=scope, candidate=candidate)
        desired_state = candidate.state
        if candidate.confidence < self.pending_confidence_threshold and candidate.layer != MemoryLayer.EPISODIC.value:
            if explicit_action not in (MemoryAction.UPDATE.value, MemoryAction.DELETE.value):
                desired_state = MemoryState.PENDING.value
        if explicit_action == MemoryAction.DELETE.value:
            desired_state = MemoryState.DELETED.value
        if explicit_action is not None:
            return ReconciliationDecision(
                action=explicit_action,
                state=desired_state,
                matched_record_id=getattr(matched, "record_id", None),
                reasoning_trace=[f"explicit action={explicit_action.lower()}"],
            )
        if matched is None:
            return ReconciliationDecision(
                action=MemoryAction.ADD.value,
                state=desired_state,
                reasoning_trace=["new semantic slot", *self._reasoning_tags(candidate=candidate)],
            )

        current_value = normalize_text(getattr(matched, "value", "")).lower()
        candidate_value = normalize_text(candidate.value).lower()
        sentiments_match = True
        if candidate.kind == MemoryKind.PREFERENCE.value:
            record_sentiment = normalize_text(str(getattr(matched, "metadata", {}).get("sentiment", ""))).lower()
            candidate_sentiment = normalize_text(str(candidate.metadata.get("sentiment", ""))).lower()
            if record_sentiment and candidate_sentiment and record_sentiment != candidate_sentiment:
                sentiments_match = False
        if current_value == candidate_value and sentiments_match:
            return ReconciliationDecision(
                action=MemoryAction.NONE.value,
                state=getattr(matched, "state", desired_state),
                matched_record_id=matched.record_id,
                reasoning_trace=["value already confirmed", *self._reasoning_tags(candidate=candidate)],
            )

        reason = self._change_reason(candidate=candidate, matched=matched)
        action = MemoryAction.UPDATE.value if getattr(matched, "state", "") == MemoryState.PENDING.value else MemoryAction.SUPERSEDE.value
        return ReconciliationDecision(
            action=action,
            state=desired_state,
            matched_record_id=matched.record_id,
            reasoning_trace=[reason, *self._reasoning_tags(candidate=candidate)],
            metadata={"matched_record_id": matched.record_id, "previous_value": getattr(matched, "value", "")},
        )

    def _find_related_record(self, *, state: Any, scope: MemoryScope, candidate: MemoryCandidate) -> Any | None:
        for record in getattr(state, "memories", {}).values():
            if getattr(record, "scope", None) is None or record.scope.to_dict() != scope.to_dict():
                continue
            if getattr(record, "state", None) == MemoryState.DELETED.value:
                continue
            if record.kind != candidate.kind or record.layer != candidate.layer:
                continue
            if record.key == candidate.key:
                return record
            if candidate.kind == MemoryKind.PREFERENCE.value and self._preference_topic_from_record(record) == self._preference_topic(candidate):
                return record
        return None

    def _reasoning_tags(self, *, candidate: MemoryCandidate) -> list[str]:
        tags = [f"kind={candidate.kind}", f"layer={candidate.layer}"]
        topic = self._preference_topic(candidate)
        if topic:
            tags.append(f"topic={topic}")
        if candidate.source_type:
            tags.append(f"source_type={candidate.source_type}")
        return tags

    def _change_reason(self, *, candidate: MemoryCandidate, matched: Any) -> str:
        if candidate.kind == MemoryKind.PROFILE_ATTRIBUTE.value and candidate.key in self.PROFILE_KEYS:
            return f"profile field updated: {candidate.key}"
        if candidate.kind == MemoryKind.PREFERENCE.value:
            old_topic = self._preference_topic_from_record(matched)
            new_topic = self._preference_topic(candidate)
            if old_topic and old_topic == new_topic:
                return f"preference updated: {new_topic}"
            return "preference contradiction resolved"
        if candidate.kind == MemoryKind.RELATION.value:
            return "relationship memory updated"
        return "superseded previous version"

    def _preference_topic(self, candidate: MemoryCandidate) -> str:
        metadata_topic = normalize_text(str(candidate.metadata.get("topic", ""))).lower()
        if metadata_topic:
            return metadata_topic
        return normalize_text(candidate.value).lower()

    def _preference_topic_from_record(self, record: Any) -> str:
        metadata_topic = normalize_text(str(getattr(record, "metadata", {}).get("topic", ""))).lower()
        if metadata_topic:
            return metadata_topic
        return normalize_text(getattr(record, "value", "")).lower()


class KeywordBoostReranker:
    def rerank(
        self,
        *,
        query: str,
        candidates: list[tuple[float, Any, dict[str, float]]],
        state: Any,
        search_filters: SearchFilters,
        intent: str,
    ) -> dict[str, RerankOutcome]:
        del state, search_filters, intent
        query_tokens = set(tokenize(query))
        outcomes: dict[str, RerankOutcome] = {}
        for _, record, breakdown in candidates:
            record_tokens = set(tokenize(getattr(record, "search_text", "")))
            overlap = len(query_tokens.intersection(record_tokens))
            score = min(overlap / max(len(query_tokens), 1), 1.0) if query_tokens else 0.0
            outcomes[record.record_id] = RerankOutcome(
                score=score,
                details={
                    "overlap": overlap,
                    "query_token_count": len(query_tokens),
                    "base_score": breakdown.get("lexical", 0.0),
                },
            )
        return outcomes


class LocalAssetProcessor:
    TEXT_FILE_SUFFIXES = {".txt", ".md", ".text", ".rst", ".json", ".jsonl"}

    def process(self, item: IngestionItem) -> list[IngestionItem]:
        if normalize_text(item.content):
            return [item]

        derived_text = self._derived_text_from_metadata(item)
        if not derived_text and item.asset_ref and item.modality in {"document", "text"}:
            derived_text = self._read_local_text(item.asset_ref)
        if not derived_text:
            return [item]

        metadata = dict(item.metadata or {})
        metadata["asset_processed_by"] = self.__class__.__name__
        metadata["asset_text_source"] = metadata.get("asset_text_source") or self._text_source(item)
        processed = IngestionItem(
            content=derived_text,
            modality=item.modality,
            role=item.role,
            metadata=metadata,
            asset_ref=item.asset_ref,
            event_type=item.event_type,
            name=item.name,
            attributes=dict(item.attributes or {}),
            occurred_at=item.occurred_at,
            source_type=item.source_type,
        )
        return [processed]

    def _text_source(self, item: IngestionItem) -> str:
        if any(item.metadata.get(key) for key in ("text", "ocr_text", "transcript", "caption", "summary")):
            return "metadata"
        if item.asset_ref:
            return "asset_ref"
        return "content"

    def _derived_text_from_metadata(self, item: IngestionItem) -> str:
        metadata = dict(item.metadata or {})
        attributes = dict(item.attributes or {})
        for key in ("text", "ocr_text", "transcript", "caption", "summary", "result", "output"):
            value = metadata.get(key)
            if value is None:
                value = attributes.get(key)
            normalized = normalize_text(str(value or ""))
            if normalized:
                return normalized
        return ""

    def _read_local_text(self, asset_ref: str) -> str:
        path = Path(asset_ref).expanduser()
        if not path.exists() or not path.is_file():
            return ""
        if path.suffix.lower() not in self.TEXT_FILE_SUFFIXES:
            return ""
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ""
        if path.suffix.lower().startswith(".json"):
            try:
                return normalize_text(json.dumps(json.loads(raw), ensure_ascii=True, sort_keys=True))
            except json.JSONDecodeError:
                return normalize_text(raw)
        return normalize_text(raw)


def _component_name(component: Any | None) -> str | None:
    if component is None:
        return None
    return component.__class__.__name__
