from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .admission import AdmissionConfig, AdmissionPolicy, DefaultAdmissionPolicy
from .extractors import DefaultMemoryExtractor, MemoryCandidate
from .types import IngestionItem, MemoryAction, MemoryState, SearchFilters
from .utils import normalize_text, tokenize


class SemanticExtractor(Protocol):
    def extract_semantic(self, evidence: Any) -> list[MemoryCandidate]:
        ...

    def build_episode_candidate(self, evidence: Any) -> MemoryCandidate | None:
        ...

    def extract(self, evidence: Any) -> list[MemoryCandidate]:
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
        scope: Any,
        candidate: MemoryCandidate,
        existing_record: Any | None,
        explicit_action: str | None = None,
    ) -> ReconciliationDecision:
        ...


class DefaultMemoryReconciler:
    def reconcile(
        self,
        *,
        state: Any,
        scope: Any,
        candidate: MemoryCandidate,
        existing_record: Any | None,
        explicit_action: str | None = None,
    ) -> ReconciliationDecision:
        del state, scope
        if explicit_action is not None:
            return ReconciliationDecision(
                action=explicit_action,
                state=MemoryState.DELETED.value if explicit_action == MemoryAction.DELETE.value else candidate.state,
                matched_record_id=getattr(existing_record, "record_id", None),
                reasoning_trace=[f"explicit action={explicit_action.lower()}"],
            )
        if existing_record is None:
            return ReconciliationDecision(
                action=MemoryAction.ADD.value,
                state=candidate.state,
                reasoning_trace=["new canonical memory slot"],
            )
        if normalize_text(existing_record.value).casefold() == normalize_text(candidate.value).casefold():
            return ReconciliationDecision(
                action=MemoryAction.MERGE.value,
                state=existing_record.state,
                matched_record_id=existing_record.record_id,
                reasoning_trace=["same value confirmed by additional evidence"],
            )
        return ReconciliationDecision(
            action=MemoryAction.SUPERSEDE.value,
            state=candidate.state,
            matched_record_id=existing_record.record_id,
            reasoning_trace=["newer or higher priority evidence changed the current value"],
            metadata={"previous_value": existing_record.value},
        )


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
    fast_path_candidate_limit: int = 96
    consolidation_window_hours: int = 168
    consolidation_min_support: int = 2
    utility_weight: float = 0.6
    retrieval_weight: float = 0.1

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
            "utility_weight": self.utility_weight,
            "retrieval_weight": self.retrieval_weight,
        }


@dataclass
class RetrievalConfig:
    rrf_k: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 0.85
    exact_weight: float = 1.25
    entity_weight: float = 0.75
    temporal_weight: float = 0.8
    authority_weight: float = 0.6
    activation_weight: float = 0.35
    policy_weight: float = 0.75
    minimum_relevance: float = 0.005
    semantic_candidate_limit: int = 128
    lane_limit: int = 100

    def describe(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class MemoryPipelineConfig:
    extractor: SemanticExtractor | Any | None = None
    admission_policy: AdmissionPolicy | None = None
    reconciler: MemoryReconciler | None = None
    reranker: Reranker | None = None
    asset_processor: AssetProcessor | None = None
    brain: BrainConfig = field(default_factory=BrainConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    semantic_promotion_threshold: float = 0.72
    pending_confidence_threshold: float = 0.72
    default_include_trace: bool = False
    max_trace_results: int = 20
    default_track_usage: bool = True

    def resolved_extractor(self) -> Any:
        return self.extractor or DefaultMemoryExtractor()

    def resolved_admission_policy(self) -> AdmissionPolicy:
        return self.admission_policy or DefaultAdmissionPolicy(self.admission)

    def resolved_reconciler(self) -> MemoryReconciler:
        return self.reconciler or DefaultMemoryReconciler()

    def describe(self) -> dict[str, Any]:
        return {
            "extractor": _component_name(self.extractor) or "DefaultMemoryExtractor",
            "admission_policy": _component_name(self.admission_policy) or "DefaultAdmissionPolicy",
            "reconciler": _component_name(self.reconciler) or "DefaultMemoryReconciler",
            "reranker": _component_name(self.reranker),
            "asset_processor": _component_name(self.asset_processor),
            "brain": self.brain.describe(),
            "retrieval": self.retrieval.describe(),
            "admission": self.admission.__dict__.copy(),
            "semantic_promotion_threshold": self.semantic_promotion_threshold,
            "pending_confidence_threshold": self.pending_confidence_threshold,
            "default_include_trace": self.default_include_trace,
            "max_trace_results": self.max_trace_results,
            "default_track_usage": self.default_track_usage,
        }


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
                    "base_score": breakdown.get("rrf", 0.0),
                },
            )
        return outcomes


class LocalAssetProcessor:
    TEXT_FILE_SUFFIXES = {".txt", ".md", ".text", ".rst", ".json", ".jsonl", ".csv"}

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
        return [
            IngestionItem(
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
                writer_id=item.writer_id,
                trust_level=item.trust_level,
                source_uri=item.source_uri,
                is_retrieved_memory=item.is_retrieved_memory,
            )
        ]

    def _text_source(self, item: IngestionItem) -> str:
        if any(item.metadata.get(key) for key in ("text", "ocr_text", "transcript", "caption", "summary")):
            return "metadata"
        return "asset_ref" if item.asset_ref else "content"

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
        if not path.exists() or not path.is_file() or path.suffix.lower() not in self.TEXT_FILE_SUFFIXES:
            return ""
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return ""
        if path.suffix.lower() in {".json", ".jsonl"}:
            try:
                return normalize_text(json.dumps(json.loads(raw), ensure_ascii=False, sort_keys=True))
            except json.JSONDecodeError:
                return normalize_text(raw)
        return normalize_text(raw)


def _component_name(component: Any | None) -> str | None:
    if component is None:
        return None
    return component.__class__.__name__
