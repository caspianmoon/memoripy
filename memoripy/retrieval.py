from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable

from .pipeline import RetrievalConfig
from .repository import EngineState
from .types import (
    Durability,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryState,
    RetrievalReceipt,
    SearchFilters,
    TrustLevel,
)
from .utils import (
    cosine_similarity,
    extract_entities,
    normalize_key,
    normalize_text,
    parse_timestamp,
    tokenize,
    unique_tokens,
)


TRUST_SCORE = {
    TrustLevel.QUARANTINED.value: 0.0,
    TrustLevel.UNTRUSTED_EXTERNAL.value: 0.15,
    TrustLevel.DERIVED.value: 0.4,
    TrustLevel.OBSERVED.value: 0.65,
    TrustLevel.USER_STATED.value: 0.85,
    TrustLevel.AUTHORITATIVE.value: 1.0,
}

DURABILITY_SCORE = {
    Durability.EPHEMERAL.value: 0.15,
    Durability.SESSION.value: 0.45,
    Durability.DURABLE.value: 0.75,
    Durability.PINNED.value: 1.0,
}

SCOPE_ORDER = ("exact_run", "user_agent", "user", "project", "organization", "namespace", "global")
SCOPE_SCORE = {
    "exact_run": 1.0,
    "user_agent": 0.92,
    "user": 0.82,
    "project": 0.7,
    "organization": 0.6,
    "namespace": 0.55,
    "global": 0.4,
}


def rank_records(
    *,
    state: EngineState,
    query: str,
    filters: SearchFilters,
    intent: str,
    make_embedding: Callable[[str], list[float]],
    config: RetrievalConfig,
    brain_enabled: bool,
) -> tuple[list[tuple[float, MemoryRecord, dict[str, float], RetrievalReceipt]], dict[str, Any]]:
    normalized_query = normalize_text(query)
    records, scope_trace = _eligible_records(state=state, filters=filters, intent=intent)
    if not records:
        return [], {"scope": scope_trace, "lanes": {}, "candidate_count": 0}

    query_tokens = unique_tokens(normalized_query)
    query_entities = {normalize_key(item) for item in extract_entities(normalized_query)}
    query_embedding = make_embedding(normalized_query) if normalized_query else []
    lane_raw: dict[str, dict[str, float]] = {}

    lexical = _bm25_scores(query_tokens=query_tokens, records=records)
    if lexical:
        lane_raw["lexical"] = lexical

    semantic: dict[str, float] = {}
    if query_embedding:
        for record in records:
            score = max(cosine_similarity(query_embedding, record.embedding), 0.0)
            if score > 0:
                semantic[record.record_id] = score
    if semantic:
        lane_raw["semantic"] = semantic

    exact: dict[str, float] = {}
    query_key = normalize_key(normalized_query)
    for record in records:
        value = normalize_key(record.value)
        key = normalize_key(record.key)
        search_text = normalize_key(record.search_text)
        score = 0.0
        if query_key and query_key in {value, key}:
            score = 1.0
        elif query_key and query_key in search_text:
            score = 0.92
        elif query_tokens:
            overlap = len(set(query_tokens).intersection(tokenize(record.search_text))) / max(len(set(query_tokens)), 1)
            if overlap >= 0.7:
                score = min(0.65 + (overlap * 0.3), 0.9)
        if score > 0:
            exact[record.record_id] = score
    if exact:
        lane_raw["exact"] = exact

    entities: dict[str, float] = {}
    if query_entities:
        for record in records:
            record_entities = {normalize_key(item) for item in record.entity_names}
            overlap = len(query_entities.intersection(record_entities))
            if overlap:
                entities[record.record_id] = min(overlap / max(len(query_entities), 1), 1.0)
    if entities:
        lane_raw["entity"] = entities

    temporal: dict[str, float] = {}
    for record in records:
        score = _temporal_score(record=record, as_of=filters.as_of, intent=intent)
        if score > 0:
            temporal[record.record_id] = score
    if temporal:
        lane_raw["temporal"] = temporal

    authority = {
        record.record_id: (TRUST_SCORE.get(record.trust_level, 0.2) * 0.7) + (
            DURABILITY_SCORE.get(record.durability, 0.5) * 0.3
        )
        for record in records
    }
    lane_raw["authority"] = authority

    general = {record.record_id: _general_score(record) for record in records}
    lane_raw["general"] = general

    policy: dict[str, float] = {}
    desired_kinds = _desired_kinds_for_intent(intent)
    if desired_kinds:
        for record in records:
            if record.kind in desired_kinds:
                policy[record.record_id] = 1.0
    if policy:
        lane_raw["policy"] = policy

    activation: dict[str, float] = {}
    if brain_enabled:
        projection = state.projections.get("activation") or {}
        for record in records:
            entry = projection.get(_base_record_id(record.record_id)) or {}
            score = float(entry.get("activation_score", 0.0))
            utility = _utility_score(record)
            activation[record.record_id] = max(score * 0.45 + utility * 0.55, 0.0)
        lane_raw["activation"] = activation

    lane_weights = {
        "lexical": config.lexical_weight,
        "semantic": config.semantic_weight,
        "exact": config.exact_weight,
        "entity": config.entity_weight,
        "temporal": config.temporal_weight,
        "authority": config.authority_weight,
        "general": 0.55,
        "policy": config.policy_weight,
        "activation": config.activation_weight,
    }

    lane_rankings: dict[str, list[tuple[str, float]]] = {}
    lane_positions: dict[str, dict[str, int]] = {}
    for lane_name, scores in lane_raw.items():
        ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[: config.lane_limit]
        lane_rankings[lane_name] = ordered
        lane_positions[lane_name] = {record_id: index + 1 for index, (record_id, _) in enumerate(ordered)}

    record_map = {record.record_id: record for record in records}
    fused: dict[str, float] = defaultdict(float)
    breakdowns: dict[str, dict[str, float]] = defaultdict(dict)
    receipt_lanes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for lane_name, ordered in lane_rankings.items():
        weight = lane_weights.get(lane_name, 1.0)
        for rank, (record_id, raw_score) in enumerate(ordered, start=1):
            contribution = weight / (config.rrf_k + rank)
            fused[record_id] += contribution
            breakdowns[record_id][lane_name] = raw_score
            breakdowns[record_id][f"{lane_name}_rrf"] = contribution
            receipt_lanes[record_id][lane_name] = {
                "rank": rank,
                "raw_score": raw_score,
                "weight": weight,
                "rrf_contribution": contribution,
            }

    ranked: list[tuple[float, MemoryRecord, dict[str, float], RetrievalReceipt]] = []
    for record_id, rrf_score in fused.items():
        record = record_map[record_id]
        scope_tier = str(record.metadata.get("_retrieval_scope_tier", "global"))
        scope_bonus = SCOPE_SCORE.get(scope_tier, 0.3) * 0.02
        trust_bonus = TRUST_SCORE.get(record.trust_level, 0.2) * 0.003
        utility_bonus = _utility_score(record) * 0.002
        dormancy_penalty = 0.002 if record.state == MemoryState.DORMANT.value and exact.get(record_id, 0.0) < 0.85 else 0.0
        contradiction_penalty = min(len(record.contradicted_by) * 0.0015, 0.0045)
        final_score = rrf_score + scope_bonus + trust_bonus + utility_bonus - dormancy_penalty - contradiction_penalty
        if final_score < config.minimum_relevance:
            continue
        breakdown = breakdowns[record_id]
        breakdown.update(
            {
                "rrf": rrf_score,
                "scope_bonus": scope_bonus,
                "trust_bonus": trust_bonus,
                "utility_bonus": utility_bonus,
                "dormancy_penalty": dormancy_penalty,
                "contradiction_penalty": contradiction_penalty,
                "final": final_score,
            }
        )
        reasons = [
            f"found_by:{lane}"
            for lane in receipt_lanes[record_id]
            if lane not in ("authority", "general", "activation")
        ]
        if not reasons:
            reasons.append("selected_by:general_memory_context")
        receipt = RetrievalReceipt(
            memory_id=_base_record_id(record.record_id),
            included=True,
            retrieval_lanes=receipt_lanes[record_id],
            final_score=final_score,
            scope_tier=scope_tier,
            reason_codes=reasons,
            evidence_ids=list(record.citation_evidence_ids or record.evidence_ids),
            current_at_query_time=not bool(record.metadata.get("_historical_version")),
        )
        ranked.append((final_score, record, breakdown, receipt))

    ranked.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
    trace = {
        "scope": scope_trace,
        "candidate_count": len(records),
        "lanes": {
            lane: [
                {"memory_id": _base_record_id(record_id), "rank": index + 1, "score": score}
                for index, (record_id, score) in enumerate(items[:20])
            ]
            for lane, items in lane_rankings.items()
        },
        "rrf_k": config.rrf_k,
    }
    return ranked, trace


def _eligible_records(
    *,
    state: EngineState,
    filters: SearchFilters,
    intent: str,
) -> tuple[list[MemoryRecord], dict[str, Any]]:
    target = filters.scope
    records: list[MemoryRecord] = []
    tier_buckets: dict[str, list[MemoryRecord]] = defaultdict(list)
    for record in state.memories.values():
        materialized_records = _materialize_records(state=state, record=record, filters=filters, intent=intent)
        for materialized in materialized_records:
            if filters.kinds and materialized.kind not in filters.kinds:
                continue
            if filters.layers and materialized.layer not in filters.layers:
                continue
            if filters.source_types and materialized.source_type not in filters.source_types:
                continue
            if filters.trust_levels and materialized.trust_level not in filters.trust_levels:
                continue
            if filters.durabilities and materialized.durability not in filters.durabilities:
                continue
            if filters.tags and not set(filters.tags).intersection(materialized.tags):
                continue
            if any(materialized.metadata.get(key) != value for key, value in filters.metadata.items()):
                continue
            tier = scope_tier(materialized.scope, target, hierarchical=filters.hierarchical_scope)
            if tier is None:
                continue
            materialized.metadata = {**materialized.metadata, "_retrieval_scope_tier": tier}
            tier_buckets[tier].append(materialized)

    if not filters.adaptive_scope or target is None or target.is_empty():
        for tier in SCOPE_ORDER:
            records.extend(tier_buckets.get(tier, []))
        return records, {
            "adaptive": False,
            "included_tiers": [tier for tier in SCOPE_ORDER if tier_buckets.get(tier)],
            "tier_counts": {tier: len(values) for tier, values in tier_buckets.items()},
        }

    included_tiers: list[str] = []
    for tier in SCOPE_ORDER:
        bucket = tier_buckets.get(tier, [])
        if not bucket:
            continue
        records.extend(bucket)
        included_tiers.append(tier)
        if len(records) >= max(filters.minimum_scope_results, 1):
            break
    return records, {
        "adaptive": True,
        "included_tiers": included_tiers,
        "tier_counts": {tier: len(values) for tier, values in tier_buckets.items()},
    }


def _materialize_records(
    *,
    state: EngineState,
    record: MemoryRecord,
    filters: SearchFilters,
    intent: str,
) -> list[MemoryRecord]:
    if record.state == MemoryState.DELETED.value:
        return []
    if record.state == MemoryState.QUARANTINED.value and not filters.include_quarantined:
        return []
    if record.state == MemoryState.PENDING.value and not filters.include_pending:
        return []
    if record.state not in (
        MemoryState.ACTIVE.value,
        MemoryState.DORMANT.value,
        MemoryState.PENDING.value,
        MemoryState.QUARANTINED.value,
    ):
        return []
    if filters.states and record.state not in filters.states:
        return []

    if filters.as_of:
        as_of = parse_timestamp(filters.as_of)
        if as_of is None:
            return [replace(record, metadata=dict(record.metadata))]
        candidates = []
        for version_id in record.version_ids:
            version = state.versions.get(version_id)
            if version is None or version.action == "DELETE":
                continue
            valid_from = parse_timestamp(version.valid_from or version.observed_at or version.created_at)
            valid_to = parse_timestamp(version.valid_to)
            if valid_from and valid_from > as_of:
                continue
            if valid_to and as_of >= valid_to:
                continue
            candidates.append(version)
        if not candidates:
            return []
        version = max(
            candidates,
            key=lambda item: parse_timestamp(item.valid_from or item.observed_at or item.created_at)
            or datetime.min.replace(tzinfo=timezone.utc),
        )
        return [_record_from_version(record=record, version=version, historical=version.version_id != record.current_version_id)]

    output = [replace(record, metadata=dict(record.metadata))]
    if filters.include_historical or intent == "historical":
        for version_id in record.version_ids:
            if version_id == record.current_version_id:
                continue
            version = state.versions.get(version_id)
            if version is None or version.action == "DELETE":
                continue
            output.append(_record_from_version(record=record, version=version, historical=True))
    return output


def _record_from_version(*, record: MemoryRecord, version: Any, historical: bool) -> MemoryRecord:
    materialized = replace(
        record,
        value=version.value,
        summary=version.summary,
        current_version_id=version.version_id,
        confidence=version.confidence,
        salience=version.salience,
        source_type=version.source_type,
        trust_level=version.trust_level,
        durability=version.durability,
        layer=version.layer,
        kind=version.kind,
        subject=version.subject,
        observed_at=version.observed_at,
        valid_from=version.valid_from,
        valid_to=version.valid_to,
        citation_evidence_ids=list(version.citation_evidence_ids or version.evidence_ids),
        state=MemoryState.SUPERSEDED.value if historical else record.state,
        metadata={**record.metadata, **version.metadata, "_historical_version": historical},
    )
    materialized.record_id = f"{record.record_id}@{version.version_id}" if historical else record.record_id
    materialized.search_text = normalize_text(
        " ".join(
            [
                materialized.layer,
                materialized.source_type,
                materialized.trust_level,
                materialized.durability,
                materialized.kind,
                materialized.key,
                materialized.subject or "",
                materialized.summary,
                materialized.value,
                " ".join(materialized.tags),
                " ".join(materialized.entity_names),
            ]
        )
    )
    return materialized


def scope_tier(record_scope: MemoryScope, target_scope: MemoryScope | None, *, hierarchical: bool) -> str | None:
    if target_scope is None or target_scope.is_empty():
        return "global" if record_scope.is_empty() else _broadest_scope(record_scope)
    if not hierarchical:
        return "exact_run" if record_scope.to_dict() == target_scope.to_dict() else None

    for field_name in ("user_id", "agent_id", "run_id", "project_id", "organization_id", "namespace"):
        expected = getattr(target_scope, field_name)
        actual = getattr(record_scope, field_name)
        if expected is None:
            if actual is not None:
                return None
            continue
        if actual is not None and actual != expected:
            return None

    if target_scope.run_id and record_scope.run_id == target_scope.run_id:
        return "exact_run"
    if target_scope.user_id and target_scope.agent_id and record_scope.user_id == target_scope.user_id and record_scope.agent_id == target_scope.agent_id:
        return "user_agent"
    if target_scope.user_id and record_scope.user_id == target_scope.user_id:
        return "user"
    if target_scope.project_id and record_scope.project_id == target_scope.project_id:
        return "project"
    if target_scope.organization_id and record_scope.organization_id == target_scope.organization_id:
        return "organization"
    if target_scope.namespace and record_scope.namespace == target_scope.namespace:
        return "namespace"
    if record_scope.is_empty():
        return "global"
    return None


def _broadest_scope(scope: MemoryScope) -> str:
    if scope.run_id:
        return "exact_run"
    if scope.user_id and scope.agent_id:
        return "user_agent"
    if scope.user_id:
        return "user"
    if scope.project_id:
        return "project"
    if scope.organization_id:
        return "organization"
    if scope.namespace:
        return "namespace"
    return "global"


def _bm25_scores(*, query_tokens: list[str], records: list[MemoryRecord]) -> dict[str, float]:
    if not query_tokens or not records:
        return {}
    documents = [tokenize(record.search_text) for record in records]
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))
    average_length = sum(len(tokens) for tokens in documents) / max(len(documents), 1)
    k1 = 1.5
    b = 0.75
    scores: dict[str, float] = {}
    total_documents = len(records)
    for record, tokens in zip(records, documents):
        counts = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1 + ((total_documents - df + 0.5) / (df + 0.5)))
            denominator = frequency + k1 * (1 - b + b * (len(tokens) / max(average_length, 1.0)))
            score += idf * ((frequency * (k1 + 1)) / denominator)
        if score > 0:
            scores[record.record_id] = score
    if scores:
        maximum = max(scores.values()) or 1.0
        scores = {record_id: score / maximum for record_id, score in scores.items()}
    return scores


def _temporal_score(*, record: MemoryRecord, as_of: str | None, intent: str) -> float:
    if as_of:
        return 1.0
    if intent == "historical":
        return 1.0 if record.metadata.get("_historical_version") else 0.35
    if record.metadata.get("_historical_version"):
        return 0.05
    updated = parse_timestamp(record.updated_at)
    if updated is None:
        return 0.2
    age_days = max((datetime.now(timezone.utc) - updated).total_seconds() / 86400.0, 0.0)
    return 1.0 / (1.0 + (age_days / 30.0))


def _general_score(record: MemoryRecord) -> float:
    trust = TRUST_SCORE.get(record.trust_level, 0.2)
    durability = DURABILITY_SCORE.get(record.durability, 0.5)
    utility = _utility_score(record)
    return min((record.salience * 0.35) + (trust * 0.25) + (durability * 0.2) + (utility * 0.2), 1.0)


def _utility_score(record: MemoryRecord) -> float:
    positive = (
        record.used_in_answer_count
        + record.confirmed_by_user_count
        + record.associated_success_count
        + (record.included_in_context_count * 0.25)
    )
    negative = (
        record.corrected_count
        + record.rejected_count
        + (record.caused_failure_count * 2)
    )
    return max(0.0, min((math.log1p(positive) - math.log1p(negative)) / math.log(8.0) + 0.5, 1.0))


def _desired_kinds_for_intent(intent: str) -> set[str]:
    mapping = {
        "profile": {MemoryKind.PROFILE_ATTRIBUTE.value, MemoryKind.FACT.value},
        "preference": {MemoryKind.PREFERENCE.value},
        "relationship": {MemoryKind.RELATION.value, MemoryKind.ENTITY.value},
        "historical": {MemoryKind.PROFILE_ATTRIBUTE.value, MemoryKind.FACT.value, MemoryKind.DECISION.value},
        "policy": {MemoryKind.POLICY.value, MemoryKind.CONSTRAINT.value},
        "commitment": {MemoryKind.COMMITMENT.value},
        "procedure": {MemoryKind.PROCEDURE.value},
        "decision": {MemoryKind.DECISION.value},
        "episodic": {MemoryKind.EPISODIC_SUMMARY.value},
    }
    return mapping.get(intent, set())


def _base_record_id(record_id: str) -> str:
    return record_id.split("@", 1)[0]
