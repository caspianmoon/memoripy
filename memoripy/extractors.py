from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .types import (
    Durability,
    EventType,
    EvidenceItem,
    MemoryKind,
    MemoryLayer,
    MemoryState,
    SourceType,
    TrustLevel,
)
from .utils import extract_entities, normalize_key, normalize_text, stable_hash, summarize_text, tokenize


@dataclass
class MemoryCandidate:
    kind: str
    key: str
    value: str
    summary: str
    confidence: float
    state: str = MemoryState.ACTIVE.value
    metadata: dict[str, Any] = field(default_factory=dict)
    entity_names: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    layer: str = MemoryLayer.SEMANTIC.value
    salience: float = 0.0
    source_type: str = SourceType.UNKNOWN.value
    trust_level: str = TrustLevel.DERIVED.value
    durability: str = Durability.DURABLE.value
    subject: str | None = None
    observed_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    evidence_spans: list[dict[str, Any]] = field(default_factory=list)

    def canonical_value(self) -> str:
        return normalize_key(self.value)


class DefaultMemoryExtractor:
    """
    Transparent, deterministic extraction for common statements.

    It is intentionally conservative. Applications that need broad semantic
    extraction should provide a custom extractor through MemoryPipelineConfig.
    """

    CLAUSE_BOUNDARY_PATTERN = re.compile(
        r"\s+(?:and|but|because|while|although|though|then|ve|ama|çünkü)\s+"
        r"(?=(?:i\b|i'm\b|i’d\b|my\b|we\b|our\b|he\b|she\b|they\b|the\b|a\b|an\b|ben\b|biz\b))",
        re.IGNORECASE | re.UNICODE,
    )
    TRAILING_JOINER_PATTERN = re.compile(r"\b(?:and|but|or|ve|ama|ya da)$", re.IGNORECASE | re.UNICODE)
    NAME_PATTERNS = (
        re.compile(r"\bmy name is (?P<value>[^,.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\bcall me (?P<value>[^,.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\bbenim adım (?P<value>[^,.!?;]+)", re.IGNORECASE | re.UNICODE),
    )
    AGE_PATTERNS = (
        re.compile(r"\bi(?: am|'m) (?P<value>\d{1,3}) years old\b", re.IGNORECASE),
        re.compile(r"\b(?P<value>\d{1,3}) yaşındayım\b", re.IGNORECASE | re.UNICODE),
    )
    LOCATION_PATTERNS = (
        re.compile(r"\bi (?:live in|am from|moved to|relocated to) (?P<value>[^,.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\b(?:ben\s+)?(?P<value>[^,.!?;]+?)(?:'da|'de|da|de) yaşıyorum\b", re.IGNORECASE | re.UNICODE),
    )
    EMPLOYER_PATTERNS = (
        re.compile(r"\bi work (?:at|for) (?P<value>[^,.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\b(?P<value>[^,.!?;]+?)(?:'da|'de|da|de) çalışıyorum\b", re.IGNORECASE | re.UNICODE),
    )
    OCCUPATION_PATTERNS = (
        re.compile(r"\bi work as (?:an? )?(?P<value>[^,.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\bmesleğim (?P<value>[^,.!?;]+)", re.IGNORECASE | re.UNICODE),
    )
    FAVORITE_PATTERNS = (
        re.compile(r"\bmy favorite (?P<key>[^,.!?;]+?) is (?P<value>[^,.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\ben sevdiğim (?P<key>[^,.!?;]+?) (?P<value>[^,.!?;]+)", re.IGNORECASE | re.UNICODE),
    )
    POSITIVE_PREF_PATTERN = re.compile(
        r"\bi (?:really )?(?:like|love|enjoy|prefer) (?P<value>[^,.!?;]+)",
        re.IGNORECASE | re.UNICODE,
    )
    NEGATIVE_PREF_PATTERN = re.compile(
        r"\bi (?:(?:do not|don't|no longer) (?:really )?(?:like|love|enjoy|prefer)|"
        r"(?:stopped|have stopped) (?:liking|loving|enjoying|preferring)|dislike|hate) "
        r"(?P<value>[^,.!?;]+)",
        re.IGNORECASE | re.UNICODE,
    )
    CONSTRAINT_PATTERNS = (
        re.compile(r"\b(?:please )?(?:never|do not|don't) (?P<value>[^.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\b(?:please )?(?:always|make sure to) (?P<value>[^.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\bavoid (?P<value>[^.!?;]+)", re.IGNORECASE | re.UNICODE),
    )
    COMMITMENT_PATTERNS = (
        re.compile(r"\bi will (?P<value>[^.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\bwe will (?P<value>[^.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\bremind me to (?P<value>[^.!?;]+)", re.IGNORECASE | re.UNICODE),
    )
    DECISION_PATTERNS = (
        re.compile(r"\b(?:we|i) decided (?:to|that) (?P<value>[^.!?;]+)", re.IGNORECASE | re.UNICODE),
        re.compile(r"\b(?:we|i) chose (?P<value>[^.!?;]+)", re.IGNORECASE | re.UNICODE),
    )
    RELATION_PATTERN = re.compile(
        r"\b(?P<left>[^,.!?;]{2,60}?)\s+"
        r"(?P<relation>works at|knows|likes|visited|is friends with|is married to|is dating)\s+"
        r"(?P<right>[^,.!?;]{2,60})",
        re.IGNORECASE | re.UNICODE,
    )

    def extract_semantic(self, evidence: EvidenceItem) -> list[MemoryCandidate]:
        text = normalize_text(evidence.text)
        if not text:
            return []

        explicit = self._explicit_candidate(evidence)
        if explicit is not None:
            return [explicit]

        candidates: list[MemoryCandidate] = []
        subject = str(evidence.metadata.get("subject") or evidence.scope.user_id or "user")
        entities = extract_entities(text)

        def add(
            *,
            kind: str,
            key: str,
            value: str,
            summary: str,
            confidence: float,
            match: re.Match[str] | None = None,
            metadata: dict[str, Any] | None = None,
            tags: list[str] | None = None,
            durability: str = Durability.DURABLE.value,
            layer: str = MemoryLayer.SEMANTIC.value,
            salience: float = 0.7,
        ) -> None:
            cleaned = self._clean_captured_value(value)
            if not cleaned:
                return
            spans: list[dict[str, Any]] = []
            if match is not None:
                spans.append(
                    {
                        "start": match.start("value"),
                        "end": match.end("value"),
                        "text": match.group("value"),
                    }
                )
            candidates.append(
                MemoryCandidate(
                    kind=kind,
                    key=normalize_key(key).replace(" ", "_"),
                    value=cleaned,
                    summary=summary.format(value=cleaned),
                    confidence=confidence,
                    metadata={
                        "extractor": "deterministic",
                        "topic": normalize_key(key),
                        **dict(metadata or {}),
                    },
                    entity_names=list(dict.fromkeys(entities + extract_entities(cleaned))),
                    tags=list(dict.fromkeys(tags or [kind, normalize_key(key)])),
                    layer=layer,
                    salience=salience,
                    source_type=evidence.source_type,
                    trust_level=evidence.trust_level,
                    durability=durability,
                    subject=subject,
                    observed_at=evidence.occurred_at or evidence.created_at,
                    valid_from=evidence.metadata.get("valid_from") or evidence.occurred_at or evidence.created_at,
                    valid_to=evidence.metadata.get("valid_to"),
                    evidence_spans=spans,
                )
            )

        for key_name, patterns in (
            ("name", self.NAME_PATTERNS),
            ("age", self.AGE_PATTERNS),
            ("location", self.LOCATION_PATTERNS),
            ("employer", self.EMPLOYER_PATTERNS),
            ("occupation", self.OCCUPATION_PATTERNS),
        ):
            for pattern in patterns:
                for match in pattern.finditer(text):
                    add(
                        kind=MemoryKind.PROFILE_ATTRIBUTE.value,
                        key=key_name,
                        value=match.group("value"),
                        summary=f"{key_name.replace('_', ' ').title()}: {{value}}",
                        confidence=0.94,
                        match=match,
                        salience=0.9,
                    )

        for pattern in self.FAVORITE_PATTERNS:
            for match in pattern.finditer(text):
                raw_key = self._clean_captured_value(match.group("key"))
                add(
                    kind=MemoryKind.PREFERENCE.value,
                    key=f"favorite_{raw_key}",
                    value=match.group("value"),
                    summary=f"Favorite {raw_key}: {{value}}",
                    confidence=0.9,
                    match=match,
                    metadata={"sentiment": "positive"},
                    tags=["preference", raw_key, "positive"],
                    salience=0.86,
                )

        for sentiment, pattern in (
            ("positive", self.POSITIVE_PREF_PATTERN),
            ("negative", self.NEGATIVE_PREF_PATTERN),
        ):
            for match in pattern.finditer(text):
                value = self._clean_captured_value(match.group("value"))
                add(
                    kind=MemoryKind.PREFERENCE.value,
                    key=f"preference_{stable_hash(normalize_key(value))[:16]}",
                    value=value,
                    summary=f"{sentiment.title()} preference: {{value}}",
                    confidence=0.82,
                    match=match,
                    metadata={"sentiment": sentiment, "topic": normalize_key(value)},
                    tags=["preference", sentiment],
                    salience=0.76,
                )

        for pattern in self.CONSTRAINT_PATTERNS:
            for match in pattern.finditer(text):
                value = self._clean_captured_value(match.group("value"))
                add(
                    kind=MemoryKind.CONSTRAINT.value,
                    key=f"constraint_{stable_hash(normalize_key(value))[:16]}",
                    value=value,
                    summary="Constraint: {value}",
                    confidence=0.86,
                    match=match,
                    durability=Durability.PINNED.value,
                    salience=0.92,
                )

        for pattern in self.COMMITMENT_PATTERNS:
            for match in pattern.finditer(text):
                value = self._clean_captured_value(match.group("value"))
                add(
                    kind=MemoryKind.COMMITMENT.value,
                    key=f"commitment_{stable_hash(normalize_key(value))[:16]}",
                    value=value,
                    summary="Commitment: {value}",
                    confidence=0.8,
                    match=match,
                    metadata={"commitment_state": "open"},
                    durability=Durability.SESSION.value,
                    salience=0.82,
                )

        for pattern in self.DECISION_PATTERNS:
            for match in pattern.finditer(text):
                value = self._clean_captured_value(match.group("value"))
                add(
                    kind=MemoryKind.DECISION.value,
                    key=f"decision_{stable_hash(normalize_key(value))[:16]}",
                    value=value,
                    summary="Decision: {value}",
                    confidence=0.84,
                    match=match,
                    salience=0.84,
                )

        for match in self.RELATION_PATTERN.finditer(text):
            left = self._clean_captured_value(match.group("left"))
            right = self._clean_captured_value(match.group("right"))
            relation = normalize_key(match.group("relation")).replace(" ", "_")
            if not left or not right:
                continue
            candidates.append(
                MemoryCandidate(
                    kind=MemoryKind.RELATION.value,
                    key=f"{normalize_key(left)}::{relation}::{normalize_key(right)}",
                    value=f"{left} {relation.replace('_', ' ')} {right}",
                    summary=f"Relation: {left} {relation.replace('_', ' ')} {right}",
                    confidence=0.84,
                    metadata={
                        "extractor": "deterministic",
                        "left": left,
                        "right": right,
                        "relation": relation,
                    },
                    entity_names=[left, right],
                    tags=["relation", relation],
                    layer=MemoryLayer.SEMANTIC.value,
                    salience=0.78,
                    source_type=evidence.source_type,
                    trust_level=evidence.trust_level,
                    durability=Durability.DURABLE.value,
                    subject=subject,
                    observed_at=evidence.occurred_at or evidence.created_at,
                    valid_from=evidence.metadata.get("valid_from") or evidence.occurred_at or evidence.created_at,
                    valid_to=evidence.metadata.get("valid_to"),
                    evidence_spans=[
                        {"start": match.start(), "end": match.end(), "text": match.group(0)}
                    ],
                )
            )

        return self._dedupe_candidates(candidates)

    def build_episode_candidate(self, evidence: EvidenceItem) -> MemoryCandidate | None:
        text = normalize_text(evidence.text)
        if not text:
            return None
        token_count = len(tokenize(text))
        base_salience = min(0.1 + (token_count / 40.0), 0.82)
        if evidence.source_type == SourceType.TOOL_RESULT.value:
            base_salience = max(base_salience, 0.78)
        elif evidence.source_type == SourceType.TOOL_CALL.value:
            base_salience = max(base_salience, 0.58)
        elif evidence.source_type == SourceType.ASSISTANT_ACTION.value:
            base_salience = max(base_salience, 0.65)
        elif evidence.source_type == SourceType.USER_MESSAGE.value:
            base_salience = max(base_salience, 0.4)
        elif evidence.source_type in (
            SourceType.RETRIEVED_MEMORY.value,
            SourceType.GENERATED_SUMMARY.value,
        ):
            return None
        if base_salience < 0.38:
            return None
        generic_key = stable_hash(
            text[:300],
            evidence.modality,
            evidence.event_type,
            evidence.name,
            evidence.scope.to_dict(),
        )[:20]
        prefix = {
            SourceType.TOOL_CALL.value: "Tool call",
            SourceType.TOOL_RESULT.value: "Tool result",
            SourceType.ASSISTANT_ACTION.value: "Assistant action",
            SourceType.EXTERNAL_DOCUMENT.value: "External evidence",
        }.get(evidence.source_type, "Episode")
        return MemoryCandidate(
            kind=MemoryKind.EPISODIC_SUMMARY.value,
            key=f"episode_{generic_key}",
            value=text,
            summary=f"{prefix}: {summarize_text(text, max_words=22)}",
            confidence=min(0.55 + (base_salience * 0.4), 0.92),
            state=MemoryState.ACTIVE.value,
            metadata={
                "extractor": "deterministic",
                "modality": evidence.modality,
                "event_type": evidence.event_type,
                "name": evidence.name,
            },
            entity_names=extract_entities(text),
            tags=["episodic", evidence.modality, evidence.event_type],
            layer=MemoryLayer.EPISODIC.value,
            salience=base_salience,
            source_type=evidence.source_type,
            trust_level=evidence.trust_level,
            durability=(
                Durability.EPHEMERAL.value
                if evidence.source_type == SourceType.TOOL_RESULT.value
                else Durability.SESSION.value
            ),
            subject=str(evidence.metadata.get("subject") or evidence.scope.user_id or "user"),
            observed_at=evidence.occurred_at or evidence.created_at,
            valid_from=evidence.occurred_at or evidence.created_at,
            valid_to=evidence.metadata.get("valid_to"),
            evidence_spans=[{"start": 0, "end": len(evidence.text), "text": evidence.text}],
        )

    def extract(self, evidence: EvidenceItem) -> list[MemoryCandidate]:
        semantic = self.extract_semantic(evidence)
        episode = self.build_episode_candidate(evidence)
        return [*semantic, *([episode] if episode is not None else [])]

    def _explicit_candidate(self, evidence: EvidenceItem) -> MemoryCandidate | None:
        metadata = dict(evidence.metadata or {})
        explicit = metadata.get("memory")
        if not isinstance(explicit, dict):
            if not metadata.get("memory_type") and not metadata.get("memory_key"):
                return None
            explicit = metadata
        value = normalize_text(str(explicit.get("value") or evidence.text))
        if not value:
            return None
        kind = str(explicit.get("kind") or explicit.get("memory_type") or MemoryKind.FACT.value)
        key = str(explicit.get("key") or explicit.get("memory_key") or f"fact_{stable_hash(value)[:16]}")
        summary = normalize_text(str(explicit.get("summary") or f"{kind.replace('_', ' ').title()}: {value}"))
        layer = str(explicit.get("layer") or MemoryLayer.SEMANTIC.value)
        return MemoryCandidate(
            kind=kind,
            key=key,
            value=value,
            summary=summary,
            confidence=float(explicit.get("confidence", 1.0)),
            state=str(explicit.get("state", MemoryState.ACTIVE.value)),
            metadata={"extractor": "explicit", **dict(explicit.get("metadata") or {})},
            entity_names=list(explicit.get("entity_names") or extract_entities(value)),
            tags=list(explicit.get("tags") or [kind]),
            layer=layer,
            salience=float(explicit.get("salience", 1.0)),
            source_type=evidence.source_type,
            trust_level=str(explicit.get("trust_level") or evidence.trust_level),
            durability=str(explicit.get("durability") or Durability.DURABLE.value),
            subject=explicit.get("subject") or evidence.metadata.get("subject") or evidence.scope.user_id,
            observed_at=explicit.get("observed_at") or evidence.occurred_at or evidence.created_at,
            valid_from=explicit.get("valid_from") or evidence.occurred_at or evidence.created_at,
            valid_to=explicit.get("valid_to"),
            evidence_spans=[{"start": 0, "end": len(evidence.text), "text": evidence.text}],
        )

    def _clean_captured_value(self, value: str) -> str:
        cleaned = normalize_text(value)
        if not cleaned:
            return ""
        cleaned = self.CLAUSE_BOUNDARY_PATTERN.split(cleaned, maxsplit=1)[0]
        cleaned = cleaned.strip(" ,.!?;:()[]{}\"'")
        cleaned = self.TRAILING_JOINER_PATTERN.sub("", cleaned).strip()
        return normalize_text(cleaned)

    def _dedupe_candidates(self, candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
        seen: set[tuple[str, str, str]] = set()
        output: list[MemoryCandidate] = []
        for candidate in candidates:
            key = (candidate.kind, normalize_key(candidate.key), normalize_key(candidate.value))
            if key in seen:
                continue
            seen.add(key)
            output.append(candidate)
        return output
