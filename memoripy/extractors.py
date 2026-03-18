from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .types import EventType, EvidenceItem, MemoryKind, MemoryLayer, MemoryState
from .utils import extract_entities, normalize_text, stable_hash, summarize_text, tokenize


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
    source_type: str = EventType.INGESTION.value


class DefaultMemoryExtractor:
    NAME_PATTERN = re.compile(r"\bmy name is (?P<value>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b", re.IGNORECASE)
    AGE_PATTERN = re.compile(r"\bi(?: am|'m) (?P<value>\d{1,3}) years old\b", re.IGNORECASE)
    LOCATION_PATTERN = re.compile(r"\bi (?:live in|am from|moved to) (?P<value>[^,.!?]+)", re.IGNORECASE)
    EMPLOYER_PATTERN = re.compile(r"\bi work (?:at|for) (?P<value>[^,.!?]+)", re.IGNORECASE)
    OCCUPATION_PATTERN = re.compile(r"\bi work as (?P<value>[^,.!?]+)", re.IGNORECASE)
    FAVORITE_PATTERN = re.compile(r"\bmy favorite (?P<key>[a-zA-Z ]+?) is (?P<value>[^,.!?]+)", re.IGNORECASE)
    POSITIVE_PREF_PATTERN = re.compile(
        r"\bi (?:really )?(?:like|love|enjoy|prefer) (?P<value>[^,.!?]+)",
        re.IGNORECASE,
    )
    NEGATIVE_PREF_PATTERN = re.compile(
        r"\bi (?:do not|don't|dislike|hate) (?P<value>[^,.!?]+)",
        re.IGNORECASE,
    )
    RELATION_PATTERN = re.compile(
        r"\b(?P<left>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?P<relation>works at|knows|likes|visited)\s+(?P<right>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)",
        re.IGNORECASE,
    )

    def extract_semantic(self, evidence: EvidenceItem) -> list[MemoryCandidate]:
        text = normalize_text(evidence.text)
        if not text:
            return []

        candidates: list[MemoryCandidate] = []
        entities = extract_entities(text)

        def add_candidate(candidate: MemoryCandidate) -> None:
            if candidate.entity_names:
                candidate.entity_names = list(dict.fromkeys(candidate.entity_names))
            else:
                candidate.entity_names = entities
            candidates.append(candidate)

        for pattern, key_name, kind in (
            (self.NAME_PATTERN, "name", MemoryKind.PROFILE_ATTRIBUTE.value),
            (self.AGE_PATTERN, "age", MemoryKind.PROFILE_ATTRIBUTE.value),
            (self.LOCATION_PATTERN, "location", MemoryKind.PROFILE_ATTRIBUTE.value),
            (self.EMPLOYER_PATTERN, "employer", MemoryKind.PROFILE_ATTRIBUTE.value),
            (self.OCCUPATION_PATTERN, "occupation", MemoryKind.PROFILE_ATTRIBUTE.value),
        ):
            for match in pattern.finditer(text):
                value = normalize_text(match.group("value"))
                if not value:
                    continue
                add_candidate(
                    MemoryCandidate(
                        kind=kind,
                        key=key_name,
                        value=value,
                        summary=f"{key_name.replace('_', ' ').title()}: {value}",
                        confidence=0.94,
                        metadata={"source": "rule", "pattern": key_name},
                        entity_names=entities + extract_entities(value),
                        tags=[key_name],
                        salience=0.9,
                        source_type=evidence.source_type,
                    )
                )

        for match in self.FAVORITE_PATTERN.finditer(text):
            raw_key = normalize_text(match.group("key")).lower().replace(" ", "_")
            value = normalize_text(match.group("value"))
            if not raw_key or not value:
                continue
            add_candidate(
                MemoryCandidate(
                    kind=MemoryKind.PREFERENCE.value,
                    key=f"favorite_{raw_key}",
                    value=value,
                    summary=f"Favorite {raw_key.replace('_', ' ')}: {value}",
                    confidence=0.9,
                    metadata={"source": "rule", "pattern": "favorite"},
                    entity_names=entities + extract_entities(value),
                    tags=["preference", raw_key],
                    salience=0.88,
                    source_type=evidence.source_type,
                )
            )

        for match in self.POSITIVE_PREF_PATTERN.finditer(text):
            value = normalize_text(match.group("value"))
            if not value:
                continue
            key = f"prefers_{stable_hash(value)[:12]}"
            add_candidate(
                MemoryCandidate(
                    kind=MemoryKind.PREFERENCE.value,
                    key=key,
                    value=value,
                    summary=f"Positive preference: {value}",
                    confidence=0.82,
                    metadata={"source": "rule", "sentiment": "positive"},
                    entity_names=entities + extract_entities(value),
                    tags=["preference", "positive"],
                    salience=0.74,
                    source_type=evidence.source_type,
                )
            )

        for match in self.NEGATIVE_PREF_PATTERN.finditer(text):
            value = normalize_text(match.group("value"))
            if not value:
                continue
            key = f"avoids_{stable_hash(value)[:12]}"
            add_candidate(
                MemoryCandidate(
                    kind=MemoryKind.PREFERENCE.value,
                    key=key,
                    value=value,
                    summary=f"Negative preference: {value}",
                    confidence=0.82,
                    metadata={"source": "rule", "sentiment": "negative"},
                    entity_names=entities + extract_entities(value),
                    tags=["preference", "negative"],
                    salience=0.74,
                    source_type=evidence.source_type,
                )
            )

        for match in self.RELATION_PATTERN.finditer(text):
            left = normalize_text(match.group("left"))
            right = normalize_text(match.group("right"))
            relation = normalize_text(match.group("relation")).lower().replace(" ", "_")
            if not left or not right:
                continue
            add_candidate(
                MemoryCandidate(
                    kind=MemoryKind.RELATION.value,
                    key=f"{left.lower().replace(' ', '_')}::{relation}::{right.lower().replace(' ', '_')}",
                    value=f"{left} {relation.replace('_', ' ')} {right}",
                    summary=f"Relation: {left} {relation.replace('_', ' ')} {right}",
                    confidence=0.86,
                    metadata={"source": "rule", "left": left, "right": right, "relation": relation},
                    entity_names=[left, right],
                    tags=["relation", relation],
                    salience=0.8,
                    source_type=evidence.source_type,
                )
            )

        return candidates

    def build_episode_candidate(self, evidence: EvidenceItem) -> MemoryCandidate | None:
        text = normalize_text(evidence.text)
        if not text:
            return None

        token_count = len(tokenize(text))
        base_salience = min(0.08 + (token_count / 25.0), 0.85)
        if evidence.event_type == EventType.TOOL_RESULT.value:
            base_salience = max(base_salience, 0.82)
        elif evidence.event_type == EventType.TOOL_CALL.value:
            base_salience = max(base_salience, 0.65)
        elif evidence.event_type == EventType.ASSISTANT_ACTION.value:
            base_salience = max(base_salience, 0.7)
        elif evidence.role == "assistant":
            base_salience = min(base_salience, 0.55)

        if base_salience < 0.35:
            return None

        generic_key = stable_hash(text[:200], evidence.modality, evidence.event_type, evidence.name)[:16]
        summary_prefix = {
            EventType.MESSAGE.value: "Episode",
            EventType.TOOL_CALL.value: "Tool call",
            EventType.TOOL_RESULT.value: "Tool result",
            EventType.ASSISTANT_ACTION.value: "Assistant action",
        }.get(evidence.event_type, "Episode")
        summary = f"{summary_prefix}: {summarize_text(text, max_words=18)}"
        state = MemoryState.ACTIVE.value if base_salience >= 0.45 or evidence.event_type != EventType.MESSAGE.value else MemoryState.PENDING.value
        return MemoryCandidate(
            kind=MemoryKind.EPISODIC_SUMMARY.value,
            key=f"episode_{generic_key}",
            value=text,
            summary=summary,
            confidence=min(0.55 + (base_salience * 0.45), 0.95),
            state=state,
            metadata={
                "source": "episodic",
                "modality": evidence.modality,
                "event_type": evidence.event_type,
                "name": evidence.name,
            },
            entity_names=extract_entities(text),
            tags=["episodic", evidence.modality, evidence.event_type],
            layer=MemoryLayer.EPISODIC.value,
            salience=base_salience,
            source_type=evidence.source_type,
        )

    def extract(self, evidence: EvidenceItem) -> list[MemoryCandidate]:
        candidates = self.extract_semantic(evidence)
        if candidates:
            return candidates

        episode = self.build_episode_candidate(evidence)
        if episode is None:
            return []
        return [
            MemoryCandidate(
                kind=episode.kind,
                key=episode.key,
                value=episode.value,
                summary=episode.summary,
                confidence=episode.confidence,
                state=episode.state,
                metadata=episode.metadata,
                entity_names=episode.entity_names,
                tags=episode.tags,
                layer=episode.layer,
                salience=episode.salience,
                source_type=episode.source_type,
            )
        ]
