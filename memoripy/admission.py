from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from .extractors import MemoryCandidate
from .repository import EngineState
from .types import (
    AdmissionDecision,
    AdmissionReason,
    Durability,
    EvidenceItem,
    MemoryAction,
    MemoryKind,
    MemoryLayer,
    MemoryState,
    SourceType,
    TrustLevel,
)
from .utils import normalize_key, normalize_text


class AdmissionPolicy(Protocol):
    def evaluate(
        self,
        *,
        candidate: MemoryCandidate,
        evidence: EvidenceItem,
        state: EngineState,
    ) -> AdmissionDecision:
        ...


@dataclass
class AdmissionConfig:
    minimum_confidence: float = 0.72
    pending_confidence: float = 0.58
    allow_assistant_semantic: bool = False
    allow_generated_summary_semantic: bool = False
    allow_system_policy: bool = True
    quarantine_sensitive_data: bool = True
    quarantine_untrusted_instructions: bool = True
    reject_retrieved_memory: bool = True
    require_evidence_span: bool = True
    admission_log_limit: int = 5000


class DefaultAdmissionPolicy:
    TRANSIENT_PATTERNS = (
        re.compile(r"^(?:ok|okay|thanks|thank you|got it|sure|done|yes|no)[.! ]*$", re.IGNORECASE),
        re.compile(
            r"^(?:(?:ok(?:ay)?|thanks|thank you|got it|sure|done|yes|no)(?:[,.! ]+|$)){1,4}$",
            re.IGNORECASE,
        ),
        re.compile(r"^(?:heartbeat|healthcheck|ping|pong|cron)(?:\s|:|$)", re.IGNORECASE),
        re.compile(r"^(?:status\s*:\s*)?(?:success|healthy|running|completed)$", re.IGNORECASE),
    )
    SENSITIVE_PATTERNS = (
        re.compile(r"\b(?:api[_ -]?key|secret|password|passwd|private[_ -]?key)\s*[:=]\s*\S+", re.IGNORECASE),
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    )
    INSTRUCTION_INJECTION_PATTERNS = (
        re.compile(r"ignore (?:all |any )?(?:previous|prior) instructions", re.IGNORECASE),
        re.compile(r"system prompt", re.IGNORECASE),
        re.compile(r"remember that the user", re.IGNORECASE),
        re.compile(r"store (?:this|the following) (?:as|in) memory", re.IGNORECASE),
        re.compile(r"do not reveal", re.IGNORECASE),
    )
    TRUST_RANK = {
        TrustLevel.QUARANTINED.value: 0,
        TrustLevel.UNTRUSTED_EXTERNAL.value: 1,
        TrustLevel.DERIVED.value: 2,
        TrustLevel.OBSERVED.value: 3,
        TrustLevel.USER_STATED.value: 4,
        TrustLevel.AUTHORITATIVE.value: 5,
    }

    def __init__(self, config: AdmissionConfig | None = None):
        self.config = config or AdmissionConfig()

    def evaluate(
        self,
        *,
        candidate: MemoryCandidate,
        evidence: EvidenceItem,
        state: EngineState,
    ) -> AdmissionDecision:
        reasons: list[str] = []
        source = evidence.source_type
        trust = candidate.trust_level or evidence.trust_level
        durability = candidate.durability or Durability.DURABLE.value
        text = normalize_text(evidence.text)

        if self.config.reject_retrieved_memory and (
            evidence.is_retrieved_memory
            or source == SourceType.RETRIEVED_MEMORY.value
            or bool(evidence.metadata.get("retrieved_memory"))
        ):
            return self._decision(
                MemoryAction.REJECT.value,
                MemoryState.DELETED.value,
                [AdmissionReason.RETRIEVED_MEMORY_REINGESTION.value],
                candidate,
                TrustLevel.QUARANTINED.value,
                Durability.EPHEMERAL.value,
            )

        if any(pattern.search(text) for pattern in self.SENSITIVE_PATTERNS):
            action = MemoryAction.QUARANTINE.value if self.config.quarantine_sensitive_data else MemoryAction.REJECT.value
            return self._decision(
                action,
                MemoryState.QUARANTINED.value,
                [AdmissionReason.SENSITIVE_DATA.value],
                candidate,
                TrustLevel.QUARANTINED.value,
                Durability.EPHEMERAL.value,
            )

        if source == SourceType.EXTERNAL_DOCUMENT.value and any(
            pattern.search(text) for pattern in self.INSTRUCTION_INJECTION_PATTERNS
        ):
            action = MemoryAction.QUARANTINE.value if self.config.quarantine_untrusted_instructions else MemoryAction.REJECT.value
            return self._decision(
                action,
                MemoryState.QUARANTINED.value,
                [AdmissionReason.UNTRUSTED_INSTRUCTION.value],
                candidate,
                TrustLevel.QUARANTINED.value,
                Durability.EPHEMERAL.value,
            )

        if source == SourceType.ASSISTANT_MESSAGE.value and candidate.layer != MemoryLayer.EPISODIC.value:
            explicit = candidate.metadata.get("extractor") == "explicit"
            if not self.config.allow_assistant_semantic and not explicit:
                return self._decision(
                    MemoryAction.REJECT.value,
                    MemoryState.DELETED.value,
                    [AdmissionReason.ASSISTANT_SELF_REPORT.value],
                    candidate,
                    TrustLevel.DERIVED.value,
                    Durability.EPHEMERAL.value,
                )

        if source == SourceType.GENERATED_SUMMARY.value and candidate.layer != MemoryLayer.EPISODIC.value:
            if not self.config.allow_generated_summary_semantic:
                return self._decision(
                    MemoryAction.REJECT.value,
                    MemoryState.DELETED.value,
                    [AdmissionReason.SYSTEM_PROMPT_ECHO.value],
                    candidate,
                    TrustLevel.DERIVED.value,
                    Durability.EPHEMERAL.value,
                )

        if source == SourceType.SYSTEM_INSTRUCTION.value:
            allowed_kind = candidate.kind in (MemoryKind.POLICY.value, MemoryKind.CONSTRAINT.value)
            if not self.config.allow_system_policy or not allowed_kind:
                return self._decision(
                    MemoryAction.REJECT.value,
                    MemoryState.DELETED.value,
                    [AdmissionReason.SYSTEM_PROMPT_ECHO.value],
                    candidate,
                    TrustLevel.AUTHORITATIVE.value,
                    Durability.PINNED.value,
                )
            trust = TrustLevel.AUTHORITATIVE.value
            durability = Durability.PINNED.value

        if candidate.layer == MemoryLayer.EPISODIC.value and any(
            pattern.search(text) for pattern in self.TRANSIENT_PATTERNS
        ):
            return self._decision(
                MemoryAction.REJECT.value,
                MemoryState.DELETED.value,
                [AdmissionReason.TRANSIENT_STATE.value],
                candidate,
                trust,
                Durability.EPHEMERAL.value,
            )

        if not candidate.value or not evidence.evidence_id:
            return self._decision(
                MemoryAction.REJECT.value,
                MemoryState.DELETED.value,
                [AdmissionReason.INSUFFICIENT_EVIDENCE.value],
                candidate,
                trust,
                durability,
            )

        if (
            self.config.require_evidence_span
            and candidate.layer != MemoryLayer.EPISODIC.value
            and candidate.metadata.get("extractor") != "explicit"
            and not candidate.evidence_spans
        ):
            return self._decision(
                MemoryAction.DEFER.value,
                MemoryState.PENDING.value,
                [AdmissionReason.INSUFFICIENT_EVIDENCE.value],
                candidate,
                trust,
                durability,
            )

        if source == SourceType.EXPLICIT_APPLICATION_WRITE.value:
            reasons.append(AdmissionReason.EXPLICIT_WRITE.value)
            trust = TrustLevel.AUTHORITATIVE.value

        if source == SourceType.EXTERNAL_DOCUMENT.value:
            trust = TrustLevel.UNTRUSTED_EXTERNAL.value
            if candidate.layer != MemoryLayer.EPISODIC.value:
                return self._decision(
                    MemoryAction.DEFER.value,
                    MemoryState.PENDING.value,
                    [AdmissionReason.SOURCE_NOT_TRUSTED.value],
                    candidate,
                    trust,
                    durability,
                )

        if source == SourceType.TOOL_RESULT.value and candidate.layer == MemoryLayer.SEMANTIC.value:
            trust = TrustLevel.OBSERVED.value
            if durability == Durability.DURABLE.value and candidate.metadata.get("extractor") != "explicit":
                durability = Durability.SESSION.value
                reasons.append(AdmissionReason.LOW_DURABILITY.value)

        authority_conflict = self._higher_authority_conflict(
            state=state,
            candidate=candidate,
            incoming_trust=trust,
        )
        if authority_conflict is not None:
            return self._decision(
                MemoryAction.QUARANTINE.value,
                MemoryState.QUARANTINED.value,
                [AdmissionReason.CONTRADICTS_HIGHER_AUTHORITY.value],
                candidate,
                TrustLevel.QUARANTINED.value,
                durability,
                metadata={"conflicting_memory_id": authority_conflict},
            )

        if candidate.confidence < self.config.pending_confidence:
            return self._decision(
                MemoryAction.REJECT.value,
                MemoryState.DELETED.value,
                [AdmissionReason.LOW_CONFIDENCE.value],
                candidate,
                trust,
                durability,
            )
        if candidate.confidence < self.config.minimum_confidence:
            return self._decision(
                MemoryAction.DEFER.value,
                MemoryState.PENDING.value,
                [AdmissionReason.LOW_CONFIDENCE.value],
                candidate,
                trust,
                durability,
            )

        reasons.append(AdmissionReason.ACCEPTED.value)
        return self._decision(
            MemoryAction.ADD.value,
            MemoryState.ACTIVE.value,
            reasons,
            candidate,
            trust,
            durability,
        )

    def _higher_authority_conflict(
        self,
        *,
        state: EngineState,
        candidate: MemoryCandidate,
        incoming_trust: str,
    ) -> str | None:
        incoming_rank = self.TRUST_RANK.get(incoming_trust, 0)
        candidate_subject = normalize_key(candidate.subject or "")
        candidate_key = normalize_key(candidate.key)
        candidate_value = normalize_key(candidate.value)
        for record in state.memories.values():
            if record.state != MemoryState.ACTIVE.value:
                continue
            if record.kind != candidate.kind or normalize_key(record.key) != candidate_key:
                continue
            if normalize_key(record.subject or "") != candidate_subject:
                continue
            if normalize_key(record.value) == candidate_value:
                continue
            if self.TRUST_RANK.get(record.trust_level, 0) > incoming_rank:
                return record.record_id
        return None

    def _decision(
        self,
        action: str,
        state: str,
        reasons: list[str],
        candidate: MemoryCandidate,
        trust_level: str,
        durability: str,
        metadata: dict[str, Any] | None = None,
    ) -> AdmissionDecision:
        return AdmissionDecision(
            action=action,
            state=state,
            reason_codes=list(dict.fromkeys(reasons)),
            confidence=float(candidate.confidence),
            trust_level=trust_level,
            durability=durability,
            metadata=dict(metadata or {}),
        )
