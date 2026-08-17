from __future__ import annotations

from collections import defaultdict
from typing import Any

from .admission import DefaultAdmissionPolicy
from .repository import EngineState
from .types import (
    AuditFinding,
    AuditReport,
    MemoryKind,
    MemoryLayer,
    MemoryState,
    SourceType,
)
from .utils import normalize_key, parse_timestamp, utc_now


def audit_state(state: EngineState) -> AuditReport:
    findings: list[AuditFinding] = []
    active_records = [
        record
        for record in state.memories.values()
        if record.state not in (MemoryState.DELETED.value, MemoryState.SUPERSEDED.value)
    ]

    exact_groups: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    slot_groups: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    for record in active_records:
        scope_key = str(sorted(record.scope.to_dict().items()))
        exact_key = (
            scope_key,
            record.kind,
            normalize_key(record.subject or ""),
            normalize_key(record.key),
            normalize_key(record.value),
        )
        exact_groups[exact_key].append(record.record_id)
        slot_groups[exact_key[:-1]].append(record)

        missing_evidence = [item for item in record.citation_evidence_ids if item not in state.evidence]
        if not record.citation_evidence_ids:
            findings.append(
                AuditFinding(
                    code="UNSUPPORTED_MEMORY",
                    severity="high",
                    message="Memory has no citation evidence.",
                    memory_ids=[record.record_id],
                    suggested_action="Quarantine or attach authoritative evidence before retrieval.",
                )
            )
        elif missing_evidence:
            findings.append(
                AuditFinding(
                    code="MISSING_EVIDENCE",
                    severity="high",
                    message="Memory references evidence that is not present in the store.",
                    memory_ids=[record.record_id],
                    evidence_ids=missing_evidence,
                    suggested_action="Restore the missing evidence or quarantine the memory.",
                )
            )

        if record.source_type == SourceType.RETRIEVED_MEMORY.value:
            findings.append(
                AuditFinding(
                    code="RETRIEVED_MEMORY_FEEDBACK_LOOP",
                    severity="critical",
                    message="Retrieved memory was written back as fresh memory.",
                    memory_ids=[record.record_id],
                    suggested_action="Delete the derived copy and enable the v4 write barrier.",
                )
            )

        if record.layer != MemoryLayer.EPISODIC.value and record.source_type in (
            SourceType.ASSISTANT_MESSAGE.value,
            SourceType.GENERATED_SUMMARY.value,
        ):
            findings.append(
                AuditFinding(
                    code="SELF_AUTHORED_SEMANTIC_MEMORY",
                    severity="high",
                    message="Assistant-generated text became durable semantic memory.",
                    memory_ids=[record.record_id],
                    suggested_action="Require user or tool evidence for durable semantic writes.",
                )
            )

        valid_to = parse_timestamp(record.valid_to)
        if (
            record.state == MemoryState.ACTIVE.value
            and valid_to is not None
            and valid_to <= parse_timestamp(utc_now())
        ):
            findings.append(
                AuditFinding(
                    code="EXPIRED_MEMORY_ACTIVE",
                    severity="high",
                    message="An expired memory is still marked active.",
                    memory_ids=[record.record_id],
                    details={"valid_to": record.valid_to},
                    suggested_action="Mark it historical or superseded.",
                )
            )

        if record.kind in (MemoryKind.PROFILE_ATTRIBUTE.value, MemoryKind.PREFERENCE.value) and not record.scope.user_id:
            findings.append(
                AuditFinding(
                    code="AMBIGUOUS_IDENTITY_SCOPE",
                    severity="medium",
                    message="A user-specific memory has no user_id scope.",
                    memory_ids=[record.record_id],
                    suggested_action="Assign a user scope to prevent identity bleed.",
                )
            )

    for memory_ids in exact_groups.values():
        if len(memory_ids) > 1:
            findings.append(
                AuditFinding(
                    code="EXACT_DUPLICATE_CLUSTER",
                    severity="medium",
                    message=f"{len(memory_ids)} active memories represent the same canonical value.",
                    memory_ids=memory_ids,
                    suggested_action="Merge the records while preserving all evidence.",
                )
            )

    for records in slot_groups.values():
        values = {normalize_key(record.value) for record in records}
        if len(records) > 1 and len(values) > 1:
            findings.append(
                AuditFinding(
                    code="CONFLICTING_CURRENT_FACTS",
                    severity="critical",
                    message="Multiple active values occupy the same canonical memory slot.",
                    memory_ids=[record.record_id for record in records],
                    details={"values": sorted(values)},
                    suggested_action="Resolve the valid-time order or quarantine the conflict.",
                )
            )

    sensitive_evidence: list[str] = []
    instruction_evidence: list[str] = []
    for evidence in state.evidence.values():
        if any(pattern.search(evidence.text) for pattern in DefaultAdmissionPolicy.SENSITIVE_PATTERNS):
            sensitive_evidence.append(evidence.evidence_id)
        if evidence.source_type == SourceType.EXTERNAL_DOCUMENT.value and any(
            pattern.search(evidence.text) for pattern in DefaultAdmissionPolicy.INSTRUCTION_INJECTION_PATTERNS
        ):
            instruction_evidence.append(evidence.evidence_id)
    if sensitive_evidence:
        findings.append(
            AuditFinding(
                code="SENSITIVE_EVIDENCE",
                severity="critical",
                message="Potential secrets or financial identifiers exist in raw evidence.",
                evidence_ids=sensitive_evidence,
                suggested_action="Redact, encrypt, or delete sensitive evidence.",
            )
        )
    if instruction_evidence:
        findings.append(
            AuditFinding(
                code="UNTRUSTED_MEMORY_INSTRUCTION",
                severity="high",
                message="External evidence contains instructions that could poison memory.",
                evidence_ids=instruction_evidence,
                suggested_action="Keep the evidence quarantined and never execute it as instruction.",
            )
        )

    retrieval_total = sum(record.retrieval_count for record in active_records)
    dominant_records = [
        record.record_id
        for record in active_records
        if retrieval_total > 20 and record.retrieval_count / retrieval_total >= 0.5
    ]
    if dominant_records:
        findings.append(
            AuditFinding(
                code="RETRIEVAL_DOMINANCE",
                severity="medium",
                message="One memory accounts for at least half of all retrievals.",
                memory_ids=dominant_records,
                suggested_action="Check for duplicate queries, feedback loops, or over-weighted activation.",
            )
        )

    never_retrieved = [record.record_id for record in active_records if record.retrieval_count == 0]
    rejected_count = sum(
        1
        for item in state.admission_log
        if item.get("action") in ("REJECT", "QUARANTINE")
    )
    duplicate_count = sum(
        max(len(ids) - 1, 0)
        for ids in exact_groups.values()
    )
    supported_count = sum(bool(record.citation_evidence_ids) for record in active_records)
    metrics = {
        "active_memory_count": len(active_records),
        "historical_memory_count": sum(
            record.state in (MemoryState.SUPERSEDED.value, MemoryState.DELETED.value)
            for record in state.memories.values()
        ),
        "citation_coverage": supported_count / max(len(active_records), 1),
        "duplicate_rate": duplicate_count / max(len(active_records), 1),
        "never_retrieved_count": len(never_retrieved),
        "admission_log_count": len(state.admission_log),
        "rejected_or_quarantined_count": rejected_count,
        "retrieval_total": retrieval_total,
    }
    return AuditReport(
        schema_version=state.schema_version,
        generated_at=utc_now(),
        memory_count=len(state.memories),
        evidence_count=len(state.evidence),
        finding_count=len(findings),
        findings=sorted(findings, key=_finding_sort_key),
        metrics=metrics,
    )


def _finding_sort_key(finding: AuditFinding) -> tuple[int, str]:
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    return severity_order.get(finding.severity, 9), finding.code
