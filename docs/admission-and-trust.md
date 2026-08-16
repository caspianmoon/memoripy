# Admission and trust

## Why admission exists

Retrieval quality cannot repair a polluted memory store. V4 therefore treats automatic writing as a policy decision rather than a side effect of extraction.

## Default source classes

- `user_message`
- `assistant_message`
- `system_instruction`
- `tool_result`
- `tool_call`
- `assistant_action`
- `external_document`
- `retrieved_memory`
- `generated_summary`
- `imported_record`
- `explicit_application_write`

Applications should label sources accurately. Unknown source data receives conservative treatment.

## Default trust levels

From strongest to weakest:

1. `authoritative`
2. `user_stated`
3. `observed`
4. `derived`
5. `untrusted_external`
6. `quarantined`

A lower-authority source cannot silently replace a conflicting higher-authority current memory.

## Default decisions

The default policy can return:

- `ADD`
- `DEFER`
- `REJECT`
- `QUARANTINE`

Reconciliation can later convert an accepted candidate into `MERGE` or `SUPERSEDE` based on the canonical slot.

## Important reason codes

- `RETRIEVED_MEMORY_REINGESTION`
- `ASSISTANT_SELF_REPORT`
- `SYSTEM_PROMPT_ECHO`
- `TRANSIENT_STATE`
- `INSUFFICIENT_EVIDENCE`
- `LOW_CONFIDENCE`
- `SENSITIVE_DATA`
- `UNTRUSTED_INSTRUCTION`
- `CONTRADICTS_HIGHER_AUTHORITY`
- `SUBJECT_AMBIGUOUS`

## Custom policy

```python
from memoripy import AdmissionDecision, MemoryAction, MemoryState


class MyPolicy:
    def evaluate(self, *, candidate, evidence, state):
        if candidate.kind == "policy" and evidence.source_type != "explicit_application_write":
            return AdmissionDecision(
                action=MemoryAction.QUARANTINE.value,
                state=MemoryState.QUARANTINED.value,
                reason_codes=["POLICY_REQUIRES_APPROVAL"],
                trust_level="quarantined",
                durability=candidate.durability,
            )
        # Delegate to another policy or return an explicit decision.
```

Pass it through `MemoryPipelineConfig(admission_policy=MyPolicy())`.

## Explicit writes

`MemoryClient.write(...)` is the preferred path for application-owned policies, constraints, and normalized facts. It still creates evidence and immutable versions, but the source is marked authoritative and explicit.

## Sensitive evidence

The default policy can quarantine common secret patterns. This is a guardrail, not a complete data-loss-prevention system. Applications handling regulated or highly sensitive information should sanitize input before capture and implement retention rules outside the default library.
