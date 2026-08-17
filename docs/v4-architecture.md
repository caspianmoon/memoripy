# Memoripy v4 architecture

## Design goals

V4 is designed around four requirements:

- Durable memory must be supported by evidence.
- Current truth must remain distinguishable from historical truth.
- Retrieval signals must not silently exclude one another.
- Memory lifecycle decisions must be inspectable.

## Core objects

### EvidenceItem

An evidence item records the source material encountered by the system. It includes source type, role, writer, trust, scope, occurrence time, URI, content hash, and whether the content was itself retrieved from memory.

Evidence is not automatically a durable memory.

### MemoryCandidate

An extractor produces a candidate with a typed key, value, subject, confidence, durability, validity period, evidence spans, and proposed memory layer.

### AdmissionDecision

The admission policy decides whether a candidate is accepted, deferred, rejected, or quarantined. It returns reason codes and the proposed trust and durability classifications.

### MemoryRecord

A record is the current view of a canonical memory slot. It points to immutable versions and supporting evidence.

### MemoryVersion

A version records an add, merge, supersession, correction, quarantine, or deletion. It contains its own value, state, validity, evidence, reasoning, and links to the version it superseded.

### RetrievalReceipt

A receipt records which retrieval lanes found a result, the rank and contribution from each lane, the scope tier, the final fused score, and inclusion reasons.

### ContextPack

A context pack is the scoped grounding surface returned to an application. It separates profile, preferences, relationships, policies, constraints, commitments, decisions, procedures, recent episodes, and tool observations.

## Ingestion path

```text
input
  -> normalize source envelope
  -> persist evidence
  -> extract candidates
  -> admission policy
  -> reconcile canonical slot
  -> write immutable version
  -> rebuild projections
```

A rejection still produces an admission-log entry. Quarantined content remains inspectable without becoming normal grounding material.

## Retrieval path

```text
query
  -> classify intent
  -> resolve scope tiers
  -> run independent lanes
  -> reciprocal-rank fusion
  -> filter by state, trust, and time
  -> build diverse context pack
  -> attach receipts and citations
```

Lexical matching does not gate semantic matching. Policy and temporal lanes can surface records that ordinary similarity would miss.

## Bitemporal behavior

V4 distinguishes:

- `observed_at`: when the underlying source says the event was observed
- `recorded_at`: when Memoripy persisted it
- `valid_from`: when the memory became true
- `valid_to`: when it stopped being true

A current query excludes expired versions. A historical query can materialize prior versions as read-only results with synthetic identifiers such as `memory_id@version_id`.

## Brain mode

`attention_fast` maintains activation and dormancy but separates retrieval frequency from utility. The projection tracks rehearsal, retrieval, context inclusion, success, correction, rejection, and failure signals.

Consolidation requires independent evidence support. Recalled copies of the same memory do not count as new evidence.

## Persistence

The file repository uses:

- a lock file
- an atomic state replacement
- a known-good backup
- a checksum
- a transaction journal
- append-only events
- state validation before commit

A malformed state file raises `MemoryCorruptionError`. It is never treated as an empty store.
