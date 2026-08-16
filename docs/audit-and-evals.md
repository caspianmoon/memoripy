# Audit and memory contracts

## Audit

`memoripy audit` inspects the current state without mutating it.

```bash
memoripy audit ./.memoripy
memoripy audit ./.memoripy --json
memoripy audit ./.memoripy --html report.html
memoripy audit ./.memoripy --fail-on high
```

Current checks include:

- evidence and version integrity
- citation coverage
- duplicate clusters
- conflicting current values
- re-ingestion loops
- assistant and generated-summary semantic writes
- untrusted external instructions
- likely secrets
- expired active records
- ambiguous user scope
- retrieval dominance

The JSON output is intended for CI. The HTML output is static and local.

## Explain

`memoripy inspect STORE --memory-id ID` prints the current record, validity, trust, evidence, and complete version history.

The Python equivalent is:

```python
explanation = client.explain(memory_id="memory_...")
```

## Memory contracts

A contract describes a sequence of evidence events followed by expected retrieval behavior. It tests the system from the outside rather than checking a private implementation detail.

```bash
memoripy eval
memoripy eval benchmarks/v4_contracts.json
```

Contract fields:

- `name`
- `description`
- `events`
- `queries`

Query assertions:

- `expect_contains`
- `expect_not_contains`
- `expect_empty`

All remaining query fields are passed to `MemoryClient.search`.

## Recommended project contracts

Every application should add contracts for:

- identity isolation
- current and historical state
- authoritative policy retrieval
- corrections
- temporary state expiration
- external-content poisoning
- tool-result durability
- multilingual names and identifiers
- deletion and privacy behavior
- representative production failures

The built-in suite is a regression baseline, not a claim that memory quality is solved universally.
