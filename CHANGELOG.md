# Changelog

## 0.4.0

Memoripy v4 rebuilds the runtime around evidence-first memory correctness.

### Added

- formal admission barrier with accept, defer, reject, and quarantine outcomes
- source and trust classification
- typed memory for policies, commitments, procedures, decisions, constraints, beliefs, artifacts, and state
- bitemporal validity and immutable version history
- independent retrieval lanes with reciprocal-rank fusion
- retrieval receipts and cited context packs
- adaptive scope expansion with user, agent, run, project, organization, and namespace isolation
- memory audit CLI and static HTML reports
- memory-contract runner
- explicit correction, explanation, feedback, and recovery APIs
- cross-platform locking and corruption-safe file persistence
- optional API-key protection for the local service

### Changed

- automatic extraction is intentionally conservative
- retrieved memory is rejected as fresh evidence
- assistant-authored user facts are rejected by default
- untrusted external instructions and likely secrets are quarantined
- activation separates retrieval frequency from utility feedback
- package metadata moved to `pyproject.toml`

### Compatibility

- v2 and v3 snapshots migrate to schema 4 on import
- v3 capture, context, chat, and maintenance namespaces remain available
- legacy storage wrappers remain best-effort compatibility surfaces
