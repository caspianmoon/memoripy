# Migrating from v3 to v4

## Snapshot migration

V2 and v3 snapshots can be imported through the normal API:

```python
snapshot = old_client.export()
new_client.import_(snapshot, mode="replace")
```

Missing v4 fields receive conservative defaults. The resulting schema version is 4.

Run an audit immediately after migration:

```bash
memoripy audit ./.memoripy --html migration-audit.html
```

## Behavioral changes

### Automatic writes are more conservative

Content that v3 stored may now be deferred, rejected, or quarantined. This is intentional. Inspect `admission_decisions` in capture results.

### Assistant messages do not normally create user facts

V4 preserves assistant messages as evidence and episodes where appropriate, but does not treat assistant self-reports as trusted semantic user memory by default.

### Retrieved memory cannot become new evidence

Any integration that feeds recalled context back into `capture` must label it as `retrieved_memory`. The default policy rejects it as a durable write.

### Temporal history is explicit

Current search excludes expired versions. Use `include_historical=True` or a historical query when prior truth is required.

### Scope supports more fields

Project, organization, and namespace are now first-class scope fields. Existing user, agent, and run behavior remains supported.

### Retrieval uses rank fusion

V4 unions independent retrieval lanes. Score values are therefore not directly comparable to v3 weighted scores.

### Corruption no longer becomes an empty store

A malformed file store raises `MemoryCorruptionError`. Restore with `memoripy recover` after inspecting the corrupt file.

## API compatibility

The main v3 assistant flow remains:

```python
client.capture(...)
client.context.build(...)
client.chat.completions.create(..., memory_strategy="v4")
client.maintenance.consolidate(...)
```

`memory_strategy="v3"` is also accepted as a compatibility alias in the chat surface.

The legacy `MemoryManager` wrapper remains best-effort compatibility. New applications should use `Memory` or `MemoryClient` directly.
