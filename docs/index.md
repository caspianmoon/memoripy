# Memoripy v4 documentation

Memoripy v4 is an evidence-first memory runtime for AI agents. The public design is organized around six operations:

1. Capture evidence.
2. Admit, defer, reject, or quarantine candidate memories.
3. Preserve current and historical versions.
4. Retrieve through independent lanes and rank fusion.
5. Build a scoped, cited context pack.
6. Audit what the system remembered and why.

Start with the repository [README](../README.md), then use these guides:

- [V4 architecture](./v4-architecture.md)
- [Admission and trust](./admission-and-trust.md)
- [Audit and memory contracts](./audit-and-evals.md)
- [Migration from v3](./migration-v3-v4.md)
- [API reference](./api-reference.md)

The older v3 concept guides remain in the repository for historical context, but v4 behavior takes precedence where they conflict.
