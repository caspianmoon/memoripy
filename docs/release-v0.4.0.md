# Memoripy 0.4.0 release notes

**Agent memory with receipts.**

Memoripy 0.4.0 introduces evidence-first admission, typed and temporal memory, explainable retrieval receipts, auditable correction and forgetting, recovery-safe persistence, an official MCP Python SDK v2 server, tenant-isolated hosting, evidence-grounded assisted extraction, retrieval profile tuning, a zero-build inspector, honest optional competitor adapters, and the scoped `@memoripy/client` TypeScript package.

The base Python runtime remains dependency-free. Optional surfaces are installed through extras.

## Verification

The release candidate was validated with the Python unit suite, memory contracts, MCP v2 smoke tests, TypeScript compilation and package dry-run, Python wheel and source distribution builds, and Twine metadata checks.

## Limitations

Assisted extraction requires a caller-supplied model and rejects proposals that cannot quote supporting evidence. The local tenant gateway uses scoped API keys and does not replace SAML, SCIM, or an enterprise identity provider.
