# Security policy

## Supported version

Security fixes are currently developed for Memoripy v4.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose memory contents, bypass scope isolation, corrupt a store, or turn untrusted content into trusted memory.

Report it privately through GitHub's security advisory flow for this repository. Include:

- affected version or commit
- reproduction steps
- expected and actual behavior
- impact
- whether the issue requires a malicious source, authenticated caller, or local file access

## Security boundaries

Memoripy provides memory admission controls, provenance, scope filtering, file integrity checks, and optional API-key authentication for the local service. It does not by itself provide:

- operating-system process isolation
- encryption at rest
- hosted multi-tenant authorization
- complete secret detection
- regulatory retention compliance
- protection from a compromised application process

Applications must authenticate identities before assigning `user_id`, `organization_id`, or other scope fields. A caller that can lie about scope can defeat isolation at the application boundary.

The bundled HTTP server should be treated as a local-development or controlled-deployment component unless the operator adds deployment-grade authentication, authorization, network policy, logging, rate limits, and secret management.
