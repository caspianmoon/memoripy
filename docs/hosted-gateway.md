# Hosted multi-tenant gateway and inspector

Memoripy v4 includes an authenticated gateway for controlled self-hosted deployments. Tenants are separated in different physical file-store directories and every request is bound to the authenticated tenant's `organization_id`.

## Create a hashed API key

```bash
memoripy key create ./registry.json customer-a \
  --scope memoripy:read \
  --scope memoripy:write
```

The plaintext bearer token is printed once. The registry stores only its SHA-256 digest. Available scopes are:

- `memoripy:read`
- `memoripy:write`
- `memoripy:admin`

List or revoke credentials:

```bash
memoripy key list ./registry.json
memoripy key revoke ./registry.json KEY_ID
```

## Run the gateway

```bash
memoripy gateway ./hosted-data ./registry.json --host 0.0.0.0 --port 8080
```

Open `/inspector` for the zero-build browser inspector. It can search memory, inspect evidence and retrieval receipts, run audits, correct records, and version-delete memories. The token is stored only in browser local storage and sent as a bearer token.

The gateway rejects attempts to supply another tenant's `organization_id`, and read-only keys cannot call mutation routes.

## Security boundary

This gateway provides API-key authentication, permission scopes, revocation, expiry, and physical tenant separation. It is not an identity provider and does not bundle SAML, Okta, or Auth0. Put it behind TLS and an appropriate reverse proxy. Enterprise identity can exchange its own authenticated session for a tenant-scoped gateway token or front the service with an identity-aware proxy.
