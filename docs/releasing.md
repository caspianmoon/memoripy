# Releasing Memoripy

The repository uses a two-job trusted-publishing workflow:

1. Build and validate the wheel and source distribution without elevated credentials.
2. Download those exact artifacts in a protected `pypi` environment and publish through PyPI OIDC trusted publishing.

Configure the PyPI trusted publisher with:

- Owner: `caspianmoon`
- Repository: `memoripy`
- Workflow: `publish-pypi.yml`
- Environment: `pypi`

Publish by creating a GitHub release whose tag starts with `v`, or by running the workflow manually with publishing explicitly enabled.

Do not add a long-lived PyPI API token unless trusted publishing is unavailable and the security trade-off has been consciously accepted.
