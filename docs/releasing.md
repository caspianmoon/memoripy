# Releasing Memoripy

Memoripy publishes to PyPI through a dedicated GitHub Actions workflow and PyPI OIDC trusted publishing. No long-lived PyPI token is stored in GitHub.

## Trust boundary

Configure the existing `memoripy` project on PyPI with this exact GitHub publisher identity:

- Owner: `caspianmoon`
- Repository: `memoripy`
- Workflow: `publish-pypi.yml`
- Environment: `pypi`

The workflow grants `id-token: write` only to the final publishing job. The build and verification jobs receive read-only repository access.

The `pypi` GitHub environment should require approval from a trusted maintainer. This adds a human release gate after artifact verification and before PyPI receives an OIDC token.

## Release process

1. Run the complete test and package-validation matrix for the intended release commit.
2. Create a non-draft GitHub release with a version tag such as `v0.4.0`.
3. Attach exactly one Memoripy wheel and one Memoripy source distribution to that release.
4. Open **Actions → Publish Python package**.
5. Select the `master` branch.
6. Enter the existing GitHub release tag.
7. Set `publish` to `true`.
8. Approve the protected `pypi` environment when GitHub requests approval.

The workflow downloads the exact artifacts already attached to the selected GitHub release. It does not rebuild the package from the current branch. Before publishing, it verifies:

- the tag format
- that the GitHub release exists and is not a draft
- that exactly one wheel and one source distribution exist
- that the package name is `memoripy`
- that the wheel version matches the release tag
- that the source-distribution filename matches the version
- that Twine accepts the package metadata
- the SHA256 hashes printed in the workflow log

Publishing is rejected when the workflow is dispatched from any branch other than `master` or when the explicit `publish=true` confirmation is absent.

## Current v0.4.0 release

The GitHub prerelease already contains:

- `memoripy-0.4.0-py3-none-any.whl`
- `memoripy-0.4.0.tar.gz`

After the PyPI trusted publisher is registered, publish it by dispatching the permanent workflow from `master` with `tag=v0.4.0` and `publish=true`.

## Rules

- Do not upload from a developer laptop.
- Do not add a repository or organization-level PyPI token.
- Do not publish artifacts rebuilt from a different commit than the GitHub release.
- Do not approve the `pypi` environment until the verification job has passed.
- Do not reuse a version already present on PyPI. Published PyPI files are immutable.
