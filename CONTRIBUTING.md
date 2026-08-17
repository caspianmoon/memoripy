# Contributing to Memoripy

Memoripy accepts focused fixes, tests, memory contracts, integrations, and documentation improvements.

## Development setup

```bash
git clone https://github.com/caspianmoon/memoripy.git
cd memoripy
git checkout v4
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

## Required checks

```bash
python -m compileall -q memoripy
python -m unittest discover -s tests -v
python -m memoripy eval
python -m build
```

## Contribution rules

- Add or update a memory contract for behavior changes.
- Preserve evidence and version history when changing memory lifecycle behavior.
- Do not weaken user, organization, project, or namespace isolation.
- Do not treat retrieved memory as fresh source evidence.
- Do not silently recover corrupt data as an empty store.
- Keep the base runtime free of required third-party dependencies.
- Label benchmark limitations and comparison conditions honestly.
- Do not commit generated memory stores, event logs, lock files, API keys, or production conversation data.

## Pull requests

Explain:

- the user-visible behavior being changed
- why the current behavior is wrong or insufficient
- the failure case covered by tests or contracts
- storage or migration impact
- validation performed

Small, reviewable changes are easier to merge than broad rewrites without evaluation.
