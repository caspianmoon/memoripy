# Memoripy v4 memory contracts

The old benchmark harness was useful as a regression scaffold, but it did not justify public comparative claims. V4 treats behavior contracts as the minimum trustworthy evaluation surface.

Run the built-in contracts:

```bash
python -m memoripy eval
```

Run the JSON examples in this directory:

```bash
python -m memoripy eval benchmarks/v4_contracts.json
```

Contracts should test externally visible behavior such as temporal updates, scope isolation, poisoning resistance, citation coverage, corrections, expiration, and procedural reuse.

A serious comparative benchmark must use equivalent models, extractors, prompts, scope rules, and datasets for every target. The repository does not present the built-in contracts as a leaderboard.
