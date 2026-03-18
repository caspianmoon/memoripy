# Memoripy Benchmarks

This directory contains a reproducible benchmark harness for Memoripy's core memory behaviors.

For the broader product and API documentation, start with [../docs/index.md](../docs/index.md).

It currently ships:

- A fixed scenario set covering fact extraction, reconciliation, retrieval, grounding, maintenance traceability, and multimodal recall
- A first-party `MemoripyBenchmarkAdapter` configured for `BrainConfig(mode="attention_fast")`
- A target placeholder for `mem0` so the same scenarios can be reused in an external Mem0 environment

Run the built-in benchmark:

```bash
python3 -m benchmarks.runner --target memoripy
python3 -m benchmarks.runner --target memoripy --json
python3 -m benchmarks.runner --latency --json
```

The latency probe builds a fixed synthetic corpus and reports search/context timing so you can compare regressions on the same machine.

Scenario definitions live in [scenarios.py](./scenarios.py). They are plain data so you can reuse them in another adapter or CI pipeline.
