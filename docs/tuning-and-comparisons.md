# Retrieval tuning and external comparisons

## Tune retrieval against contracts

```bash
memoripy tune benchmarks/v4_contracts.json \
  --output memoripy-retrieval-profile.json
```

The tuner evaluates multiple retrieval profiles against the same memory contracts and writes the best profile as JSON. Load it in the service, inspector, or gateway:

```bash
memoripy serve ./.memoripy --retrieval-profile memoripy-retrieval-profile.json
memoripy inspector ./.memoripy --retrieval-profile memoripy-retrieval-profile.json
memoripy gateway ./data ./registry.json --retrieval-profile memoripy-retrieval-profile.json
```

The tuner is deliberately bounded. It adjusts retrieval weights and candidate breadth. It cannot rewrite admission, trust, tenant-isolation, or security rules.

## Compare memory systems honestly

```bash
memoripy compare benchmarks/v4_contracts.json --json
```

Adapters are included for Memoripy, Mem0, Hindsight, LangMem, and Graphiti. External systems remain optional and may require model credentials, a database, or a running service.

Unavailable adapters are reported as unavailable and excluded from numerical scoring. Memoripy does not assign competitors fake zeroes merely because their infrastructure or credentials are missing.

Install all optional adapters with:

```bash
pip install "memoripy[comparisons]"
```
