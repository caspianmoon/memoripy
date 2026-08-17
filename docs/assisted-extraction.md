# Assisted and temporal extraction

Memoripy's dependency-free extractor is deliberately conservative. Applications that need broader language coverage can opt into `AssistedMemoryExtractor` without changing the storage or admission model.

```python
from memoripy import AssistedMemoryExtractor, MemoryClient, MemoryPipelineConfig
from memoripy.implemented_models import OpenAIChatModel

extractor = AssistedMemoryExtractor(
    OpenAIChatModel(api_key="...", model_name="gpt-5-mini")
)
client = MemoryClient(
    pipeline=MemoryPipelineConfig(extractor=extractor)
)
```

The model must return structured JSON. Every automatically accepted candidate must be grounded in an exact evidence span. Model output cannot promote its own trust level. Trust continues to come from the evidence source and the admission policy.

The supported fields are:

- `kind`
- `key`
- `value`
- `summary`
- `confidence`
- `durability`
- `layer`
- `subject`
- `valid_from`
- `valid_to`
- `tags`
- `metadata`
- `quote` or `evidence_spans`

Candidates with unknown types, insufficient confidence, invalid JSON, or missing grounded evidence are rejected.

## Deterministic temporal parsing

The default extractor recognizes explicit ISO dates, date ranges, `since`, `until`, and common relative phrases such as `yesterday`, `last week`, `last month`, and `three days ago`.

```python
client.capture(
    messages=[{
        "role": "user",
        "content": "I moved to Istanbul last month",
        "timestamp": "2026-08-17T12:00:00Z",
    }],
    user_id="u1",
)
```

The location memory receives an observation time of August 17, 2026 and a validity start of July 17, 2026. More complex temporal language should use assisted extraction or an explicit application write.
