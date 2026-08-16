# FAQ

## Does Memoripy require an LLM?

No. The core, deterministic extractor, local retrieval, file storage, audits, and contract runner have no required third-party dependencies. Applications can supply their own extraction, chat, embedding, or reranking providers.

## Can I use it only for memory management?

Yes. Use `MemoryClient.write`, `add`, `search`, `get`, `history`, `correct`, `forget`, `export`, and `import_` without using chat completions.

## Does it support remote embedding services?

Yes. Supply any object implementing `get_embedding(text)` or `embed(text)` when creating the client.

## What happens to suspicious content?

Depending on the source and policy, it is rejected, deferred, or quarantined. Evidence can remain inspectable without becoming trusted memory.

## Is the built-in HTTP service production-ready by itself?

No. Treat it as a local development surface unless the host application configures authentication, durable storage, transport security, tenancy, rate limits, and monitoring.
