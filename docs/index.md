# Memoripy Documentation

This folder contains the detailed documentation for Memoripy. The README is the landing page; this docs tree is the deeper usage and reference set.

If you are new to the project, start with [README.md](../README.md), then use one of the reading paths below.

## What Memoripy Is

Memoripy is a Python memory framework for LLM applications. It is designed to let assistants and agents:

- ingest conversations, tool calls, tool results, and standalone items
- preserve raw evidence and versioned memory records
- retrieve relevant memory with ranking and citations
- build grounded context packs for later turns
- expose the same behavior through an SDK or HTTP service

It supports both lower-level memory operations and higher-level assistant-turn workflows.

## Reading Paths

### New User Path

If you want to understand the product and get something working quickly:

1. [Concepts and mental model](./concepts.md)
2. [Getting started](./getting-started.md)
3. [Assistant-first memory guide](./assistant-memory.md)
4. [Brain mode and maintenance](./brain-mode-and-maintenance.md)

### Production Integration Path

If you already understand the idea and want to integrate it into an app or service:

1. [Getting started](./getting-started.md)
2. [Service and storage](./service-and-storage.md)
3. [API reference](./api-reference.md)
4. [Troubleshooting and FAQ](./faq.md)

### Contributor and Internals Path

If you need to understand how the implementation works:

1. [Concepts and mental model](./concepts.md)
2. [API reference](./api-reference.md)
3. [Architecture appendix](./architecture.md)
4. [Benchmarks](../benchmarks/README.md)

## Documentation Contents

- [Concepts and mental model](./concepts.md)
- [Getting started](./getting-started.md)
- [Assistant-first memory guide](./assistant-memory.md)
- [Low-level and compatibility guide](./low-level-and-compatibility.md)
- [Brain mode and maintenance](./brain-mode-and-maintenance.md)
- [Service and storage](./service-and-storage.md)
- [API reference](./api-reference.md)
- [Architecture appendix](./architecture.md)
- [Troubleshooting and FAQ](./faq.md)

## Related Repo Guides

- [Examples overview](../examples/README.md)
- [Benchmark harness](../benchmarks/README.md)
- [Top-level README](../README.md)
