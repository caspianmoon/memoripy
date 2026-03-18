# Memoripy Examples

This directory contains both the new v3 assistant-first examples and older compatibility examples.

## Start Here

- [v3_basic.py](./v3_basic.py): Capture messages plus tool events, build a context pack, and ground chat completions with `memory_strategy="v3"`.

## Compatibility Examples

- [v2_basic.py](./v2_basic.py): Basic `add/search/chat` usage with the lower-level v2-style API.
- [azure_example.py](./azure_example.py): Using Memoripy with Azure OpenAI chat and embedding models.
- [chatcompletions.py](./chatcompletions.py): Chat completion with an OpenRouter chat model and an Ollama embedding model.
- [openai_example.py](./openai_example.py): Using Memoripy with an OpenAI chat model and an Ollama embedding model.
- [openrouter.py](./openrouter.py): Using Memoripy with an OpenRouter chat model and an Ollama embedding model.
- [dynamo](./dynamo/): Using the Memoripy storage adapter to leverage AWS DynamoDB as the memory persistence layer.
