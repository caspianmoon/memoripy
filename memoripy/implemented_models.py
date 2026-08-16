from __future__ import annotations

import json
from typing import Any
from urllib import parse, request

from .model import ChatModel, EmbeddingModel
from .utils import hashed_embedding, unique_tokens


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


class SimpleKeywordEmbeddingModel(EmbeddingModel):
    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    def get_embedding(self, text: str) -> list[float]:
        return hashed_embedding(text, dimensions=self.dimensions)

    def initialize_embedding_dimension(self) -> int:
        return self.dimensions


class EchoChatModel(ChatModel):
    def __init__(self, model_name: str = "echo-chat"):
        self.model_name = model_name

    def invoke(self, messages: list[dict[str, Any]]) -> str:
        return str(messages[-1].get("content", "")) if messages else ""

    def extract_concepts(self, text: str) -> list[str]:
        return unique_tokens(text)[:10]


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def get_embedding(self, text: str) -> list[float]:
        response = _post_json(
            f"{self.base_url}/embeddings",
            {"model": self.model_name, "input": text},
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return list(response["data"][0]["embedding"])


class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "mxbai-embed-large", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def get_embedding(self, text: str) -> list[float]:
        response = _post_json(
            f"{self.base_url}/api/embeddings",
            {"model": self.model_name, "prompt": text},
            {},
        )
        return list(response["embedding"])


class ChatCompletionsModel(ChatModel):
    def __init__(self, api_endpoint: str, api_key: str, model_name: str):
        self.api_endpoint = api_endpoint.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

    def invoke(self, messages: list[dict[str, Any]]) -> str:
        response = _post_json(
            f"{self.api_endpoint}/chat/completions",
            {"model": self.model_name, "messages": messages},
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return str(response["choices"][0]["message"]["content"])

    def extract_concepts(self, text: str) -> list[str]:
        return unique_tokens(text)[:10]


class OpenAIChatModel(ChatCompletionsModel):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        super().__init__(api_endpoint=base_url, api_key=api_key, model_name=model_name)


class OpenRouterChatModel(ChatCompletionsModel):
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_endpoint="https://openrouter.ai/api/v1", api_key=api_key, model_name=model_name)


class OllamaChatModel(ChatModel):
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def invoke(self, messages: list[dict[str, Any]]) -> str:
        response = _post_json(
            f"{self.base_url}/api/chat",
            {"model": self.model_name, "messages": messages, "stream": False},
            {},
        )
        return str(response["message"]["content"])

    def extract_concepts(self, text: str) -> list[str]:
        return unique_tokens(text)[:10]


class AzureOpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        model_name: str = "text-embedding-3-small",
    ):
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint.rstrip("/")
        self.model_name = model_name

    def get_embedding(self, text: str) -> list[float]:
        query = parse.urlencode({"api-version": self.api_version})
        response = _post_json(
            f"{self.azure_endpoint}/openai/deployments/{self.model_name}/embeddings?{query}",
            {"input": text},
            {"api-key": self.api_key},
        )
        return list(response["data"][0]["embedding"])


class AzureOpenAIChatModel(ChatModel):
    def __init__(
        self,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        model_name: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint.rstrip("/")
        self.model_name = model_name

    def invoke(self, messages: list[dict[str, Any]]) -> str:
        query = parse.urlencode({"api-version": self.api_version})
        response = _post_json(
            f"{self.azure_endpoint}/openai/deployments/{self.model_name}/chat/completions?{query}",
            {"messages": messages},
            {"api-key": self.api_key},
        )
        return str(response["choices"][0]["message"]["content"])

    def extract_concepts(self, text: str) -> list[str]:
        return unique_tokens(text)[:10]
