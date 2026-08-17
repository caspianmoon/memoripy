from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from .client import MemoryClient
from .evals import MemoryContract


@dataclass(frozen=True)
class AdapterAvailability:
    available: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "reason": self.reason}


class MemoryComparisonAdapter(Protocol):
    name: str

    def availability(self) -> AdapterAvailability: ...

    def retain(self, *, scope: str, content: str, timestamp: str | None = None) -> None: ...

    def recall(self, *, scope: str, query: str, limit: int = 5) -> list[str]: ...

    def close(self) -> None: ...


class MemoripyComparisonAdapter:
    name = "memoripy"

    def __init__(self, client: MemoryClient | None = None) -> None:
        self.client = client or MemoryClient()

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability(True)

    def retain(self, *, scope: str, content: str, timestamp: str | None = None) -> None:
        self.client.capture(
            messages=[{"role": "user", "content": content, "timestamp": timestamp}],
            user_id=scope,
        )

    def recall(self, *, scope: str, query: str, limit: int = 5) -> list[str]:
        result = self.client.search(query=query, user_id=scope, limit=limit, track_usage=False)
        return [str(item["memory"]["value"]) for item in result["results"]]

    def close(self) -> None:
        return None


class Mem0ComparisonAdapter:
    name = "mem0"

    def __init__(self, memory: Any | None = None, *, config: dict[str, Any] | None = None) -> None:
        self._error: str | None = None
        self.memory = memory
        if self.memory is None:
            try:
                from mem0 import Memory

                self.memory = Memory.from_config(config) if config else Memory()
            except Exception as exc:  # optional dependency/provider configuration
                self._error = str(exc)

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability(self.memory is not None, self._error)

    def retain(self, *, scope: str, content: str, timestamp: str | None = None) -> None:
        if self.memory is None:
            raise RuntimeError(self._error or "Mem0 is unavailable")
        messages = [{"role": "user", "content": content}]
        kwargs = {"user_id": scope}
        if timestamp:
            kwargs["metadata"] = {"timestamp": timestamp}
        try:
            self.memory.add(messages, **kwargs)
        except TypeError:
            self.memory.add(content, **kwargs)

    def recall(self, *, scope: str, query: str, limit: int = 5) -> list[str]:
        if self.memory is None:
            raise RuntimeError(self._error or "Mem0 is unavailable")
        payload = self.memory.search(query=query, user_id=scope, limit=limit)
        results = payload.get("results", payload) if isinstance(payload, dict) else payload
        return [_extract_text(item) for item in list(results or [])[:limit]]

    def close(self) -> None:
        return None


class HindsightComparisonAdapter:
    name = "hindsight"

    def __init__(self, client: Any | None = None, *, base_url: str | None = None) -> None:
        self._error: str | None = None
        self.client = client
        if self.client is None:
            try:
                from hindsight_client import Hindsight

                self.client = Hindsight(base_url=base_url or os.environ.get("HINDSIGHT_BASE_URL", "http://127.0.0.1:8888"))
            except Exception as exc:
                self._error = str(exc)

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability(self.client is not None, self._error)

    def retain(self, *, scope: str, content: str, timestamp: str | None = None) -> None:
        if self.client is None:
            raise RuntimeError(self._error or "Hindsight is unavailable")
        kwargs: dict[str, Any] = {"bank_id": scope, "content": content}
        if timestamp:
            kwargs["timestamp"] = timestamp
        self.client.retain(**kwargs)

    def recall(self, *, scope: str, query: str, limit: int = 5) -> list[str]:
        if self.client is None:
            raise RuntimeError(self._error or "Hindsight is unavailable")
        payload = self.client.recall(bank_id=scope, query=query, max_tokens=4096)
        return _normalize_results(payload, limit=limit)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


class LangMemComparisonAdapter:
    name = "langmem"

    def __init__(self, *, model: str | None = None) -> None:
        self._error: str | None = None
        self._managers: dict[str, Any] = {}
        self.model = model or os.environ.get("LANGMEM_MODEL")
        self.store: Any | None = None
        self.create_manager: Any | None = None
        try:
            from langgraph.store.memory import InMemoryStore
            from langmem import create_memory_store_manager

            if not self.model:
                raise RuntimeError("LANGMEM_MODEL is required")
            self.store = InMemoryStore()
            self.create_manager = create_memory_store_manager
        except Exception as exc:
            self._error = str(exc)

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability(self.store is not None and self.create_manager is not None, self._error)

    def _manager(self, scope: str) -> Any:
        if scope not in self._managers:
            self._managers[scope] = self.create_manager(
                self.model,
                namespace=("memoripy-comparison", scope),
                store=self.store,
            )
        return self._managers[scope]

    def retain(self, *, scope: str, content: str, timestamp: str | None = None) -> None:
        if self.store is None or self.create_manager is None:
            raise RuntimeError(self._error or "LangMem is unavailable")
        message: dict[str, Any] = {"role": "user", "content": content}
        if timestamp:
            message["metadata"] = {"timestamp": timestamp}
        self._manager(scope).invoke({"messages": [message]})

    def recall(self, *, scope: str, query: str, limit: int = 5) -> list[str]:
        if self.store is None:
            raise RuntimeError(self._error or "LangMem is unavailable")
        results = self.store.search(("memoripy-comparison", scope), query=query, limit=limit)
        return [_extract_text(getattr(item, "value", item)) for item in results]

    def close(self) -> None:
        return None


class GraphitiComparisonAdapter:
    name = "graphiti"

    def __init__(
        self,
        graphiti: Any | None = None,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self._error: str | None = None
        self.graphiti = graphiti
        if self.graphiti is None:
            try:
                from graphiti_core import Graphiti

                uri = uri or os.environ.get("NEO4J_URI")
                user = user or os.environ.get("NEO4J_USER")
                password = password or os.environ.get("NEO4J_PASSWORD")
                if not all((uri, user, password)):
                    raise RuntimeError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are required")
                self.graphiti = Graphiti(uri, user, password)
                asyncio.run(self.graphiti.build_indices_and_constraints())
            except Exception as exc:
                self._error = str(exc)

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability(self.graphiti is not None, self._error)

    def retain(self, *, scope: str, content: str, timestamp: str | None = None) -> None:
        if self.graphiti is None:
            raise RuntimeError(self._error or "Graphiti is unavailable")
        from graphiti_core.nodes import EpisodeType

        reference = _parse_timestamp(timestamp)
        asyncio.run(
            self.graphiti.add_episode(
                name=f"memoripy-contract-{scope}-{reference.timestamp()}",
                episode_body=content,
                source=EpisodeType.text,
                source_description=f"memory contract scope={scope}",
                reference_time=reference,
                group_id=scope,
            )
        )

    def recall(self, *, scope: str, query: str, limit: int = 5) -> list[str]:
        if self.graphiti is None:
            raise RuntimeError(self._error or "Graphiti is unavailable")
        results = asyncio.run(self.graphiti.search(query, group_ids=[scope], num_results=limit))
        return [_extract_text(item) for item in results]

    def close(self) -> None:
        if self.graphiti is not None:
            asyncio.run(self.graphiti.close())


@dataclass
class AdapterComparisonResult:
    adapter: str
    availability: AdapterAvailability
    contract_count: int = 0
    passed_count: int = 0
    failed_count: int = 0
    score_ratio: float | None = None
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "availability": self.availability.to_dict(),
            "contract_count": self.contract_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "score_ratio": self.score_ratio,
            "failures": list(self.failures),
        }


def run_comparison(
    contracts: list[MemoryContract],
    adapters: list[MemoryComparisonAdapter],
) -> dict[str, Any]:
    results: list[AdapterComparisonResult] = []
    for adapter in adapters:
        availability = adapter.availability()
        if not availability.available:
            results.append(AdapterComparisonResult(adapter=adapter.name, availability=availability))
            continue
        passed = 0
        failures: list[str] = []
        try:
            for contract in contracts:
                contract_passed, contract_failures = _run_adapter_contract(adapter, contract)
                if contract_passed:
                    passed += 1
                else:
                    failures.extend(f"{contract.name}:{failure}" for failure in contract_failures)
        except Exception as exc:
            availability = AdapterAvailability(False, f"runtime failure: {exc}")
            results.append(AdapterComparisonResult(adapter=adapter.name, availability=availability))
        else:
            results.append(
                AdapterComparisonResult(
                    adapter=adapter.name,
                    availability=availability,
                    contract_count=len(contracts),
                    passed_count=passed,
                    failed_count=len(contracts) - passed,
                    score_ratio=passed / max(len(contracts), 1),
                    failures=failures,
                )
            )
        finally:
            try:
                adapter.close()
            except Exception:
                pass
    return {
        "contract_count": len(contracts),
        "results": [item.to_dict() for item in results],
        "available_adapter_count": sum(item.availability.available for item in results),
        "note": "Unavailable adapters are excluded rather than assigned fabricated scores.",
    }


def build_default_adapters(names: list[str]) -> list[MemoryComparisonAdapter]:
    factories = {
        "memoripy": MemoripyComparisonAdapter,
        "mem0": Mem0ComparisonAdapter,
        "hindsight": HindsightComparisonAdapter,
        "langmem": LangMemComparisonAdapter,
        "graphiti": GraphitiComparisonAdapter,
    }
    output = []
    for name in names:
        normalized = name.strip().casefold()
        if normalized not in factories:
            raise ValueError(f"Unknown comparison adapter: {name}")
        output.append(factories[normalized]())
    return output


def _run_adapter_contract(adapter: MemoryComparisonAdapter, contract: MemoryContract) -> tuple[bool, list[str]]:
    failures: list[str] = []
    default_scope = f"contract-{contract.name}"
    for event in contract.events:
        scope = str(event.get("user_id") or default_scope)
        timestamp = event.get("occurred_at")
        contents = _event_contents(event)
        for content in contents:
            adapter.retain(scope=scope, content=content, timestamp=timestamp)
    for index, query in enumerate(contract.queries):
        scope = str(query.get("user_id") or default_scope)
        values = adapter.recall(scope=scope, query=str(query.get("query", "")), limit=int(query.get("limit", 5)))
        rendered = json.dumps(values, ensure_ascii=False)
        query_failures = []
        for expected in query.get("expect_contains", []):
            if str(expected) not in rendered:
                query_failures.append(f"query[{index}]:missing:{expected}")
        for unexpected in query.get("expect_not_contains", []):
            if str(unexpected) in rendered:
                query_failures.append(f"query[{index}]:unexpected:{unexpected}")
        if query.get("expect_empty") and values:
            query_failures.append(f"query[{index}]:expected_empty")
        failures.extend(query_failures)
    return not failures, failures


def _event_contents(event: dict[str, Any]) -> list[str]:
    output = []
    for message in event.get("messages") or []:
        if str(message.get("role", "user")) == "user" and message.get("content"):
            output.append(str(message["content"]))
    for item in event.get("items") or []:
        if item.get("content"):
            output.append(str(item["content"]))
    if event.get("text"):
        output.append(str(event["text"]))
    return output


def _extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("memory", "content", "text", "fact", "value", "summary"):
            if key in item and item[key] is not None:
                value = item[key]
                return _extract_text(value) if isinstance(value, (dict, list)) else str(value)
        return json.dumps(item, ensure_ascii=False, default=str)
    for key in ("memory", "content", "text", "fact", "value", "summary"):
        value = getattr(item, key, None)
        if value is not None:
            return str(value)
    return str(item)


def _normalize_results(payload: Any, *, limit: int) -> list[str]:
    if isinstance(payload, dict):
        for key in ("results", "memories", "items"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        payload = [payload]
    return [_extract_text(item) for item in payload[:limit] if item is not None]


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
