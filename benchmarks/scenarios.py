from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkScenario:
    scenario_id: str
    category: str
    description: str
    steps: list[dict[str, Any]]
    evaluation: dict[str, Any]
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)


DEFAULT_SCENARIOS: list[BenchmarkScenario] = [
    BenchmarkScenario(
        scenario_id="fact-extraction-profile",
        category="fact_extraction",
        description="Extract profile attributes and stable preferences from a compact user utterance.",
        steps=[
            {
                "operation": "capture",
                "payload": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "My name is Khazar and I live in Istanbul and my favorite city is Tokyo",
                        }
                    ]
                },
            }
        ],
        evaluation={
            "operation": "context",
            "payload": {
                "query": "What do you know about me?",
                "include_trace": True,
            },
        },
        expected_contains=["Khazar", "Istanbul", "Tokyo", '"intent": "general"'],
    ),
    BenchmarkScenario(
        scenario_id="reconciliation-contradiction",
        category="reconciliation",
        description="Supersede a contradicted preference without leaking the stale value into the current result set.",
        steps=[
            {"operation": "capture", "payload": {"messages": [{"role": "user", "content": "I like pizza"}]}},
            {"operation": "capture", "payload": {"messages": [{"role": "user", "content": "I don't like pizza"}]}},
        ],
        evaluation={
            "operation": "search",
            "payload": {
                "query": "pizza",
                "include_trace": True,
            },
        },
        expected_contains=['"sentiment": "negative"', '"latest_action": "SUPERSEDE"'],
        expected_not_contains=['"summary": "Positive preference: pizza"'],
    ),
    BenchmarkScenario(
        scenario_id="retrieval-scope-hierarchy",
        category="retrieval",
        description="Prefer run-scoped memories over broader user-scoped ones when both exist.",
        steps=[
            {"operation": "add", "payload": {"text": "I live in Paris"}},
            {"operation": "add", "payload": {"text": "I live in Berlin", "agent_id": "jarvis", "run_id": "trip-1"}},
        ],
        evaluation={
            "operation": "context",
            "payload": {
                "query": "Where do I live?",
                "agent_id": "jarvis",
                "run_id": "trip-1",
                "include_trace": True,
            },
        },
        expected_contains=['"value": "Berlin"', '"selected_memory_ids"'],
    ),
    BenchmarkScenario(
        scenario_id="grounding-tool-observation",
        category="grounding",
        description="Keep tool observations available as a first-class grounding section with provenance.",
        steps=[
            {
                "operation": "capture",
                "payload": {
                    "messages": [{"role": "user", "content": "Please check tomorrow's weather"}],
                    "events": [
                        {
                            "event_type": "tool_result",
                            "name": "weather.lookup",
                            "content": "Tomorrow in Istanbul it will be sunny and 21 C",
                        }
                    ],
                    "agent_id": "jarvis",
                    "run_id": "weather-1",
                },
            }
        ],
        evaluation={
            "operation": "context",
            "payload": {
                "query": "What's the weather tomorrow?",
                "agent_id": "jarvis",
                "run_id": "weather-1",
                "include_trace": True,
            },
        },
        expected_contains=["tool_observations", "Istanbul", '"citation_count": 1'],
    ),
    BenchmarkScenario(
        scenario_id="grounded-chat-compact-budget",
        category="grounding",
        description="Use compact v3 grounding without leaking verbose memory ids into the prompt surface.",
        steps=[
            {
                "operation": "capture",
                "payload": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "My name is Khazar and I live in Istanbul and my favorite city is Tokyo",
                        }
                    ],
                    "agent_id": "jarvis",
                },
            }
        ],
        evaluation={
            "operation": "chat",
            "payload": {
                "messages": [{"role": "user", "content": "What do you know about me?"}],
                "agent_id": "jarvis",
                "memory_strategy": "v3",
                "include_memory_pack": True,
                "include_trace": True,
                "context_policy": "compact",
            },
        },
        expected_contains=['"memory_strategy": "v3"', '"context_policy": "compact"'],
        expected_not_contains=["memory_id="],
    ),
    BenchmarkScenario(
        scenario_id="multimodal-asset-processing",
        category="multimodal",
        description="Recover memory from a non-text document asset through the configured asset processor.",
        steps=[
            {
                "operation": "add",
                "payload": {
                    "items": [
                        {
                            "modality": "document",
                            "metadata": {"text": "My favorite city is Tokyo"},
                        }
                    ]
                },
            }
        ],
        evaluation={
            "operation": "search",
            "payload": {
                "query": "favorite city",
                "include_trace": True,
            },
        },
        expected_contains=["Tokyo", '"asset_processor": "LocalAssetProcessor"'],
    ),
    BenchmarkScenario(
        scenario_id="attention-working-memory",
        category="retrieval",
        description="Expose a working-memory section when attention_fast mode prioritizes activated items.",
        steps=[
            {
                "operation": "capture",
                "payload": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "My name is Khazar and I live in Istanbul and my favorite city is Tokyo",
                        }
                    ],
                    "agent_id": "jarvis",
                },
            }
        ],
        evaluation={
            "operation": "context",
            "payload": {
                "query": "What do you remember about me right now?",
                "agent_id": "jarvis",
                "include_trace": True,
            },
        },
        expected_contains=['"working_memory"', '"selected_memory_ids"'],
    ),
    BenchmarkScenario(
        scenario_id="maintenance-trace",
        category="grounding",
        description="Persist maintenance metadata so later trace output exposes the last consolidation run.",
        steps=[
            {
                "operation": "capture",
                "payload": {
                    "messages": [{"role": "user", "content": "My favorite city is Tokyo"}],
                },
            },
            {
                "operation": "consolidate",
                "payload": {
                    "budget_ms": 5,
                    "limit": 20,
                },
            },
        ],
        evaluation={
            "operation": "search",
            "payload": {
                "query": "favorite city",
                "include_trace": True,
            },
        },
        expected_contains=['"consolidation"', '"last_run_at"'],
    ),
]
