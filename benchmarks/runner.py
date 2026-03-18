from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Protocol

from memoripy import BrainConfig, LocalAssetProcessor, MemoryClient, MemoryPipelineConfig

from .scenarios import DEFAULT_SCENARIOS, BenchmarkScenario


CATEGORY_WEIGHTS = {
    "fact_extraction": 1.0,
    "reconciliation": 1.0,
    "retrieval": 1.0,
    "grounding": 1.0,
    "multimodal": 1.0,
}


class BenchmarkAdapter(Protocol):
    name: str

    def run_scenario(self, scenario: BenchmarkScenario) -> dict[str, Any]:
        ...


class MemoripyBenchmarkAdapter:
    name = "memoripy"

    def __init__(self) -> None:
        self.pipeline = MemoryPipelineConfig(
            asset_processor=LocalAssetProcessor(),
            default_include_trace=True,
            brain=BrainConfig(mode="attention_fast"),
        )

    def run_scenario(self, scenario: BenchmarkScenario) -> dict[str, Any]:
        client = MemoryClient(pipeline=self.pipeline)
        scope_defaults = {"user_id": f"bench-{scenario.scenario_id}", "agent_id": None, "run_id": None}
        for step in scenario.steps:
            operation = step["operation"]
            payload = {**scope_defaults, **dict(step.get("payload") or {})}
            if operation == "capture":
                client.capture(**payload)
            elif operation == "add":
                client.add(**payload)
            elif operation == "consolidate":
                client.maintenance.consolidate(**payload)
            else:
                raise ValueError(f"Unsupported benchmark step: {operation}")

        operation = scenario.evaluation["operation"]
        payload = {**scope_defaults, **dict(scenario.evaluation.get("payload") or {})}
        if operation == "search":
            return client.search(**payload)
        if operation == "context":
            return client.context.build(**payload).to_dict()
        if operation == "chat":
            return client.chat.completions.create(**payload)
        raise ValueError(f"Unsupported benchmark evaluation: {operation}")


class Mem0BenchmarkAdapter:
    name = "mem0"

    def __init__(self) -> None:
        raise RuntimeError(
            "Mem0 is not installed in this workspace. "
            "Use the shared scenarios in benchmarks/scenarios.py with your Mem0 environment."
        )

    def run_scenario(self, scenario: BenchmarkScenario) -> dict[str, Any]:
        raise NotImplementedError


def evaluate_result(scenario: BenchmarkScenario, payload: dict[str, Any], adapter_name: str) -> dict[str, Any]:
    rendered = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    matched = [needle for needle in scenario.expected_contains if needle in rendered]
    missing = [needle for needle in scenario.expected_contains if needle not in rendered]
    unexpected = [needle for needle in scenario.expected_not_contains if needle in rendered]
    passed = not missing and not unexpected
    score = CATEGORY_WEIGHTS.get(scenario.category, 1.0) if passed else 0.0
    return {
        "adapter": adapter_name,
        "scenario_id": scenario.scenario_id,
        "category": scenario.category,
        "description": scenario.description,
        "passed": passed,
        "score": score,
        "matched": matched,
        "missing": missing,
        "unexpected": unexpected,
        "payload_preview": rendered[:800],
    }


def run_benchmarks(adapter: BenchmarkAdapter, scenarios: list[BenchmarkScenario] | None = None) -> dict[str, Any]:
    scenarios = scenarios or DEFAULT_SCENARIOS
    results = [evaluate_result(scenario, adapter.run_scenario(scenario), adapter.name) for scenario in scenarios]
    max_score = sum(CATEGORY_WEIGHTS.get(scenario.category, 1.0) for scenario in scenarios)
    earned_score = sum(result["score"] for result in results)
    category_totals: dict[str, dict[str, float]] = {}
    for scenario, result in zip(scenarios, results):
        bucket = category_totals.setdefault(scenario.category, {"passed": 0.0, "total": 0.0})
        bucket["passed"] += result["score"]
        bucket["total"] += CATEGORY_WEIGHTS.get(scenario.category, 1.0)
    return {
        "adapter": adapter.name,
        "scenario_count": len(scenarios),
        "earned_score": earned_score,
        "max_score": max_score,
        "score_ratio": (earned_score / max_score) if max_score else 0.0,
        "categories": category_totals,
        "results": results,
    }


def run_latency_probe(memory_count: int = 10000, iterations: int = 10) -> dict[str, Any]:
    client = MemoryClient(pipeline=MemoryPipelineConfig(brain=BrainConfig(mode="attention_fast")))
    for index in range(memory_count):
        client.add(text=f"My project code is Atlas-{index}", user_id="latency-user")

    search_samples: list[float] = []
    context_samples: list[float] = []
    for _ in range(max(iterations, 1)):
        started = time.perf_counter()
        client.search(query="Atlas-9999", user_id="latency-user", limit=5)
        search_samples.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        client.context.build(query="What project code do you remember?", user_id="latency-user", limit=5)
        context_samples.append((time.perf_counter() - started) * 1000.0)

    def _summary(samples: list[float]) -> dict[str, float]:
        ordered = sorted(samples)
        p95_index = min(len(ordered) - 1, max(int(math.ceil(len(ordered) * 0.95)) - 1, 0))
        return {
            "min_ms": ordered[0],
            "avg_ms": sum(ordered) / len(ordered),
            "p95_ms": ordered[p95_index],
            "max_ms": ordered[-1],
        }

    return {
        "memory_count": memory_count,
        "iterations": iterations,
        "search": _summary(search_samples),
        "context": _summary(context_samples),
    }


def _adapter_from_name(name: str) -> BenchmarkAdapter:
    normalized = name.strip().lower()
    if normalized == "memoripy":
        return MemoripyBenchmarkAdapter()
    if normalized == "mem0":
        return Mem0BenchmarkAdapter()
    raise ValueError(f"Unknown benchmark target: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Memoripy benchmark scenarios.")
    parser.add_argument("--target", default="memoripy", choices=("memoripy", "mem0"))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--latency", action="store_true", help="Run the synthetic latency probe instead of the scenario suite.")
    args = parser.parse_args(argv)

    if args.latency:
        summary = run_latency_probe()
        if args.json:
            print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    summary = run_benchmarks(_adapter_from_name(args.target))
    if args.json:
        print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    print(f"adapter={summary['adapter']} score={summary['earned_score']:.1f}/{summary['max_score']:.1f}")
    for category, totals in summary["categories"].items():
        print(f"- {category}: {totals['passed']:.1f}/{totals['total']:.1f}")
    for result in summary["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['scenario_id']}: {result['description']}")
        if not result["passed"]:
            if result["missing"]:
                print(f"  missing={result['missing']}")
            if result["unexpected"]:
                print(f"  unexpected={result['unexpected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
