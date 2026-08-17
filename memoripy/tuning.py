from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .client import MemoryClient
from .evals import MemoryContract, run_contracts
from .pipeline import MemoryPipelineConfig, RetrievalConfig


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    retrieval: RetrievalConfig
    score_ratio: float
    failed_contracts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "retrieval": self.retrieval.describe(),
            "score_ratio": self.score_ratio,
            "failed_contracts": list(self.failed_contracts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalProfile":
        return cls(
            name=str(payload.get("name", "custom")),
            retrieval=RetrievalConfig.from_dict(payload.get("retrieval")),
            score_ratio=float(payload.get("score_ratio", 0.0)),
            failed_contracts=list(payload.get("failed_contracts") or []),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TuningResult:
    selected: RetrievalProfile
    candidates: list[RetrievalProfile]
    contract_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "contract_count": self.contract_count,
        }


def built_in_retrieval_candidates() -> dict[str, RetrievalConfig]:
    return {
        "balanced": RetrievalConfig(),
        "exact_and_lexical": RetrievalConfig(
            lexical_weight=1.25,
            exact_weight=1.55,
            semantic_weight=0.65,
            entity_weight=0.7,
        ),
        "semantic": RetrievalConfig(
            lexical_weight=0.75,
            exact_weight=1.1,
            semantic_weight=1.35,
            entity_weight=0.85,
        ),
        "temporal": RetrievalConfig(
            lexical_weight=0.9,
            semantic_weight=0.8,
            exact_weight=1.2,
            temporal_weight=1.55,
            authority_weight=0.8,
        ),
        "policy_and_authority": RetrievalConfig(
            lexical_weight=0.9,
            semantic_weight=0.75,
            exact_weight=1.2,
            authority_weight=1.35,
            policy_weight=1.55,
        ),
        "low_noise": RetrievalConfig(
            lexical_weight=1.0,
            semantic_weight=0.8,
            exact_weight=1.35,
            entity_weight=0.65,
            minimum_relevance=0.009,
            lane_limit=64,
        ),
    }


def tune_retrieval(
    contracts: list[MemoryContract],
    *,
    candidates: dict[str, RetrievalConfig] | None = None,
    client_factory: Callable[[RetrievalConfig], MemoryClient] | None = None,
) -> TuningResult:
    candidates = candidates or built_in_retrieval_candidates()
    if not candidates:
        raise ValueError("At least one retrieval candidate is required")
    profiles: list[RetrievalProfile] = []
    for name, config in candidates.items():
        factory = client_factory or (
            lambda current: MemoryClient(pipeline=MemoryPipelineConfig(retrieval=current))
        )
        summary = run_contracts(contracts, client_factory=lambda config=config: factory(config))
        failed = [item["name"] for item in summary["results"] if not item["passed"]]
        profiles.append(
            RetrievalProfile(
                name=name,
                retrieval=config,
                score_ratio=float(summary["score_ratio"]),
                failed_contracts=failed,
                metadata={
                    "passed_count": summary["passed_count"],
                    "failed_count": summary["failed_count"],
                },
            )
        )
    profiles.sort(key=_profile_sort_key, reverse=True)
    return TuningResult(selected=profiles[0], candidates=profiles, contract_count=len(contracts))


def save_retrieval_profile(profile: RetrievalProfile, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_retrieval_profile(path: str | Path) -> RetrievalProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RetrievalProfile.from_dict(payload)


def _profile_sort_key(profile: RetrievalProfile) -> tuple[float, float, float]:
    config = profile.retrieval
    # Prefer higher contract performance, then lower candidate breadth and a
    # less extreme total weight when multiple profiles tie.
    total_weight = sum(
        float(value)
        for key, value in asdict(config).items()
        if key.endswith("_weight")
    )
    return profile.score_ratio, -float(config.lane_limit), -total_weight
