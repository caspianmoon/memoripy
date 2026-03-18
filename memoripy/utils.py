from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']+")
ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def unique_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokenize(text):
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def extract_entities(text: str) -> list[str]:
    seen: set[str] = set()
    entities: list[str] = []
    for match in ENTITY_PATTERN.findall(text or ""):
        entity = normalize_text(match)
        if entity and entity not in seen:
            seen.add(entity)
            entities.append(entity)
    return entities


def cosine_similarity(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def hashed_embedding(text: str, dimensions: int = 32) -> list[float]:
    buckets = [0.0] * dimensions
    tokens = unique_tokens(text)
    if not tokens:
        return buckets
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index in range(dimensions):
            buckets[index] += 1.0 if digest[index % len(digest)] % 2 == 0 else -1.0
    scale = math.sqrt(sum(value * value for value in buckets)) or 1.0
    return [value / scale for value in buckets]


def summarize_text(text: str, max_words: int = 24) -> str:
    words = normalize_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def deep_copy_json(data: Any) -> Any:
    return copy.deepcopy(data)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


@contextlib.contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    ensure_parent(lock_path)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def json_ready_dict(mapping: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(mapping, ensure_ascii=True))


def flatten_text_parts(parts: Iterable[str]) -> str:
    return "\n".join(part for part in parts if normalize_text(part))
