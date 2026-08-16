from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*|\d+(?:[./:@-]\d+)*", re.UNICODE)
_ENTITY_PATTERN = re.compile(r"\b([^\W\d_][\w'’-]*(?:\s+[^\W\d_][\w'’-]*){0,4})\b", re.UNICODE)
_FALLBACK_LOCKS: dict[str, threading.RLock] = {}
_FALLBACK_LOCKS_GUARD = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timestamp_lte(left: str | None, right: str | None) -> bool:
    left_dt = parse_timestamp(left)
    right_dt = parse_timestamp(right)
    if left_dt is None or right_dt is None:
        return False
    return left_dt <= right_dt


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def payload_checksum(payload: Any) -> str:
    return stable_hash(payload)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return " ".join(normalized.strip().split())


def normalize_key(text: str) -> str:
    return normalize_text(text).casefold()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text).casefold()
    return [match.group(0) for match in _TOKEN_PATTERN.finditer(normalized)]


def unique_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokenize(text):
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def character_ngrams(text: str, minimum: int = 3, maximum: int = 5) -> list[str]:
    compact = " ".join(tokenize(text))
    if not compact:
        return []
    grams: list[str] = []
    for size in range(max(minimum, 1), max(maximum, minimum) + 1):
        if len(compact) < size:
            continue
        grams.extend(compact[index : index + size] for index in range(0, len(compact) - size + 1))
    return grams


def extract_entities(text: str) -> list[str]:
    normalized = normalize_text(text)
    seen: set[str] = set()
    entities: list[str] = []
    for match in _ENTITY_PATTERN.finditer(normalized):
        value = normalize_text(match.group(1)).strip(".,;:!?()[]{}\"'")
        if len(value) < 2:
            continue
        first = value[0]
        if not first.isupper() and not any(char.isdigit() for char in value):
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        entities.append(value)
    return entities


def cosine_similarity(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def hashed_embedding(text: str, dimensions: int = 128) -> list[float]:
    dimensions = max(int(dimensions), 8)
    buckets = [0.0] * dimensions
    features: list[tuple[str, float]] = [(token, 1.0) for token in unique_tokens(text)]
    features.extend((f"#{gram}", 0.25) for gram in character_ngrams(text, 3, 4))
    if not features:
        return buckets
    for feature, weight in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=32).digest()
        for offset in range(0, len(digest), 4):
            index = int.from_bytes(digest[offset : offset + 2], "big") % dimensions
            sign = 1.0 if digest[offset + 2] % 2 == 0 else -1.0
            buckets[index] += sign * weight
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


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    ensure_parent(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode("utf-8")
    atomic_write_bytes(path, body)


def atomic_copy(source: Path, destination: Path) -> None:
    ensure_parent(destination)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    try:
        shutil.copy2(source, tmp_name)
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
        _fsync_directory(destination.parent)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


@contextlib.contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    ensure_parent(lock_path)
    lock_key = str(lock_path.resolve())
    with _FALLBACK_LOCKS_GUARD:
        fallback_lock = _FALLBACK_LOCKS.setdefault(lock_key, threading.RLock())
    with fallback_lock:
        with lock_path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def json_ready_dict(mapping: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(mapping, ensure_ascii=False, default=str))


def flatten_text_parts(parts: Iterable[str]) -> str:
    return "\n".join(part for part in parts if normalize_text(part))


def redact_secret(value: str, keep: int = 3) -> str:
    text = str(value)
    if len(text) <= keep * 2:
        return "*" * len(text)
    return f"{text[:keep]}{'*' * (len(text) - (keep * 2))}{text[-keep:]}"
