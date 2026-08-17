from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


_ISO_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2}(?:[T ][0-9:]{5,8}(?:Z|[+-]\d{2}:?\d{2})?)?)\b")
_RANGE_PATTERN = re.compile(
    r"\bfrom\s+(?P<start>\d{4}-\d{2}-\d{2})\s+(?:to|until|through)\s+(?P<end>\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_AGO_PATTERN = re.compile(r"\b(?P<count>\d+)\s+(?P<unit>day|week|month|year)s?\s+ago\b", re.IGNORECASE)
_IN_PATTERN = re.compile(r"\bin\s+(?P<count>\d+)\s+(?P<unit>day|week|month|year)s?\b", re.IGNORECASE)
_SINCE_PATTERN = re.compile(r"\bsince\s+(?P<date>\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_UNTIL_PATTERN = re.compile(r"\buntil\s+(?P<date>\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class TemporalBounds:
    observed_at: str
    valid_from: str
    valid_to: str | None = None
    precision: str = "instant"
    source: str = "observation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "precision": self.precision,
            "source": self.source,
        }


def parse_datetime(value: str | datetime | None, *, default: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            parsed = default or datetime.now(timezone.utc)
    else:
        parsed = default or datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def infer_temporal_bounds(text: str, reference_time: str | datetime | None = None) -> TemporalBounds:
    reference = parse_datetime(reference_time)
    observed = isoformat(reference)
    normalized = " ".join((text or "").split())
    lower = normalized.casefold()

    range_match = _RANGE_PATTERN.search(normalized)
    if range_match:
        start = parse_datetime(range_match.group("start"), default=reference)
        end = parse_datetime(range_match.group("end"), default=reference) + timedelta(days=1)
        return TemporalBounds(observed, isoformat(start), isoformat(end), "day", "explicit_range")

    since_match = _SINCE_PATTERN.search(normalized)
    if since_match:
        start = parse_datetime(since_match.group("date"), default=reference)
        return TemporalBounds(observed, isoformat(start), None, "day", "explicit_since")

    until_match = _UNTIL_PATTERN.search(normalized)
    if until_match:
        end = parse_datetime(until_match.group("date"), default=reference) + timedelta(days=1)
        return TemporalBounds(observed, observed, isoformat(end), "day", "explicit_until")

    ago_match = _AGO_PATTERN.search(normalized)
    if ago_match:
        start = _shift(reference, -int(ago_match.group("count")), ago_match.group("unit"))
        return TemporalBounds(observed, isoformat(start), None, _precision(ago_match.group("unit")), "relative_ago")

    in_match = _IN_PATTERN.search(normalized)
    if in_match:
        start = _shift(reference, int(in_match.group("count")), in_match.group("unit"))
        return TemporalBounds(observed, isoformat(start), None, _precision(in_match.group("unit")), "relative_future")

    relative = {
        "today": reference,
        "yesterday": reference - timedelta(days=1),
        "tomorrow": reference + timedelta(days=1),
        "last week": reference - timedelta(days=7),
        "next week": reference + timedelta(days=7),
        "last month": _shift_months(reference, -1),
        "next month": _shift_months(reference, 1),
        "last year": _shift_years(reference, -1),
        "next year": _shift_years(reference, 1),
    }
    for phrase, value in relative.items():
        if phrase in lower:
            precision = "day" if phrase in {"today", "yesterday", "tomorrow"} else phrase.split()[-1]
            return TemporalBounds(observed, isoformat(value), None, precision, f"relative_{phrase.replace(' ', '_')}")

    iso_match = _ISO_PATTERN.search(normalized)
    if iso_match:
        start = parse_datetime(iso_match.group(1), default=reference)
        return TemporalBounds(observed, isoformat(start), None, "day", "explicit_date")

    return TemporalBounds(observed, observed, None, "instant", "observation")


def _shift(value: datetime, count: int, unit: str) -> datetime:
    unit = unit.casefold()
    if unit == "day":
        return value + timedelta(days=count)
    if unit == "week":
        return value + timedelta(weeks=count)
    if unit == "month":
        return _shift_months(value, count)
    if unit == "year":
        return _shift_years(value, count)
    return value


def _shift_months(value: datetime, count: int) -> datetime:
    month_index = (value.year * 12 + value.month - 1) + count
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _shift_years(value: datetime, count: int) -> datetime:
    year = value.year + count
    day = min(value.day, calendar.monthrange(year, value.month)[1])
    return value.replace(year=year, day=day)


def _precision(unit: str) -> str:
    return {"day": "day", "week": "week", "month": "month", "year": "year"}.get(unit.casefold(), "instant")
