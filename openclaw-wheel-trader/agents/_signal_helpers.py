"""Shared helpers for signal-based agents (bear, bull, future analogs).

Kept private (`_signal_helpers`) — these are deliberately untyped/duck-typed
because callers pass either a `WheelCandidate` dataclass or a raw dict, and
we want both to flow through identically.
"""

from __future__ import annotations

from typing import Any


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Tolerant float coercion — never raises on a None/string field."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_field(candidate: Any, *keys: str, default: Any = None) -> Any:
    """Read a field from either a dict or a dataclass-like object.

    Returns the first non-None value among ``keys``; ``default`` if none hit.
    """
    if isinstance(candidate, dict):
        for k in keys:
            if k in candidate and candidate[k] is not None:
                return candidate[k]
        return default
    for k in keys:
        v = getattr(candidate, k, None)
        if v is not None:
            return v
    return default
