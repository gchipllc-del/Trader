"""
Append-only audit logger.
Every action the bot takes is logged here BEFORE execution.
This is forensic-grade — if something goes wrong, this is the source of truth.
"""
from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
AUDIT_FILE = LOG_DIR / "audit_log.jsonl"

# Audit log contains trade decisions, position state, order IDs — owner-only.
SECURE_FILE_MODE = 0o600

_SECRET_KEYWORDS = ("key", "secret", "token", "password", "credential")


def _collect_secret_values() -> tuple[str, ...]:
    """Snapshot env-var secret VALUES at import so we can scrub them out
    of free-form strings (exception messages, etc.) before writing to disk.

    Only env vars whose NAME matches a secret keyword are collected; we
    don't iterate every env var on every log call. Values shorter than 6
    chars are skipped — too short to be a meaningful secret and we'd risk
    munching legitimate substrings.
    """
    out = []
    for name, value in os.environ.items():
        if not value or len(value) < 6:
            continue
        lowered = name.lower()
        if any(kw in lowered for kw in _SECRET_KEYWORDS):
            out.append(value)
    return tuple(out)


_SECRET_VALUES: tuple[str, ...] = _collect_secret_values()


def _scrub_string(s: str) -> str:
    """Replace any occurrence of a known env-var secret value with ***REDACTED***."""
    if not s or not _SECRET_VALUES:
        return s
    out = s
    for secret in _SECRET_VALUES:
        if secret in out:
            out = out.replace(secret, "***REDACTED***")
    return out


def _looks_like_secret_key(name: str) -> bool:
    lowered = name.lower()
    return any(kw in lowered for kw in _SECRET_KEYWORDS)


def _scrub_value(v, _seen: set | None = None):
    """Walk dicts/lists/strings recursively. At every level, fully redact
    dict entries whose KEY name matches a secret keyword; otherwise scrub
    known env-var secret VALUES out of string leaves.

    A `_seen` set of object ids guards against circular references — audit
    `details` come from exception traces and external API responses that
    can legitimately contain self-referential structures (e.g. a logger
    that loops a request object back into itself). Without this, the
    recursion would stack-overflow and crash the whole audit write.
    """
    if isinstance(v, str):
        return _scrub_string(v)
    if isinstance(v, (dict, list, tuple)):
        if _seen is None:
            _seen = set()
        if id(v) in _seen:
            return "***CYCLIC***"
        _seen = _seen | {id(v)}
    if isinstance(v, dict):
        return {
            k: ("***REDACTED***" if _looks_like_secret_key(k) else _scrub_value(x, _seen))
            for k, x in v.items()
        }
    if isinstance(v, (list, tuple)):
        return type(v)(_scrub_value(x, _seen) for x in v)
    return v


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_secure_perms(path: Path) -> None:
    """Apply 0o600 to the audit file. Idempotent; silent on permission errors."""
    try:
        if path.is_file() and (path.stat().st_mode & 0o777) != SECURE_FILE_MODE:
            os.chmod(path, SECURE_FILE_MODE)
    except OSError:
        pass


def _write_jsonl(event: dict):
    """Write a single JSON line with file locking for concurrent safety."""
    line = json.dumps(event) + "\n"
    with open(AUDIT_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    _ensure_secure_perms(AUDIT_FILE)


def log_event(
    event_type: str,
    action: str,
    details: dict | None = None,
    result: str = "pending",
) -> dict:
    """
    Log an event to the audit trail. Call BEFORE executing the action.

    Args:
        event_type: Category — "order", "monitor", "circuit_breaker",
                    "kill_switch", "config_change", "error", "startup"
        action: What's happening — "sell_csp", "close_position", "daily_check"
        details: Relevant data (ticker, strike, qty, etc.)
                 WARNING: Never include API keys, secrets, or tokens here.
        result: "pending", "success", "failed", "vetoed", "blocked"

    Returns:
        The logged event dict (with id and timestamp).
    """
    _ensure_log_dir()

    # Two-pass sanitization at every nesting level:
    #   (1) field-name based — redact whole values where the KEY name looks
    #       like a secret holder ("api_key", "auth_token", "credentials", …)
    #   (2) value scrub — replace any known env-var secret VALUE embedded
    #       inside a free-form string. Catches the common leak where an
    #       exception message includes the auth header that triggered it.
    if details:
        details = _scrub_value(details)

    event = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "action": action,
        "details": details or {},
        "result": result,
    }

    _write_jsonl(event)

    return event


def update_event_result(event_id: str, result: str, error_msg: str | None = None):
    """
    Append a follow-up log entry updating the result of a previous event.
    We don't modify the original line (append-only) — we add a resolution entry.

    Args:
        event_id: The id field from the original event.
        result: New result status.
        error_msg: Optional error message.
    """
    _ensure_log_dir()
    resolution = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "resolution",
        "action": "update_result",
        "details": {
            "original_event_id": event_id,
            "new_result": result,
        },
        "result": result,
    }
    if error_msg:
        # Same value-scrub path as log_event — exception messages
        # routinely include the auth header that triggered the failure.
        resolution["details"]["error"] = _scrub_string(error_msg)

    _write_jsonl(resolution)


def get_recent_events(n: int = 50, event_type: str | None = None) -> list[dict]:
    """Read the last N events from the audit log, optionally filtered by type.

    Uses a bounded deque to avoid loading the entire file into memory.
    """
    if not AUDIT_FILE.exists():
        return []

    # Use a deque to keep only the last N matching events in memory
    buf: deque[dict] = deque(maxlen=n)
    with open(AUDIT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event_type is None or event.get("event_type") == event_type:
                    buf.append(event)
            except json.JSONDecodeError:
                continue

    return list(buf)
