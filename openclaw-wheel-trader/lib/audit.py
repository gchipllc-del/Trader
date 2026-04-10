"""
Append-only audit logger.
Every action the bot takes is logged here BEFORE execution.
This is forensic-grade — if something goes wrong, this is the source of truth.
"""

import fcntl
import json
import os
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
AUDIT_FILE = LOG_DIR / "audit_log.jsonl"


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


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

    # Sanitize — strip any field that looks like a secret
    if details:
        sanitized = {}
        secret_keywords = {"key", "secret", "token", "password", "credential"}
        for k, v in details.items():
            if any(s in k.lower() for s in secret_keywords):
                sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = v
        details = sanitized

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
        resolution["details"]["error"] = error_msg

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
