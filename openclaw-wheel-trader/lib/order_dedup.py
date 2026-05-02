"""
Cross-process order intent deduplication.

Stores recent intent hashes in a JSON file under an exclusive fcntl lock so
two concurrent scan/monitor processes cannot both clear the same order in
the same dedup window. Replaces the in-RAM `_recent_intents` dict in
order_gate.py, which only protected within a single process.

The 2026-04-28 BAC double-buy incident was the bite that motivated this:
two scans 10 minutes apart each ran their own RAM dedup and each saw a
fresh slate, so both submitted a buy for the same ticker.

The store is a JSON file at data/intent_hashes.json mapping
{hash: expires_at_unix_timestamp}. Entries past expiry are GC'd on every
write. The window matches order_gate.DUPLICATE_WINDOW_SECONDS.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

DEDUP_FILE = Path(__file__).parent.parent / "data" / "intent_hashes.json"
SECURE_FILE_MODE = 0o600


def _ensure_parent():
    DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)


def _ensure_secure_perms(path: Path) -> None:
    """Apply 0o600 to the dedup hash store. Idempotent; silent on errors."""
    try:
        if path.is_file() and (path.stat().st_mode & 0o777) != SECURE_FILE_MODE:
            os.chmod(path, SECURE_FILE_MODE)
    except OSError:
        pass


def _read_locked(f) -> dict[str, float]:
    f.seek(0)
    raw = f.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        # Corrupted file — start clean rather than crash trades.
        return {}


def _write_locked(f, data: dict[str, float]):
    f.seek(0)
    f.truncate()
    f.write(json.dumps(data, separators=(",", ":")))
    f.flush()


def check_and_record(intent_hash: str, window_seconds: int = 60) -> tuple[bool, float]:
    """
    Atomically check if `intent_hash` was seen within `window_seconds`, and
    record it if not.

    Returns (is_duplicate, seconds_since_first_seen).
        is_duplicate=True  → caller MUST block the order
        is_duplicate=False → hash recorded, caller may proceed
    """
    _ensure_parent()
    now = time.time()

    # Open with a+ so we can lock+read+write under one fcntl handle.
    with open(DEDUP_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            store = _read_locked(f)

            # GC expired entries (kept for 2× window, matches prior in-RAM behavior).
            cutoff = now - (window_seconds * 2)
            store = {h: ts for h, ts in store.items() if ts > cutoff}

            if intent_hash in store:
                first_seen = store[intent_hash]
                age = now - first_seen
                if age < window_seconds:
                    # Persist the GC'd shape but DO NOT update the timestamp —
                    # keep the original first-seen so a flood of dups all see
                    # the same original time.
                    _write_locked(f, store)
                    return True, age

            store[intent_hash] = now
            _write_locked(f, store)
            return False, 0.0
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    _ensure_secure_perms(DEDUP_FILE)


def reset_for_tests():
    """Clear the dedup store. Test-only — never call from production paths."""
    if DEDUP_FILE.exists():
        DEDUP_FILE.unlink()
