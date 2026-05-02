"""
File-locked positions.json access (Wave 3 audit finding #15).

Every engine module previously had its own _load_positions /
_save_positions pair with NO inter-process locking. The classic race:

    1. Scan-A reads positions.json → list of N positions
    2. Scan-B reads positions.json → same N positions (stale view)
    3. Scan-A appends new entry, writes N+1
    4. Scan-B appends a different new entry, writes N+1 — silently
       overwrites Scan-A's append

This module provides the canonical interface. Use `mutate_positions()`
for the read-modify-write critical section; it holds an fcntl exclusive
lock for the entire cycle so concurrent writers serialize cleanly.

For pure reads where staleness is acceptable (dashboards, status
display) `load_positions()` returns a snapshot under a brief shared
lock — fast and safe.

The audit module (lib/audit.py) already uses this same fcntl pattern;
we mirror it here for symmetry.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"

# Trade state contains position sizes, entry prices, and order IDs — owner-only.
SECURE_FILE_MODE = 0o600


def _ensure_secure_perms(path: Path) -> None:
    """Apply 0o600 (owner read+write only) to the file if not already set.
    Idempotent — silently skips if file doesn't exist or isn't a regular file."""
    try:
        if path.is_file() and (path.stat().st_mode & 0o777) != SECURE_FILE_MODE:
            os.chmod(path, SECURE_FILE_MODE)
    except OSError:
        # Permission errors should not block trading; log via caller if needed.
        pass


def _resolve_path(path: Path | None) -> Path:
    """Default to module-level POSITIONS_PATH, but accept an override so
    callers (tests, alternate engines) can redirect without touching globals."""
    p = path or POSITIONS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_positions(path: Path | None = None) -> list[dict]:
    """Return a snapshot of positions.json. Briefly holds a shared lock
    so a concurrent writer can't tear the JSON mid-read."""
    p = _resolve_path(path)
    if not p.exists():
        return []
    with open(p, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            raw = f.read()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    if not raw.strip():
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Corrupted file — return empty rather than crash the trade
        # pipeline. The caller (e.g. monitor) should still emit an
        # audit event when it sees this.
        return []


def save_positions(positions: list[dict], path: Path | None = None) -> None:
    """Atomic-overwrite positions.json under an exclusive lock.

    Prefer mutate_positions() over a manual load + save pair — the
    pair has a window where a concurrent writer can clobber your
    in-flight changes.
    """
    p = _resolve_path(path)
    with open(p, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            f.truncate()
            f.write(json.dumps(positions, indent=2))
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    _ensure_secure_perms(p)


@contextmanager
def mutate_positions(path: Path | None = None) -> Iterator[list[dict]]:
    """
    Acquire the lock, yield the current list, write it back when the
    block exits cleanly. Any uncaught exception inside the block aborts
    the write so positions.json is not partially mutated on error.

    Usage:
        with mutate_positions() as positions:
            positions.append({...})
            # exit → write under lock
    """
    p = _resolve_path(path)
    with open(p, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            try:
                positions = json.loads(raw) if raw.strip() else []
                if not isinstance(positions, list):
                    positions = []
            except json.JSONDecodeError:
                positions = []

            yield positions

            # Caller returned cleanly — persist.
            f.seek(0)
            f.truncate()
            f.write(json.dumps(positions, indent=2))
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    _ensure_secure_perms(p)
