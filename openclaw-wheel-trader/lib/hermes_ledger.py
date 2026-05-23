"""Hermes experiment ledger — scientific-method tuning tracker.

Implements the self-improving loop from the "Hermes self-improving trading
agent" framework (video 2, May 2026):

  prompt → strategy → outcome → learn → updated prompt → ...

Every parameter change Hermes makes is treated as a SCIENTIFIC EXPERIMENT:
  1. Record a baseline snapshot (params + goal-distance + recent PnL)
  2. Apply ONE parameter change (one variable at a time)
  3. Wait for the next evaluation cycle
  4. Compare new metrics to baseline
  5. If improvement → KEEP (new baseline), else → ROLL BACK

This file is the storage + decision layer for that loop. The optimizer
calls in to record experiments and consult the ledger before proposing
new changes (don't repeat a recently-rolled-back experiment).

Storage: data/hermes_experiments.jsonl (append-only JSONL).
Each row is one experiment with status open|kept|rolled_back.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "hermes_experiments.jsonl"

ExperimentStatus = Literal["open", "kept", "rolled_back", "expired"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_all() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    out = []
    with open(LEDGER_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_all(rows: list[dict]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, LEDGER_PATH)


def open_experiment(
    *,
    param: str,
    old_value,
    new_value,
    reason: str,
    baseline_metrics: dict,
    expected_direction: Literal["up", "down", "any"] = "up",
) -> dict:
    """Record a new pending experiment. Returns the experiment dict
    (caller uses experiment_id to close it later).

    baseline_metrics should include at minimum:
      - goal_distance_pct (how far from $5k target as a fraction)
      - rolling_30d_pnl (in $)
      - win_rate (0..1)
      - n_trades_30d
    """
    rows = _read_all()
    eid = f"exp_{int(datetime.now(timezone.utc).timestamp())}_{param}"
    exp = {
        "experiment_id": eid,
        "opened_at": _now_iso(),
        "closed_at": None,
        "param": param,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason,
        "expected_direction": expected_direction,
        "baseline": baseline_metrics,
        "post_change": None,
        "verdict": None,
        "status": "open",
    }
    rows.append(exp)
    _write_all(rows)
    return exp


def list_open_experiments(
    max_age_hours: float = 168.0,  # 7 days default — auto-expire stale
) -> list[dict]:
    """Return every still-open experiment, dropping stale ones to expired."""
    rows = _read_all()
    now = datetime.now(timezone.utc)
    changed = False
    open_exp: list[dict] = []
    for r in rows:
        if r.get("status") != "open":
            continue
        try:
            opened = datetime.fromisoformat(
                r["opened_at"].replace("Z", "+00:00")
            )
            age_h = (now - opened).total_seconds() / 3600.0
        except (KeyError, ValueError, TypeError):
            age_h = 0.0
        if age_h > max_age_hours:
            r["status"] = "expired"
            r["closed_at"] = _now_iso()
            r["verdict"] = "expired_without_evaluation"
            changed = True
            continue
        open_exp.append(r)
    if changed:
        _write_all(rows)
    return open_exp


def close_experiment(
    experiment_id: str,
    *,
    post_metrics: dict,
    keep_threshold_delta: float = 0.0,
) -> dict | None:
    """Evaluate a pending experiment.

    Decision:
      - If post_metrics improves the GOAL DISTANCE by ≥ keep_threshold_delta
        (relative — e.g. 0.005 = 0.5pp closer to goal) → KEEP.
      - Else → ROLL BACK.

    Returns the updated experiment row, or None if not found.
    Caller is responsible for actually reverting the YAML value when
    verdict == "rolled_back".
    """
    rows = _read_all()
    for r in rows:
        if r.get("experiment_id") != experiment_id:
            continue
        if r.get("status") != "open":
            return r  # already closed
        baseline = r.get("baseline", {}) or {}
        b_dist = baseline.get("goal_distance_pct")
        p_dist = post_metrics.get("goal_distance_pct")
        if b_dist is None or p_dist is None:
            # Can't evaluate by goal; fall back to PnL delta
            b_pnl = baseline.get("rolling_30d_pnl", 0.0)
            p_pnl = post_metrics.get("rolling_30d_pnl", 0.0)
            improved = (p_pnl - b_pnl) > keep_threshold_delta
        else:
            # Smaller goal_distance = closer = better
            improved = (b_dist - p_dist) >= keep_threshold_delta
        r["post_change"] = post_metrics
        r["closed_at"] = _now_iso()
        r["verdict"] = "improved" if improved else "regressed"
        r["status"] = "kept" if improved else "rolled_back"
        _write_all(rows)
        return r
    return None


def recently_rolled_back(param: str, lookback_hours: float = 72.0) -> list[dict]:
    """Return experiments on `param` that rolled back in the last N hours.
    Hermes uses this to avoid re-proposing a recently-failed change.
    """
    rows = _read_all()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    out = []
    for r in rows:
        if r.get("param") != param:
            continue
        if r.get("status") != "rolled_back":
            continue
        try:
            closed = datetime.fromisoformat(
                (r.get("closed_at") or "").replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            continue
        if closed >= cutoff:
            out.append(r)
    return out


def history(limit: int = 50) -> list[dict]:
    """Most recent N experiments newest-first — for the dashboard."""
    rows = _read_all()
    rows.sort(key=lambda r: r.get("opened_at", ""), reverse=True)
    return rows[:limit]


def stats() -> dict:
    """Aggregate kept/rolled-back/open counts + improvement rate."""
    rows = _read_all()
    counts = {"open": 0, "kept": 0, "rolled_back": 0, "expired": 0}
    for r in rows:
        s = r.get("status", "open")
        if s in counts:
            counts[s] += 1
    closed = counts["kept"] + counts["rolled_back"]
    keep_rate = (counts["kept"] / closed) if closed > 0 else None
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "keep_rate": round(keep_rate, 4) if keep_rate is not None else None,
    }


__all__ = [
    "LEDGER_PATH",
    "open_experiment",
    "list_open_experiments",
    "close_experiment",
    "recently_rolled_back",
    "history",
    "stats",
]
