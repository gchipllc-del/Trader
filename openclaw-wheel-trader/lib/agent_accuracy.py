"""
Agent Accuracy Tracker — per-agent rolling accuracy + dynamic weighting.

Adapted from the **ai-hedge-fund** pattern (github.com/virattt/ai-hedge-fund,
49.6K stars): instead of equal-weight aggregation of bull/bear signals,
track each agent's recent track record and reweight votes proportionally.
An agent that nailed the last 5 selloffs gets more say than one that's
been wrong recently.

Decision-correctness rubric (per closed trade):
    Bull agent:
      score >= 4 and outcome=win  → correct
      score >= 4 and outcome=loss → wrong
      score < 4  and outcome=win  → wrong  (missed the move)
      score < 4  and outcome=loss → correct (correctly cautious)
    Bear agent:
      score >= 4 and outcome=loss → correct
      score >= 4 and outcome=win  → wrong (over-cautious)
      score < 4  and outcome=loss → wrong (missed the risk)
      score < 4  and outcome=win  → correct

This is a "directional" rubric — it doesn't try to grade VETO/DOWNSIZE
decisions on their own (those affect whether the trade happens at all,
which we can't counterfactually score). It just asks: did the agent's
score correctly anticipate the trade's outcome?

Weights are computed from a rolling window of recent closed trades.
Below MIN_SAMPLES (default 5), weights default to 0.5/0.5 — we don't
let one or two lucky/unlucky trades flip the bot's whole decision
weighting.
"""

from __future__ import annotations

import json
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from lib.audit import log_event


_AGENT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_accuracy.jsonl"

# How many recent closed trades to look at when computing weights.
DEFAULT_LOOKBACK = 20

# Below this many resolved decisions per agent, fall back to 0.5/0.5
# defaults. Two trades is not a signal.
MIN_SAMPLES = 5

# When at MIN_SAMPLES or above, blend the agent's rolling accuracy with
# the 0.5 prior using this weight. 0.0 = use raw accuracy, 1.0 = always
# 0.5. We pull toward the prior to dampen noise on small samples.
SHRINKAGE = 0.15

# Score threshold — at-or-above counts as "the agent voted strongly" for
# the directional rubric above.
SCORE_THRESHOLD = 4


def _read_log() -> list[dict]:
    """Read the JSONL log. Returns []  on missing/corrupt."""
    if not _AGENT_LOG_PATH.exists():
        return []
    rows: list[dict] = []
    try:
        with open(_AGENT_LOG_PATH) as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        log_event("agent_accuracy", "log_read_failed",
                  {"error": str(e)[:200]}, result="degraded")
    return rows


def _append_log(entry: dict) -> None:
    """Append a single JSONL entry under an exclusive flock."""
    try:
        _AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_AGENT_LOG_PATH, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry) + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        # Never raise into the trade path — accuracy logging is
        # observability, not safety-critical. A failed write loses
        # one data point; missing a write must NOT block trade exec.
        log_event("agent_accuracy", "log_write_failed",
                  {"error": str(e)[:200]}, result="degraded")


def log_decision(
    *,
    ticker: str,
    bull_score: int,
    bear_score: int,
    bull_action: str,
    bear_action: str,
    combined_decision: str,
    composite_score: int = 0,
) -> None:
    """Record bull/bear scores at trade entry. Outcome filled later by
    record_outcome(). Called from stock_engine.execute_stock_buy() after
    combine_bull_bear() and before the position is persisted.
    """
    entry = {
        "ticker": ticker,
        "bull_score": int(bull_score),
        "bear_score": int(bear_score),
        "bull_action": bull_action,
        "bear_action": bear_action,
        "combined_decision": combined_decision,
        "composite_score": int(composite_score),
        "outcome": None,         # filled when trade closes
        "pnl_pct": None,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "resolved_at": None,
    }
    _append_log(entry)


def record_outcome(
    *,
    ticker: str,
    outcome: Literal["win", "loss", "flat"],
    pnl_pct: float,
) -> bool:
    """Fill in the outcome on the most-recent unresolved entry for this
    ticker. Returns True if a matching entry was updated.

    This rewrites the JSONL — slower than append but the log is small
    (one row per trade) and we only resolve a few times a day. Correctness
    over throughput.
    """
    rows = _read_log()
    target_idx = None
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        if r.get("ticker") == ticker and r.get("outcome") is None:
            target_idx = i
            break
    if target_idx is None:
        return False

    rows[target_idx]["outcome"] = outcome
    rows[target_idx]["pnl_pct"] = float(pnl_pct)
    rows[target_idx]["resolved_at"] = datetime.now(timezone.utc).isoformat()

    try:
        _AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _AGENT_LOG_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        tmp.replace(_AGENT_LOG_PATH)
        return True
    except OSError as e:
        log_event("agent_accuracy", "outcome_write_failed",
                  {"ticker": ticker, "error": str(e)[:200]}, result="degraded")
        return False


def _score_agent(rows: list[dict], agent: Literal["bull", "bear"]) -> tuple[int, int]:
    """Return (correct, total) for the agent across resolved rows."""
    correct = 0
    total = 0
    score_key = f"{agent}_score"
    for r in rows:
        outcome = r.get("outcome")
        if outcome not in ("win", "loss"):  # "flat" doesn't count either way
            continue
        score = int(r.get(score_key, 0) or 0)
        is_strong = score >= SCORE_THRESHOLD
        won = outcome == "win"

        if agent == "bull":
            # Bull strong + win = correct; bull weak + loss = correct (cautious)
            is_correct = (is_strong and won) or (not is_strong and not won)
        else:  # bear
            # Bear strong + loss = correct (correctly cautious); bear weak + win = correct
            is_correct = (is_strong and not won) or (not is_strong and won)

        if is_correct:
            correct += 1
        total += 1
    return correct, total


def get_agent_weights(lookback: int = DEFAULT_LOOKBACK) -> dict[str, float]:
    """Compute normalized rolling weights for bull and bear agents.

    Returns ``{"bull_weight": float, "bear_weight": float, "n_bull": int,
    "n_bear": int, "is_default": bool}`` where the two weights sum to 1.0.

    Falls back to {0.5, 0.5} when either agent has fewer than
    MIN_SAMPLES resolved trades.
    """
    rows = _read_log()
    # Only the most recent `lookback` resolved trades.
    resolved = [r for r in rows if r.get("outcome") in ("win", "loss")]
    resolved = resolved[-lookback:]

    bull_correct, bull_total = _score_agent(resolved, "bull")
    bear_correct, bear_total = _score_agent(resolved, "bear")

    if bull_total < MIN_SAMPLES or bear_total < MIN_SAMPLES:
        return {
            "bull_weight": 0.5,
            "bear_weight": 0.5,
            "n_bull": bull_total,
            "n_bear": bear_total,
            "is_default": True,
            "reason": (
                f"insufficient data — need {MIN_SAMPLES} resolved trades "
                f"per agent, have bull={bull_total} bear={bear_total}"
            ),
        }

    # Raw accuracy + shrinkage toward 0.5 prior to dampen noise.
    bull_acc_raw = bull_correct / bull_total
    bear_acc_raw = bear_correct / bear_total
    bull_acc = (1 - SHRINKAGE) * bull_acc_raw + SHRINKAGE * 0.5
    bear_acc = (1 - SHRINKAGE) * bear_acc_raw + SHRINKAGE * 0.5

    # Normalize so the two weights sum to 1.0.
    denom = bull_acc + bear_acc
    if denom <= 0:
        return {
            "bull_weight": 0.5, "bear_weight": 0.5,
            "n_bull": bull_total, "n_bear": bear_total,
            "is_default": True,
            "reason": "zero combined accuracy",
        }

    bull_w = bull_acc / denom
    bear_w = bear_acc / denom
    return {
        "bull_weight": round(bull_w, 4),
        "bear_weight": round(bear_w, 4),
        "bull_accuracy": round(bull_acc_raw, 4),
        "bear_accuracy": round(bear_acc_raw, 4),
        "n_bull": bull_total,
        "n_bear": bear_total,
        "is_default": False,
    }


def summary_report() -> str:
    """Human-readable summary for the CLI / dashboard."""
    rows = _read_log()
    resolved = [r for r in rows if r.get("outcome") in ("win", "loss")]
    w = get_agent_weights()
    lines = [
        "AGENT ACCURACY REPORT",
        f"  Resolved decisions: {len(resolved)}  (unresolved: {len(rows) - len(resolved)})",
        f"  Bull: {w.get('bull_accuracy', 'n/a')} accuracy on {w['n_bull']} trades  "
        f"→ weight {w['bull_weight']}",
        f"  Bear: {w.get('bear_accuracy', 'n/a')} accuracy on {w['n_bear']} trades  "
        f"→ weight {w['bear_weight']}",
    ]
    if w["is_default"]:
        lines.append(f"  (using default 0.5/0.5 — {w.get('reason', '')})")
    return "\n".join(lines)
