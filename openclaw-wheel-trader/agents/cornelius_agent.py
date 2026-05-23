"""Cornelius — secondary self-improving agent for FILTER THRESHOLDS.

Companion to Hermes (the primary strategy-parameter tuner). The video-2
framework runs them on offset cadences:
  - Hermes weekly, owns portfolio mechanics + scoring weights
  - Cornelius weekly with a 3-day offset, owns the entry filter thresholds

The separation matters because filter thresholds (min_composite_score,
min_bayesian_win_prob, correlation_threshold) interact with strategy
params in tricky ways — tuning both in the same cycle makes attribution
impossible. By offsetting, each agent's experiments run on a parameter
landscape that the other already settled.

Cornelius writes to the SAME ledger as Hermes (lib.hermes_ledger) so the
operator sees one unified experiment timeline. Conflicts (both agents
proposing changes to the same param on the same day) resolve to Hermes.

What Cornelius watches (filter side):
  - min_composite_score : how strict the screener is on candidate quality
  - bayesian_min_win_prob : Bayesian sniff-test floor for entries
  - correlation_threshold : how aggressively we down-weight correlated names
  - markov.min_abs_signal : (optional) Markov signal magnitude floor

Each cycle Cornelius:
  1. Reads the recent miss rate per filter (what % of skipped candidates
     would have been profitable?)
  2. If miss rate > target → loosen the filter (one step)
  3. If false-positive rate > target → tighten the filter (one step)
  4. Opens an experiment in the ledger so Hermes' weekly review evaluates it
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STRATEGY_PATH = ROOT / "config" / "wheel_strategy.yaml"
TRADE_HISTORY_PATH = ROOT / "data" / "trade_history.json"
SCAN_LOG_PATH = ROOT / "logs" / "cron_scan.log"  # optional — may not exist

# Filter parameters Cornelius owns + their safety bounds
# (mirror legacy hermes_optimizer BOUNDS for the params it doesn't touch).
FILTER_BOUNDS = {
    "min_composite_score":   (2, 8),
    "bayesian_min_win_prob": (0.45, 0.75),
    "correlation_threshold": (0.40, 0.85),
}
FILTER_STEPS = {
    "min_composite_score":   1,
    "bayesian_min_win_prob": 0.02,
    "correlation_threshold": 0.05,
}


def _load_strategy() -> dict:
    with open(STRATEGY_PATH) as f:
        return yaml.safe_load(f) or {}


def _save_strategy(s: dict) -> None:
    with open(STRATEGY_PATH, "w") as f:
        yaml.safe_dump(s, f, default_flow_style=False, sort_keys=False)


def _load_trades(window_days: int = 14) -> list[dict]:
    if not TRADE_HISTORY_PATH.exists():
        return []
    try:
        with open(TRADE_HISTORY_PATH) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    trades = raw.get("trades", raw) if isinstance(raw, dict) else raw
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    out = []
    for t in trades:
        ts = t.get("closed_at") or t.get("completed_at") or t.get("opened_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt >= cutoff:
                out.append(t)
        except (ValueError, TypeError):
            continue
    return out


def _clamp(value, param: str):
    lo, hi = FILTER_BOUNDS[param]
    if isinstance(lo, int):
        return max(lo, min(hi, int(round(value))))
    return max(lo, min(hi, round(value, 4)))


def diagnose_filters(window_days: int = 14) -> list[dict]:
    """Heuristic per-filter recommendations.

    Without per-skip telemetry (which the scanner doesn't fully emit
    today), we use win-rate of recent ENTRIES as a proxy:
      - if WR is very high (≥75%) on few trades → filter is too strict
        (we passed up borderline winners) → loosen one filter
      - if WR is poor (≤45%) on many trades → filter is too loose
        (we let in losers) → tighten one filter
      - in between → hold

    This is a deliberately conservative single-axis heuristic so
    Cornelius doesn't fight Hermes. A future revision can ingest scanner
    skip-counter logs for per-filter attribution.
    """
    trades = _load_trades(window_days)
    n = len(trades)
    recs: list[dict] = []
    if n < 5:
        return [{
            "param": "none", "direction": "hold",
            "reason": f"insufficient_trades({n}<5)",
            "confidence": 0.0,
        }]
    wins = sum(
        1 for t in trades
        if (t.get("total_pnl") or t.get("realized_pnl") or 0) > 0
    )
    wr = wins / n if n else 0.0

    if wr >= 0.75:
        # Filters too tight — try loosening one. Composite-score is the
        # most reversible (integer step).
        recs.append({
            "param": "min_composite_score",
            "direction": "decrease",
            "reason": f"wr_high({wr:.0%}) on n={n} → loosen composite floor",
            "confidence": min(1.0, (wr - 0.75) * 4),
        })
    elif wr <= 0.45:
        # Filters too loose — tighten the win-prob floor (smallest
        # increments, lowest collateral damage).
        recs.append({
            "param": "bayesian_min_win_prob",
            "direction": "increase",
            "reason": f"wr_low({wr:.0%}) on n={n} → raise bayesian floor",
            "confidence": min(1.0, (0.45 - wr) * 4),
        })
    else:
        recs.append({
            "param": "none", "direction": "hold",
            "reason": f"wr({wr:.0%}) within tolerance band on n={n}",
            "confidence": 0.0,
        })
    return recs


def apply_one_filter_change(rec: dict) -> dict | None:
    """Apply a single filter-threshold change. Returns the change dict
    {param, old, new, reason} or None if nothing applied.
    """
    if rec.get("param") not in FILTER_BOUNDS or rec.get("direction") == "hold":
        return None
    strategy = _load_strategy()
    sp = strategy.setdefault("stock_params", {})
    param = rec["param"]
    current = sp.get(param, FILTER_BOUNDS[param][0])
    step = FILTER_STEPS[param]
    if rec["direction"] == "increase":
        new = _clamp(current + step, param)
    else:
        new = _clamp(current - step, param)
    if new == current:
        return None
    sp[param] = new
    _save_strategy(strategy)
    return {"param": param, "old": current, "new": new, "reason": rec["reason"]}


def run_cornelius_cycle(window_days: int = 14, dry_run: bool = False) -> dict:
    """Single Cornelius pass:
      1. Diagnose filter health
      2. Pick the top recommendation
      3. If live: apply ONE change + open ledger experiment
    """
    from lib.hermes_ledger import open_experiment, recently_rolled_back
    from lib.hermes_goal_score import compute_goal_metrics

    recs = diagnose_filters(window_days=window_days)
    eligible = [
        r for r in recs
        if r.get("param") not in (None, "none")
        and not recently_rolled_back(r["param"])
    ]
    pick = eligible[0] if eligible else None
    baseline = compute_goal_metrics()

    applied = None
    exp = None
    if pick and not dry_run:
        applied = apply_one_filter_change(pick)
        if applied:
            exp = open_experiment(
                param=applied["param"],
                old_value=applied["old"],
                new_value=applied["new"],
                reason=applied["reason"],
                baseline_metrics=baseline,
                expected_direction="up",
            )

    return {
        "cycle_at": datetime.now(timezone.utc).isoformat(),
        "agent": "cornelius",
        "window_days": window_days,
        "diagnoses": recs,
        "pick": pick,
        "applied": applied,
        "experiment": exp,
        "dry_run": dry_run,
    }


def render_cornelius_report(report: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"CORNELIUS FILTER CYCLE  —  window={report['window_days']}d")
    lines.append("=" * 70)
    for d in report.get("diagnoses", []):
        lines.append(f"  - {d.get('param', '?')}: {d.get('direction')} "
                     f"(conf={d.get('confidence', 0):.2f})  {d.get('reason')}")
    pick = report.get("pick")
    applied = report.get("applied")
    if applied:
        lines.append("")
        lines.append(
            f"APPLIED: {applied['param']}  "
            f"{applied['old']} → {applied['new']}"
        )
    elif pick:
        lines.append("")
        lines.append(
            f"PICKED (dry-run): {pick['param']} → {pick['direction']}"
        )
    else:
        lines.append("")
        lines.append("No change picked.")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "FILTER_BOUNDS", "FILTER_STEPS",
    "diagnose_filters", "apply_one_filter_change",
    "run_cornelius_cycle", "render_cornelius_report",
]
