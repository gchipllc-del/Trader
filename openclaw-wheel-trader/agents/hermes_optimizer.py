"""
Hermes Self-Optimization Agent — adaptive parameter tuning.

Named after the Greek god of trade and commerce. Hermes reviews trade
history, analyzes what's working (and what isn't), and adjusts strategy
parameters to maximize growth. Runs after market close each day.

Self-optimization loop:
  1. REVIEW: Analyze last N trades (wins, losses, by ticker/pattern/momentum)
  2. DIAGNOSE: Identify which parameters contribute to wins vs losses
  3. TUNE: Adjust parameters within safe bounds
  4. LOG: Record every change with reasoning for audit trail
  5. VALIDATE: Ensure parameters stay within safety bounds

Tunable parameters (in wheel_strategy.yaml → stock_params):
  - stop_loss_pct: 0.02-0.08
  - default_target_pct: 0.05-0.20
  - min_composite_score: 2-6
  - max_position_pct: 0.10-0.30
  - max_trades_per_scan: 1-5
  - trailing_stop_pct: 0.0-0.06
  - allow_momentum_only: true/false
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

from lib.audit import log_event
from lib.memory_palace import diary_write

STRATEGY_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"
SETTINGS_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"
OPTIMIZATION_LOG = Path(__file__).parent.parent / "data" / "hermes_log.jsonl"

# Safety bounds — Hermes can NEVER exceed these.
# 2026-05-01 widening pass: with Wave 1-3 guards in place (broker dedup,
# auto-trim, fill polling, persistent dedup), Hermes can safely explore
# more of the space without breakers ever silently stranding a position.
# Each widening below stays inside the circuit-breaker ceiling.
BOUNDS = {
    "stop_loss_pct":         (0.02, 0.08),  # unchanged — 2-8% is the right range
    "default_target_pct":    (0.05, 0.30),  # was 0.20 — let runners run with trailing stop
    "min_composite_score":   (2, 8),        # universe rarely scores >8.5/10 (saw starvation 2026-04-27)
    "max_position_pct":      (0.10, 0.30),  # tied to circuit_breakers.max_position_pct
    "max_trades_per_scan":   (1, 7),        # was 5 — high-edge days fire more entries
    "trailing_stop_pct":     (0.0, 0.08),   # was 0.06 — more flexibility
    "max_concurrent_positions": (2, 10),    # was 8 — more book breadth as bankroll grows
    "partial_exit_threshold": (0.05, 0.30), # was 0.08 — allow faster partials
    "partial_exit_fraction":  (0.25, 0.75), # unchanged
    "kelly_fraction":         (0.10, 0.50), # half-Kelly cap stays — full Kelly = ruin risk
    "bayesian_min_win_prob":  (0.45, 0.75), # was 0.50 — slightly more permissive
    "correlation_threshold":  (0.40, 0.85), # was 0.50 — more sensitive to clustering
}

# How much to adjust per optimization cycle (conservative steps)
STEP_SIZES = {
    "stop_loss_pct":         0.005,
    "default_target_pct":    0.01,
    "min_composite_score":   1,
    "max_position_pct":      0.025,
    "max_trades_per_scan":   1,
    "trailing_stop_pct":     0.005,
    "partial_exit_threshold": 0.02,
    "partial_exit_fraction":  0.10,
    "kelly_fraction":         0.05,
    "bayesian_min_win_prob":  0.02,
    "correlation_threshold":  0.05,
    "max_concurrent_positions": 1,
}


def _load_strategy() -> dict:
    with open(STRATEGY_PATH) as f:
        return yaml.safe_load(f)


def _save_strategy(strategy: dict):
    with open(STRATEGY_PATH, "w") as f:
        yaml.safe_dump(strategy, f, default_flow_style=False, sort_keys=False)


def _load_positions() -> list[dict]:
    """Locked snapshot via the canonical store (audit finding #5)."""
    from lib.positions_store import load_positions as _store_load
    return _store_load(POSITIONS_PATH)


def _log_optimization(entry: dict):
    """Append to the Hermes optimization log."""
    OPTIMIZATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(OPTIMIZATION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _clamp(value, param_name: str):
    """Clamp a value within safety bounds."""
    lo, hi = BOUNDS.get(param_name, (value, value))
    if isinstance(lo, int) and isinstance(hi, int):
        return max(lo, min(hi, int(round(value))))
    return max(lo, min(hi, round(value, 4)))


# ============================================================
# STEP 1: REVIEW — Analyze recent trade performance
# ============================================================

def review_trades(lookback_days: int = 14, trade_type: str = "stock") -> dict:
    """
    Analyze closed trades from the last N days for a specific strategy type.

    Wave 3 #12: Hermes tunes `stock_params`, so CSP/CC trades must be
    excluded from the input — their P/L distribution and stop/target
    semantics differ. Pass trade_type="csp" or "cc" if/when those have
    their own Hermes tuners.

    Returns performance metrics broken down by:
    - Overall win rate, avg win, avg loss
    - By pattern (which candlestick signals work best)
    - By momentum score (do high-momentum entries outperform)
    - Stop loss effectiveness (are stops too tight or too loose)
    """
    positions = _load_positions()
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    closed = []
    skipped_other_types = 0
    for p in positions:
        if p.get("status") != "closed":
            continue
        # Filter to the strategy type Hermes is tuning (Wave 3 #12).
        # Default "stock" type — older entries may have no `type` field;
        # treat those as stock for backward compat.
        ptype = p.get("type", "stock")
        if ptype != trade_type:
            skipped_other_types += 1
            continue
        closed_at = p.get("closed_at", "")
        if not closed_at:
            continue
        try:
            close_dt = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
            if close_dt < cutoff:
                continue
        except (ValueError, TypeError):
            continue
        closed.append(p)

    if not closed:
        return {
            "total_trades": 0,
            "trade_type": trade_type,
            "skipped_other_types": skipped_other_types,
            "message": (
                f"No closed {trade_type} trades in last {lookback_days}d "
                f"(skipped {skipped_other_types} of other types)"
            ),
        }

    wins = []
    losses = []
    stop_outs = []
    target_hits = []
    patterns = {}

    for p in closed:
        entry = p.get("entry_price", 0)
        # Estimate exit price from close reason
        reason = p.get("close_reason", "")
        stop = p.get("stop_loss", 0)
        target = p.get("target_price", 0)
        score = p.get("composite_score", 0)
        ticker = p.get("ticker", "?")

        # Calculate approximate P/L
        if reason == "stop_loss":
            exit_price = stop
            stop_outs.append(p)
        elif reason == "take_profit":
            exit_price = target
            target_hits.append(p)
        else:
            # Estimate: midpoint between entry and target for wins, entry and stop for losses
            exit_price = entry  # Unknown, assume breakeven

        pnl_pct = (exit_price - entry) / entry if entry > 0 else 0

        if pnl_pct > 0:
            wins.append({"ticker": ticker, "pnl_pct": pnl_pct, "score": score, "reason": reason})
        else:
            losses.append({"ticker": ticker, "pnl_pct": pnl_pct, "score": score, "reason": reason})

    total = len(closed)
    win_rate = len(wins) / total if total > 0 else 0
    avg_win = sum(w["pnl_pct"] for w in wins) / len(wins) if wins else 0
    avg_loss = sum(l["pnl_pct"] for l in losses) / len(losses) if losses else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Score breakdown: what composite scores produce wins?
    win_scores = [w["score"] for w in wins]
    loss_scores = [l["score"] for l in losses]
    avg_win_score = sum(win_scores) / len(win_scores) if win_scores else 0
    avg_loss_score = sum(loss_scores) / len(loss_scores) if loss_scores else 0

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 3),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "expectancy": round(expectancy, 4),
        "stop_outs": len(stop_outs),
        "target_hits": len(target_hits),
        "avg_win_score": round(avg_win_score, 1),
        "avg_loss_score": round(avg_loss_score, 1),
        "profit_factor": round(abs(avg_win * len(wins)) / abs(avg_loss * len(losses)), 2) if losses and avg_loss != 0 else 999,
    }


# ============================================================
# STEP 2: DIAGNOSE — Identify what needs tuning
# ============================================================

def diagnose(review: dict, current_params: dict) -> list[dict]:
    """
    Given trade review results, produce a list of recommended adjustments.

    Each recommendation: {"param": str, "direction": "increase"|"decrease", "reason": str}
    """
    # Small-sample guard. Hermes used to recommend changes on samples as
    # small as 3 trades, which is statistically meaningless. On 2026-04-27
    # this drove min_composite_score to 10/10 (max) on a 3-trade sample
    # with 33% win rate, gating the bot to zero entries the next day.
    # Now we hold below 8 trades, and below 15 we suppress entry-tightening
    # specifically (it's the single most catastrophic miscalibration).
    MIN_TRADES_FOR_ENTRY_TIGHTENING = 15
    MIN_TRADES_ANY_CHANGE = 8

    if review["total_trades"] < MIN_TRADES_ANY_CHANGE:
        return [{
            "param": "none", "direction": "hold",
            "reason": (
                f"Only {review['total_trades']} trades in window "
                f"(need ≥{MIN_TRADES_ANY_CHANGE} for any change)."
            ),
        }]

    # `increase` on min_composite_score means "be more selective" — tightening
    # the entry filter. With a small sample, a single bad streak can push
    # this to a level the universe can't actually achieve. Suppress until
    # we have ≥15 trades.
    suppress_entry_tightening = (
        review["total_trades"] < MIN_TRADES_FOR_ENTRY_TIGHTENING
    )

    recommendations = []

    # Too many stop-outs → stop may be too tight
    stop_out_rate = review["stop_outs"] / review["total_trades"]
    if stop_out_rate > 0.50:
        recommendations.append({
            "param": "stop_loss_pct",
            "direction": "increase",
            "reason": f"Stop-out rate {stop_out_rate:.0%} is high — stops may be too tight",
        })
    elif stop_out_rate < 0.15 and review["avg_loss_pct"] < -0.06:
        recommendations.append({
            "param": "stop_loss_pct",
            "direction": "decrease",
            "reason": f"Few stop-outs but avg loss {review['avg_loss_pct']:+.1%} is large — tighten stops",
        })

    # Win rate too low → raise minimum score or reduce trades
    if review["win_rate"] < 0.40 and not suppress_entry_tightening:
        recommendations.append({
            "param": "min_composite_score",
            "direction": "increase",
            "reason": f"Win rate {review['win_rate']:.0%} is low — be more selective",
        })
    elif review["win_rate"] > 0.65 and review["total_trades"] < 8:
        # High win rate but few trades → can be less selective
        recommendations.append({
            "param": "min_composite_score",
            "direction": "decrease",
            "reason": f"Win rate {review['win_rate']:.0%} is high with few trades — can take more setups",
        })

    # Avg win too small vs avg loss → widen targets or tighten stops
    if review["avg_win_pct"] > 0 and abs(review["avg_loss_pct"]) > 0:
        rr_ratio = review["avg_win_pct"] / abs(review["avg_loss_pct"])
        if rr_ratio < 1.0:
            recommendations.append({
                "param": "default_target_pct",
                "direction": "increase",
                "reason": f"R:R ratio {rr_ratio:.1f} is < 1 — need wider targets",
            })

    # Good performance → increase position sizing for faster growth
    if review["expectancy"] > 0.02 and review["win_rate"] > 0.50:
        recommendations.append({
            "param": "max_position_pct",
            "direction": "increase",
            "reason": f"Positive expectancy {review['expectancy']:+.2%} — increase position size for growth",
        })
    elif review["expectancy"] < 0:
        recommendations.append({
            "param": "max_position_pct",
            "direction": "decrease",
            "reason": f"Negative expectancy {review['expectancy']:+.2%} — reduce risk per trade",
        })

    # Good win rate + positive expectancy → increase trade frequency
    if review["expectancy"] > 0.01 and review["win_rate"] > 0.45:
        recommendations.append({
            "param": "max_trades_per_scan",
            "direction": "increase",
            "reason": f"Strategy is profitable — increase trade frequency",
        })

    # Enable trailing stop if we're leaving money on the table
    # (lots of target hits = price is running past our targets)
    if review["target_hits"] > review["total_trades"] * 0.4:
        current_trailing = current_params.get("trailing_stop_pct", 0)
        if current_trailing == 0:
            recommendations.append({
                "param": "trailing_stop_pct",
                "direction": "increase",
                "reason": f"{review['target_hits']}/{review['total_trades']} hit targets — add trailing stop to catch bigger moves",
            })

    return recommendations


# ============================================================
# STEP 3: TUNE — Apply adjustments within bounds
# ============================================================

def apply_adjustments(recommendations: list[dict]) -> dict:
    """
    Apply recommended parameter adjustments to wheel_strategy.yaml.

    Returns dict of changes made.
    """
    strategy = _load_strategy()
    if "stock_params" not in strategy:
        strategy["stock_params"] = {}

    params = strategy["stock_params"]
    changes = {}

    for rec in recommendations:
        param = rec["param"]
        direction = rec["direction"]

        if param == "none" or direction == "hold":
            continue
        if param not in BOUNDS:
            continue

        current = params.get(param, BOUNDS[param][0])
        step = STEP_SIZES.get(param, 0)

        if direction == "increase":
            new_value = current + step
        elif direction == "decrease":
            new_value = current - step
        else:
            continue

        new_value = _clamp(new_value, param)

        if new_value != current:
            old_value = current
            params[param] = new_value
            changes[param] = {
                "old": old_value,
                "new": new_value,
                "reason": rec["reason"],
            }

    # Also update circuit breaker max_position_pct if it changed
    if "max_position_pct" in changes:
        with open(SETTINGS_PATH) as f:
            settings = yaml.safe_load(f)
        settings["circuit_breakers"]["max_position_pct"] = changes["max_position_pct"]["new"]
        with open(SETTINGS_PATH, "w") as f:
            yaml.safe_dump(settings, f, default_flow_style=False, sort_keys=False)

    if changes:
        _save_strategy(strategy)

    return changes


# ============================================================
# STEP 4+5: OPTIMIZE — Full cycle with logging and validation
# ============================================================

def run_optimization(lookback_days: int = 14, dry_run: bool = False) -> dict:
    """
    Full Hermes optimization cycle:
      1. Review recent trades
      2. Diagnose issues
      3. Tune parameters (unless dry_run)
      4. Log everything
      5. Validate bounds

    Returns full optimization report.
    """
    strategy = _load_strategy()
    current_params = strategy.get("stock_params", {})

    # Step 1: Review (stock-only — Wave 3 #12)
    review = review_trades(lookback_days, trade_type="stock")

    # Step 2: Diagnose
    recommendations = diagnose(review, current_params)

    # Step 3: Tune
    skip_reason = None
    if dry_run:
        changes = {}
        skip_reason = "dry_run"
    elif review["total_trades"] < 3:
        changes = {}
        skip_reason = (
            f"insufficient_trades: have {review['total_trades']} closed stock "
            f"trades in last {lookback_days}d, need 3"
        )
    elif not recommendations:
        changes = {}
        skip_reason = "no_recommendations: metrics within tolerance bands"
    else:
        changes = apply_adjustments(recommendations)
        if not changes:
            skip_reason = "all_adjustments_at_bounds: recommendations would push past hermes_bounds"

    # Step 4: Log
    report = {
        "cycle": "hermes_optimization",
        "lookback_days": lookback_days,
        "dry_run": dry_run,
        "review": review,
        "recommendations": recommendations,
        "changes": changes,
        "skip_reason": skip_reason,
        "params_after": strategy.get("stock_params", {}),
    }

    _log_optimization(report)

    # Wave 3 #11: always emit a diagnosis line to the audit trail so an
    # operator can answer "why didn't Hermes change anything?" without
    # spelunking through trade history.
    log_event("hermes", "optimization_complete", {
        "trades_reviewed": review["total_trades"],
        "trade_type": review.get("trade_type", "stock"),
        "skipped_other_types": review.get("skipped_other_types", 0),
        "recommendations": len(recommendations),
        "changes_applied": len(changes),
        "skip_reason": skip_reason,
        "win_rate": review.get("win_rate", 0),
        "expectancy": review.get("expectancy", 0),
    })

    diary_summary = (
        f"OPT|trades_{review['total_trades']}|wr_{review.get('win_rate', 0):.0%}|"
        f"exp_{review.get('expectancy', 0):+.2%}|changes_{len(changes)}"
    )
    if skip_reason:
        diary_summary += f"|skip_{skip_reason.split(':')[0]}"
    diary_write("hermes_agent", diary_summary)

    return report


def print_optimization_report(report: dict):
    """Print a human-readable optimization report."""
    review = report["review"]
    recs = report["recommendations"]
    changes = report["changes"]

    print("=" * 55)
    print("  HERMES SELF-OPTIMIZATION REPORT")
    print("=" * 55)

    if review["total_trades"] == 0:
        print("  No trades to analyze yet.")
        print("  Hermes needs at least 3 closed trades to start optimizing.")
        print()
        return

    print(f"\n  PERFORMANCE REVIEW ({report['lookback_days']}d)")
    print(f"  {'Total trades:':<25s} {review['total_trades']}")
    print(f"  {'Win rate:':<25s} {review['win_rate']:.0%} ({review['wins']}W / {review['losses']}L)")
    print(f"  {'Avg win:':<25s} {review.get('avg_win_pct', 0):+.2%}")
    print(f"  {'Avg loss:':<25s} {review.get('avg_loss_pct', 0):+.2%}")
    print(f"  {'Expectancy:':<25s} {review.get('expectancy', 0):+.2%}")
    print(f"  {'Profit factor:':<25s} {review.get('profit_factor', 0):.1f}")
    print(f"  {'Stop-outs:':<25s} {review.get('stop_outs', 0)}")
    print(f"  {'Target hits:':<25s} {review.get('target_hits', 0)}")

    print(f"\n  RECOMMENDATIONS ({len(recs)})")
    for r in recs:
        arrow = "↑" if r["direction"] == "increase" else ("↓" if r["direction"] == "decrease" else "–")
        print(f"  {arrow} {r['param']}: {r['reason']}")

    if changes:
        print(f"\n  CHANGES APPLIED ({len(changes)})")
        for param, detail in changes.items():
            print(f"  • {param}: {detail['old']} → {detail['new']}")
            print(f"    Reason: {detail['reason']}")
    elif report.get("dry_run"):
        print("\n  [DRY RUN — no changes applied]")
    else:
        print("\n  No parameter changes needed.")

    # Current params
    params = report.get("params_after", {})
    if params:
        print(f"\n  CURRENT PARAMETERS")
        for k, v in sorted(params.items()):
            bounds = BOUNDS.get(k, ("?", "?"))
            # Cast to str — some params are lists (e.g. support_distance_tiers)
            # which can't be formatted with the f-string {:<10} alignment spec.
            v_str = str(v)
            print(f"  {k:<30s} {v_str:<10} (bounds: {bounds[0]}-{bounds[1]})")

    print()


def get_optimization_history(limit: int = 10) -> list[dict]:
    """Read the last N optimization entries from the log."""
    if not OPTIMIZATION_LOG.exists():
        return []

    entries = []
    with open(OPTIMIZATION_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return entries[-limit:]
