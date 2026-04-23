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

# Safety bounds — Hermes can NEVER exceed these
BOUNDS = {
    "stop_loss_pct":         (0.02, 0.08),
    "default_target_pct":    (0.05, 0.20),
    "min_composite_score":   (2, 10),  # Expanded 6→10 for high-conviction growth mode
    "max_position_pct":      (0.10, 0.30),
    "max_trades_per_scan":   (1, 5),
    "trailing_stop_pct":     (0.0, 0.06),
    "max_concurrent_positions": (2, 8),
    "partial_exit_threshold": (0.08, 0.30),  # New: scale-out trigger (8%-30%)
    "partial_exit_fraction":  (0.25, 0.75),  # New: % to sell at scale-out
    # New polybot integration parameters
    "kelly_fraction":         (0.10, 0.50),  # Fractional Kelly multiplier
    "bayesian_min_win_prob":  (0.50, 0.75),  # Bayesian veto threshold
    "correlation_threshold":  (0.50, 0.85),  # Price correlation flag point
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
    if not POSITIONS_PATH.exists():
        return []
    with open(POSITIONS_PATH) as f:
        return json.load(f)


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

def review_trades(lookback_days: int = 14) -> dict:
    """
    Analyze closed trades from the last N days.

    Returns performance metrics broken down by:
    - Overall win rate, avg win, avg loss
    - By pattern (which candlestick signals work best)
    - By momentum score (do high-momentum entries outperform)
    - Stop loss effectiveness (are stops too tight or too loose)
    """
    positions = _load_positions()
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    closed = []
    for p in positions:
        if p.get("status") != "closed":
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
        return {"total_trades": 0, "message": "No closed trades in lookback period"}

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
    if review["total_trades"] < 3:
        return [{"param": "none", "direction": "hold", "reason": f"Only {review['total_trades']} trades — insufficient data"}]

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
    if review["win_rate"] < 0.40:
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

    # Step 1: Review
    review = review_trades(lookback_days)

    # Step 2: Diagnose
    recommendations = diagnose(review, current_params)

    # Step 3: Tune
    if dry_run or review["total_trades"] < 3:
        changes = {}
    else:
        changes = apply_adjustments(recommendations)

    # Step 4: Log
    report = {
        "cycle": "hermes_optimization",
        "lookback_days": lookback_days,
        "dry_run": dry_run,
        "review": review,
        "recommendations": recommendations,
        "changes": changes,
        "params_after": strategy.get("stock_params", {}),
    }

    _log_optimization(report)

    log_event("hermes", "optimization_complete", {
        "trades_reviewed": review["total_trades"],
        "recommendations": len(recommendations),
        "changes_applied": len(changes),
        "win_rate": review.get("win_rate", 0),
        "expectancy": review.get("expectancy", 0),
    })

    diary_write("hermes_agent",
        f"OPT|trades_{review['total_trades']}|wr_{review.get('win_rate', 0):.0%}|"
        f"exp_{review.get('expectancy', 0):+.2%}|changes_{len(changes)}")

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
            print(f"  {k:<30s} {v:<10} (bounds: {bounds[0]}-{bounds[1]})")

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
