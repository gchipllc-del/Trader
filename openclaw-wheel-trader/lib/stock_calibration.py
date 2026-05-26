"""
Stock Calibration — measure prediction accuracy over time.

Adapted from polybot's calibration.py for stock trading.
Tracks: "When we scored a stock 8/13 composite, how often did it hit target?"

This is how Hermes knows if our edge is real or if we're just lucky.

Key metrics:
    - Win Rate by Score Bucket: Do high-score trades win more?
    - Avg P/L by Score Bucket: Do high-score trades earn more?
    - Signal Accuracy: Which signals (candlestick, momentum, Kronos, news) predict best?
    - Kronos Accuracy: When Kronos says bullish, does the stock go up?
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CALIBRATION_FILE = DATA_DIR / "stock_calibration.json"
TRADE_HISTORY_FILE = DATA_DIR / "trade_history.json"


def _load_trade_history() -> list[dict]:
    if not TRADE_HISTORY_FILE.exists():
        return []
    with open(TRADE_HISTORY_FILE, "r") as f:
        return json.load(f)


def _load_calibration() -> list[dict]:
    if not CALIBRATION_FILE.exists():
        return []
    with open(CALIBRATION_FILE, "r") as f:
        return json.load(f)


def _save_calibration(entries: list[dict]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def record_prediction(
    ticker: str,
    composite_score: int,
    kronos_direction: str | None = None,
    kronos_expected_return: float | None = None,
    news_sentiment: float | None = None,
    pattern: str | None = None,
    momentum_score: int = 0,
    entry_price: float = 0.0,
    target_price: float = 0.0,
    stop_loss: float = 0.0,
    bayesian_win_prob: float | None = None,
    bayesian_sources: dict | None = None,
) -> dict:
    """
    Record a prediction at trade entry for later accuracy analysis.
    Called by stock_engine when a buy is executed.
    """
    entries = _load_calibration()

    entry = {
        "ticker": ticker,
        "composite_score": composite_score,
        "kronos_direction": kronos_direction,
        "kronos_expected_return": kronos_expected_return,
        "news_sentiment": news_sentiment,
        "pattern": pattern,
        "momentum_score": momentum_score,
        "entry_price": entry_price,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "bayesian_win_prob": bayesian_win_prob,
        "bayesian_sources": bayesian_sources or {},
        "outcome": None,  # Filled when trade closes
        "pnl_pct": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    entries.append(entry)
    _save_calibration(entries)
    return entry


def record_outcome(
    ticker: str,
    outcome: str,    # "win" or "loss"
    pnl_pct: float,
    close_reason: str = "",
):
    """
    Record the outcome of a prediction. Called when a stock trade closes.
    Matches the most recent unresolved prediction for this ticker.
    """
    entries = _load_calibration()

    # Find most recent unresolved entry for this ticker
    for i in range(len(entries) - 1, -1, -1):
        if entries[i]["ticker"] == ticker and entries[i]["outcome"] is None:
            entries[i]["outcome"] = outcome
            entries[i]["pnl_pct"] = pnl_pct
            entries[i]["close_reason"] = close_reason
            entries[i]["resolved_at"] = datetime.now(timezone.utc).isoformat()
            _save_calibration(entries)
            return entries[i]

    return None


def win_rate_by_score() -> dict[str, dict]:
    """
    Win rate by composite score bucket.

    Returns: {"0-4": {"wins": 2, "total": 5, "win_rate": 0.40}, ...}
    """
    history = _load_trade_history()
    if not history:
        return {}

    buckets: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})

    for trade in history:
        score = trade.get("composite_score", 0)
        pnl = trade.get("realized_pnl", 0)

        if score <= 4:
            bucket = "0-4"
        elif score <= 7:
            bucket = "5-7"
        elif score <= 10:
            bucket = "8-10"
        else:
            bucket = "11-13"

        buckets[bucket]["total"] += 1
        if pnl > 0:
            buckets[bucket]["wins"] += 1

    results = {}
    for bucket in ["0-4", "5-7", "8-10", "11-13"]:
        data = buckets.get(bucket, {"wins": 0, "total": 0})
        if data["total"] > 0:
            data["win_rate"] = round(data["wins"] / data["total"], 4)
        else:
            data["win_rate"] = 0.0
        results[bucket] = dict(data)

    return results


def avg_pnl_by_score() -> dict[str, dict]:
    """
    Average P/L percentage by composite score bucket.

    Returns: {"0-4": {"avg_pnl_pct": -0.02, "count": 5}, ...}
    """
    history = _load_trade_history()
    if not history:
        return {}

    buckets: dict[str, list] = defaultdict(list)

    for trade in history:
        score = trade.get("composite_score", 0)
        pnl_pct = trade.get("pnl_pct", 0)

        if score <= 4:
            bucket = "0-4"
        elif score <= 7:
            bucket = "5-7"
        elif score <= 10:
            bucket = "8-10"
        else:
            bucket = "11-13"

        buckets[bucket].append(pnl_pct)

    results = {}
    for bucket in ["0-4", "5-7", "8-10", "11-13"]:
        pnls = buckets.get(bucket, [])
        if pnls:
            results[bucket] = {
                "avg_pnl_pct": round(sum(pnls) / len(pnls), 4),
                "count": len(pnls),
                "best": round(max(pnls), 4),
                "worst": round(min(pnls), 4),
            }
        else:
            results[bucket] = {"avg_pnl_pct": 0.0, "count": 0, "best": 0.0, "worst": 0.0}

    return results


def kronos_accuracy() -> dict:
    """
    How accurate is Kronos? When it says bullish, does the stock go up?

    Returns: {
        "bullish": {"correct": 5, "total": 8, "accuracy": 0.625},
        "bearish": {"correct": 3, "total": 4, "accuracy": 0.750},
    }
    """
    entries = _load_calibration()
    resolved = [e for e in entries if e.get("outcome") and e.get("kronos_direction")]

    if not resolved:
        return {}

    results: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for e in resolved:
        direction = e["kronos_direction"]
        outcome = e["outcome"]
        pnl_pct = e.get("pnl_pct", 0)

        results[direction]["total"] += 1

        # Kronos was correct if:
        # - Said bullish and trade was a win (positive P/L)
        # - Said bearish and trade was a loss (or skipped — but we don't record skips)
        if direction == "bullish" and pnl_pct > 0:
            results[direction]["correct"] += 1
        elif direction == "bearish" and pnl_pct <= 0:
            results[direction]["correct"] += 1
        elif direction == "neutral" and abs(pnl_pct) < 0.02:
            results[direction]["correct"] += 1

    for direction in results:
        total = results[direction]["total"]
        if total > 0:
            results[direction]["accuracy"] = round(results[direction]["correct"] / total, 4)
        else:
            results[direction]["accuracy"] = 0.0

    return dict(results)


def signal_accuracy() -> dict[str, dict]:
    """
    Break down win rate by which signal triggered the entry.

    Returns: {"bullish_engulfing": {"wins": 3, "total": 5, "win_rate": 0.60}, ...}
    """
    entries = _load_calibration()
    resolved = [e for e in entries if e.get("outcome") and e.get("pattern")]

    if not resolved:
        return {}

    patterns: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})

    for e in resolved:
        pattern = e["pattern"]
        patterns[pattern]["total"] += 1
        if e["outcome"] == "win":
            patterns[pattern]["wins"] += 1

    results = {}
    for pattern, data in sorted(patterns.items(), key=lambda x: -x[1]["total"]):
        if data["total"] > 0:
            data["win_rate"] = round(data["wins"] / data["total"], 4)
        results[pattern] = dict(data)

    return results


def brier_score() -> float:
    """
    Brier Score — mean squared error of probability forecasts.

    Perfect calibration = 0.0
    Random guessing = 0.25
    Always wrong with confidence = 1.0

    Lower is better.
    """
    entries = _load_calibration()
    resolved = [e for e in entries
                if e.get("outcome") is not None
                and e.get("bayesian_win_prob") is not None]
    if not resolved:
        return -1.0

    total = 0.0
    for e in resolved:
        p = e["bayesian_win_prob"]
        o = 1.0 if e["outcome"] == "win" else 0.0
        total += (p - o) ** 2

    return total / len(resolved)


def log_loss() -> float:
    """
    Logarithmic scoring rule — penalizes confident wrong predictions heavily.

    Lower is better. 0 = perfect.
    """
    entries = _load_calibration()
    resolved = [e for e in entries
                if e.get("outcome") is not None
                and e.get("bayesian_win_prob") is not None]
    if not resolved:
        return -1.0

    total = 0.0
    eps = 1e-10

    for e in resolved:
        p = max(min(e["bayesian_win_prob"], 1.0 - eps), eps)
        o = 1.0 if e["outcome"] == "win" else 0.0
        total -= o * math.log(p) + (1.0 - o) * math.log(1.0 - p)

    return total / len(resolved)


def calibration_curve(n_bins: int = 10) -> dict[str, dict]:
    """
    Calibration curve — predicted vs actual win rate by probability bucket.

    Well-calibrated = predicted ~= actual for each bucket.
    If we said 70% for a bunch of trades, they should win ~70% of the time.
    """
    entries = _load_calibration()
    resolved = [e for e in entries
                if e.get("outcome") is not None
                and e.get("bayesian_win_prob") is not None]
    if not resolved:
        return {}

    bins: dict[str, list] = defaultdict(list)
    bin_width = 1.0 / n_bins

    for e in resolved:
        p = e["bayesian_win_prob"]
        bin_idx = min(int(p / bin_width), n_bins - 1)
        low = bin_idx * bin_width
        high = low + bin_width
        bin_key = f"{low:.1f}-{high:.1f}"
        bins[bin_key].append({
            "predicted": p,
            "actual": 1.0 if e["outcome"] == "win" else 0.0,
        })

    curve = {}
    for bin_key, items in sorted(bins.items()):
        predicted_mean = sum(x["predicted"] for x in items) / len(items)
        actual_rate = sum(x["actual"] for x in items) / len(items)
        curve[bin_key] = {
            "predicted_mean": round(predicted_mean, 4),
            "actual_rate": round(actual_rate, 4),
            "count": len(items),
            "gap": round(abs(predicted_mean - actual_rate), 4),
        }

    return curve


def source_accuracy() -> dict[str, dict]:
    """
    Break down accuracy by individual Bayesian source signal.
    (trend, level, pattern, momentum, kronos, news)

    Hermes uses this to adjust source weights in the future.
    """
    entries = _load_calibration()
    resolved = [e for e in entries
                if e.get("outcome") is not None
                and e.get("bayesian_sources")]
    if not resolved:
        return {}

    source_scores: dict[str, list[float]] = defaultdict(list)

    for e in resolved:
        outcome = 1.0 if e["outcome"] == "win" else 0.0
        for source_name, source_prob in e.get("bayesian_sources", {}).items():
            score = (source_prob - outcome) ** 2
            source_scores[source_name].append(score)

    results = {}
    for source_name, scores in source_scores.items():
        results[source_name] = {
            "brier": round(sum(scores) / len(scores), 4),
            "count": len(scores),
        }

    return dict(sorted(results.items(), key=lambda x: x[1]["brier"]))


def print_calibration_report():
    """Print a formatted calibration report to terminal."""
    history = _load_trade_history()
    entries = _load_calibration()
    resolved_cal = [e for e in entries if e.get("outcome")]

    print("=" * 60)
    print("  OPENCLAW STOCK CALIBRATION REPORT")
    print("=" * 60)
    print(f"  Total trades:       {len(history)}")
    print(f"  Calibration entries: {len(entries)} ({len(resolved_cal)} resolved)")

    if not history:
        print("\n  No completed trades yet. Keep trading!")
        return

    # Overall win rate
    wins = sum(1 for t in history if t.get("realized_pnl", 0) > 0)
    losses = len(history) - wins
    wr = wins / len(history) if history else 0
    print(f"\n  Win Rate:           {wr:.1%} ({wins}W / {losses}L)")

    # Average P/L
    total_pnl = sum(t.get("realized_pnl", 0) for t in history)
    avg_pnl = total_pnl / len(history) if history else 0
    print(f"  Total P/L:          ${total_pnl:+,.2f}")
    print(f"  Avg P/L per trade:  ${avg_pnl:+,.2f}")

    # Win rate by score
    wr_score = win_rate_by_score()
    if wr_score:
        print(f"\n  --- Win Rate by Score ---")
        for bucket, data in wr_score.items():
            if data["total"] > 0:
                bar_len = int(data["win_rate"] * 20)
                bar = "#" * bar_len + "." * (20 - bar_len)
                print(f"  {bucket:>6s}: {data['win_rate']:.0%} [{bar}] "
                      f"({data['wins']}/{data['total']})")

    # P/L by score
    pnl_score = avg_pnl_by_score()
    if pnl_score:
        print(f"\n  --- Avg P/L % by Score ---")
        for bucket, data in pnl_score.items():
            if data["count"] > 0:
                print(f"  {bucket:>6s}: {data['avg_pnl_pct']:+.2%} "
                      f"(best: {data['best']:+.2%}, worst: {data['worst']:+.2%}, "
                      f"n={data['count']})")

    # Brier score / Log loss (polybot-style)
    bs = brier_score()
    ll = log_loss()
    if bs >= 0:
        quality = "Excellent" if bs < 0.10 else "Good" if bs < 0.15 else "Fair" if bs < 0.20 else "Poor"
        print(f"\n  --- Bayesian Calibration ---")
        print(f"  Brier Score:  {bs:.4f} ({quality})  [perfect=0, random=0.25]")
        print(f"  Log Loss:     {ll:.4f}  [lower=better, penalizes overconfident misses]")

        # Calibration curve
        curve = calibration_curve()
        if curve:
            print(f"\n  --- Calibration Curve ---")
            for bin_key, data in curve.items():
                bar_len = int(data["actual_rate"] * 20)
                bar = "#" * bar_len + "." * (20 - bar_len)
                print(f"  {bin_key}: pred={data['predicted_mean']:.2f} "
                      f"actual={data['actual_rate']:.2f} [{bar}] n={data['count']}")

        # Source accuracy
        sa = source_accuracy()
        if sa:
            print(f"\n  --- Source Accuracy (lower Brier = better) ---")
            for source, data in sa.items():
                print(f"  {source:12s}: Brier={data['brier']:.4f}  (n={data['count']})")

    # Kronos accuracy
    ka = kronos_accuracy()
    if ka:
        print(f"\n  --- Kronos Prediction Accuracy ---")
        for direction, data in ka.items():
            if data["total"] > 0:
                print(f"  {direction:10s}: {data['accuracy']:.0%} "
                      f"({data['correct']}/{data['total']})")

    # Signal accuracy
    sa = signal_accuracy()
    if sa:
        print(f"\n  --- Candlestick Pattern Accuracy ---")
        for pattern, data in sa.items():
            if data["total"] >= 2:  # Only show patterns with enough data
                print(f"  {pattern:22s}: {data['win_rate']:.0%} "
                      f"({data['wins']}/{data['total']})")

    print("=" * 60)
