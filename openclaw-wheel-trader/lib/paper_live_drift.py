"""Paper-vs-live drift detector — guard against the Kalshi-pattern bug
where paper-mode pricing diverged from what live would have actually
done.

For traderbot the same risk exists in subtler form: backtest fills at
the bar's close, live market orders fill at the ask (buys) or bid (sells)
plus a few bps of impact. When you flip live, surprises compound.

This module periodically compares:
  - The bot's INTENDED fill (from order_gate.intent.limit_price or
    the bar's close at signal time)
  - The actual EXECUTED fill (Alpaca fill_price from the broker)

Drift = (executed - intended) / intended

Logged per fill to data/paper_live_drift.jsonl. The CLI summarizes
mean/median drift, the worst-case ticker, and whether the realism
knobs in wheel_strategy.yaml need recalibration.

Run:
  python main.py paper-live-drift                  # show summary
  python main.py paper-live-drift --record         # add today's fills
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "paper_live_drift.jsonl"
POSITIONS_PATH = ROOT / "data" / "positions.json"


def _load_positions() -> list[dict]:
    if not POSITIONS_PATH.exists():
        return []
    try:
        with open(POSITIONS_PATH) as f:
            raw = json.load(f)
        return raw.get("positions", raw) if isinstance(raw, dict) else raw
    except (OSError, json.JSONDecodeError):
        return []


def _append(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _read_all() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    out = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def record_fills() -> dict:
    """Walk current positions; for any with a recorded fill_price that
    differs from intent/signal-time price, log the drift.

    The position structure (from execute_stock_buy):
      ticker, entry_price, expected_entry, shares, opened_at, fill_price,
      fill_at, intent_price
    Some fields may be absent on legacy positions — handle gracefully.
    """
    positions = _load_positions()
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)

    seen_ids: set[str] = {r.get("position_id") for r in _read_all()}
    recorded = 0
    for p in positions:
        pid = (p.get("order_id") or
               f"{p.get('ticker','?')}_{p.get('opened_at','?')}")
        if pid in seen_ids:
            continue
        intent = p.get("intent_price") or p.get("expected_entry")
        fill = p.get("fill_price") or p.get("entry_price")
        if intent is None or fill is None:
            continue
        opened_at = p.get("opened_at")
        if not opened_at:
            continue
        try:
            dt = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if dt < cutoff:
            continue
        try:
            intent = float(intent)
            fill = float(fill)
        except (TypeError, ValueError):
            continue
        if intent <= 0:
            continue
        drift_pct = (fill - intent) / intent
        _append({
            "position_id": pid,
            "ticker": p.get("ticker"),
            "opened_at": opened_at,
            "side": "buy" if (p.get("type") or "").startswith("stock") else "?",
            "intent_price": round(intent, 4),
            "fill_price": round(fill, 4),
            "drift_pct": round(drift_pct, 6),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
        recorded += 1
    return {"recorded": recorded, "total_in_log": len(_read_all())}


def summary(window_days: int = 30) -> dict:
    """Aggregate drift stats over the last `window_days`."""
    rows = _read_all()
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    window: list[dict] = []
    for r in rows:
        ts = r.get("opened_at") or r.get("recorded_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if dt >= cutoff:
            window.append(r)
    if not window:
        return {"n": 0, "window_days": window_days}

    drifts = [r["drift_pct"] for r in window if "drift_pct" in r]
    drifts.sort()
    mean = sum(drifts) / len(drifts)
    median = drifts[len(drifts) // 2]
    worst_above = max(drifts)
    worst_below = min(drifts)

    # Per-ticker median drift
    by_ticker: dict[str, list[float]] = {}
    for r in window:
        t = r.get("ticker", "?")
        by_ticker.setdefault(t, []).append(r.get("drift_pct", 0.0))
    ticker_medians = sorted(
        [(t, sorted(v)[len(v) // 2], len(v)) for t, v in by_ticker.items()],
        key=lambda x: -abs(x[1]),
    )[:10]

    # Calibration check: how does the bot's currently-modeled slippage
    # compare to actual? We want bot's entry_slippage_pct ≈ |mean drift|
    # if the bot is buying. If actual drift is 2× the modeled value, the
    # backtest is overstating returns and the operator should widen the
    # slippage knob in wheel_strategy.yaml.
    return {
        "n": len(window),
        "window_days": window_days,
        "mean_drift_pct": round(mean, 6),
        "median_drift_pct": round(median, 6),
        "worst_above_pct": round(worst_above, 6),
        "worst_below_pct": round(worst_below, 6),
        "per_ticker_median": [
            {"ticker": t, "median_drift": round(m, 6), "n": n}
            for t, m, n in ticker_medians
        ],
        "recommendation": _recommend(mean, median),
    }


def _recommend(mean: float, median: float) -> str:
    """Compare observed drift to the bot's modeled slippage."""
    try:
        import yaml
        path = ROOT / "config" / "wheel_strategy.yaml"
        with open(path) as f:
            strat = yaml.safe_load(f) or {}
        sp = strat.get("stock_params", {})
        modeled = float(sp.get("entry_slippage_pct", 0.0005))
    except Exception:
        modeled = 0.0005

    observed = max(abs(mean), abs(median))
    if observed < modeled * 0.5:
        return f"OK: observed drift ({observed:.4f}) is below modeled ({modeled:.4f}). Realism knob can stay or even tighten."
    if observed < modeled * 1.5:
        return f"OK: observed drift ({observed:.4f}) is within ±50% of modeled ({modeled:.4f}). Calibration looks right."
    if observed < modeled * 3:
        return f"WIDEN: observed drift ({observed:.4f}) is 1.5-3× modeled ({modeled:.4f}). Consider widening entry/exit_slippage_pct to {round(observed, 4)}."
    return f"URGENT: observed drift ({observed:.4f}) is >3× modeled ({modeled:.4f}). Backtest is materially overstating fills. Widen slippage to {round(observed, 4)} immediately."


def render(s: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("PAPER-VS-LIVE FILL DRIFT")
    lines.append("=" * 70)
    if s.get("n", 0) == 0:
        lines.append(f"No drift records in last {s.get('window_days', 30)} days.")
        return "\n".join(lines + [""])
    lines.append(f"Window:         {s['window_days']} days, {s['n']} fills")
    lines.append(f"Mean drift:     {s['mean_drift_pct']:+.4%}")
    lines.append(f"Median drift:   {s['median_drift_pct']:+.4%}")
    lines.append(f"Worst above:    {s['worst_above_pct']:+.4%}")
    lines.append(f"Worst below:    {s['worst_below_pct']:+.4%}")
    lines.append("")
    lines.append("Per-ticker median drift (top 10 by |drift|):")
    for r in s.get("per_ticker_median", []):
        lines.append(f"  {r['ticker']:<6}  {r['median_drift']:+.4%}  (n={r['n']})")
    lines.append("")
    lines.append("Calibration check:")
    lines.append(f"  {s.get('recommendation', '')}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["record_fills", "summary", "render"]


if __name__ == "__main__":
    import sys
    if "--record" in sys.argv:
        r = record_fills()
        print(f"Recorded {r['recorded']} new fill(s). Log size: {r['total_in_log']}")
    print(render(summary()))
