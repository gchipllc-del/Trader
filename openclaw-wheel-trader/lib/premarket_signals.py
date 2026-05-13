"""
Pre-market signal computation.

Two roles:
  1. Direct input to ``cmd_premarket_scan`` — find stocks gapping down
     overnight that look like dip-buy candidates.
  2. Persisted signal that the regular-hours screener can read at
     market open — overnight gap is a leading indicator that decays
     fast but is useful in the first hour of regular trading.

Pre-market quote feeds: Alpaca's IEX feed includes pre-market data
free. We grab the latest trade (which works in extended hours) and
compare to the prior session close. Gap = (current - prior_close) /
prior_close.

The signal output is persisted to ``data/premarket_signals.json``
right after computation so the next screener pass picks it up without
having to recompute. Persistence is intentionally a flat JSON file —
day-time consumers shouldn't pay for the broker API again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from lib.audit import log_event

SIGNAL_PATH = Path(__file__).parent.parent / "data" / "premarket_signals.json"
# How long a stored signal stays usable before downstream consumers
# treat it as stale. 8h covers the gap from pre-market open (4 AM ET)
# through the first regular hour and a bit (10:30 AM ET / 9:30 AM CT).
SIGNAL_TTL_HOURS = 8


@dataclass
class PremarketSignal:
    """One ticker's overnight gap reading."""
    ticker: str
    prior_close: float
    last_price: float
    gap_pct: float          # signed; negative = gapped down
    premarket_volume: int   # cumulative volume since pre-market open
    fetched_at: str         # ISO timestamp


def compute_overnight_gap(ticker: str, client) -> PremarketSignal | None:
    """Compute the current overnight gap for one ticker.

    Returns None if we can't get either the prior close or a current
    price — e.g. data fetch failure, missing daily bar, or asset that
    Alpaca doesn't trade. Best-effort: a return of None is never an
    error from the caller's perspective.
    """
    try:
        # Most recent daily bar = prior session close (today's bar
        # won't exist yet in pre-market).
        bars = client.get_bars([ticker], timeframe="1Day", limit=2)
        df = bars.get(ticker) if isinstance(bars, dict) else None
        if df is None or len(df) == 0:
            return None
        prior_close = float(df["close"].iloc[-1])
    except Exception:
        return None

    try:
        # Latest trade pulls extended-hours data automatically when in pre-market
        from alpaca.data.requests import StockLatestTradeRequest
        sc = client._get_stock_data_client()
        client.limiter.wait_if_needed()
        req = StockLatestTradeRequest(symbol_or_symbols=ticker)
        resp = sc.get_stock_latest_trade(req)
        trade = resp.get(ticker) if isinstance(resp, dict) else None
        if trade is None:
            return None
        last_price = float(trade.price)
    except Exception:
        return None

    gap_pct = (last_price - prior_close) / prior_close if prior_close > 0 else 0.0
    # Approximate pre-market volume — without ticking through every
    # extended-hours minute bar, we just leave it as 0 for now.
    # Day-time consumers care about the *gap*, not the volume; if a
    # downstream caller wants accurate volume, add a 1Min bar fetch.
    return PremarketSignal(
        ticker=ticker,
        prior_close=round(prior_close, 4),
        last_price=round(last_price, 4),
        gap_pct=round(gap_pct, 4),
        premarket_volume=0,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def persist_signals(signals: list[PremarketSignal]) -> None:
    """Write the signals to disk for the regular-hours screener to read.

    Overwrites the file — pre-market signals are time-bound and an
    older snapshot shouldn't pollute the next day's screen.
    """
    SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "signals": {s.ticker: asdict(s) for s in signals},
    }
    tmp = SIGNAL_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(SIGNAL_PATH)


def load_signals(*, max_age_hours: float = SIGNAL_TTL_HOURS) -> dict[str, dict]:
    """Read persisted signals if fresh enough. Returns ``{ticker: dict}``
    keyed by ticker. Empty dict when missing, stale, or unparseable —
    callers should always handle the empty case.
    """
    if not SIGNAL_PATH.exists():
        return {}
    try:
        with open(SIGNAL_PATH) as f:
            payload = json.load(f)
        computed = datetime.fromisoformat(
            payload.get("computed_at", "").replace("Z", "+00:00")
        )
        age_hours = (datetime.now(timezone.utc) - computed).total_seconds() / 3600
        if age_hours > max_age_hours:
            return {}
        return payload.get("signals", {})
    except (ValueError, json.JSONDecodeError, OSError):
        return {}


def get_gap_for(ticker: str) -> float | None:
    """Convenience: return the signed gap_pct for ``ticker``, or None
    if no fresh signal is available.
    """
    signals = load_signals()
    sig = signals.get(ticker.upper())
    if not sig:
        return None
    return float(sig.get("gap_pct", 0))


def compute_all(tickers: list[str], client) -> list[PremarketSignal]:
    """Compute + persist signals for the given ticker list. Returns
    the list (may be shorter than input — missing tickers are dropped).
    """
    signals: list[PremarketSignal] = []
    for t in tickers:
        s = compute_overnight_gap(t, client)
        if s is not None:
            signals.append(s)
    if signals:
        persist_signals(signals)
        log_event("premarket_signals", "computed", {
            "n_tickers": len(signals),
            "gaps": {s.ticker: s.gap_pct for s in signals},
        })
    return signals
