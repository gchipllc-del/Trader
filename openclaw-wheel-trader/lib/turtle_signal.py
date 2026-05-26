"""Turtle Trading System — Richard Dennis's 1980s breakout strategy.

The classic version, adapted for our equity universe and Alpaca daily bars.
Source: Mato Conti video summary (2026-05-24), referencing the original
Dennis/Eckhardt Turtle program.

The rules:

  REGIME FILTER (trend direction)
    - Price > 200-period SMA  →  long-only regime
    - Price < 200-period SMA  →  short-only regime (we ignore — long-only bot)
    - Bot only fires when in a CONFIRMED long regime.

  ENTRY (Donchian breakout)
    - LONG when today's close > rolling 40-bar high (excluding today)
    - This is the "System 2" Turtle entry; the 20-bar variant ("System 1")
      is also implemented as an optional shorter-horizon mode.

  EXIT (ATR-based stop)
    - Compute the Average True Range over the last 14 bars
    - Stop loss = entry_price - (atr_multiplier × ATR), default multiplier=2.0
    - This is "Volatility Stop" — dynamically wider in choppy markets, tighter
      in quiet ones. The Turtles used 2N (2 ATRs).

  POSITION SIZING (volatility-targeted)
    - The Turtles called this "Unit Size":
        unit = (bankroll × risk_per_trade) / (atr_multiplier × ATR × shares_per_dollar)
    - Default risk_per_trade = 0.02 (2% of bankroll at risk per Unit)

Per the source video, this exact rule set on 60-min NQ futures produced
~30% win rate but +$4,850 avg win vs -$1,700 avg loss over a multi-decade
backtest — the asymmetric edge that defines breakout systems.

This module is PURE PYTHON (no pandas / numpy) so it runs in the current
NumPy-2-fragile env. Math is exact via list math. Daily bars are pulled
through ``lib.markov_regime._fetch_alpaca_daily_closes`` — same path as
the Markov module — so the data dependency is shared.

PUBLIC API
    classify_regime(prices, window=200) -> "long" | "short" | "flat"
    donchian_break(prices, window=40) -> "breakout_up" | "breakout_down" | "none"
    atr(highs, lows, closes, window=14) -> float
    unit_size(bankroll, atr_value, atr_multiplier=2.0, risk_per_trade=0.02,
              price=...) -> int  # shares
    turtle_signal(ticker, ...) -> dict
    render_summary(result) -> str
    universe_scan(tickers, ...) -> list[dict]   # rank by signal strength
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal


# Defaults match the canonical Turtle System 2 (long-term version)
DEFAULT_REGIME_WINDOW = 200    # SMA period for the trend filter
DEFAULT_BREAKOUT_WINDOW = 40   # Donchian breakout lookback
DEFAULT_ATR_WINDOW = 14        # ATR computation window
DEFAULT_ATR_MULTIPLIER = 2.0   # Stop = N × ATR below entry (Turtles used 2N)
DEFAULT_RISK_PER_TRADE = 0.02  # 2% of bankroll per Unit

State = Literal["long", "short", "flat"]
Break = Literal["breakout_up", "breakout_down", "none"]


# ─── 1. Regime filter ────────────────────────────────────────────────

def sma(values: list[float], window: int) -> float | None:
    """Simple moving average over the last ``window`` values."""
    if window < 1 or len(values) < window:
        return None
    tail = values[-window:]
    return sum(tail) / window


def classify_regime(prices: list[float], window: int = DEFAULT_REGIME_WINDOW) -> State:
    """Direction of the trend filter.

    Returns "long" if today's close > SMA(window), "short" if below,
    "flat" only when we lack enough history.
    """
    if len(prices) <= window:
        return "flat"
    today = prices[-1]
    avg = sma(prices, window)
    if avg is None:
        return "flat"
    if today > avg:
        return "long"
    if today < avg:
        return "short"
    return "flat"


# ─── 2. Donchian breakout ────────────────────────────────────────────

def donchian_break(prices: list[float], window: int = DEFAULT_BREAKOUT_WINDOW) -> Break:
    """Has today's close broken out of the prior ``window``-bar range?

    The Turtles compared today's close to the high/low of the PRIOR N
    bars (excluding today). A 40-bar System-2 break is the canonical
    longer-term version; 20 is the shorter-term System-1.
    """
    if len(prices) <= window:
        return "none"
    today = prices[-1]
    prior = prices[-(window + 1):-1]
    if not prior:
        return "none"
    if today > max(prior):
        return "breakout_up"
    if today < min(prior):
        return "breakout_down"
    return "none"


# ─── 3. ATR (volatility-adaptive stop) ───────────────────────────────

def _true_range(high: float, low: float, prev_close: float) -> float:
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    )


def atr(highs: list[float], lows: list[float], closes: list[float],
        window: int = DEFAULT_ATR_WINDOW) -> float | None:
    """Average True Range over the last ``window`` bars.

    Uses Wilder's TR (max of three classic ranges). Requires
    len(closes) >= window + 1 because TR needs the prior close.
    """
    n = min(len(highs), len(lows), len(closes))
    if n < window + 1:
        return None
    trs = []
    for i in range(n - window, n):
        if i <= 0:
            continue
        trs.append(_true_range(highs[i], lows[i], closes[i - 1]))
    if not trs:
        return None
    return sum(trs) / len(trs)


# ─── 4. Position sizing (Unit) ───────────────────────────────────────

def unit_size(
    bankroll: float, atr_value: float, price: float,
    atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    max_position_pct: float = 0.30,
) -> dict:
    """Compute the Turtle Unit size in shares.

    Risk per share = atr_multiplier × ATR (price distance to stop).
    Unit_dollars  = bankroll × risk_per_trade / risk_per_share
    Shares        = floor(unit_dollars / share_price), bounded by the
                    position-size cap from circuit_breaker settings.

    Returns a dict with shares + the intermediate math so the caller can
    log it.
    """
    if atr_value <= 0 or price <= 0:
        return {
            "shares": 0, "reason": "invalid_inputs",
            "risk_per_share": 0, "unit_dollars": 0,
        }
    risk_per_share = atr_multiplier * atr_value
    risk_budget = bankroll * risk_per_trade
    unit_dollars = risk_budget / risk_per_share * price  # convert to $ size
    cap_dollars = bankroll * max_position_pct
    sized_dollars = min(unit_dollars, cap_dollars)
    shares = int(sized_dollars / price)
    return {
        "shares": max(0, shares),
        "risk_per_share": round(risk_per_share, 4),
        "unit_dollars": round(unit_dollars, 2),
        "cap_dollars": round(cap_dollars, 2),
        "sized_dollars": round(sized_dollars, 2),
        "reason": "ok" if shares > 0 else "size_below_one_share",
    }


# ─── 5. End-to-end signal ────────────────────────────────────────────

def turtle_signal(
    ticker: str,
    lookback_days: int = 365,
    bankroll: float = 1500.0,
    regime_window: int = DEFAULT_REGIME_WINDOW,
    breakout_window: int = DEFAULT_BREAKOUT_WINDOW,
    atr_window: int = DEFAULT_ATR_WINDOW,
    atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
) -> dict:
    """End-to-end signal for one ticker.

    Output dict:
      ticker, regime, break, fired (bool), entry_price, stop_price,
      atr, n_bars, position_sizing (dict), reasoning (str)

    The signal "fires" only when:
      regime == "long" AND break == "breakout_up"
    Anything else → fired=False with a reason.
    """
    from lib.markov_regime import _fetch_alpaca_daily_closes

    # We need OHLC, not just close. The Alpaca raw API returns h/l/o/c —
    # extend the helper here to grab them all in one call.
    closes = _fetch_alpaca_daily_closes(ticker, lookback_days=lookback_days)
    if len(closes) < max(regime_window, breakout_window) + atr_window + 5:
        return {
            "ticker": ticker, "fired": False,
            "reason": f"insufficient_history({len(closes)} bars)",
            "n_bars": len(closes),
        }

    # Need OHLC for ATR. Re-fetch with bar metadata.
    highs, lows, full_closes = _fetch_alpaca_daily_ohlc(ticker, lookback_days)
    if len(full_closes) != len(closes):
        # Fall back to using closes for ATR (degraded mode)
        highs = full_closes if full_closes else closes
        lows = full_closes if full_closes else closes
        full_closes = full_closes if full_closes else closes

    regime = classify_regime(full_closes, window=regime_window)
    brk = donchian_break(full_closes, window=breakout_window)
    atr_value = atr(highs, lows, full_closes, window=atr_window)

    today_price = full_closes[-1]
    fired = (regime == "long" and brk == "breakout_up")
    reason = "fired" if fired else f"regime={regime}, break={brk}"

    stop_price = None
    sizing = None
    if fired and atr_value is not None:
        stop_price = round(today_price - atr_multiplier * atr_value, 2)
        sizing = unit_size(
            bankroll=bankroll, atr_value=atr_value, price=today_price,
            atr_multiplier=atr_multiplier,
            risk_per_trade=risk_per_trade,
        )

    return {
        "ticker": ticker,
        "n_bars": len(full_closes),
        "today_price": round(today_price, 2),
        "regime": regime,
        "sma200": (round(sma(full_closes, regime_window), 2)
                    if sma(full_closes, regime_window) else None),
        "donchian_break": brk,
        "donchian_high_prior": (round(max(full_closes[-(breakout_window + 1):-1]), 2)
                                 if len(full_closes) > breakout_window else None),
        "atr": round(atr_value, 4) if atr_value is not None else None,
        "atr_window": atr_window,
        "fired": fired,
        "stop_price": stop_price,
        "stop_distance_pct": (round((today_price - stop_price) / today_price, 4)
                              if stop_price else None),
        "sizing": sizing,
        "reason": reason,
        "asof": datetime.now(timezone.utc).isoformat(),
    }


def _fetch_alpaca_daily_ohlc(
    ticker: str, lookback_days: int = 365,
) -> tuple[list[float], list[float], list[float]]:
    """Variant of _fetch_alpaca_daily_closes that returns (highs, lows, closes)
    so we can compute ATR. Falls back to empty lists on failure.
    """
    import os
    import requests
    from datetime import datetime, timedelta, timezone
    api_key = (
        os.environ.get("ALPACA_API_KEY")
        or os.environ.get("APCA_API_KEY_ID")
    )
    secret = (
        os.environ.get("ALPACA_SECRET_KEY")
        or os.environ.get("APCA_API_SECRET_KEY")
    )
    if not api_key or not secret:
        # Re-read from .env the same way markov_regime does
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for raw in env_path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k in ("ALPACA_API_KEY", "APCA_API_KEY_ID") and not api_key:
                    api_key = v
                elif k in ("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY") and not secret:
                    secret = v
    if not api_key or not secret:
        return [], [], []
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=lookback_days)
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    highs, lows, closes = [], [], []
    page_token = None
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}
    while True:
        params = {
            "timeframe": "1Day",
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 10000,
            "adjustment": "all",
        }
        if page_token:
            params["page_token"] = page_token
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return highs, lows, closes
        bars = data.get("bars", []) or []
        for b in bars:
            try:
                highs.append(float(b["h"]))
                lows.append(float(b["l"]))
                closes.append(float(b["c"]))
            except (KeyError, TypeError, ValueError):
                continue
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return highs, lows, closes


# ─── 6. Universe scan ────────────────────────────────────────────────

def universe_scan(
    tickers: list[str],
    bankroll: float = 1500.0,
    **kwargs,
) -> list[dict]:
    """Run turtle_signal across a list of tickers and return the ones
    that fired, sorted by stop-distance-percent (smaller = tighter
    R:R, less risk per share)."""
    fired = []
    for t in tickers:
        try:
            s = turtle_signal(t, bankroll=bankroll, **kwargs)
            if s.get("fired"):
                fired.append(s)
        except Exception as e:
            fired.append({
                "ticker": t, "fired": False,
                "reason": f"error: {str(e)[:120]}",
            })
    # Sort fired-signals by smallest stop_distance_pct (best R:R)
    fired_only = [f for f in fired if f.get("fired")]
    fired_only.sort(key=lambda f: f.get("stop_distance_pct") or 1.0)
    return fired_only


# ─── 7. Renderer ─────────────────────────────────────────────────────

def render_summary(result: dict) -> str:
    if "error" in result:
        return f"{result.get('ticker', '?')}: ERROR — {result['error']}"
    lines = []
    lines.append("=" * 70)
    lines.append(f"TURTLE SIGNAL — {result['ticker']}")
    lines.append("=" * 70)
    lines.append(
        f"Bars used: {result['n_bars']}  Today's close: "
        f"${result['today_price']:.2f}"
    )
    lines.append("")
    sma_v = result.get("sma200")
    sma_str = f"${sma_v:.2f}" if sma_v is not None else "n/a"
    lines.append(f"REGIME ({DEFAULT_REGIME_WINDOW}d SMA):  "
                 f"{result['regime'].upper()}  "
                 f"(price ${result['today_price']:.2f} vs SMA {sma_str})")
    lines.append("")
    lines.append(f"BREAKOUT ({DEFAULT_BREAKOUT_WINDOW}-bar Donchian):  "
                 f"{result['donchian_break']}")
    if result.get("donchian_high_prior") is not None:
        lines.append(f"  Prior {DEFAULT_BREAKOUT_WINDOW}-bar high: "
                     f"${result['donchian_high_prior']:.2f}")
    lines.append("")
    atr_v = result.get("atr")
    if atr_v is not None:
        lines.append(f"ATR ({result['atr_window']}d): ${atr_v:.4f}")
    lines.append("")
    if result.get("fired"):
        sizing = result.get("sizing", {})
        lines.append(f"SIGNAL FIRED ✓  long entry at ${result['today_price']:.2f}")
        lines.append(f"  Stop loss:  ${result['stop_price']:.2f}  "
                     f"({result['stop_distance_pct']:+.2%} from entry)")
        lines.append(f"  Sizing:     {sizing.get('shares', 0)} shares "
                     f"(${sizing.get('sized_dollars', 0):.2f} notional)")
        lines.append(f"  Risk/share: ${sizing.get('risk_per_share', 0):.4f}  "
                     f"Cap:        ${sizing.get('cap_dollars', 0):.2f}")
    else:
        lines.append(f"NO SIGNAL — {result['reason']}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "classify_regime", "donchian_break", "atr", "unit_size",
    "turtle_signal", "universe_scan", "render_summary",
    "DEFAULT_REGIME_WINDOW", "DEFAULT_BREAKOUT_WINDOW",
    "DEFAULT_ATR_WINDOW", "DEFAULT_ATR_MULTIPLIER",
    "DEFAULT_RISK_PER_TRADE",
]


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else 1500.0
    print(render_summary(turtle_signal(ticker, bankroll=bankroll)))
