"""Dip-buy / mean-reversion entry sleeve — the complement to Turtle.

THESIS
  Turtle is a BREAKOUT system: buys at NEW HIGHS (close > 40-bar high)
  expecting trend continuation. It wins in the 30-40% of months where
  the market trends; it underperforms in the 60-70% where price chops
  or pulls back to its mean.

  Dip-buy is the OPPOSITE: buys at OVERSOLD pullbacks WITHIN an
  established uptrend. The setup is:
    - long regime intact (close > 200-MA, same as Turtle)
    - BUT pulled back to a meaningful oversold reading (RSI < 30
      or close > 200-MA but ≤ 1.02× 200-MA — "kissing the mean")
    - Volume confirming the reversal (today's vol >= avg)

  Why both? They win in different regimes. Turtle compounds in bull
  rallies; dip-buy harvests during the more common chop and minor
  pullbacks. Running them in PARALLEL captures more of the year.

  Critically: dip-buy DOES NOT activate in bear regimes (close <
  200-MA). A "dip" below the 200-MA isn't a dip — it's a downtrend.
  We're not catching falling knives. Long-regime filter is mandatory.

USAGE
    from lib.dipbuy_signal import classify_dipbuy_setup, dipbuy_signal

    sig = dipbuy_signal(closes, highs, lows, volumes)
    if sig["fire"]:
        # entry: today's close
        # stop: sig["stop"] (below recent swing low or ATR-based)
        # target: sig["target"] (back to recent swing high or 200-MA reclaim)

SIGNAL SHAPE
  Returns:
    {
      "fire": bool,           # True if dip-buy setup is valid
      "kind": str,            # "rsi_oversold" | "ma_touch" | "swing_low"
      "regime": str,          # "long" | "short" | "flat"
      "rsi": float,           # current RSI(14)
      "distance_to_200ma": float,  # % above/below 200-MA
      "stop_pct": float,      # recommended stop loss %
      "target_pct": float,    # recommended take-profit %
      "reason": str,          # human-readable why
    }

COMPLEMENTARITY WITH TURTLE
  Turtle requires:    close > 40-bar high (BREAKOUT UP)
  Dip-buy requires:   close <= 40-bar high AND RSI <= 30
  The two are MUTUALLY EXCLUSIVE — a single bar can never satisfy both.
  No double-counting risk.
"""
from __future__ import annotations

from typing import Literal


State = Literal["long", "short", "flat"]
DipKind = Literal["rsi_oversold", "ma_touch", "swing_low", "none"]


# ─── Indicators (pure-python, no numpy/pandas) ────────────────────────

def _sma(values: list[float], window: int) -> float | None:
    if len(values) < window or window < 1:
        return None
    return sum(values[-window:]) / window


def _rsi(closes: list[float], window: int = 14) -> float | None:
    """Wilder's RSI. Returns 0-100 or None if insufficient data."""
    if len(closes) < window + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    if len(gains) < window:
        return None
    # Wilder's smoothing
    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window
    for i in range(window, len(gains)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(highs: list[float], lows: list[float], closes: list[float],
         window: int = 14) -> float | None:
    """Average True Range over the last ``window`` bars."""
    if min(len(highs), len(lows), len(closes)) < window + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < window:
        return None
    # Simple average of last ``window`` TRs
    return sum(trs[-window:]) / window


# ─── Regime classification (mirrors Turtle's) ─────────────────────────

def _regime(closes: list[float], window: int = 200) -> State:
    if len(closes) <= window:
        return "flat"
    avg = _sma(closes, window)
    if avg is None:
        return "flat"
    today = closes[-1]
    if today > avg:
        return "long"
    if today < avg:
        return "short"
    return "flat"


# ─── Dip-buy setup classification ─────────────────────────────────────

def classify_dipbuy_setup(
    closes: list[float],
    rsi_threshold: float = 30.0,
    ma_touch_pct: float = 0.02,
    regime_window: int = 200,
) -> DipKind:
    """Identify which (if any) dip-buy setup is present in the latest bar.

    "rsi_oversold": RSI(14) <= rsi_threshold AND long regime
    "ma_touch":      close within ma_touch_pct of 200-MA from ABOVE
                     (e.g., 200MA $100, close $101 = 1% above = TOUCH)
    "swing_low":     today's close is the lowest in the last 5 bars
                     AND long regime AND RSI(14) <= rsi_threshold + 10
                     (a secondary safety net)

    Returns "none" if no dip-buy setup applies.
    """
    if len(closes) < regime_window + 1:
        return "none"
    state = _regime(closes, regime_window)
    if state != "long":
        return "none"  # never catch falling knives

    rsi = _rsi(closes)
    sma200 = _sma(closes, regime_window)
    today = closes[-1]

    # rsi_oversold (the strongest setup)
    if rsi is not None and rsi <= rsi_threshold:
        return "rsi_oversold"

    # ma_touch (pullback to the trend line)
    if sma200 is not None:
        distance = (today - sma200) / sma200
        if 0 < distance <= ma_touch_pct:
            return "ma_touch"

    # swing_low (5-bar low while still in long regime and softer RSI)
    if len(closes) >= 5:
        recent_low = min(closes[-5:])
        if today == recent_low and rsi is not None and rsi <= rsi_threshold + 10:
            return "swing_low"

    return "none"


# ─── Stop / target levels (ATR-based) ─────────────────────────────────

def _stop_pct(highs: list[float], lows: list[float], closes: list[float],
              atr_mult: float = 2.0, min_stop_pct: float = 0.03,
              max_stop_pct: float = 0.07) -> float:
    """ATR-based stop loss as a percentage of current price.

    Defaults: 2 × ATR(14), clamped to [3%, 7%].

    Wider than Turtle's 3.5% because mean-reversion entries are inside
    pullbacks — we need room to weather the rest of the dip before the
    bounce. Capped at 7% to avoid runaway risk on high-vol names.
    """
    atr = _atr(highs, lows, closes)
    if atr is None or closes[-1] == 0:
        return min_stop_pct
    pct = (atr * atr_mult) / closes[-1]
    return max(min_stop_pct, min(pct, max_stop_pct))


# ─── Public entry signal ──────────────────────────────────────────────

def dipbuy_signal(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float] | None = None,
    rsi_threshold: float = 30.0,
    ma_touch_pct: float = 0.02,
    regime_window: int = 200,
    require_volume_confirm: bool = True,
    volume_window: int = 20,
    volume_multiplier: float = 1.0,
) -> dict:
    """Full dip-buy entry signal — classify setup, compute regime,
    apply volume confirmation, return entry/stop/target levels.

    The signal is INTENTIONALLY ORTHOGONAL to Turtle (which fires on
    breakouts up). A bar that satisfies dip-buy can never satisfy
    Turtle — close <= 40-bar high is implied by an oversold pullback.
    """
    state = _regime(closes, regime_window)
    kind = classify_dipbuy_setup(
        closes,
        rsi_threshold=rsi_threshold,
        ma_touch_pct=ma_touch_pct,
        regime_window=regime_window,
    )
    rsi = _rsi(closes) or 50.0
    sma200 = _sma(closes, regime_window)
    today = closes[-1]
    distance = ((today - sma200) / sma200) if sma200 else 0.0

    if kind == "none":
        return {
            "fire": False, "kind": "none", "regime": state,
            "rsi": rsi, "distance_to_200ma": distance,
            "stop_pct": 0.0, "target_pct": 0.0,
            "reason": f"no_dipbuy_setup (regime={state}, RSI={rsi:.1f})",
        }

    # Volume confirmation — same logic as Turtle's volume gate
    if require_volume_confirm and volumes and len(volumes) > volume_window:
        today_vol = volumes[-1]
        avg_vol = sum(volumes[-(volume_window + 1):-1]) / volume_window
        if avg_vol > 0 and today_vol < avg_vol * volume_multiplier:
            return {
                "fire": False, "kind": kind, "regime": state,
                "rsi": rsi, "distance_to_200ma": distance,
                "stop_pct": 0.0, "target_pct": 0.0,
                "reason": (
                    f"{kind}_but_low_volume "
                    f"(today {int(today_vol)} < {volume_multiplier:.1f}x "
                    f"avg {int(avg_vol)})"
                ),
            }

    # Stop: ATR-based (or default 4% if ATR unavailable)
    stop_pct = _stop_pct(highs, lows, closes)
    # Target: 1.5x the stop (positive expectancy R:R)
    target_pct = stop_pct * 1.5

    return {
        "fire": True, "kind": kind, "regime": state,
        "rsi": rsi, "distance_to_200ma": distance,
        "stop_pct": stop_pct, "target_pct": target_pct,
        "reason": (
            f"{kind} in {state} regime, RSI={rsi:.1f}, "
            f"distance={distance:+.2%}, stop={stop_pct:.2%}, "
            f"target={target_pct:.2%}"
        ),
    }


__all__ = [
    "dipbuy_signal",
    "classify_dipbuy_setup",
]
