"""
Candlestick Pattern Detection — The Candlestick Trading Bible

Detects 10 key patterns:
  Bullish: engulfing, morning star, hammer, dragonfly doji, bullish harami
  Bearish: engulfing, evening star, shooting star, gravestone doji, bearish harami
  Neutral: doji, tweezers (classified by context)

Source: The Candlestick Trading Bible, chapters on each pattern
Key principle: A pattern is only valid at a KEY LEVEL (support/resistance)
in alignment with the TREND.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class CandlestickSignal:
    """A detected candlestick pattern."""
    pattern: str
    direction: Literal["bullish", "bearish", "neutral"]
    strength: int          # 1-3 (1=weak, 2=moderate, 3=strong)
    bar_index: int         # Index in the dataframe
    date: str
    description: str


def _body(row) -> float:
    return abs(row["close"] - row["open"])


def _upper_wick(row) -> float:
    return row["high"] - max(row["open"], row["close"])


def _lower_wick(row) -> float:
    return min(row["open"], row["close"]) - row["low"]


def _is_bullish(row) -> bool:
    return row["close"] > row["open"]


def _is_bearish(row) -> bool:
    return row["close"] < row["open"]


def _range(row) -> float:
    return row["high"] - row["low"]


def _avg_body(df: pd.DataFrame, lookback: int = 14) -> float:
    """Average body size over lookback period."""
    bodies = (df["close"] - df["open"]).abs()
    return bodies.rolling(lookback).mean().iloc[-1]


# ============================================================
# INDIVIDUAL PATTERN DETECTORS
# ============================================================


def detect_engulfing(df: pd.DataFrame, i: int) -> CandlestickSignal | None:
    """
    Bullish engulfing: bearish candle followed by larger bullish candle
    that completely engulfs the previous body.
    Bearish engulfing: bullish candle followed by larger bearish candle.
    """
    if i < 1:
        return None

    prev, curr = df.iloc[i - 1], df.iloc[i]

    # Bullish engulfing
    if (_is_bearish(prev) and _is_bullish(curr)
            and curr["open"] <= prev["close"]
            and curr["close"] >= prev["open"]
            and _body(curr) > _body(prev)):
        return CandlestickSignal(
            pattern="bullish_engulfing", direction="bullish", strength=3,
            bar_index=i, date=str(df.index[i]),
            description="Bullish engulfing: current candle fully wraps previous bearish body"
        )

    # Bearish engulfing
    if (_is_bullish(prev) and _is_bearish(curr)
            and curr["open"] >= prev["close"]
            and curr["close"] <= prev["open"]
            and _body(curr) > _body(prev)):
        return CandlestickSignal(
            pattern="bearish_engulfing", direction="bearish", strength=3,
            bar_index=i, date=str(df.index[i]),
            description="Bearish engulfing: current candle fully wraps previous bullish body"
        )

    return None


def detect_doji(df: pd.DataFrame, i: int) -> CandlestickSignal | None:
    """
    Doji: open and close are nearly identical (body < 10% of range).
    Indicates indecision.
    """
    row = df.iloc[i]
    rng = _range(row)
    if rng == 0:
        return None

    if _body(row) / rng < 0.10:
        return CandlestickSignal(
            pattern="doji", direction="neutral", strength=1,
            bar_index=i, date=str(df.index[i]),
            description="Doji: open ≈ close, market indecision"
        )
    return None


def detect_dragonfly_doji(df: pd.DataFrame, i: int) -> CandlestickSignal | None:
    """
    Dragonfly doji: long lower wick, no/tiny upper wick, tiny body at top.
    Bullish reversal signal at support.
    """
    row = df.iloc[i]
    rng = _range(row)
    if rng == 0:
        return None

    body_pct = _body(row) / rng
    lower_pct = _lower_wick(row) / rng
    upper_pct = _upper_wick(row) / rng

    if body_pct < 0.10 and lower_pct > 0.65 and upper_pct < 0.10:
        return CandlestickSignal(
            pattern="dragonfly_doji", direction="bullish", strength=2,
            bar_index=i, date=str(df.index[i]),
            description="Dragonfly doji: long lower shadow, buyers rejected sellers"
        )
    return None


def detect_gravestone_doji(df: pd.DataFrame, i: int) -> CandlestickSignal | None:
    """
    Gravestone doji: long upper wick, no/tiny lower wick, tiny body at bottom.
    Bearish reversal signal at resistance.
    """
    row = df.iloc[i]
    rng = _range(row)
    if rng == 0:
        return None

    body_pct = _body(row) / rng
    upper_pct = _upper_wick(row) / rng
    lower_pct = _lower_wick(row) / rng

    if body_pct < 0.10 and upper_pct > 0.65 and lower_pct < 0.10:
        return CandlestickSignal(
            pattern="gravestone_doji", direction="bearish", strength=2,
            bar_index=i, date=str(df.index[i]),
            description="Gravestone doji: long upper shadow, sellers rejected buyers"
        )
    return None


def detect_morning_star(df: pd.DataFrame, i: int) -> CandlestickSignal | None:
    """
    Morning star (3-bar pattern):
    1. Large bearish candle
    2. Small body (gap down preferred) — the star
    3. Large bullish candle closing into bar 1's body
    """
    if i < 2:
        return None

    bar1, bar2, bar3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    avg = _avg_body(df.iloc[:i], 14) if i >= 14 else _body(bar1)

    if (_is_bearish(bar1) and _body(bar1) > avg * 0.8
            and _body(bar2) < avg * 0.5
            and _is_bullish(bar3) and _body(bar3) > avg * 0.8
            and bar3["close"] > (bar1["open"] + bar1["close"]) / 2):
        return CandlestickSignal(
            pattern="morning_star", direction="bullish", strength=3,
            bar_index=i, date=str(df.index[i]),
            description="Morning star: 3-bar bullish reversal (bearish→small→bullish)"
        )
    return None


def detect_evening_star(df: pd.DataFrame, i: int) -> CandlestickSignal | None:
    """
    Evening star (3-bar pattern):
    1. Large bullish candle
    2. Small body (gap up preferred)
    3. Large bearish candle closing into bar 1's body
    """
    if i < 2:
        return None

    bar1, bar2, bar3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    avg = _avg_body(df.iloc[:i], 14) if i >= 14 else _body(bar1)

    if (_is_bullish(bar1) and _body(bar1) > avg * 0.8
            and _body(bar2) < avg * 0.5
            and _is_bearish(bar3) and _body(bar3) > avg * 0.8
            and bar3["close"] < (bar1["open"] + bar1["close"]) / 2):
        return CandlestickSignal(
            pattern="evening_star", direction="bearish", strength=3,
            bar_index=i, date=str(df.index[i]),
            description="Evening star: 3-bar bearish reversal (bullish→small→bearish)"
        )
    return None


def detect_hammer(df: pd.DataFrame, i: int) -> CandlestickSignal | None:
    """
    Hammer: small body at top, long lower wick (≥2x body), tiny upper wick.
    Bullish reversal at support. Body color less important but bullish body stronger.
    """
    row = df.iloc[i]
    body = _body(row)
    lower = _lower_wick(row)
    upper = _upper_wick(row)

    if body == 0:
        return None

    if lower >= body * 2 and upper <= body * 0.5:
        strength = 3 if _is_bullish(row) else 2
        return CandlestickSignal(
            pattern="hammer", direction="bullish", strength=strength,
            bar_index=i, date=str(df.index[i]),
            description="Hammer: long lower wick shows buyer rejection of lower prices"
        )
    return None


def detect_shooting_star(df: pd.DataFrame, i: int) -> CandlestickSignal | None:
    """
    Shooting star: small body at bottom, long upper wick (≥2x body), tiny lower wick.
    Bearish reversal at resistance.
    """
    row = df.iloc[i]
    body = _body(row)
    upper = _upper_wick(row)
    lower = _lower_wick(row)

    if body == 0:
        return None

    if upper >= body * 2 and lower <= body * 0.5:
        strength = 3 if _is_bearish(row) else 2
        return CandlestickSignal(
            pattern="shooting_star", direction="bearish", strength=strength,
            bar_index=i, date=str(df.index[i]),
            description="Shooting star: long upper wick shows seller rejection of higher prices"
        )
    return None


def detect_harami(df: pd.DataFrame, i: int) -> CandlestickSignal | None:
    """
    Harami: small candle body contained entirely within previous large candle's body.
    Bullish harami: bearish candle → small bullish candle inside
    Bearish harami: bullish candle → small bearish candle inside
    """
    if i < 1:
        return None

    prev, curr = df.iloc[i - 1], df.iloc[i]

    prev_top = max(prev["open"], prev["close"])
    prev_bot = min(prev["open"], prev["close"])
    curr_top = max(curr["open"], curr["close"])
    curr_bot = min(curr["open"], curr["close"])

    if curr_top <= prev_top and curr_bot >= prev_bot and _body(curr) < _body(prev) * 0.5:
        if _is_bearish(prev) and _is_bullish(curr):
            return CandlestickSignal(
                pattern="bullish_harami", direction="bullish", strength=2,
                bar_index=i, date=str(df.index[i]),
                description="Bullish harami: small bullish candle inside previous bearish body"
            )
        elif _is_bullish(prev) and _is_bearish(curr):
            return CandlestickSignal(
                pattern="bearish_harami", direction="bearish", strength=2,
                bar_index=i, date=str(df.index[i]),
                description="Bearish harami: small bearish candle inside previous bullish body"
            )
    return None


def detect_tweezers(df: pd.DataFrame, i: int) -> CandlestickSignal | None:
    """
    Tweezers tops: two candles with nearly identical highs at resistance.
    Tweezers bottoms: two candles with nearly identical lows at support.
    """
    if i < 1:
        return None

    prev, curr = df.iloc[i - 1], df.iloc[i]
    rng = max(_range(prev), _range(curr))
    if rng == 0:
        return None

    tolerance = rng * 0.05  # Highs/lows within 5% of range

    # Tweezers bottom
    if abs(prev["low"] - curr["low"]) <= tolerance and _is_bearish(prev) and _is_bullish(curr):
        return CandlestickSignal(
            pattern="tweezers_bottom", direction="bullish", strength=2,
            bar_index=i, date=str(df.index[i]),
            description="Tweezers bottom: matching lows show strong support rejection"
        )

    # Tweezers top
    if abs(prev["high"] - curr["high"]) <= tolerance and _is_bullish(prev) and _is_bearish(curr):
        return CandlestickSignal(
            pattern="tweezers_top", direction="bearish", strength=2,
            bar_index=i, date=str(df.index[i]),
            description="Tweezers top: matching highs show strong resistance rejection"
        )

    return None


# ============================================================
# MAIN SCANNER
# ============================================================

ALL_DETECTORS = [
    detect_engulfing,
    detect_doji,
    detect_dragonfly_doji,
    detect_gravestone_doji,
    detect_morning_star,
    detect_evening_star,
    detect_hammer,
    detect_shooting_star,
    detect_harami,
    detect_tweezers,
]


def scan_patterns(
    df: pd.DataFrame,
    lookback: int = 5,
    direction_filter: str | None = None,
) -> list[CandlestickSignal]:
    """
    Scan the last N bars for candlestick patterns.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        lookback: How many recent bars to scan
        direction_filter: "bullish", "bearish", or None for all

    Returns:
        List of detected signals sorted by bar_index (most recent first)
    """
    signals = []
    start = max(0, len(df) - lookback)

    for i in range(start, len(df)):
        for detector in ALL_DETECTORS:
            signal = detector(df, i)
            if signal is not None:
                if direction_filter is None or signal.direction == direction_filter:
                    signals.append(signal)

    # Most recent first
    signals.sort(key=lambda s: s.bar_index, reverse=True)
    return signals


def get_latest_signal(
    df: pd.DataFrame,
    direction: str,
    allowed_patterns: list[str] | None = None,
) -> CandlestickSignal | None:
    """
    Get the most recent signal matching direction and pattern filter.
    Used by order gate to confirm trade entries.
    """
    signals = scan_patterns(df, lookback=3, direction_filter=direction)

    if allowed_patterns:
        signals = [s for s in signals if s.pattern in allowed_patterns]

    return signals[0] if signals else None
