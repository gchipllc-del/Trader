"""
Trend Identification — Candlestick Trading Bible Framework

Three questions before any trade:
1. What is the market doing? (trending up, down, ranging, choppy)
2. What are the key levels? (handled by zones.py)
3. What is the best signal? (handled by candlestick.py)

This module answers question #1 using multiple timeframes.
Source: Candlestick Trading Bible Ch. 51-70
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class TrendAnalysis:
    """Result of trend analysis for a single timeframe."""
    timeframe: str
    direction: Literal["uptrend", "downtrend", "ranging", "choppy"]
    strength: int            # 0-3
    description: str
    higher_highs: bool
    higher_lows: bool
    lower_highs: bool
    lower_lows: bool


def _detect_swing_sequence(df: pd.DataFrame, window: int = 10) -> dict:
    """
    Detect if price is making higher highs/lows (uptrend)
    or lower highs/lows (downtrend).
    """
    highs = []
    lows = []

    for i in range(window, len(df) - window):
        if df["high"].iloc[i] == df["high"].iloc[i - window : i + window + 1].max():
            highs.append(df["high"].iloc[i])
        if df["low"].iloc[i] == df["low"].iloc[i - window : i + window + 1].min():
            lows.append(df["low"].iloc[i])

    result = {
        "higher_highs": False,
        "higher_lows": False,
        "lower_highs": False,
        "lower_lows": False,
    }

    if len(highs) >= 2:
        result["higher_highs"] = highs[-1] > highs[-2]
        result["lower_highs"] = highs[-1] < highs[-2]

    if len(lows) >= 2:
        result["higher_lows"] = lows[-1] > lows[-2]
        result["lower_lows"] = lows[-1] < lows[-2]

    return result


def _sma_trend(df: pd.DataFrame) -> str:
    """Quick trend check using 20 and 50 SMA."""
    if len(df) < 50:
        return "insufficient_data"

    sma20 = df["close"].rolling(20).mean().iloc[-1]
    sma50 = df["close"].rolling(50).mean().iloc[-1]
    price = df["close"].iloc[-1]

    if price > sma20 > sma50:
        return "uptrend"
    elif price < sma20 < sma50:
        return "downtrend"
    else:
        return "mixed"


def _adr_choppy_check(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    Check if market is choppy by comparing average daily range
    to net movement. High ADR but low net = choppy.
    """
    if len(df) < lookback:
        return False

    recent = df.iloc[-lookback:]
    adr = (recent["high"] - recent["low"]).mean()
    net_move = abs(recent["close"].iloc[-1] - recent["close"].iloc[0])

    # If net movement is less than 1 ADR over the period, it's choppy
    return net_move < adr


def analyze_trend(df: pd.DataFrame, timeframe: str = "daily") -> TrendAnalysis:
    """
    Analyze the trend for a given timeframe's data.

    Args:
        df: OHLCV DataFrame
        timeframe: Label ("weekly", "daily", "4hour")

    Returns:
        TrendAnalysis with direction, strength, and structure details
    """
    if len(df) < 20:
        return TrendAnalysis(
            timeframe=timeframe, direction="choppy", strength=0,
            description="Insufficient data for trend analysis",
            higher_highs=False, higher_lows=False,
            lower_highs=False, lower_lows=False,
        )

    swings = _detect_swing_sequence(df)
    sma = _sma_trend(df)
    is_choppy = _adr_choppy_check(df)

    # Classify
    if is_choppy:
        direction = "choppy"
        strength = 0
        desc = "Choppy: high range but low net movement — stay away"

    elif swings["higher_highs"] and swings["higher_lows"]:
        direction = "uptrend"
        strength = 3 if sma == "uptrend" else 2
        desc = "Uptrend: higher highs and higher lows confirmed"
        if sma == "uptrend":
            desc += " + price above rising SMAs"

    elif swings["lower_highs"] and swings["lower_lows"]:
        direction = "downtrend"
        strength = 3 if sma == "downtrend" else 2
        desc = "Downtrend: lower highs and lower lows confirmed"
        if sma == "downtrend":
            desc += " + price below falling SMAs"

    else:
        direction = "ranging"
        strength = 1
        desc = "Ranging: no clear trend — price moving between boundaries"

    return TrendAnalysis(
        timeframe=timeframe,
        direction=direction,
        strength=strength,
        description=desc,
        higher_highs=swings["higher_highs"],
        higher_lows=swings["higher_lows"],
        lower_highs=swings["lower_highs"],
        lower_lows=swings["lower_lows"],
    )


def multi_timeframe_analysis(
    weekly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None = None,
) -> dict:
    """
    Top-down analysis across timeframes.
    Weekly = trend direction, Daily = levels, 4H = entry timing.

    Returns dict with each timeframe's analysis and overall alignment score.
    """
    weekly = analyze_trend(weekly_df, "weekly")
    daily = analyze_trend(daily_df, "daily")

    results = {
        "weekly": weekly,
        "daily": daily,
        "intraday": None,
        "alignment_score": 0,  # 0-3
        "tradeable": False,
    }

    if intraday_df is not None and len(intraday_df) > 20:
        intraday = analyze_trend(intraday_df, "4hour")
        results["intraday"] = intraday

    # Score alignment
    score = 0
    directions = [weekly.direction, daily.direction]
    if results["intraday"]:
        directions.append(results["intraday"].direction)

    # All agree on uptrend or downtrend = max score
    if all(d == "uptrend" for d in directions):
        score = 3
    elif all(d == "downtrend" for d in directions):
        score = 3
    # Weekly + daily agree
    elif weekly.direction == daily.direction and weekly.direction in ("uptrend", "downtrend"):
        score = 2
    # At least weekly has a clear trend
    elif weekly.direction in ("uptrend", "downtrend"):
        score = 1

    results["alignment_score"] = score
    # Tradeable if score ≥2 and no timeframe is "choppy"
    results["tradeable"] = score >= 2 and "choppy" not in directions

    return results
