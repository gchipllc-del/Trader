"""
Momentum Indicators — RSI, MACD, Volume Surge, Rate of Change.

Used by the stock engine to identify high-momentum entries
that don't require traditional candlestick confirmation.
Fast-moving stocks with strong momentum can generate signals
on their own when combined with trend alignment.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MomentumSignal:
    """Aggregated momentum assessment."""
    rsi: float
    rsi_signal: str          # "oversold", "neutral", "overbought"
    macd_histogram: float
    macd_cross: str          # "bullish_cross", "bearish_cross", "none"
    volume_surge: float      # ratio vs 20-day average
    volume_signal: str       # "surge", "normal", "dry"
    roc_5d: float            # 5-day rate of change
    roc_20d: float           # 20-day rate of change
    momentum_score: int      # 0-4 composite momentum score


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_volume_ratio(volumes: pd.Series, period: int = 20) -> pd.Series:
    """Current volume / average volume over period."""
    avg = volumes.rolling(period).mean()
    return (volumes / avg.replace(0, np.nan)).fillna(1.0)


def compute_rate_of_change(prices: pd.Series, period: int = 5) -> pd.Series:
    """Rate of change over N days."""
    shifted = prices.shift(period)
    return ((prices - shifted) / shifted.replace(0, np.nan)).fillna(0)


def analyze_momentum(daily_df: pd.DataFrame) -> MomentumSignal | None:
    """
    Full momentum analysis on a daily DataFrame.

    Returns MomentumSignal with a 0-4 composite score:
      +1 RSI recovering from oversold (<35) or strong but not overbought (40-65)
      +1 MACD bullish cross (histogram flipping positive)
      +1 Volume surge (>1.5x average)
      +1 Positive short-term rate of change (5d ROC > 1%)
    """
    if len(daily_df) < 30:
        return None

    close = daily_df["close"]
    volume = daily_df["volume"]

    # RSI
    rsi_series = compute_rsi(close)
    rsi = float(rsi_series.iloc[-1])
    if rsi < 30:
        rsi_signal = "oversold"
    elif rsi > 70:
        rsi_signal = "overbought"
    else:
        rsi_signal = "neutral"

    # MACD
    macd_line, signal_line, histogram = compute_macd(close)
    macd_hist = float(histogram.iloc[-1])
    macd_hist_prev = float(histogram.iloc[-2]) if len(histogram) > 1 else 0

    if macd_hist > 0 and macd_hist_prev <= 0:
        macd_cross = "bullish_cross"
    elif macd_hist < 0 and macd_hist_prev >= 0:
        macd_cross = "bearish_cross"
    else:
        macd_cross = "none"

    # Volume
    vol_ratio = compute_volume_ratio(volume)
    vol_current = float(vol_ratio.iloc[-1])
    if vol_current >= 1.5:
        vol_signal = "surge"
    elif vol_current < 0.5:
        vol_signal = "dry"
    else:
        vol_signal = "normal"

    # Rate of change
    roc5 = float(compute_rate_of_change(close, 5).iloc[-1])
    roc20 = float(compute_rate_of_change(close, 20).iloc[-1])

    # --- Composite Momentum Score (0-4) ---
    score = 0

    # RSI: recovering from oversold or in bullish neutral zone
    if rsi < 35 or (40 <= rsi <= 65):
        score += 1

    # MACD: bullish cross or positive and increasing histogram
    if macd_cross == "bullish_cross" or (macd_hist > 0 and macd_hist > macd_hist_prev):
        score += 1

    # Volume: surge confirms move
    if vol_current >= 1.5:
        score += 1

    # Short-term momentum: 5-day ROC > 1%
    if roc5 > 0.01:
        score += 1

    return MomentumSignal(
        rsi=round(rsi, 1),
        rsi_signal=rsi_signal,
        macd_histogram=round(macd_hist, 4),
        macd_cross=macd_cross,
        volume_surge=round(vol_current, 2),
        volume_signal=vol_signal,
        roc_5d=round(roc5, 4),
        roc_20d=round(roc20, 4),
        momentum_score=score,
    )
