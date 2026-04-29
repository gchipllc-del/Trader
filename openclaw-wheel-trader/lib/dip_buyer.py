"""
Dip Buyer — find oversold-but-trend-intact setups for the buy-the-dip play.

The pattern: a stock in a long-term uptrend that's pulled back to RSI<35,
sitting below its 50-day SMA but above its 200-day SMA, with the most recent
bar showing a bounce signal. This is the classic "healthy pullback" vs.
"falling knife": we want pullbacks IN uptrends, never breakdowns from them.

Five-feature composite score (0-1 each, weighted sum):
    1. Oversold strength    (35%) — how deeply oversold (RSI 35→25→15)
    2. Trend intact          (25%) — price above SMA200, SMA50 above SMA200
    3. Pullback magnitude    (15%) — drawdown from 20d high, sweet spot 5-12%
    4. Bounce signal         (15%) — recent bar green + close above prior low
    5. Volume confirmation   (10%) — recent down-days had above-avg volume
                                    (real selling, not noise)

Triggered when:
    composite >= 0.55 AND
    price > SMA200 (long-term uptrend) AND
    price < SMA50 (in pullback) AND
    20d drawdown >= 0.05 AND
    20d drawdown <= 0.25 (else it's a real breakdown, not a dip)

This complements anomaly_detector.py: anomaly detects breakouts going UP,
dip_buyer detects pullbacks ready to BOUNCE up. Same long-only direction.

Usage:
    python main.py dipbuy
    python main.py dipbuy --watchlist NVDA,AMD,COIN --threshold 0.5
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lib.audit import log_event

# --- Tunable thresholds ---------------------------------------------------
DEFAULT_COMPOSITE_THRESHOLD = 0.55  # 0..1 scale, sum of weighted features
RSI_OVERSOLD = 35.0                  # RSI ≤ this = oversold
RSI_DEEP_OVERSOLD = 25.0             # full credit at this level
MIN_DRAWDOWN_PCT = 0.05              # need >= 5% pullback from 20d high
MAX_DRAWDOWN_PCT = 0.25              # >25% = breakdown, not dip
MIN_AVG_VOLUME = 1_000_000           # liquidity floor
MIN_PRICE = 5.0                      # no penny stocks

W_OVERSOLD = 0.35
W_TREND = 0.25
W_PULLBACK = 0.15
W_BOUNCE = 0.15
W_VOLUME = 0.10

DIP_LOG_PATH = Path(__file__).parent.parent / "data" / "dip_log.jsonl"


@dataclass
class DipScore:
    symbol: str
    timestamp: str
    last_price: float
    rsi_14: float
    sma_50: float
    sma_200: float
    drawdown_from_20d_high: float
    composite: float
    oversold_score: float
    trend_score: float
    pullback_score: float
    bounce_score: float
    volume_score: float
    triggered: bool
    skip_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --- Indicators -----------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI on close-price series. Returns last value (NaN-safe)."""
    if len(close) < period + 1:
        return float("nan")
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period, min_periods=period).mean()
    avg_loss = losses.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else float("nan")


def _sma(close: pd.Series, period: int) -> float:
    if len(close) < period:
        return float("nan")
    return float(close.tail(period).mean())


def _drawdown_from_high(close: pd.Series, lookback: int = 20) -> float:
    """Pct drawdown of latest close from rolling-N-day high. Positive = drop."""
    if len(close) < lookback:
        return 0.0
    high = float(close.tail(lookback).max())
    last = float(close.iloc[-1])
    if high <= 0:
        return 0.0
    return (high - last) / high


# --- Composite scoring ----------------------------------------------------

def _scaled_oversold(rsi: float) -> float:
    """1.0 at RSI 25 (deep oversold), 0.0 at RSI 35 (just-touched), <0 above."""
    if np.isnan(rsi):
        return 0.0
    if rsi <= RSI_DEEP_OVERSOLD:
        return 1.0
    if rsi >= RSI_OVERSOLD:
        return 0.0
    # Linear between 25→35 maps to 1.0→0.0
    return max(0.0, (RSI_OVERSOLD - rsi) / (RSI_OVERSOLD - RSI_DEEP_OVERSOLD))


def _trend_intact_score(last: float, sma50: float, sma200: float) -> float:
    """1.0 if price > SMA200 AND SMA50 > SMA200 (golden cross territory).
    0.5 if price > SMA200 but SMA50 < SMA200 (recovering from death cross).
    0.0 if price < SMA200 (long-term downtrend — fail this gate)."""
    if np.isnan(sma50) or np.isnan(sma200) or last < sma200:
        return 0.0
    if sma50 > sma200:
        return 1.0
    return 0.5


def _pullback_score(drawdown: float) -> float:
    """1.0 in the sweet spot (5-12% pullback), tapering at edges."""
    if drawdown < MIN_DRAWDOWN_PCT or drawdown > MAX_DRAWDOWN_PCT:
        return 0.0
    if drawdown <= 0.12:
        # Linear ramp from 5% (0.0) → 12% (1.0)
        return (drawdown - MIN_DRAWDOWN_PCT) / (0.12 - MIN_DRAWDOWN_PCT)
    # 12-25%: taper back from 1.0 to 0.0 (deeper = riskier)
    return max(0.0, (MAX_DRAWDOWN_PCT - drawdown) / (MAX_DRAWDOWN_PCT - 0.12))


def _bounce_score(df: pd.DataFrame) -> float:
    """1.0 if last bar green AND closes above prior 2-bar low. Else fractional."""
    if df is None or len(df) < 4:
        return 0.0
    last = df.iloc[-1]
    prev_low = float(min(df.iloc[-3]["low"], df.iloc[-2]["low"]))
    last_close = float(last["close"])
    last_open = float(last["open"])
    bar_green = last_close > last_open
    above_prior_low = last_close > prev_low
    if bar_green and above_prior_low:
        return 1.0
    if bar_green or above_prior_low:
        return 0.5
    return 0.0


def _volume_confirm_score(df: pd.DataFrame) -> float:
    """1.0 if recent down-days showed above-avg volume (real selling, ready to
    exhaust). 0.0 if down-days were quiet (no panic = no bounce yet)."""
    if df is None or len(df) < 21:
        return 0.0
    last20 = df.tail(20)
    avg_vol = float(last20["volume"].mean())
    down_days = last20[last20["close"] < last20["open"]]
    if down_days.empty or avg_vol <= 0:
        return 0.0
    down_avg_vol = float(down_days["volume"].mean())
    ratio = down_avg_vol / avg_vol
    # 1.0 at ratio=1.5 (50% above avg on down days), 0.0 at ratio=1.0
    return float(np.clip((ratio - 1.0) / 0.5, 0.0, 1.0))


# --- Top-level scoring ----------------------------------------------------

def compute_features(
    symbol: str,
    daily_df: pd.DataFrame,
    composite_threshold: float = DEFAULT_COMPOSITE_THRESHOLD,
    min_avg_volume: int = MIN_AVG_VOLUME,
    min_price: float = MIN_PRICE,
) -> DipScore:
    """Score a single symbol for dip-buy setup."""
    now_iso = datetime.now(timezone.utc).isoformat()

    if daily_df is None or len(daily_df) < 200:
        return DipScore(
            symbol=symbol, timestamp=now_iso,
            last_price=float("nan"), rsi_14=float("nan"),
            sma_50=float("nan"), sma_200=float("nan"),
            drawdown_from_20d_high=float("nan"),
            composite=float("nan"), oversold_score=0,
            trend_score=0, pullback_score=0, bounce_score=0, volume_score=0,
            triggered=False, skip_reason="insufficient_history_200",
        )

    last = daily_df.iloc[-1]
    last_close = float(last["close"])

    rsi14 = _rsi(daily_df["close"], 14)
    sma50 = _sma(daily_df["close"], 50)
    sma200 = _sma(daily_df["close"], 200)
    drawdown = _drawdown_from_high(daily_df["close"], 20)

    # Component scores
    oversold = _scaled_oversold(rsi14)
    trend = _trend_intact_score(last_close, sma50, sma200)
    pullback = _pullback_score(drawdown)
    bounce = _bounce_score(daily_df)
    volume = _volume_confirm_score(daily_df)

    composite = (
        W_OVERSOLD * oversold
        + W_TREND * trend
        + W_PULLBACK * pullback
        + W_BOUNCE * bounce
        + W_VOLUME * volume
    )

    # Quality / safety filters
    skip_reason: str | None = None
    avg_vol = float(daily_df["volume"].tail(20).mean())
    if avg_vol < min_avg_volume:
        skip_reason = f"low_avg_volume_{int(avg_vol):,}"
    elif last_close < min_price:
        skip_reason = f"price_too_low_{last_close:.2f}"
    elif last_close < sma200:
        skip_reason = "below_sma200_long_term_downtrend"
    elif last_close > sma50:
        skip_reason = "above_sma50_no_pullback"
    elif drawdown < MIN_DRAWDOWN_PCT:
        skip_reason = f"drawdown_too_small_{drawdown*100:.1f}%"
    elif drawdown > MAX_DRAWDOWN_PCT:
        skip_reason = f"drawdown_too_large_{drawdown*100:.1f}%"

    triggered = bool(composite >= composite_threshold and skip_reason is None)

    return DipScore(
        symbol=symbol,
        timestamp=now_iso,
        last_price=last_close,
        rsi_14=round(rsi14, 2) if not np.isnan(rsi14) else float("nan"),
        sma_50=round(sma50, 2) if not np.isnan(sma50) else float("nan"),
        sma_200=round(sma200, 2) if not np.isnan(sma200) else float("nan"),
        drawdown_from_20d_high=round(drawdown, 4),
        composite=round(composite, 3),
        oversold_score=round(oversold, 3),
        trend_score=round(trend, 3),
        pullback_score=round(pullback, 3),
        bounce_score=round(bounce, 3),
        volume_score=round(volume, 3),
        triggered=triggered,
        skip_reason=skip_reason,
    )


# --- Universe scan --------------------------------------------------------

def scan_universe(
    client,
    symbols: list[str],
    bars_lookback: int = 220,
    composite_threshold: float = DEFAULT_COMPOSITE_THRESHOLD,
) -> list[DipScore]:
    """Scan a list of symbols for dip-buy setups."""
    if not symbols:
        return []

    bars = client.get_bars(symbols, timeframe="1Day", limit=bars_lookback)

    scores: list[DipScore] = []
    for sym in symbols:
        df = bars.get(sym)
        scores.append(
            compute_features(sym, df, composite_threshold=composite_threshold)
        )

    def _key(s: DipScore) -> float:
        return -999.0 if np.isnan(s.composite) else s.composite

    scores.sort(key=_key, reverse=True)
    return scores


# --- Output ---------------------------------------------------------------

def print_dip_report(scores: list[DipScore], top: int = 15) -> None:
    print("=" * 84)
    print("  DIP BUYER — TOP CANDIDATES (oversold + trend-intact)")
    print("=" * 84)
    print(
        f"  {'Sym':<6} {'Px':>8} {'RSI':>6} {'DD':>7} {'OS':>5} {'Tr':>5} "
        f"{'Pb':>5} {'Bo':>5} {'Vol':>5} {'Comp':>6} {'Hit':>4}  Notes"
    )
    print("  " + "-" * 80)
    for s in scores[:top]:
        if isinstance(s.composite, float) and np.isnan(s.composite):
            print(f"  {s.symbol:<6} insufficient history")
            continue
        hit = "✓" if s.triggered else " "
        notes = s.skip_reason or ""
        print(
            f"  {s.symbol:<6} {s.last_price:>8.2f} {s.rsi_14:>6.1f} "
            f"{s.drawdown_from_20d_high*100:>6.1f}% "
            f"{s.oversold_score:>5.2f} {s.trend_score:>5.2f} "
            f"{s.pullback_score:>5.2f} {s.bounce_score:>5.2f} "
            f"{s.volume_score:>5.2f} {s.composite:>6.3f} "
            f" {hit:<3} {notes}"
        )
    print("  " + "-" * 80)
    triggered = [s for s in scores if s.triggered]
    print(f"  Triggered: {len(triggered)} of {len(scores)} symbols")
    print()


def persist_scores(scores: list[DipScore]) -> int:
    triggered = [s for s in scores if s.triggered]
    if not triggered:
        return 0
    DIP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DIP_LOG_PATH, "a") as f:
        for s in triggered:
            f.write(json.dumps(s.to_dict()) + "\n")
    for s in triggered:
        log_event("dip_buyer", "triggered", s.to_dict(), result="info")
    return len(triggered)


# --- Default watchlist ----------------------------------------------------

# High-quality names where dips have historically been buyable (vs structural
# breakdowns). Bias toward growth + tech mega-caps + crypto-correlated.
DEFAULT_WATCHLIST = [
    "NVDA", "AMD", "TSLA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "AVGO",
    "AVAGO", "INTC", "MU", "TSM", "AMAT", "LRCX",
    "COIN", "MSTR", "HOOD", "PLTR", "SOFI",
    "CRM", "ADBE", "NOW", "SNOW", "DDOG", "NET", "OKTA",
    "SHOP", "SQ", "PYPL", "AFRM",
    "DIS", "NFLX", "RBLX", "U", "DKNG",
]
