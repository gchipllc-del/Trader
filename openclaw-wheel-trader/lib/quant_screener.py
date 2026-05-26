"""
Quantitative Stock Screener — data-driven universe selection.

Ranks tickers by risk-adjusted metrics BEFORE the technical analysis
(trend + zones + candlestick) kicks in. This is the first gate:

  1. Quant screen (this module) → filter universe by Sharpe, drawdown, volatility
  2. Technical screen (stock_engine / screener) → timing entries with signals
  3. Order gate → safety checks and execution

Used at every phase:
  Phase 1: rank stocks for buying
  Phase 2: rank underlyings for CSP selling
  Phase 3: rank full Wheel candidates
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from lib.audit import log_event


@dataclass
class QuantScore:
    """Quantitative assessment of a ticker."""
    ticker: str
    price: float
    total_return_1y: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float           # Annualized
    avg_daily_volume: float
    # Composite quant score (0-10)
    quant_score: float
    verdict: str                # STRONG, OK, WEAK, AVOID


def score_ticker(ticker: str, daily_df: pd.DataFrame) -> QuantScore | None:
    """
    Score a single ticker on quantitative metrics.

    Requires at least 100 trading days of data.
    """
    if len(daily_df) < 100:
        return None

    prices = daily_df["close"]
    returns = prices.pct_change().dropna()

    if len(returns) < 50:
        return None

    price = float(prices.iloc[-1])
    start_price = float(prices.iloc[0])

    # Core metrics
    total_return = (price - start_price) / start_price
    trading_days = len(returns)
    years = trading_days / 252
    annualized_return = (1 + total_return) ** (1 / max(years, 0.1)) - 1

    volatility = float(returns.std() * np.sqrt(252))
    avg_volume = float(daily_df["volume"].mean())

    # Max drawdown
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    max_dd = float(drawdown.min())

    # Sharpe ratio (risk-free ≈ 0 for simplicity)
    mean_return = float(returns.mean() * 252)
    sharpe = mean_return / volatility if volatility > 0 else 0

    # Sortino ratio (downside deviation only)
    downside = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 10 else volatility
    sortino = mean_return / downside_vol if downside_vol > 0 else sharpe

    # --- Composite Quant Score (0-10) ---
    score = 0.0

    # Sharpe contribution (0-3)
    if sharpe >= 1.5:
        score += 3
    elif sharpe >= 0.75:
        score += 2
    elif sharpe >= 0.3:
        score += 1

    # Drawdown contribution (0-3)
    if max_dd > -0.15:
        score += 3
    elif max_dd > -0.25:
        score += 2
    elif max_dd > -0.40:
        score += 1
    # > -40% drawdown gets 0

    # Return contribution (0-2)
    if annualized_return > 0.20:
        score += 2
    elif annualized_return > 0.05:
        score += 1

    # Volume/liquidity contribution (0-1)
    if avg_volume > 5_000_000:
        score += 1
    elif avg_volume > 1_000_000:
        score += 0.5

    # Volatility penalty (0-1 bonus for low vol)
    if volatility < 0.30:
        score += 1
    elif volatility < 0.50:
        score += 0.5

    # Verdict
    if score >= 7 and max_dd > -0.35:
        verdict = "STRONG"
    elif score >= 5 and max_dd > -0.45:
        verdict = "OK"
    elif max_dd < -0.55 or sharpe < -0.5:
        verdict = "AVOID"
    else:
        verdict = "WEAK"

    return QuantScore(
        ticker=ticker,
        price=round(price, 2),
        total_return_1y=round(total_return, 4),
        annualized_return=round(annualized_return, 4),
        max_drawdown=round(max_dd, 4),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        volatility=round(volatility, 4),
        avg_daily_volume=round(avg_volume, 0),
        quant_score=round(score, 1),
        verdict=verdict,
    )


def screen_universe(
    daily_data: dict[str, pd.DataFrame],
    max_price: float | None = None,
    min_sharpe: float = -0.5,
    max_drawdown: float = -0.55,
    exclude_avoid: bool = True,
) -> list[QuantScore]:
    """
    Screen all tickers and return ranked list.

    Args:
        daily_data: {ticker: DataFrame} from data pipeline
        max_price: Filter out stocks above this price (for affordability)
        min_sharpe: Minimum Sharpe ratio to include
        max_drawdown: Maximum acceptable drawdown (e.g., -0.55 = -55%)
        exclude_avoid: Remove tickers with AVOID verdict

    Returns:
        List of QuantScore sorted by quant_score descending
    """
    scores = []

    for ticker, df in daily_data.items():
        qs = score_ticker(ticker, df)
        if qs is None:
            continue

        # Apply filters
        if max_price and qs.price > max_price:
            continue
        if qs.sharpe_ratio < min_sharpe:
            continue
        if qs.max_drawdown < max_drawdown:
            continue
        if exclude_avoid and qs.verdict == "AVOID":
            continue

        scores.append(qs)

    # Rank by quant score, then Sharpe as tiebreaker
    scores.sort(key=lambda s: (s.quant_score, s.sharpe_ratio), reverse=True)

    log_event("quant_screener", "screen_complete", {
        "total_tickers": len(daily_data),
        "passed": len(scores),
        "top_3": [s.ticker for s in scores[:3]],
    })

    return scores


def get_optimal_tickers(
    daily_data: dict[str, pd.DataFrame],
    max_picks: int = 8,
    max_price: float | None = None,
    min_quant_score: float = 4.0,
) -> list[str]:
    """
    Return the best N tickers from the universe.
    Used by the scan pipeline to dynamically select what to trade.
    """
    scores = screen_universe(daily_data, max_price=max_price)
    filtered = [s for s in scores if s.quant_score >= min_quant_score]
    return [s.ticker for s in filtered[:max_picks]]


def print_screening_report(scores: list[QuantScore]):
    """Print a formatted screening report."""
    if not scores:
        print("  No tickers passed screening.")
        return

    print(f"  {'Ticker':6s} {'Price':>7s} {'1Y Ret':>7s} {'MaxDD':>7s} {'Sharpe':>7s} "
          f"{'Vol':>6s} {'QScore':>7s} {'Verdict':>8s}")
    print("  " + "-" * 65)

    for s in scores:
        print(f"  {s.ticker:6s} ${s.price:>6.2f} {s.total_return_1y:>+6.0%} "
              f"{s.max_drawdown:>+6.0%} {s.sharpe_ratio:>7.2f} "
              f"{s.volatility:>5.0%} {s.quant_score:>7.1f} {s.verdict:>8s}")
