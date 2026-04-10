"""
Implied Volatility Rank — determines if options premiums are "expensive" or "cheap."

IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV)

We sell options when IV Rank is HIGH (>30%) because premiums are inflated.
Source: Advanced Algorithmic Trading concepts + standard options practice
"""

import numpy as np
import pandas as pd


def calculate_historical_volatility(
    closes: pd.Series, window: int = 20
) -> pd.Series:
    """
    Calculate rolling historical (realized) volatility.
    HV = std(log returns) * sqrt(252) annualized
    """
    log_returns = np.log(closes / closes.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(252)


def calculate_iv_rank(
    current_iv: float,
    iv_history: pd.Series,
    lookback_days: int = 252,
) -> float:
    """
    IV Rank: where is current IV relative to its 52-week range?

    Args:
        current_iv: Current implied volatility (as decimal, e.g., 0.35 for 35%)
        iv_history: Series of historical IV values
        lookback_days: How many trading days to look back (252 = 1 year)

    Returns:
        IV Rank as a float between 0.0 and 1.0
        0.0 = IV at 52-week low (cheap premiums)
        1.0 = IV at 52-week high (expensive premiums)
    """
    if len(iv_history) < lookback_days:
        recent = iv_history
    else:
        recent = iv_history.iloc[-lookback_days:]

    iv_high = recent.max()
    iv_low = recent.min()

    if iv_high == iv_low:
        return 0.5  # Can't determine rank

    rank = (current_iv - iv_low) / (iv_high - iv_low)
    return max(0.0, min(1.0, rank))


def calculate_iv_percentile(
    current_iv: float,
    iv_history: pd.Series,
    lookback_days: int = 252,
) -> float:
    """
    IV Percentile: what percentage of days had IV below current level?
    
    More robust than IV Rank since it accounts for distribution shape.
    """
    if len(iv_history) < lookback_days:
        recent = iv_history
    else:
        recent = iv_history.iloc[-lookback_days:]

    days_below = (recent < current_iv).sum()
    return days_below / len(recent)


def evaluate_premium_environment(
    current_iv: float,
    iv_history: pd.Series,
    min_iv_rank: float = 0.30,
) -> dict:
    """
    Evaluate whether the current premium environment is favorable for selling.

    Returns:
        dict with rank, percentile, and sell recommendation
    """
    rank = calculate_iv_rank(current_iv, iv_history)
    percentile = calculate_iv_percentile(current_iv, iv_history)

    favorable = rank >= min_iv_rank

    return {
        "current_iv": round(current_iv, 4),
        "iv_rank": round(rank, 4),
        "iv_percentile": round(percentile, 4),
        "favorable_for_selling": favorable,
        "assessment": (
            f"IV Rank {rank:.0%} — {'FAVORABLE' if favorable else 'UNFAVORABLE'} "
            f"for selling premium (threshold: {min_iv_rank:.0%})"
        ),
    }
