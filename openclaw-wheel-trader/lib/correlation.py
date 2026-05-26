"""
Correlation Detection — prevent concentrated bets on correlated stocks.

Adapted from polybot's market_scanner.py correlation detection.

In stocks, if F and NIO are both EV-adjacent, buying both = effectively doubling
exposure to the EV sector. Correlation detection flags this and lets the risk
agent veto or size down one of them.

Uses:
    1. Sector/industry grouping (from config or yfinance)
    2. Historical price correlation (Pearson on log returns)
    3. Manual correlation groups for known clusters
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

import numpy as np
import pandas as pd

from lib.audit import log_event

CONFIG_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"


def _load_strategy() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# Manual correlation groups — tickers that are highly correlated
# These are heuristic; price correlation will also be computed
CORRELATION_GROUPS = {
    "ev_autos": ["NIO", "RIVN", "TSLA", "LCID", "F"],           # EV + legacy
    "banks": ["BAC", "JPM", "WFC", "C", "GS"],
    "airlines": ["AAL", "UAL", "DAL", "LUV"],
    "cruise": ["CCL", "NCLH", "RCL"],
    "chinese_adr": ["NIO", "GRAB", "BABA", "JD", "PDD"],
    "fintech": ["SOFI", "NU", "PYPL", "SQ", "HOOD"],
    "mobility": ["UBER", "LYFT", "DASH", "GRAB"],
    "steel_mining": ["CLF", "VALE", "X", "NUE", "FCX"],
    "pharma": ["PFE", "MRK", "JNJ", "BMY", "LLY"],
    "consumer": ["KO", "PEP", "MCD", "SBUX"],
    "big_tech": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD"],
    "ai_defense": ["PLTR", "AI", "ANET"],
    "entertainment": ["WBD", "DIS", "NFLX", "PARA"],
    "intel_semi": ["INTC", "AMD", "NVDA", "MU", "QCOM"],
}


@dataclass
class CorrelationResult:
    """Output of correlation check."""
    ticker_a: str
    ticker_b: str
    price_correlation: float      # Pearson correlation on log returns
    same_sector_group: str | None # Named group if they share one
    is_correlated: bool
    reason: str = ""


def _price_correlation(df_a: pd.DataFrame, df_b: pd.DataFrame, lookback: int = 60) -> float:
    """
    Compute Pearson correlation of log returns between two tickers.

    Returns correlation in [-1, 1]. >0.7 = strongly correlated.
    """
    try:
        a_close = df_a["close"].tail(lookback).values
        b_close = df_b["close"].tail(lookback).values

        min_len = min(len(a_close), len(b_close))
        if min_len < 20:
            return 0.0

        a_close = a_close[-min_len:]
        b_close = b_close[-min_len:]

        a_ret = np.diff(np.log(a_close))
        b_ret = np.diff(np.log(b_close))

        if len(a_ret) < 2 or a_ret.std() == 0 or b_ret.std() == 0:
            return 0.0

        corr = np.corrcoef(a_ret, b_ret)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def _find_shared_group(ticker_a: str, ticker_b: str) -> str | None:
    """Find if two tickers share a predefined correlation group."""
    for group_name, members in CORRELATION_GROUPS.items():
        if ticker_a in members and ticker_b in members:
            return group_name
    return None


def check_correlation(
    ticker_a: str,
    ticker_b: str,
    df_a: pd.DataFrame | None = None,
    df_b: pd.DataFrame | None = None,
    threshold: float = 0.70,
) -> CorrelationResult:
    """
    Check if two tickers are highly correlated.

    Args:
        ticker_a: First ticker
        ticker_b: Second ticker
        df_a: Daily OHLCV for ticker_a (optional — skips price corr if None)
        df_b: Daily OHLCV for ticker_b
        threshold: Correlation above this = flagged

    Returns:
        CorrelationResult with decision.
    """
    group = _find_shared_group(ticker_a, ticker_b)

    price_corr = 0.0
    if df_a is not None and df_b is not None:
        price_corr = _price_correlation(df_a, df_b)

    # Flag if either: shared group OR strong price correlation
    is_correlated = group is not None or abs(price_corr) >= threshold

    reason = ""
    if group:
        reason = f"both in '{group}' sector"
        if abs(price_corr) >= threshold:
            reason += f" + {price_corr:+.2f} price correlation"
    elif abs(price_corr) >= threshold:
        reason = f"{price_corr:+.2f} price correlation (threshold {threshold})"

    return CorrelationResult(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        price_correlation=round(price_corr, 4),
        same_sector_group=group,
        is_correlated=is_correlated,
        reason=reason,
    )


def check_portfolio_correlation(
    new_ticker: str,
    held_tickers: list[str],
    daily_data: dict,
    threshold: float = 0.70,
) -> dict:
    """
    Check if a new candidate is highly correlated with any held position.

    Args:
        new_ticker: Proposed ticker
        held_tickers: List of currently-held tickers
        daily_data: dict of ticker → DataFrame
        threshold: Correlation flag threshold

    Returns:
        {
            "correlated": bool,
            "conflicts": list of CorrelationResult,
            "reason": str,
        }
    """
    conflicts = []
    new_df = daily_data.get(new_ticker)

    for held in held_tickers:
        if held == new_ticker:
            continue
        held_df = daily_data.get(held)

        result = check_correlation(new_ticker, held, new_df, held_df, threshold)
        if result.is_correlated:
            conflicts.append(result)

    correlated = len(conflicts) > 0

    if correlated:
        groups = set(c.same_sector_group for c in conflicts if c.same_sector_group)
        max_corr = max((abs(c.price_correlation) for c in conflicts), default=0)
        reason = (
            f"{len(conflicts)} correlated holding(s): "
            f"{', '.join(c.ticker_b for c in conflicts)} "
            f"(max corr {max_corr:+.2f}, groups: {groups or 'none'})"
        )
    else:
        reason = "no correlated holdings"

    log_event("correlation", "checked", {
        "new_ticker": new_ticker,
        "held": held_tickers,
        "correlated": correlated,
        "conflicts": [c.ticker_b for c in conflicts],
    })

    return {
        "correlated": correlated,
        "conflicts": [
            {
                "ticker": c.ticker_b,
                "group": c.same_sector_group,
                "correlation": c.price_correlation,
                "reason": c.reason,
            }
            for c in conflicts
        ],
        "reason": reason,
    }


def get_sector_groups(tickers: list[str]) -> dict[str, list[str]]:
    """Group tickers by known correlation group. Useful for diversification check."""
    groups: dict[str, list[str]] = {}
    ungrouped = []

    for ticker in tickers:
        found = False
        for group_name, members in CORRELATION_GROUPS.items():
            if ticker in members:
                groups.setdefault(group_name, []).append(ticker)
                found = True
                break
        if not found:
            ungrouped.append(ticker)

    if ungrouped:
        groups["ungrouped"] = ungrouped

    return groups
