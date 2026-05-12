"""
Stock & Options Screener — finds Wheel Strategy candidates.

Pipeline:
1. Filter universe for liquid, optionable stocks in our price range
2. Run trend analysis (Candlestick Bible framework)
3. Detect S/R zones (Naked Forex method)
4. Check IV environment (favorable for selling?)
5. Score and rank CSP / CC candidates
6. Return composite-scored trade ideas

This is the orchestrator — it calls zones.py, trend.py, candlestick.py, iv_rank.py
"""

import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import pandas as pd

from lib.zones import detect_zones, get_nearest_support, get_nearest_resistance
from lib.trend import analyze_trend, multi_timeframe_analysis
from lib.candlestick import get_latest_signal
from lib.iv_rank import evaluate_premium_environment

CONFIG_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"


def _load_strategy_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@dataclass
class WheelCandidate:
    """A scored trade candidate for the Wheel Strategy."""
    ticker: str
    trade_type: Literal["csp", "cc"]         # Cash-secured put or covered call
    strike: float
    expiration: str
    premium: float
    delta: float
    dte: int
    annualized_return: float
    # Composite scoring (Candlestick Bible framework)
    trend_score: int        # 0-3
    level_score: int        # 0-3
    signal_score: int       # 0-3
    composite_score: int    # 0-9
    # Details
    zone_level: float
    zone_touches: int
    iv_rank: float
    candlestick_pattern: str | None
    tradeable: bool         # Meets all thresholds


def score_csp_candidate(
    ticker: str,
    option: dict,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    iv_data: dict,
    config: dict,
) -> WheelCandidate | None:
    """
    Score a put option as a Wheel CSP candidate.

    Args:
        ticker: Stock symbol
        option: Dict with keys: strike, expiration, bid, ask, delta, dte, open_interest
        daily_df: Daily OHLCV data
        weekly_df: Weekly OHLCV data
        iv_data: Output from evaluate_premium_environment()
        config: Strategy config dict

    Returns:
        WheelCandidate or None if it doesn't pass filters
    """
    csp_cfg = config["csp"]

    # --- Basic filters ---
    delta = option.get("delta", 0)
    dte = option.get("dte", 0)
    oi = option.get("open_interest", 0)
    bid = option.get("bid", 0)
    ask = option.get("ask", 0)
    strike = option.get("strike", 0)

    # Delta range
    if not (csp_cfg["delta_min"] <= delta <= csp_cfg["delta_max"]):
        return None

    # DTE range
    if not (csp_cfg["dte_min"] <= dte <= csp_cfg["dte_max"]):
        return None

    # Liquidity
    if oi < csp_cfg["min_open_interest"]:
        return None

    spread = ask - bid
    if spread > csp_cfg["max_bid_ask_spread"]:
        return None

    # Premium calculation
    mid_price = (bid + ask) / 2
    collateral = strike * 100
    if collateral == 0:
        return None

    annualized = (mid_price * 100 / collateral) * (365 / max(dte, 1))
    if annualized < csp_cfg["min_annualized_return"]:
        return None
    max_ann = csp_cfg.get("max_annualized_return")
    if max_ann is not None and annualized > max_ann:
        # Likely bad data (stale quote, corporate action, special div) —
        # reject to avoid putting capital behind a phantom yield.
        return None

    # --- Trend Score (0-3) ---
    mtf = multi_timeframe_analysis(weekly_df, daily_df)
    trend_score = mtf["alignment_score"]

    # For CSPs we want uptrend or ranging (selling puts in downtrend is dangerous)
    weekly_trend = mtf["weekly"]
    if weekly_trend.direction == "downtrend" and weekly_trend.strength >= 2:
        return None  # Don't sell puts into a strong downtrend

    # --- Level Score (0-3) ---
    current_price = daily_df["close"].iloc[-1]
    zones = detect_zones(daily_df, current_price)
    nearest_support = get_nearest_support(zones, current_price)

    level_score = 0
    zone_level = 0.0
    zone_touches = 0

    if nearest_support:
        # Is the strike at or near the support zone?
        distance_to_zone = abs(strike - nearest_support.level) / nearest_support.level
        if distance_to_zone < 0.03:  # Within 3% of zone
            level_score = 3
        elif distance_to_zone < 0.05:
            level_score = 2
        elif distance_to_zone < 0.08:
            level_score = 1

        zone_level = nearest_support.level
        zone_touches = nearest_support.touches

    # --- Signal Score (0-3) ---
    signal_score = 0
    pattern_name = None

    bullish_patterns = config["confirmation"]["bullish_patterns"]
    signal = get_latest_signal(daily_df, "bullish", bullish_patterns)

    if signal:
        signal_score = signal.strength
        pattern_name = signal.pattern

    # --- Composite ---
    composite = trend_score + level_score + signal_score
    min_score = config["confirmation"]["min_composite_score"]
    tradeable = (
        composite >= min_score
        and iv_data.get("favorable_for_selling", False)
        and mtf.get("tradeable", False)
    )

    return WheelCandidate(
        ticker=ticker,
        trade_type="csp",
        strike=strike,
        expiration=option.get("expiration", ""),
        premium=round(mid_price, 2),
        delta=round(delta, 3),
        dte=dte,
        annualized_return=round(annualized, 4),
        trend_score=trend_score,
        level_score=level_score,
        signal_score=signal_score,
        composite_score=composite,
        zone_level=zone_level,
        zone_touches=zone_touches,
        iv_rank=iv_data.get("iv_rank", 0),
        candlestick_pattern=pattern_name,
        tradeable=tradeable,
    )


def score_cc_candidate(
    ticker: str,
    option: dict,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    iv_data: dict,
    cost_basis: float,
    config: dict,
    original_purchase_price: float | None = None,
) -> WheelCandidate | None:
    """
    Score a call option as a Wheel CC candidate.
    Similar to CSP scoring but for calls at resistance zones, above cost basis.

    Two floors are enforced on the strike:
      * adjusted ``cost_basis`` (existing behavior, reduced by accumulated CC
        premiums) — so each CC nets income above the running basis
      * ``original_purchase_price`` (new, optional) — the unadjusted broker
        fill price at the time of CSP assignment. Prevents the strike from
        ratcheting below the actual capital invested as cost_basis drifts
        down across multiple CC cycles. Falls back to cost_basis if not
        provided (back-compat).
    """
    cc_cfg = config["cc"]

    delta = option.get("delta", 0)
    dte = option.get("dte", 0)
    oi = option.get("open_interest", 0)
    bid = option.get("bid", 0)
    ask = option.get("ask", 0)
    strike = option.get("strike", 0)

    # Must be above adjusted cost basis (running, includes CC-premium credit)
    if strike <= cost_basis:
        return None
    # Must also be above the original purchase price (hard floor that does
    # not drift). Defaults to cost_basis if caller didn't pass it, which
    # makes this a no-op for back-compat.
    hard_floor = original_purchase_price if original_purchase_price is not None else cost_basis
    if strike < hard_floor:
        return None

    if not (cc_cfg["delta_min"] <= delta <= cc_cfg["delta_max"]):
        return None
    if not (cc_cfg["dte_min"] <= dte <= cc_cfg["dte_max"]):
        return None
    if oi < cc_cfg["min_open_interest"]:
        return None

    spread = ask - bid
    if spread > cc_cfg["max_bid_ask_spread"]:
        return None

    mid_price = (bid + ask) / 2
    share_value = strike * 100
    if share_value == 0:
        return None

    annualized = (mid_price * 100 / share_value) * (365 / max(dte, 1))
    if annualized < cc_cfg["min_annualized_return"]:
        return None
    max_ann = cc_cfg.get("max_annualized_return")
    if max_ann is not None and annualized > max_ann:
        # Likely bad data (stale quote, corporate action, special div).
        return None

    # --- Trend Score ---
    mtf = multi_timeframe_analysis(weekly_df, daily_df)
    trend_score = mtf["alignment_score"]

    # --- Level Score (resistance zone) ---
    current_price = daily_df["close"].iloc[-1]
    zones = detect_zones(daily_df, current_price)
    nearest_resistance = get_nearest_resistance(zones, current_price)

    level_score = 0
    zone_level = 0.0
    zone_touches = 0

    if nearest_resistance:
        distance = abs(strike - nearest_resistance.level) / nearest_resistance.level
        if distance < 0.03:
            level_score = 3
        elif distance < 0.05:
            level_score = 2
        elif distance < 0.08:
            level_score = 1

        zone_level = nearest_resistance.level
        zone_touches = nearest_resistance.touches

    # --- Signal Score (bearish at resistance for CCs) ---
    signal_score = 0
    pattern_name = None

    bearish_patterns = config["confirmation"]["bearish_patterns"]
    signal = get_latest_signal(daily_df, "bearish", bearish_patterns)

    if signal:
        signal_score = signal.strength
        pattern_name = signal.pattern

    composite = trend_score + level_score + signal_score
    min_score = config["confirmation"]["min_composite_score"]
    tradeable = composite >= min_score and iv_data.get("favorable_for_selling", False)

    return WheelCandidate(
        ticker=ticker,
        trade_type="cc",
        strike=strike,
        expiration=option.get("expiration", ""),
        premium=round(mid_price, 2),
        delta=round(delta, 3),
        dte=dte,
        annualized_return=round(annualized, 4),
        trend_score=trend_score,
        level_score=level_score,
        signal_score=signal_score,
        composite_score=composite,
        zone_level=zone_level,
        zone_touches=zone_touches,
        iv_rank=iv_data.get("iv_rank", 0),
        candlestick_pattern=pattern_name,
        tradeable=tradeable,
    )


def _risk_adjusted_yield(c: WheelCandidate) -> float:
    """Annualized premium yield discounted by assignment-probability.

    A 0.35-delta candidate ~35% likely to be assigned has its yield
    weighted by ~0.65, vs a 0.20-delta candidate weighted by ~0.80.
    Compare two candidates of identical raw annualized return: the
    safer one ranks higher.

    Pattern from alpacahq/options-wheel scoring — they use
    ``(1 - |Δ|) * (250 / (dte + 5)) * (bid/strike)``; we already have
    ``annualized_return`` precomputed, so the simpler equivalent is
    ``annualized_return * (1 - |Δ|)``.
    """
    return float(c.annualized_return or 0) * max(0.0, 1.0 - abs(float(c.delta or 0)))


def rank_candidates(candidates: list[WheelCandidate]) -> list[WheelCandidate]:
    """
    Rank candidates by composite score, then risk-adjusted yield.
    Only returns tradeable candidates.

    Tiebreaker change (alpacahq/options-wheel pattern): raw
    annualized_return doesn't account for assignment probability, so a
    juicy 0.35-delta candidate would outrank a safer 0.20-delta one at
    equal raw yield. The risk-adjusted score multiplies yield by
    ``(1 - |Δ|)`` so the safer trade wins ties.
    """
    tradeable = [c for c in candidates if c.tradeable]
    tradeable.sort(
        key=lambda c: (c.composite_score, _risk_adjusted_yield(c)),
        reverse=True,
    )
    return tradeable
