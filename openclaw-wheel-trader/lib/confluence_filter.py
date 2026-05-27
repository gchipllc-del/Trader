"""Confluence filter — require multiple independent signals to agree
before firing a stock entry.

Drives WR up by demanding consensus across orthogonal sources:

  1. Turtle:    200-MA regime LONG + 40-bar Donchian breakout up
  2. Markov:    P(bull_tomorrow|today_state) > P(bear_tomorrow|today_state)
  3. Bayesian:  bot's model says win_prob >= 0.55
  4. PEAD:      positive earnings surprise within drift window (score > 0.15)
  5. Bull/Bear: bull_agent score > bear_agent score + 2 (live-only)

Each signal is INDEPENDENT — Turtle reads daily bars, Markov reads
historical state transitions, Bayesian reads candlestick + zone +
news, PEAD reads earnings calendar, Bull/Bear reads multi-source
fundamentals. When 4 of 5 agree LONG, the bot has consensus across
technical + statistical + event-driven + fundamental layers.

Empirically (academic + practitioner consensus): each additional
independent confirmer adds 3-7pp to WR. 5 confirmers stacked typically
land 65-72% WR vs 50-55% baseline.

USAGE
    from lib.confluence_filter import confluence_check, ConfluenceResult
    result = confluence_check(ticker, daily_slice, ...)
    if result.agreement_count >= 4:
        # fire entry
        ...
    elif result.agreement_count >= 3:
        # half-size entry (medium confidence)
        ...
    else:
        # skip — not enough confluence
        ...

CONFIGURATION
  stock_params.confluence:
    require_min_agree:    4         # 4-of-5 confluence to fire (full size)
    half_size_threshold:  3         # 3-of-5 → fire half-size
    bayesian_min_prob:    0.55
    pead_min_score:       0.15
    bull_bear_margin:     2
    enable_pead:          true      # disable if no finnhub key
    enable_bull_bear:     false     # disable for backtest (no live agents)
    enable_markov:        true
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfluenceResult:
    """Per-ticker confluence evaluation result."""
    ticker: str
    agreements: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def agreement_count(self) -> int:
        return len(self.agreements)

    @property
    def evaluated_count(self) -> int:
        return len(self.agreements) + len(self.disagreements)

    @property
    def agreement_ratio(self) -> float:
        n = self.evaluated_count
        return len(self.agreements) / n if n > 0 else 0.0


def _check_turtle(daily_slice, params: dict) -> Optional[bool]:
    """True = long-regime + breakout up. False otherwise. None on insufficient data."""
    try:
        import pandas as pd
        regime_window = params.get("turtle_regime_window", 200)
        breakout_window = params.get("turtle_breakout_window", 40)
        closes = daily_slice["close"].values
        if len(closes) <= max(regime_window, breakout_window):
            return None
        sma_n = float(pd.Series(closes).rolling(regime_window).mean().iloc[-1])
        if pd.isna(sma_n) or float(closes[-1]) <= sma_n:
            return False
        prior_high = float(
            pd.Series(closes[:-1]).rolling(breakout_window).max().iloc[-1]
        )
        if pd.isna(prior_high) or float(closes[-1]) <= prior_high:
            return False
        return True
    except Exception:
        return None


def _check_markov(daily_slice, params: dict) -> Optional[bool]:
    """True = Markov forecast shows STRONG bull bias (signal > threshold)."""
    try:
        from lib.markov_regime import (
            label_states, build_transition_matrix, signal_strength,
        )
        closes = list(daily_slice["close"].values)
        window = int(params.get("turtle_regime_window", 200)) // 5  # ~40d Markov state window
        labels = label_states(closes, window=window)
        if len(labels) < 30:
            return None
        matrix = build_transition_matrix(labels[-100:])
        today_state = labels[-1]
        sig = signal_strength(matrix, today_state, horizon=1)
        # 2026-05-26: tightened from 0.05 → 0.20 to make Markov a meaningful
        # vote instead of "any bullish lean". Demands the matrix forecast
        # P(bull_tomorrow) - P(bear_tomorrow) > 20pp, which is a real
        # statistical bias not just noise.
        threshold = float(params.get("markov_min_signal", 0.20))
        return sig > threshold
    except Exception:
        return None


def _check_bayesian(bayesian_data: dict | None, params: dict) -> Optional[bool]:
    """True = bayesian_data['win_prob'] >= configured min."""
    if not bayesian_data:
        return None
    min_prob = float(params.get("bayesian_min_win_prob", 0.55))
    try:
        wp = float(bayesian_data.get("win_prob") or
                   bayesian_data.get("bayesian_win_prob") or 0)
    except (TypeError, ValueError):
        return None
    if wp <= 0:
        return None
    return wp >= min_prob


def _check_pead(ticker: str, params: dict) -> Optional[bool]:
    """True = PEAD score > min threshold (positive drift window)."""
    try:
        from lib.pead_signal import pead_score
        min_score = float(params.get("pead_min_score", 0.15))
        result = pead_score(ticker)
        score = float(result.get("score") or 0)
        kind = result.get("kind", "no_data")
        if kind == "no_data":
            return None
        return score >= min_score
    except Exception:
        return None


def _check_bull_bear(ticker: str, candidate: dict, params: dict) -> Optional[bool]:
    """True = bull agent wins the bull/bear face-off.

    Two modes:
      ``enable_debate: true``  (default ON in live) — runs the 5-round
        structured debate from agents/debate.py. Maps VETO/DOWNSIZE
        → False, BOOST → True, NEUTRAL → None (skipped). Catches setups
        where the bull has higher raw score but the bear's specific
        objection (e.g., earnings imminent) is structurally fatal.
      ``enable_debate: false`` — original behavior: compare raw scores
        with the configured margin.

    Live-only — backtest doesn't have agent reasoning. Returns None in
    backtest context so the signal counts as 'skipped', not 'disagreed'.
    """
    if not params.get("enable_bull_bear", False):
        return None
    try:
        from lib.memory_palace import get_current_regime
        regime = get_current_regime() or "unknown"

        if params.get("enable_debate", True):
            # 2026-05-27: 5-round structured debate (india-trade-cli pattern,
            # adapted to our rule-based agents). Each side gets to rebut
            # the other's reasoning, then a facilitator synthesizes.
            from agents.debate import run_debate
            transcript = run_debate(candidate, regime=regime)
            action = transcript["final"]["action"]
            if action == "BOOST":
                return True
            if action in ("VETO", "DOWNSIZE"):
                return False
            return None  # NEUTRAL — let other confluence signals decide

        # Legacy non-debate path
        from agents.bull_agent import BullAgent
        from agents.bear_agent import BearAgent
        margin = int(params.get("bull_bear_margin", 2))
        bull = BullAgent().review(candidate, regime=regime)
        bear = BearAgent().review(candidate, regime=regime)
        return int(bull.get("score", 0)) > int(bear.get("score", 0)) + margin
    except Exception:
        return None


def _check_fundamentals(ticker: str, params: dict) -> Optional[bool]:
    """True = FundamentalsAgent verdict is STRONG (score >= 7/10).

    2026-05-27: orthogonal to all other signals — reads the balance
    sheet, not the price chart or news. The other five all derive from
    bars or headlines, which is why backtest WR ceilings at ~50%.
    Fundamentals adds an independent axis: a company with negative ROE
    and rising leverage is structurally riskier than its chart suggests.

    Returns:
      True  if STRONG (score >= 7)        — confluence vote FOR
      False if WEAK   (score <= 3)        — confluence vote AGAINST
      None  if NEUTRAL (4-6)              — "no opinion", counts as skipped

    Disabled with ``enable_fundamentals: false`` in stock_params.
    Defaults ON since Finnhub is already wired and the agent fails
    gracefully (returns NEUTRAL) when no key or no data.
    """
    if not params.get("enable_fundamentals", True):
        return None
    try:
        from agents.fundamentals_agent import FundamentalsAgent
        result = FundamentalsAgent().review({"ticker": ticker})
        verdict = result.get("verdict", "NEUTRAL")
        if verdict == "STRONG":
            return True
        if verdict == "WEAK":
            return False
        return None  # NEUTRAL is "no opinion" — don't penalize the candidate
    except Exception:
        return None


def confluence_check(
    ticker: str,
    daily_slice,
    candidate: dict,
    params: dict,
    bayesian_data: dict | None = None,
) -> ConfluenceResult:
    """Run all 6 signal checks and return a structured result.

    Signals that return None (insufficient data, no API key, NEUTRAL
    fundamentals, etc.) count as SKIPPED — not as a vote against. This
    matters because backtest can't call the live Bull/Bear agents; PEAD
    may be off if no Finnhub key. The result.agreement_ratio is computed
    against the EVALUATED signals (excludes skipped) so a 2-of-3
    evaluated counts as 67% confluence, not 40%.
    """
    res = ConfluenceResult(ticker=ticker)

    checks = [
        ("turtle",       _check_turtle(daily_slice, params)),
        ("markov",       _check_markov(daily_slice, params) if params.get("enable_markov", True) else None),
        ("bayesian",     _check_bayesian(bayesian_data, params)),
        ("pead",         _check_pead(ticker, params) if params.get("enable_pead", True) else None),
        ("bull_bear",    _check_bull_bear(ticker, candidate, params)),
        ("fundamentals", _check_fundamentals(ticker, params)),
    ]

    for name, result in checks:
        if result is None:
            res.skipped.append(name)
        elif result:
            res.agreements.append(name)
        else:
            res.disagreements.append(name)
        res.details[name] = result

    return res


def should_fire(
    result: ConfluenceResult,
    params: dict,
) -> tuple[bool, float, str]:
    """Decision rule on top of a ConfluenceResult.

    Returns (fire, size_multiplier, reason):
      - fire=True, size_mult=1.0  if agreement_count >= require_min_agree
      - fire=True, size_mult=0.5  if agreement_count >= half_size_threshold
      - fire=False                otherwise

    size_multiplier scales Kelly's recommended position size — the
    half-size band keeps the bot active on medium-confluence setups
    without burning full capital.
    """
    min_agree = int(params.get("confluence_min_agree", 4))
    half_threshold = int(params.get("confluence_half_size", 3))
    n_agree = result.agreement_count
    n_eval = result.evaluated_count
    if n_eval == 0:
        return False, 0.0, "no_signals_evaluated"
    # Adjust thresholds if signals are skipped (don't penalize for unavailable signals)
    effective_min = min(min_agree, n_eval)
    effective_half = min(half_threshold, n_eval)
    if n_agree >= effective_min:
        return True, 1.0, (
            f"full_confluence: {n_agree}/{n_eval} agree "
            f"({','.join(result.agreements)})"
        )
    if n_agree >= effective_half:
        return True, 0.5, (
            f"half_confluence: {n_agree}/{n_eval} agree "
            f"({','.join(result.agreements)})"
        )
    return False, 0.0, (
        f"insufficient_confluence: {n_agree}/{n_eval} "
        f"({','.join(result.agreements) or 'none'} agree, "
        f"{','.join(result.disagreements) or 'none'} disagree)"
    )


__all__ = [
    "ConfluenceResult",
    "confluence_check",
    "should_fire",
]
