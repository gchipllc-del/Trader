"""
Bayesian Stock Forecaster — multi-signal probability aggregation.

Adapted from polybot's forecaster.py for stock trading.

Instead of simply adding signals (trend + level + signal + momentum = composite),
this module applies Bayesian updates to produce a calibrated probability that
a trade will hit its target before stop.

Pipeline:
    1. Start with base rate prior (55% for bullish zone + trend setups)
    2. Bayesian update with candlestick signal
    3. Bayesian update with momentum
    4. Bayesian update with Kronos price forecast
    5. Bayesian update with news sentiment
    6. Light anchor toward market (bid/ask indicates conviction)

Output: calibrated win probability for Kelly sizing and trade scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml

from lib.audit import log_event

CONFIG_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"


def _load_strategy() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@dataclass
class StockForecast:
    """Result of Bayesian stock forecasting."""
    ticker: str
    win_probability: float              # Final Bayesian probability (0.0 - 1.0)
    confidence: float                   # How sure we are in the estimate
    sources: dict[str, float] = field(default_factory=dict)  # Individual signal probs
    weights: dict[str, float] = field(default_factory=dict)  # Source weights
    bayesian_chain: list[dict] = field(default_factory=list) # Audit trail of updates
    evidence_summary: str = ""
    expected_return: float = 0.0
    recommended: bool = False
    reason: str = ""


def bayesian_update(prior: float, likelihood: float, base_rate: float = 0.5) -> float:
    """
    Apply Bayes' theorem to update a probability.

    P(H|E) = P(E|H) * P(H) / P(E)

    Args:
        prior: Current probability estimate (0.0 - 1.0)
        likelihood: How likely the evidence if hypothesis is true (0.0 - 1.0)
        base_rate: How common the evidence is in general (0.0 - 1.0)

    Returns:
        Updated probability, clamped to [0.05, 0.95].
    """
    if prior <= 0 or prior >= 1:
        return prior

    # Likelihood ratio: how much more likely is E given H than given ~H
    evidence_rate = likelihood * prior + base_rate * (1 - prior)
    if evidence_rate <= 0:
        return prior

    posterior = (likelihood * prior) / evidence_rate
    return max(0.05, min(0.95, posterior))


def _pattern_to_likelihood(pattern: str | None, signal_score: int) -> float:
    """
    Map candlestick pattern to likelihood of success.

    Strong patterns (hammer at support, bullish engulfing) have higher
    probability of working than weak ones.
    """
    if not pattern:
        return 0.50

    # Empirical win rates for bullish patterns (will tune with calibration)
    pattern_likelihoods = {
        "bullish_engulfing": 0.65,
        "morning_star": 0.68,
        "hammer": 0.62,
        "dragonfly_doji": 0.58,
        "bullish_harami": 0.55,
        "tweezers_bottom": 0.60,
        "pin_bar": 0.63,
    }

    base = pattern_likelihoods.get(pattern, 0.55)
    # Strength modifier: signal_score is 0-3
    if signal_score >= 3:
        return min(0.80, base + 0.10)
    elif signal_score == 2:
        return base
    elif signal_score == 1:
        return max(0.45, base - 0.08)
    else:
        return 0.50


def _momentum_to_likelihood(momentum_score: int) -> float:
    """Map momentum score (0-4) to likelihood."""
    # 0/4: weak → more likely to fail (0.40)
    # 1/4: 0.48
    # 2/4: 0.55 (neutral)
    # 3/4: 0.62
    # 4/4: strong → more likely to succeed (0.70)
    return 0.40 + (momentum_score / 4.0) * 0.30


def _kronos_to_likelihood(kronos_expected_return: float | None, confidence: float = 0.5) -> float:
    """Map Kronos expected return to likelihood of trade success."""
    if kronos_expected_return is None:
        return 0.50

    # Blend Kronos confidence — low confidence pulls toward 0.50
    # Expected return +5% → likelihood 0.65, confidence 1.0
    # Expected return -5% → likelihood 0.35, confidence 1.0
    raw = 0.50 + (kronos_expected_return * 3.0)  # Amplify the signal
    raw = max(0.20, min(0.80, raw))

    # Apply confidence weighting
    return 0.50 + (raw - 0.50) * confidence


def _news_to_likelihood(news_sentiment: float | None, news_confidence: float = 0.5) -> float:
    """Map news sentiment (0-1) to likelihood."""
    if news_sentiment is None:
        return 0.50

    # News 0.0 (bearish) → 0.30 likelihood
    # News 0.5 (neutral) → 0.50
    # News 1.0 (bullish) → 0.70
    raw = 0.30 + (news_sentiment * 0.40)

    # Apply confidence weighting
    return 0.50 + (raw - 0.50) * news_confidence


def _insider_to_likelihood(
    insider_sentiment: float | None,
    insider_confidence: float = 0.0,
) -> float:
    """
    Map insider-flow signal (0-1) to likelihood of bullish outcome.

    Empirical basis: Cohen/Malloy/Pomorski (2012) + Jeng/Metrick/Zeckhauser
    (2003) — insider cluster buys predict 5.5-7% CAPM-adjusted abnormal
    returns over 6 months. We scale accordingly but apply the bot's
    native confidence so that low-confidence signals don't drift prior
    away from 0.50.

    Asymmetric by design: sells (captured by insider_flow as neutral)
    don't trigger negative bias because 10b5-1 plans, diversification,
    and tax selling are noisy. Only BUY signals move the needle.
    """
    if insider_sentiment is None:
        return 0.50

    # Sentiment scale (from insider_flow.py):
    #   0.85 = bullish_cluster (3+ insiders in 30d)
    #   0.65 = bullish (single material buy)
    #   0.50 = neutral
    # Map to likelihood, then weight by confidence so a weak signal
    # (confidence ~ 0.1) barely perturbs the prior.
    raw = 0.30 + (insider_sentiment * 0.40)
    return 0.50 + (raw - 0.50) * insider_confidence


def _congress_to_likelihood(
    congress_sentiment: float | None,
    congress_confidence: float = 0.0,
) -> float:
    """
    Map Senate-flow signal (0-1) to likelihood of bullish outcome.

    Empirical basis: Ziobrowski et al. (JFQA 2004) — senator stock
    portfolios outperformed by ~12% annually 1993-1998. Effect narrowed
    post-STOCK Act but remains positive for cluster buys. congress_flow
    already applies a stale-data suppressor (forces sentiment=0.5,
    confidence~0.05 when feed is older than 60 days), so we can trust
    the confidence field.
    """
    if congress_sentiment is None:
        return 0.50

    # Same shape as insider but slightly softer (noisier data source):
    raw = 0.32 + (congress_sentiment * 0.36)
    return 0.50 + (raw - 0.50) * congress_confidence


def _trend_to_likelihood(trend_score: int, weekly_direction: str) -> float:
    """Trend alignment likelihood."""
    # Downtrend strongly disfavors bullish trade
    if weekly_direction == "downtrend":
        return 0.30

    # Bull alignment (0-3 score)
    return 0.45 + (trend_score / 3.0) * 0.25


def _level_to_likelihood(level_score: int, zone_touches: int) -> float:
    """Support zone proximity likelihood."""
    if level_score == 0:
        return 0.45

    base = 0.50 + (level_score / 3.0) * 0.15

    # Bonus for well-tested zones
    if zone_touches >= 3:
        base += 0.05

    return min(0.75, base)


def forecast_stock(
    ticker: str,
    composite_score: int,
    trend_score: int,
    level_score: int,
    signal_score: int,
    momentum_score: int,
    pattern: str | None = None,
    zone_touches: int = 0,
    weekly_direction: str = "sideways",
    kronos_expected_return: float | None = None,
    kronos_confidence: float = 0.5,
    news_sentiment: float | None = None,
    news_confidence: float = 0.5,
    insider_sentiment: float | None = None,
    insider_confidence: float = 0.0,
    congress_sentiment: float | None = None,
    congress_confidence: float = 0.0,
    base_rate: float = 0.55,
) -> StockForecast:
    """
    Produce a Bayesian-updated win probability for a stock trade setup.

    Args:
        ticker: Stock symbol
        composite_score: 0-13 composite from screener
        trend_score: 0-3 trend alignment
        level_score: 0-3 support zone quality
        signal_score: 0-3 candlestick signal strength
        momentum_score: 0-4 momentum score
        pattern: Candlestick pattern name (e.g., "bullish_engulfing")
        zone_touches: Number of times the support zone has been tested
        weekly_direction: "uptrend", "downtrend", or "sideways"
        kronos_expected_return: Kronos AI forecast (e.g., +0.05 = +5%)
        kronos_confidence: Kronos confidence 0-1
        news_sentiment: News sentiment 0-1 (0=bearish, 1=bullish)
        news_confidence: News signal confidence 0-1
        insider_sentiment: Form-4 insider-buy signal 0-1 (from lib.insider_flow)
        insider_confidence: Strength of insider signal 0-1
        congress_sentiment: Senate PTR signal 0-1 (from lib.congress_flow)
        congress_confidence: Strength of congress signal 0-1
        base_rate: Base rate prior for bullish stock setups (default 55%)

    Returns:
        StockForecast with win_probability, confidence, and evidence chain.
    """
    chain = []
    sources = {}

    # Start with base rate
    prior = base_rate
    chain.append({"step": "prior", "prob": round(prior, 4), "note": f"base_rate={base_rate}"})

    # Update with trend
    trend_lh = _trend_to_likelihood(trend_score, weekly_direction)
    sources["trend"] = round(trend_lh, 4)
    prior = bayesian_update(prior, trend_lh, base_rate=0.50)
    chain.append({"step": "trend", "likelihood": round(trend_lh, 4), "posterior": round(prior, 4)})

    # Update with level
    level_lh = _level_to_likelihood(level_score, zone_touches)
    sources["level"] = round(level_lh, 4)
    prior = bayesian_update(prior, level_lh, base_rate=0.50)
    chain.append({"step": "level", "likelihood": round(level_lh, 4), "posterior": round(prior, 4)})

    # Update with pattern
    pattern_lh = _pattern_to_likelihood(pattern, signal_score)
    sources["pattern"] = round(pattern_lh, 4)
    prior = bayesian_update(prior, pattern_lh, base_rate=0.50)
    chain.append({"step": "pattern", "pattern": pattern, "likelihood": round(pattern_lh, 4), "posterior": round(prior, 4)})

    # Update with momentum
    mom_lh = _momentum_to_likelihood(momentum_score)
    sources["momentum"] = round(mom_lh, 4)
    prior = bayesian_update(prior, mom_lh, base_rate=0.50)
    chain.append({"step": "momentum", "likelihood": round(mom_lh, 4), "posterior": round(prior, 4)})

    # Update with Kronos AI
    kronos_lh = _kronos_to_likelihood(kronos_expected_return, kronos_confidence)
    sources["kronos"] = round(kronos_lh, 4)
    prior = bayesian_update(prior, kronos_lh, base_rate=0.50)
    chain.append({"step": "kronos", "likelihood": round(kronos_lh, 4), "posterior": round(prior, 4)})

    # Update with insider flow (Form 4 clusters — leading indicator,
    # updated before news hits)
    insider_lh = _insider_to_likelihood(insider_sentiment, insider_confidence)
    sources["insider"] = round(insider_lh, 4)
    prior = bayesian_update(prior, insider_lh, base_rate=0.50)
    chain.append({"step": "insider", "likelihood": round(insider_lh, 4), "posterior": round(prior, 4)})

    # Update with Senate flow (STOCK Act PTRs — same informed-flow premise
    # as insider but at a different layer; low confidence on stale feed)
    congress_lh = _congress_to_likelihood(congress_sentiment, congress_confidence)
    sources["congress"] = round(congress_lh, 4)
    prior = bayesian_update(prior, congress_lh, base_rate=0.50)
    chain.append({"step": "congress", "likelihood": round(congress_lh, 4), "posterior": round(prior, 4)})

    # Update with news
    news_lh = _news_to_likelihood(news_sentiment, news_confidence)
    sources["news"] = round(news_lh, 4)
    prior = bayesian_update(prior, news_lh, base_rate=0.50)
    chain.append({"step": "news", "likelihood": round(news_lh, 4), "posterior": round(prior, 4)})

    win_prob = prior

    # Confidence: based on how many signals strongly agreed (all above 0.55 or all below 0.45)
    signal_strengths = [abs(v - 0.50) for v in sources.values()]
    avg_strength = sum(signal_strengths) / len(signal_strengths) if signal_strengths else 0
    confidence = min(1.0, avg_strength * 3.0)  # Scale 0-0.33 → 0-1.0

    # Build evidence summary
    summary_parts = []
    if trend_score >= 2: summary_parts.append(f"strong trend ({trend_score}/3)")
    if level_score >= 2: summary_parts.append(f"strong support ({level_score}/3, {zone_touches} touches)")
    if pattern: summary_parts.append(f"{pattern} pattern")
    if momentum_score >= 3: summary_parts.append(f"hot momentum ({momentum_score}/4)")
    if kronos_expected_return is not None and abs(kronos_expected_return) > 0.02:
        emoji = "↗" if kronos_expected_return > 0 else "↘"
        summary_parts.append(f"Kronos {emoji}{kronos_expected_return:+.1%}")
    if news_sentiment is not None and abs(news_sentiment - 0.5) > 0.15:
        tag = "bullish" if news_sentiment > 0.5 else "bearish"
        summary_parts.append(f"{tag} news")
    if (insider_sentiment is not None and insider_confidence >= 0.30
            and insider_sentiment > 0.60):
        summary_parts.append(f"insider buying ({insider_confidence:.0%} conf)")
    if (congress_sentiment is not None and congress_confidence >= 0.30
            and congress_sentiment > 0.60):
        summary_parts.append(f"Senate buying ({congress_confidence:.0%} conf)")

    summary = "; ".join(summary_parts) if summary_parts else "mixed signals"

    # Recommendation
    strategy = _load_strategy()
    min_win_prob = strategy.get("stock_params", {}).get("bayesian_min_win_prob", 0.58)
    recommended = win_prob >= min_win_prob

    forecast = StockForecast(
        ticker=ticker,
        win_probability=round(win_prob, 4),
        confidence=round(confidence, 4),
        sources=sources,
        weights={},  # Equal weighting in this simple Bayesian update
        bayesian_chain=chain,
        evidence_summary=summary,
        recommended=recommended,
        reason=f"win_prob {win_prob:.0%} {'≥' if recommended else '<'} threshold {min_win_prob:.0%}",
    )

    log_event("bayesian_forecaster", "forecast_complete", {
        "ticker": ticker,
        "win_probability": forecast.win_probability,
        "confidence": forecast.confidence,
        "recommended": forecast.recommended,
    })

    return forecast
