"""
Bull Agent — counterpart to BearAgent, scoring the bullish case.

Inspired by the Bullish/Bearish researcher pair in TauricResearch's
TradingAgents. Where their paper uses LLM debate, we use deterministic
signal scoring on both sides — no LLM cost, no per-trade dollar bleed.

Role: argue FOR the trade. Identify specific bullish signals on the
same candidate fields the strategy + bear agent saw. The output is a
confidence score 0-10 used to:

  • Upsize Kelly when bull is significantly stronger than bear
  • Annotate the audit trail with both sides of the debate

Bull cannot veto and cannot override the bear. If bear downsizes or
vetoes, bull's "BOOST" is informational only — risk is asymmetric and
we always defer to the bearish signal when the two disagree.

Scoring (each 1-2 points, total clamped to 10):

  composite_score ≥ 8 ........................ +1   strong upstream conviction
  regime == "bull" ........................... +2   tailwind from market regime
  kronos_direction == "bullish" OR ER > 2% ... +1   forecasting agrees
  news_sentiment > 0.30 ...................... +1   positive macro / company news
  bayesian_win_prob > 0.65 ................... +2   high model confidence
  earnings_days > 30 ......................... +1   no imminent gap risk
  correlation_penalty == 1.0 ................. +1   no portfolio clustering drag
  candlestick_pattern in BULLISH_PATTERNS .... +1   technical confirmation

Output:

    {
        "score": int,                    # 0-10
        "signals": [{"name", "weight", "evidence"}],
        "action": "BOOST" | "NEUTRAL" | "WEAK",
        "size_multiplier": float,        # 1.25 | 1.0 | 1.0  (no downsize from bull)
        "reasoning": str,
    }

The DOWNSIZE/VETO direction lives entirely in BearAgent — bull stays in
the "neutral or upsize" lane to preserve asymmetric risk discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lib.audit import log_event
from lib.memory_palace import diary_write

# Candlestick patterns that indicate bullish reversal/continuation.
# Mirror of agents.bear_agent.BEARISH_PATTERNS.
BULLISH_PATTERNS = {
    "bullish_engulfing",
    "morning_star",
    "hammer",
    "inverted_hammer",
    "piercing_line",
    "tweezers_bottom",
    "dragonfly_doji",
    "three_white_soldiers",
}


@dataclass
class BullSignal:
    """One bullish indicator hit."""
    name: str
    weight: int
    evidence: str


def _coerce(value: Any, default: float = 0.0) -> float:
    """Tolerant float coercion — never crash on a None/string field."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract(candidate: Any, *keys: str, default: Any = None) -> Any:
    """Read a field from either a dict or a dataclass-like object."""
    if isinstance(candidate, dict):
        for k in keys:
            if k in candidate and candidate[k] is not None:
                return candidate[k]
        return default
    for k in keys:
        v = getattr(candidate, k, None)
        if v is not None:
            return v
    return default


class BullAgent:
    """
    Argues FOR the proposed trade. Cannot veto. Output is informational
    + a possible upsize multiplier if the bullish case is overwhelming
    AND bear didn't itself downsize.

    Thresholds:
      score 0-3  → WEAK    (size_multiplier 1.0, no boost)
      score 4-6  → NEUTRAL (size_multiplier 1.0)
      score ≥ 7  → BOOST   (size_multiplier 1.25, capped by max_position)
    """

    name = "bull_agent"

    BOOST_THRESHOLD = 7
    NEUTRAL_THRESHOLD = 4

    BOOST_MULTIPLIER = 1.25

    def review(self, candidate: Any, regime: str = "unknown") -> dict:
        """Score the bull case for a single candidate."""
        signals: list[BullSignal] = []

        composite = _coerce(_extract(candidate, "composite_score"))
        if composite >= 8:
            signals.append(BullSignal(
                "strong_composite_score",
                weight=1,
                evidence=f"composite={composite} ≥ 8",
            ))

        if str(regime).lower() == "bull":
            signals.append(BullSignal(
                "bull_regime",
                weight=2,
                evidence="market regime is bull",
            ))

        kronos_dir = str(_extract(candidate, "kronos_direction") or "").lower()
        kronos_ret = _coerce(_extract(candidate, "kronos_expected_return"))
        if kronos_dir == "bullish" or kronos_ret > 0.02:
            signals.append(BullSignal(
                "kronos_bullish",
                weight=1,
                evidence=f"kronos_direction={kronos_dir or 'n/a'}, "
                         f"expected_return={kronos_ret:+.2%}",
            ))

        news_sent = _coerce(_extract(candidate, "news_sentiment"))
        if news_sent > 0.3:
            signals.append(BullSignal(
                "positive_news",
                weight=1,
                evidence=f"news_sentiment={news_sent:.2f} (> 0.30)",
            ))

        bayes = _extract(candidate, "bayesian_win_prob")
        if bayes is not None and _coerce(bayes) > 0.65:
            signals.append(BullSignal(
                "high_bayesian_win_prob",
                weight=2,
                evidence=f"bayesian_win_prob={_coerce(bayes):.2f} (> 0.65)",
            ))

        earnings_days = _extract(candidate, "earnings_days")
        if earnings_days is not None:
            d = _coerce(earnings_days, default=0)
            if d > 30:
                signals.append(BullSignal(
                    "earnings_far",
                    weight=1,
                    evidence=f"earnings in {int(d)} days (no imminent risk)",
                ))

        # Only count "no correlation drag" when the field was EXPLICITLY set
        # by upstream gates — a missing field means we don't know, not "good".
        corr_pen_raw = _extract(candidate, "correlation_penalty")
        if corr_pen_raw is not None and _coerce(corr_pen_raw) >= 0.99:
            signals.append(BullSignal(
                "no_correlation_drag",
                weight=1,
                evidence=f"correlation_penalty={_coerce(corr_pen_raw):.2f} (no clustering)",
            ))

        pattern = str(_extract(candidate, "candlestick_pattern", "pattern") or "").lower()
        if pattern in BULLISH_PATTERNS:
            signals.append(BullSignal(
                "bullish_candle",
                weight=1,
                evidence=f"pattern={pattern}",
            ))

        score = sum(s.weight for s in signals)
        score = min(score, 10)

        if score >= self.BOOST_THRESHOLD:
            action = "BOOST"
            multiplier = self.BOOST_MULTIPLIER
        elif score >= self.NEUTRAL_THRESHOLD:
            action = "NEUTRAL"
            multiplier = 1.0
        else:
            action = "WEAK"
            multiplier = 1.0

        ticker = _extract(candidate, "ticker") or "?"
        reasoning = (
            f"{ticker} bull_score={score}/10 → {action}: "
            + ", ".join(s.name for s in signals)
            if signals else
            f"{ticker} bull_score=0/10 → WEAK: no bullish signals"
        )

        result = {
            "agent": self.name,
            "score": score,
            "signals": [
                {"name": s.name, "weight": s.weight, "evidence": s.evidence}
                for s in signals
            ],
            "action": action,
            "size_multiplier": multiplier,
            "reasoning": reasoning,
        }

        log_event("bull_agent", "review", {
            "ticker": ticker,
            "score": score,
            "action": action,
            "signal_names": [s.name for s in signals],
        })

        if action == "BOOST":
            diary_write(self.name,
                f"{ticker}|BULL_BOOST|score_{score}/10|"
                f"{','.join(s.name for s in signals)[:120]}")

        return result


def combine_bull_bear(
    bull_review: dict,
    bear_review: dict,
    cap_multiplier: float = 1.25,
) -> dict:
    """
    Reconcile bull + bear into a single sizing multiplier.

    Asymmetric: bear's downsize/veto ALWAYS wins. Bull's boost only fires
    when bear is silent (PASS) AND bull's score significantly exceeds
    bear's. This preserves the "risk-first" discipline — we never let
    bull's enthusiasm override a bear veto.

    Returns:
        {
            "size_multiplier": float,    # final combined
            "decision": "VETO" | "DOWNSIZE" | "BOOST" | "PASS",
            "delta": int,                # bull_score - bear_score
            "reasoning": str,
        }
    """
    bear_action = bear_review.get("action", "PASS")
    bear_mult = float(bear_review.get("size_multiplier", 1.0))
    bull_score = int(bull_review.get("score", 0))
    bear_score = int(bear_review.get("score", 0))
    delta = bull_score - bear_score

    if bear_action == "VETO":
        return {
            "size_multiplier": 0.0,
            "decision": "VETO",
            "delta": delta,
            "reasoning": (
                f"bear VETO score={bear_score} (bull was {bull_score}) — "
                f"asymmetric risk discipline: bear always wins"
            ),
        }
    if bear_action == "DOWNSIZE":
        return {
            "size_multiplier": bear_mult,
            "decision": "DOWNSIZE",
            "delta": delta,
            "reasoning": (
                f"bear DOWNSIZE score={bear_score} (bull was {bull_score}) — "
                f"bull boost suppressed by bear caution"
            ),
        }

    # Bear is PASS — bull may boost if its case is materially stronger.
    if bull_review.get("action") == "BOOST" and delta >= 4:
        return {
            "size_multiplier": min(cap_multiplier,
                                   float(bull_review.get("size_multiplier", 1.0))),
            "decision": "BOOST",
            "delta": delta,
            "reasoning": (
                f"bull BOOST score={bull_score} vs bear {bear_score} "
                f"(delta {delta:+d} ≥ 4) — upsizing within cap"
            ),
        }

    return {
        "size_multiplier": 1.0,
        "decision": "PASS",
        "delta": delta,
        "reasoning": (
            f"bull {bull_score}/bear {bear_score} (delta {delta:+d}) → no adjustment"
        ),
    }
