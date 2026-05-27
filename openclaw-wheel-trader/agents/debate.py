"""5-round structured Bull/Bear debate orchestrator.

Pattern adapted from hopit-ai/india-trade-cli (Vibe Trading), which uses
a multi-round LLM debate to surface arguments that one-shot scorers miss.
Our agents are rule-based, not LLM-based, so the "rebuttal" rounds are
implemented by re-examining the candidate for counter-evidence to each
opponent reason rather than via LLM reasoning — but the structural
benefit (each side gets to RESPOND to the other) is preserved.

THE FIVE ROUNDS

  Round 1: Bull opening    — BullAgent.review()
  Round 2: Bear opening    — BearAgent.review()
  Round 3: Bull rebuttal   — re-examine candidate for bullish counters
                             to each bear reason
  Round 4: Bear rebuttal   — re-examine candidate for bearish counters
                             to each bull reason
  Round 5: Facilitator     — synthesize net score → final action

WHY THIS HELPS
  Our one-shot bull/bear are STATIC: each sees only the candidate, never
  the opponent's argument. A 5-round structure lets each side concede
  some points and amplify others when the opponent has surfaced specific
  weaknesses. Net effect: more conviction-tuned sizing on hard cases.

WHEN TO CALL
  Heavyweight enough that we don't want to run on every screen-stage
  candidate — only on FINALISTS (candidates that already passed Turtle +
  composite + confluence). Caller pattern:

      from agents.debate import run_debate
      transcript = run_debate(candidate, regime="bull")
      if transcript["action"] == "VETO":
          # skip this trade
      elif transcript["action"] == "DOWNSIZE":
          # halve position
      elif transcript["action"] == "BOOST":
          # use bull's boost multiplier
      else:  # NEUTRAL
          # normal sizing

CONTRACT (debate result)
  {
      "ticker": str,
      "rounds": {
          "1_bull_opening":   <bull.review result>,
          "2_bear_opening":   <bear.review result>,
          "3_bull_rebuttal":  {"counters": [str], "score_delta": int},
          "4_bear_rebuttal":  {"counters": [str], "score_delta": int},
      },
      "final": {
          "bull_score":   int,   # opening + rebuttal delta
          "bear_score":   int,
          "net_score":    int,   # bull - bear
          "action":       "VETO" | "DOWNSIZE" | "NEUTRAL" | "BOOST",
          "size_multiplier": float,
          "reasoning":    str,
      },
  }
"""
from __future__ import annotations

from typing import Any


# Rebuttal map: "if opponent cited THIS reason, look for THESE candidate
# fields/conditions to mount a counter". Each tuple is (counter_name,
# candidate-field, check-fn, weight). The check_fn receives the field
# value and returns True if the counter applies.
_BULL_REBUTTALS_TO_BEAR = [
    # Bear says low composite — Bull counters with high Bayesian win prob
    ("bear_low_composite_but_high_bayes", "low_composite_score",
     "bayesian_win_prob", lambda v: v is not None and float(v) >= 0.65, 2),

    # Bear says Kronos bearish — Bull counters with strong positive news
    ("bear_kronos_but_news_strong", "kronos_bearish",
     "news_sentiment", lambda v: v is not None and float(v) >= 0.4, 1),

    # Bear says negative news — Bull counters with strong Bayesian
    ("bear_news_but_bayes_strong", "negative_news",
     "bayesian_win_prob", lambda v: v is not None and float(v) >= 0.65, 1),

    # Bear says bearish candle — Bull counters with bull market regime
    ("bear_candle_but_bull_regime", "bearish_candle",
     "_regime_is_bull", lambda v: v is True, 1),

    # Bear says low Bayesian — Bull counters with strong composite (>= 8)
    ("bear_bayes_but_composite_strong", "low_bayesian_win_prob",
     "composite_score", lambda v: v is not None and float(v) >= 8, 1),

    # Bear says correlated_with_held — Bull counters with hot momentum
    ("bear_corr_but_momentum_hot", "correlated_with_held",
     "momentum_score", lambda v: v is not None and float(v) >= 3, 1),

    # Bear says gap down — Bull counters with very positive news
    ("bear_gap_but_news_strong", "overnight_gap_down",
     "news_sentiment", lambda v: v is not None and float(v) >= 0.5, 1),
]

_BEAR_REBUTTALS_TO_BULL = [
    # Bull says strong composite — Bear counters with imminent earnings
    ("bull_composite_but_earnings_close", "strong_composite_score",
     "earnings_days", lambda v: v is not None and 0 <= float(v) <= 7, 2),

    # Bull says positive news — Bear counters with imminent earnings
    # (news could reverse on print)
    ("bull_news_but_earnings_close", "positive_news",
     "earnings_days", lambda v: v is not None and 0 <= float(v) <= 7, 1),

    # Bull says Kronos bullish — Bear counters with correlated_with_held
    # (concentration risk overrides single-name forecast)
    ("bull_kronos_but_concentrated", "kronos_bullish",
     "correlation_penalty", lambda v: v is not None and float(v) < 0.7, 1),

    # Bull says high Bayesian — Bear counters with bear market regime
    # (model may not generalize to this regime)
    ("bull_bayes_but_bear_regime", "high_bayesian_win_prob",
     "_regime_is_bear", lambda v: v is True, 2),

    # Bull says bullish candle — Bear counters with strongly negative news
    ("bull_candle_but_news_bad", "bullish_candle",
     "news_sentiment", lambda v: v is not None and float(v) <= -0.4, 1),

    # Bull says gap up — Bear counters with imminent earnings (gap could
    # be front-running and reverse on the actual print)
    ("bull_gap_but_earnings_close", "overnight_gap_up",
     "earnings_days", lambda v: v is not None and 0 <= float(v) <= 7, 1),
]


def _coerce(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _extract(candidate: Any, key: str) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(key)
    return getattr(candidate, key, None)


def _build_rebuttal(
    opponent_review: dict,
    candidate: Any,
    regime: str,
    rebuttal_map: list[tuple],
) -> dict:
    """Apply the rebuttal map: for each opponent reason that matches,
    check whether the candidate provides counter-evidence; if so, accrue
    score delta and record the counter.

    BullAgent / BearAgent return a ``signals`` list of dicts
    ({name, weight, evidence}). We extract the set of signal names for
    matching against the rebuttal map. Fall back to ``reasons`` for any
    future agents that follow the simpler list-of-strings convention.
    """
    signals = opponent_review.get("signals") or []
    if signals and isinstance(signals[0], dict):
        opponent_reasons = {s.get("name", "") for s in signals}
    else:
        opponent_reasons = set(opponent_review.get("reasons", []))
    counters: list[str] = []
    evidence: list[str] = []
    score_delta = 0

    for counter_name, opp_reason, field, predicate, weight in rebuttal_map:
        if opp_reason not in opponent_reasons:
            continue

        # Special pseudo-fields handle regime checks
        if field == "_regime_is_bull":
            value = (str(regime).lower() == "bull")
        elif field == "_regime_is_bear":
            value = (str(regime).lower() == "bear")
        else:
            value = _extract(candidate, field)

        try:
            applies = bool(predicate(value))
        except (TypeError, ValueError):
            applies = False

        if applies:
            counters.append(counter_name)
            evidence.append(f"counter '{opp_reason}' with {field}={value!r}")
            score_delta += weight

    return {
        "counters": counters,
        "evidence": evidence,
        "score_delta": score_delta,
    }


def _synthesize(
    bull_review: dict,
    bear_review: dict,
    bull_rebut: dict,
    bear_rebut: dict,
) -> dict:
    """Round 5: facilitator synthesizes net score and recommended action.

    Action mapping (net = bull_final - bear_final):
      net <= -3  → VETO     (size_multiplier 0.0; bear convincingly won)
      -2 ≤ net ≤ -1 → DOWNSIZE (size_multiplier 0.5)
      0  ≤ net ≤ 2  → NEUTRAL  (size_multiplier 1.0)
      net ≥ 3   → BOOST    (size_multiplier 1.25; bull convincingly won)
    """
    bull_open = int(bull_review.get("score", 0))
    bear_open = int(bear_review.get("score", 0))
    bull_final = bull_open + int(bull_rebut.get("score_delta", 0))
    bear_final = bear_open + int(bear_rebut.get("score_delta", 0))
    net = bull_final - bear_final

    if net <= -3:
        action, mult = "VETO", 0.0
    elif net <= -1:
        action, mult = "DOWNSIZE", 0.5
    elif net >= 3:
        action, mult = "BOOST", 1.25
    else:
        action, mult = "NEUTRAL", 1.0

    reasoning_parts = [
        f"bull {bull_open}+{bull_rebut.get('score_delta', 0):+d}={bull_final}",
        f"bear {bear_open}+{bear_rebut.get('score_delta', 0):+d}={bear_final}",
        f"net={net:+d} → {action}",
    ]
    if bull_rebut.get("counters"):
        reasoning_parts.append("bull_counters: " + ", ".join(bull_rebut["counters"]))
    if bear_rebut.get("counters"):
        reasoning_parts.append("bear_counters: " + ", ".join(bear_rebut["counters"]))

    return {
        "bull_score": bull_final,
        "bear_score": bear_final,
        "net_score": net,
        "action": action,
        "size_multiplier": mult,
        "reasoning": " | ".join(reasoning_parts),
    }


def run_debate(
    candidate: Any,
    regime: str = "unknown",
    bull_cls=None,
    bear_cls=None,
) -> dict:
    """Run the full 5-round debate and return the structured transcript.

    Optional ``bull_cls`` / ``bear_cls`` args let tests inject stubs.
    Defaults import BullAgent / BearAgent from agents.
    """
    if bull_cls is None:
        from agents.bull_agent import BullAgent
        bull_cls = BullAgent
    if bear_cls is None:
        from agents.bear_agent import BearAgent
        bear_cls = BearAgent

    bull = bull_cls()
    bear = bear_cls()

    # Rounds 1-2: opening arguments
    bull_review = bull.review(candidate, regime=regime)
    bear_review = bear.review(candidate, regime=regime)

    # Rounds 3-4: rebuttals
    bull_rebuttal = _build_rebuttal(
        opponent_review=bear_review,
        candidate=candidate,
        regime=regime,
        rebuttal_map=_BULL_REBUTTALS_TO_BEAR,
    )
    bear_rebuttal = _build_rebuttal(
        opponent_review=bull_review,
        candidate=candidate,
        regime=regime,
        rebuttal_map=_BEAR_REBUTTALS_TO_BULL,
    )

    # Round 5: synthesis
    final = _synthesize(bull_review, bear_review, bull_rebuttal, bear_rebuttal)

    ticker = _extract(candidate, "ticker") or "?"
    return {
        "ticker": ticker,
        "regime": regime,
        "rounds": {
            "1_bull_opening":  bull_review,
            "2_bear_opening":  bear_review,
            "3_bull_rebuttal": bull_rebuttal,
            "4_bear_rebuttal": bear_rebuttal,
        },
        "final": final,
    }


__all__ = ["run_debate"]
