"""
Bear Agent — adversarial stress-test before trade execution.

Inspired by the bull/bear researcher debate pattern in TauricResearch's
TradingAgents (https://github.com/TauricResearch/TradingAgents). Adapted
to our flat consensus architecture: deterministic signal-based scoring
instead of LLM debate, so it adds zero latency and zero supply-chain
risk.

Role: take the same candidate the strategy agent proposed, and score
how strong the BEAR case is. If the bear case is strong (≥ 6/10), the
trade is vetoed. Moderate bear (3-5/10) downsizes Kelly to half. Weak
bear (≤ 2/10) passes through.

Signals examined (each candidate ALREADY carries these fields from the
existing scoring pipeline — bear agent doesn't fetch new data):

  • composite_score below a configurable floor
  • current regime is "bear"
  • Kronos forecast is bearish
  • news sentiment is materially negative
  • Bayesian win probability is below 50%
  • earnings event imminent (within 7 days)
  • correlation_penalty is severe (< 0.7)
  • candlestick pattern is in the bearish set

Each signal contributes 1 point (some 2). Final score 0-10. Output:

    {
        "score": int,                    # 0-10
        "signals": [{"name", "weight", "evidence"}],
        "action": "PASS" | "DOWNSIZE" | "VETO",
        "size_multiplier": float,        # 1.0 for PASS, 0.5 for DOWNSIZE, 0.0 for VETO
        "reasoning": str,
    }

Wired into agents/consensus.seek_consensus AFTER compliance, BEFORE the
LLM analyst — cheaper than the LLM, can short-circuit when the bear
case is overwhelming. Also called directly from stock_engine and
crypto_engine entry paths since those don't go through seek_consensus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lib.audit import log_event
from lib.memory_palace import diary_write

# Candlestick patterns that indicate bearish reversal/continuation.
# Keep this in sync with lib/candlestick.py's pattern names.
BEARISH_PATTERNS = {
    "bearish_engulfing",
    "evening_star",
    "shooting_star",
    "hanging_man",
    "dark_cloud_cover",
    "tweezers_top",
    "gravestone_doji",
}


@dataclass
class BearSignal:
    """One bearish indicator hit."""
    name: str
    weight: int
    evidence: str


def _coerce(value: Any, default: float = 0.0) -> float:
    """Tolerant float coercion — bears never crash on a None/string field."""
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


class BearAgent:
    """
    Argues AGAINST the proposed trade. Cannot execute. Output drives a
    sizing multiplier and optional veto applied upstream.

    Thresholds (tunable later via Hermes):
      score 0-2  → PASS    (size_multiplier = 1.0)
      score 3-5  → DOWNSIZE (size_multiplier = 0.5)
      score ≥ 6  → VETO    (size_multiplier = 0.0)
    """

    name = "bear_agent"

    DOWNSIZE_THRESHOLD = 3
    VETO_THRESHOLD = 6

    # Score floor below which composite_score itself counts as bearish
    # (independent of the strategy's own min_composite_score gate).
    LOW_SCORE_FLOOR = 5  # of /9 for options or /13 for stocks; matches both ranges

    def review(self, candidate: Any, regime: str = "unknown") -> dict:
        """Score the bear case for a single candidate.

        Args:
            candidate: dict or object with the standard scoring fields
                (composite_score, kronos_direction, news_sentiment,
                bayesian_win_prob, earnings_days, correlation_penalty,
                candlestick_pattern / pattern). Missing fields are treated
                as neutral (0 points).
            regime: current market regime — "bull" / "bear" / "sideways" /
                "unknown". Pass from memory_palace.get_current_regime().

        Returns:
            Decision dict (see module docstring).
        """
        signals: list[BearSignal] = []

        composite = _coerce(_extract(candidate, "composite_score"))
        if composite > 0 and composite < self.LOW_SCORE_FLOOR:
            signals.append(BearSignal(
                "low_composite_score",
                weight=1,
                evidence=f"composite={composite} < floor {self.LOW_SCORE_FLOOR}",
            ))

        if str(regime).lower() == "bear":
            signals.append(BearSignal(
                "bear_regime",
                weight=2,
                evidence=f"market regime is bear",
            ))

        kronos_dir = str(_extract(candidate, "kronos_direction") or "").lower()
        kronos_ret = _coerce(_extract(candidate, "kronos_expected_return"))
        if kronos_dir == "bearish" or kronos_ret < -0.02:
            signals.append(BearSignal(
                "kronos_bearish",
                weight=1,
                evidence=f"kronos_direction={kronos_dir or 'n/a'}, "
                         f"expected_return={kronos_ret:+.2%}",
            ))

        news_sent = _coerce(_extract(candidate, "news_sentiment"))
        if news_sent < -0.3:
            signals.append(BearSignal(
                "negative_news",
                weight=1,
                evidence=f"news_sentiment={news_sent:.2f} (< -0.30)",
            ))

        bayes = _extract(candidate, "bayesian_win_prob")
        if bayes is not None and _coerce(bayes) < 0.50:
            signals.append(BearSignal(
                "low_bayesian_win_prob",
                weight=2,
                evidence=f"bayesian_win_prob={_coerce(bayes):.2f} (< 0.50)",
            ))

        earnings_days = _extract(candidate, "earnings_days")
        if earnings_days is not None:
            d = _coerce(earnings_days, default=999)
            if 0 <= d <= 7:
                signals.append(BearSignal(
                    "earnings_imminent",
                    weight=1,
                    evidence=f"earnings in {int(d)} days",
                ))

        corr_pen = _coerce(_extract(candidate, "correlation_penalty"), default=1.0)
        if corr_pen < 0.7:
            signals.append(BearSignal(
                "correlated_with_held",
                weight=1,
                evidence=f"correlation_penalty={corr_pen:.2f} (< 0.70)",
            ))

        pattern = str(_extract(candidate, "candlestick_pattern", "pattern") or "").lower()
        if pattern in BEARISH_PATTERNS:
            signals.append(BearSignal(
                "bearish_candle",
                weight=1,
                evidence=f"pattern={pattern}",
            ))

        # Learning-loop signal: prior outcomes on this ticker. If we've
        # lost on >= 60% of the last 5+ resolved trades, weight the bear
        # case higher. Cheap deterministic read against MemPalace KG;
        # no LLM call. Pattern from TradingAgents v0.2.4 — "agents learn
        # from past trades" realized as a numeric signal here.
        ticker_for_history = _extract(candidate, "ticker")
        if ticker_for_history:
            try:
                from lib.memory_palace import prior_loss_rate
                losses, total, rate = prior_loss_rate(str(ticker_for_history),
                                                     lookback_n=10)
                if total >= 5 and rate >= 0.60:
                    signals.append(BearSignal(
                        "prior_loss_history",
                        weight=1,
                        evidence=f"{losses}/{total} recent trades lost "
                                 f"({rate:.0%}) — pattern likely repeats",
                    ))
            except Exception:
                pass  # never block on memory lookup failure

        score = sum(s.weight for s in signals)
        score = min(score, 10)  # clamp

        if score >= self.VETO_THRESHOLD:
            action = "VETO"
            multiplier = 0.0
        elif score >= self.DOWNSIZE_THRESHOLD:
            action = "DOWNSIZE"
            multiplier = 0.5
        else:
            action = "PASS"
            multiplier = 1.0

        ticker = _extract(candidate, "ticker") or "?"
        reasoning = (
            f"{ticker} bear_score={score}/10 → {action}: "
            + ", ".join(s.name for s in signals)
            if signals else
            f"{ticker} bear_score=0/10 → PASS: no bearish signals"
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

        log_event("bear_agent", "review", {
            "ticker": ticker,
            "score": score,
            "action": action,
            "signal_names": [s.name for s in signals],
        })

        if action != "PASS":
            diary_write(self.name,
                f"{ticker}|BEAR_{action}|score_{score}/10|"
                f"{','.join(s.name for s in signals)[:120]}")

        return result
