"""Fundamentals Analyst — orthogonal vote from financial-statement quality.

Inspired by the TradingAgents (TauricResearch/TradingAgents) Analyst Team
pattern, adapted to our rule-based agent contract. Fundamentals is the
missing pillar in our confluence stack — Turtle / Markov / Bayesian /
PEAD / Bull-Bear all read price or news. Nothing reads the balance
sheet. This agent does, using metrics we already pull via Finnhub.

WHY THIS HELPS WIN RATE
  Bars-only signals have an empirical ceiling around 50% WR because
  they're all reading the same underlying tape. Fundamentals are an
  INDEPENDENT axis — a company with negative ROE and rising debt-to-
  equity is structurally riskier than one with strong margins and
  improving free-cash-flow, regardless of what its price chart looks
  like. Adding fundamentals as a confluence vote gives the bot a real
  reason to skip a chart that "looks great" but rests on bad financials.

CONTRACT (matches BullAgent / BearAgent shape)
  review(candidate, regime="unknown") -> {
      "agent": "fundamentals_agent",
      "score": int 0-10,
      "verdict": "STRONG" | "NEUTRAL" | "WEAK",
      "reasons": list[str],
      "details": dict,
  }

SCORING (additive 0-10, capped at 10)
  +2  ROE TTM ≥ 15%   (strong returns on equity)
  +1  ROE TTM ≥ 10%   (decent returns)
  -2  ROE TTM < 0     (negative — losing money on equity)
  +1  PE TTM 5-25     (reasonable valuation, neither cheap-trap nor bubble)
  -1  PE TTM > 60     (priced for perfection — bad asymmetry)
  -2  PE TTM < 0      (no earnings — risky)
  +2  debt/equity < 0.5  (low leverage)
  +1  debt/equity < 1.0  (manageable leverage)
  -2  debt/equity > 2.0  (highly leveraged — fragile)
  +1  current_ratio ≥ 1.5  (strong short-term liquidity)
  -1  current_ratio < 1.0  (can't cover short-term obligations)
  +1  beta < 1.0      (less volatile than market)

  Final action thresholds match BullAgent's NEUTRAL_THRESHOLD/BOOST_THRESHOLD:
    score ≥ 7 → STRONG  (fundamentals back the trade)
    score 4-6 → NEUTRAL (no strong opinion)
    score ≤ 3 → WEAK    (fundamentals don't support; veto in confluence)

NOTES
  * Returns NEUTRAL on Finnhub miss/error rather than vetoing — we don't
    want a flaky API to block trades. Use confluence to demand multiple
    positive signals, not this agent alone.
  * Cached at the Finnhub layer (24h TTL) so a full scan only pays the
    API cost once per ticker per day.
  * No async / no LLM calls — pure rule-based, sub-millisecond per call
    after cache warm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FundamentalSignal:
    name: str
    weight: int   # can be negative for bearish signals
    evidence: str


def _coerce(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion (matches bull_agent._coerce)."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class FundamentalsAgent:
    """Score the fundamentals quality of a single ticker."""

    name = "fundamentals_agent"

    STRONG_THRESHOLD = 7
    NEUTRAL_THRESHOLD = 4
    SCORE_CEILING = 10
    SCORE_FLOOR = 0

    def __init__(self, finnhub_getter=None):
        # Dependency injection lets tests swap in a stub Finnhub
        if finnhub_getter is None:
            try:
                from lib.finnhub_client import get_basic_financials
                self._get_financials = get_basic_financials
            except Exception:
                self._get_financials = lambda _: None
        else:
            self._get_financials = finnhub_getter

    def review(self, candidate: Any, regime: str = "unknown") -> dict:
        """Score fundamentals for a single candidate.

        The candidate dict's only required field is ``ticker``. We don't
        read any technicals here — fundamentals is meant to be an
        ORTHOGONAL vote.
        """
        ticker = self._extract(candidate, "ticker") or "?"

        signals: list[FundamentalSignal] = []
        details: dict = {}

        financials = None
        try:
            financials = self._get_financials(ticker)
        except Exception as e:
            return self._neutral_result(
                ticker, reasons=[f"finnhub_error: {str(e)[:80]}"], details={}
            )

        if not financials:
            return self._neutral_result(
                ticker, reasons=["no_financial_data"], details={}
            )

        details = financials

        # --- Profitability: ROE ---
        roe = _coerce(financials.get("roe_ttm"))
        if roe >= 15:
            signals.append(FundamentalSignal(
                "strong_roe", weight=2,
                evidence=f"ROE TTM {roe:.1f}% ≥ 15%",
            ))
        elif roe >= 10:
            signals.append(FundamentalSignal(
                "decent_roe", weight=1,
                evidence=f"ROE TTM {roe:.1f}% ≥ 10%",
            ))
        elif roe < 0:
            signals.append(FundamentalSignal(
                "negative_roe", weight=-2,
                evidence=f"ROE TTM {roe:.1f}% — losing money on equity",
            ))

        # --- Valuation: PE ratio ---
        pe = _coerce(financials.get("pe_ttm"))
        if 5 <= pe <= 25:
            signals.append(FundamentalSignal(
                "reasonable_pe", weight=1,
                evidence=f"PE TTM {pe:.1f} (5-25 sweet spot)",
            ))
        elif pe > 60:
            signals.append(FundamentalSignal(
                "high_pe", weight=-1,
                evidence=f"PE TTM {pe:.1f} > 60 — priced for perfection",
            ))
        elif pe < 0:
            signals.append(FundamentalSignal(
                "negative_earnings", weight=-2,
                evidence=f"PE TTM {pe:.1f} — no earnings",
            ))

        # --- Leverage: debt / equity ---
        de = _coerce(financials.get("debt_to_equity"))
        if 0 < de < 0.5:
            signals.append(FundamentalSignal(
                "low_leverage", weight=2,
                evidence=f"D/E {de:.2f} < 0.5",
            ))
        elif 0.5 <= de < 1.0:
            signals.append(FundamentalSignal(
                "ok_leverage", weight=1,
                evidence=f"D/E {de:.2f} < 1.0",
            ))
        elif de >= 2.0:
            signals.append(FundamentalSignal(
                "high_leverage", weight=-2,
                evidence=f"D/E {de:.2f} ≥ 2.0 — fragile balance sheet",
            ))

        # --- Liquidity: current ratio ---
        cr = _coerce(financials.get("current_ratio"))
        if cr >= 1.5:
            signals.append(FundamentalSignal(
                "strong_liquidity", weight=1,
                evidence=f"current ratio {cr:.2f} ≥ 1.5",
            ))
        elif 0 < cr < 1.0:
            signals.append(FundamentalSignal(
                "weak_liquidity", weight=-1,
                evidence=f"current ratio {cr:.2f} < 1.0",
            ))

        # --- Volatility: beta ---
        beta = _coerce(financials.get("beta"))
        if 0 < beta < 1.0:
            signals.append(FundamentalSignal(
                "low_beta", weight=1,
                evidence=f"beta {beta:.2f} < 1.0",
            ))

        # Aggregate
        raw_score = sum(s.weight for s in signals)
        # Clip to the [0, 10] range to match the BullAgent/BearAgent
        # contract — negative scores don't make sense for confluence.
        score = max(self.SCORE_FLOOR, min(raw_score, self.SCORE_CEILING))

        if score >= self.STRONG_THRESHOLD:
            verdict = "STRONG"
        elif score >= self.NEUTRAL_THRESHOLD:
            verdict = "NEUTRAL"
        else:
            verdict = "WEAK"

        reasons = [s.name for s in signals]
        return {
            "agent": self.name,
            "ticker": ticker,
            "score": int(score),
            "raw_score": int(raw_score),
            "verdict": verdict,
            "reasons": reasons,
            "evidence": [s.evidence for s in signals],
            "details": details,
        }

    def _neutral_result(self, ticker: str, reasons: list[str], details: dict) -> dict:
        """Standard NEUTRAL response when we can't read fundamentals.

        We don't VETO on missing data — that would silently block trades
        for any ticker Finnhub doesn't cover. Confluence callers can
        choose to treat NEUTRAL as a no-vote (which is the current
        confluence_filter behavior for missing signals).
        """
        return {
            "agent": self.name,
            "ticker": ticker,
            "score": 5,             # neutral midpoint
            "raw_score": 0,
            "verdict": "NEUTRAL",
            "reasons": reasons,
            "evidence": [],
            "details": details,
        }

    @staticmethod
    def _extract(candidate: Any, *keys: str) -> Any:
        """Mirror of bull_agent._extract — handle dict OR object access."""
        for key in keys:
            if isinstance(candidate, dict):
                v = candidate.get(key)
            else:
                v = getattr(candidate, key, None)
            if v is not None:
                return v
        return None


__all__ = ["FundamentalsAgent", "FundamentalSignal"]
