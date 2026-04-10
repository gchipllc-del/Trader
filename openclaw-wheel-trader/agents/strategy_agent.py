"""
Sprint 7: Strategy Agent — proposes trades based on market analysis.

Part of the 3-agent governance system:
  Strategy Agent → proposes trades
  Risk Agent → validates risk, can VETO
  Compliance Agent → checks regulatory rules

Trade proceeds ONLY with unanimous consent.
"""

from datetime import datetime, timezone
from lib.memory_palace import diary_write, diary_read, get_current_regime, recall_ticker_history
from lib.screener import WheelCandidate
from lib.audit import log_event


class StrategyAgent:
    """Proposes trades. Cannot execute — only Risk + Compliance can approve."""

    name = "strategy_agent"

    def propose_csp(self, candidate: WheelCandidate) -> dict:
        """Build a trade proposal for agent consensus."""
        history = recall_ticker_history(candidate.ticker)
        regime = get_current_regime()

        # Check own diary for recent activity on this ticker
        recent = diary_read(self.name, last_n=20)
        recent_ticker = [e for e in recent if candidate.ticker in e.get("entry", "")]

        proposal = {
            "agent": self.name,
            "action": "sell_csp",
            "ticker": candidate.ticker,
            "strike": candidate.strike,
            "expiration": candidate.expiration,
            "premium": candidate.premium,
            "composite_score": candidate.composite_score,
            "trend_score": candidate.trend_score,
            "level_score": candidate.level_score,
            "signal_score": candidate.signal_score,
            "iv_rank": candidate.iv_rank,
            "zone_level": candidate.zone_level,
            "zone_touches": candidate.zone_touches,
            "pattern": candidate.candlestick_pattern,
            "regime": regime,
            "ticker_history_entries": len(history.get("kg_facts", [])),
            "recent_activity": len(recent_ticker),
            "proposed_at": datetime.now(timezone.utc).isoformat(),
        }

        diary_write(self.name,
            f"PROPOSE|{candidate.ticker}|CSP_{candidate.strike}P|"
            f"score_{candidate.composite_score}/9|regime_{regime}")

        log_event("agent", "strategy_proposed", {
            "ticker": candidate.ticker,
            "strike": candidate.strike,
            "score": candidate.composite_score,
        })

        return proposal

    def propose_cc(self, candidate: WheelCandidate, cost_basis: float) -> dict:
        """Build a covered call proposal."""
        proposal = {
            "agent": self.name,
            "action": "sell_cc",
            "ticker": candidate.ticker,
            "strike": candidate.strike,
            "expiration": candidate.expiration,
            "premium": candidate.premium,
            "cost_basis": cost_basis,
            "above_basis_by": candidate.strike - cost_basis,
            "composite_score": candidate.composite_score,
            "proposed_at": datetime.now(timezone.utc).isoformat(),
        }

        diary_write(self.name,
            f"PROPOSE|{candidate.ticker}|CC_{candidate.strike}C|"
            f"above_basis_{candidate.strike - cost_basis:.2f}")

        return proposal
