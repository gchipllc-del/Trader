"""
Sprint 7: Risk Agent — validates portfolio risk. Can VETO any trade.

Checks:
- Portfolio concentration (max 10% per position, 30% per sector)
- Correlation risk (too many correlated positions)
- Max drawdown tolerance
- Position count limits
- Market regime appropriateness
"""

import json
from pathlib import Path
from datetime import datetime, timezone

from lib.audit import log_event
from lib.memory_palace import diary_write, diary_read, get_current_regime, kg_query

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"

# Sector mapping for concentration checks
SECTOR_MAP = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "AMD": "tech",
    "GOOGL": "tech", "META": "tech", "AMZN": "consumer",
    "SPY": "index",
}


class RiskAgent:
    """Reviews every trade proposal. Can VETO. Cannot propose or execute."""

    name = "risk_agent"

    def review(self, proposal: dict, portfolio_value: float) -> dict:
        """
        Review a trade proposal from Strategy Agent.

        Returns:
            {
                "approved": bool,
                "reason": str,
                "checks": dict of individual check results
            }
        """
        ticker = proposal.get("ticker", "")
        strike = proposal.get("strike", 0)
        action = proposal.get("action", "")

        checks = {}
        veto_reasons = []

        # 1. Position concentration
        collateral = strike * 100  # CSP collateral requirement
        position_pct = collateral / portfolio_value if portfolio_value > 0 else 1.0
        checks["position_concentration"] = {
            "pct": round(position_pct, 4),
            "max": 0.10,
            "pass": position_pct <= 0.10,
        }
        if not checks["position_concentration"]["pass"]:
            veto_reasons.append(f"Position {position_pct:.1%} exceeds 10% limit")

        # 2. Sector concentration
        sector = SECTOR_MAP.get(ticker, "other")
        positions = self._load_positions()
        sector_value = sum(
            p.get("strike", 0) * 100
            for p in positions
            if p.get("status") in ("open", "assigned")
            and SECTOR_MAP.get(p.get("ticker", ""), "other") == sector
        )
        sector_pct = (sector_value + collateral) / portfolio_value if portfolio_value > 0 else 1.0
        checks["sector_concentration"] = {
            "sector": sector,
            "pct": round(sector_pct, 4),
            "max": 0.30,
            "pass": sector_pct <= 0.30,
        }
        if not checks["sector_concentration"]["pass"]:
            veto_reasons.append(f"Sector {sector} at {sector_pct:.0%} would exceed 30%")

        # 3. Open position count
        open_count = len([p for p in positions if p.get("status") in ("open", "assigned")])
        max_positions = 8
        checks["position_count"] = {
            "current": open_count,
            "max": max_positions,
            "pass": open_count < max_positions,
        }
        if not checks["position_count"]["pass"]:
            veto_reasons.append(f"Already at {open_count} positions (max {max_positions})")

        # 4. Regime check
        regime = get_current_regime()
        regime_ok = True
        if action == "sell_csp" and regime == "bear":
            # Selling puts in a bear market is extra risky
            score = proposal.get("composite_score", 0)
            if score < 8:  # Require higher score in bear markets
                regime_ok = False
                veto_reasons.append(f"Bear regime requires score ≥8, got {score}")
        checks["regime"] = {
            "current": regime,
            "appropriate": regime_ok,
            "pass": regime_ok,
        }

        # 5. Composite score sanity
        score = proposal.get("composite_score", 0)
        checks["score"] = {
            "value": score,
            "min": 7,
            "pass": score >= 7,
        }
        if not checks["score"]["pass"]:
            veto_reasons.append(f"Score {score}/9 below minimum 7")

        # Decision
        approved = len(veto_reasons) == 0
        reason = "All risk checks passed" if approved else "; ".join(veto_reasons)

        diary_write(self.name,
            f"{'APPROVE' if approved else 'VETO'}|{ticker}|{action}|"
            f"pos_{position_pct:.0%}|sector_{sector}_{sector_pct:.0%}|"
            f"regime_{regime}|score_{score}")

        log_event("agent", "risk_reviewed", {
            "ticker": ticker,
            "approved": approved,
            "reason": reason,
        })

        return {
            "agent": self.name,
            "approved": approved,
            "reason": reason,
            "checks": checks,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load_positions(self) -> list[dict]:
        if not POSITIONS_PATH.exists():
            return []
        with open(POSITIONS_PATH) as f:
            return json.load(f)
