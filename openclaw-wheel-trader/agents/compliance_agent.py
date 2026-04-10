"""
Sprint 7: Compliance Agent — regulatory and rule compliance.

Checks:
- Wash sale rule (30-day window after a loss on same ticker)
- Earnings date filter (don't sell through earnings)
- Pattern Day Trader rules (if applicable)
- Trading during restricted periods
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from lib.audit import log_event
from lib.memory_palace import diary_write, kg_query

TRADE_HISTORY_PATH = Path(__file__).parent.parent / "data" / "trade_history.json"


class ComplianceAgent:
    """Checks regulatory compliance. Cannot propose or execute."""

    name = "compliance_agent"

    def review(self, proposal: dict) -> dict:
        """
        Review a trade proposal for compliance issues.

        Returns:
            {"approved": bool, "reason": str, "checks": dict}
        """
        ticker = proposal.get("ticker", "")
        checks = {}
        issues = []

        # 1. Wash sale check
        wash = self._check_wash_sale(ticker)
        checks["wash_sale"] = wash
        if not wash["pass"]:
            issues.append(wash["reason"])

        # 2. Earnings filter
        earnings = self._check_earnings(ticker, proposal.get("expiration", ""))
        checks["earnings"] = earnings
        if not earnings["pass"]:
            issues.append(earnings["reason"])

        # 3. Market hours check
        hours = self._check_market_hours()
        checks["market_hours"] = hours
        if not hours["pass"]:
            issues.append(hours["reason"])

        approved = len(issues) == 0
        reason = "All compliance checks passed" if approved else "; ".join(issues)

        diary_write(self.name,
            f"{'CLEAR' if approved else 'BLOCK'}|{ticker}|"
            f"wash_{'ok' if checks['wash_sale']['pass'] else 'FAIL'}|"
            f"earnings_{'ok' if checks['earnings']['pass'] else 'FAIL'}")

        log_event("agent", "compliance_reviewed", {
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

    def _check_wash_sale(self, ticker: str) -> dict:
        """
        Wash sale rule: Cannot claim a tax loss if you repurchase
        the same or substantially identical security within 30 days.

        We check: was there a loss on this ticker in the last 30 days?
        If so, selling a new put (which could result in buying shares)
        triggers wash sale territory.
        """
        history = self._load_trade_history()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        recent_losses = [
            t for t in history
            if t.get("ticker") == ticker
            and t.get("total_pnl", 0) < 0
            and t.get("completed_at", "") > cutoff
        ]

        if recent_losses:
            last_loss = recent_losses[-1]
            return {
                "pass": False,
                "reason": f"Wash sale risk: {ticker} had a ${last_loss['total_pnl']:.2f} loss "
                         f"on {last_loss.get('completed_at', 'unknown')[:10]} (within 30 days)",
                "last_loss_date": last_loss.get("completed_at", ""),
                "loss_amount": last_loss.get("total_pnl", 0),
            }

        return {"pass": True, "reason": "No recent losses on this ticker"}

    def _check_earnings(self, ticker: str, expiration: str) -> dict:
        """
        Don't sell options expiring through an earnings date.
        
        TODO: Wire to earnings calendar API (Sprint 8).
        For now, checks KG for known earnings dates.
        """
        # Check knowledge graph for earnings info
        facts = kg_query(ticker, current_only=True)
        earnings_facts = [f for f in facts if "earnings" in f.get("predicate", "").lower()]

        if earnings_facts:
            for ef in earnings_facts:
                earnings_date = ef.get("object", "")
                if expiration and earnings_date and expiration >= earnings_date:
                    return {
                        "pass": False,
                        "reason": f"Option expires {expiration} through earnings on {earnings_date}",
                    }

        return {"pass": True, "reason": "No earnings conflict detected"}

    def _check_market_hours(self) -> dict:
        """Basic check that we're in market hours (9:30 AM - 4:00 PM ET)."""
        now = datetime.now(timezone.utc)
        # Rough ET conversion (UTC-4 or UTC-5)
        et_hour = (now.hour - 4) % 24  # Approximate EDT

        is_weekday = now.weekday() < 5
        is_market_hours = 9 <= et_hour < 16

        if not is_weekday:
            return {"pass": False, "reason": "Weekend — market closed"}

        if not is_market_hours:
            return {"pass": False, "reason": f"Outside market hours (est ~{et_hour}:00 ET)"}

        return {"pass": True, "reason": "Within market hours"}

    def _load_trade_history(self) -> list[dict]:
        if not TRADE_HISTORY_PATH.exists():
            return []
        with open(TRADE_HISTORY_PATH) as f:
            return json.load(f)
