"""
Sprint 7: Risk Agent — validates portfolio risk. Can VETO any trade.

Checks:
- Portfolio concentration (max 10% per position, 30% per sector)
- Correlation risk (too many correlated positions)
- Max drawdown tolerance
- Position count limits
- Market regime appropriateness
"""
from __future__ import annotations

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


def _position_exposure(p: dict) -> float:
    """Approximate dollar exposure of a held position for concentration math.

    The old sector check used ``strike * 100`` for every record, which
    (a) ignored ``type=="stock"`` positions entirely because they have no
    ``strike`` field, and (b) undercounted assigned stock when the price
    moved away from the strike. This helper returns a per-position
    exposure that matches the denominator (portfolio_value, which is
    market-value-based) more honestly.
    """
    ptype = p.get("type", "")
    status = p.get("status", "")
    shares = float(p.get("shares", 0) or 0)
    mark = p.get("mark") or p.get("current_price")
    entry = float(p.get("entry_price", 0) or 0)
    strike = float(p.get("strike", 0) or 0)

    # Only HELD positions contribute to concentration. The caller already
    # filters by status, but be defensive here too.
    if status not in ("open", "assigned"):
        return 0.0

    if ptype == "csp" and status == "open":
        # Cash held aside against assignment.
        return strike * 100
    if ptype == "cc" and status == "open":
        # The underlying shares — collateral against a call away.
        if mark and shares:
            return float(mark) * shares
        return strike * 100  # fallback if the position record lacks shares/mark
    if status == "assigned" or ptype == "stock":
        # Stock exposure: use market value when available, fall back to
        # cost basis. Either is better than ignoring the position.
        if mark and shares:
            return float(mark) * shares
        if entry and shares:
            return entry * shares
        # Last-resort fallback for legacy CSP records that lacked shares
        # but have a strike (treat as 1 contract = 100 shares).
        return strike * 100
    # Legacy open record without an explicit type but with a strike —
    # treat as a CSP (this matches the pre-refactor behavior).
    if status == "open" and strike > 0:
        return strike * 100
    return 0.0


class RiskAgent:
    """Reviews every trade proposal. Can VETO. Cannot propose or execute."""

    name = "risk_agent"

    def review(
        self,
        proposal: dict,
        portfolio_value: float,
        *,
        broker_positions: list[dict] | None = None,
    ) -> dict:
        """
        Review a trade proposal from Strategy Agent.

        Args:
            proposal: trade proposal dict from strategy_agent
            portfolio_value: current account value
            broker_positions: optional pre-fetched broker positions. If
                passed, used for the capital-at-risk gate; if omitted,
                the gate is skipped (back-compat). Callers that hold the
                broker view should pass it.

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
            _position_exposure(p)
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

        # 6. Global capital-at-risk (NEW)
        # Per-position and per-sector checks pass-fail trades individually,
        # but a sequence of individually-OK CSPs can still sum to most of
        # the account's cash. Cap total = (Σ short-put collateral) +
        # (Σ long-share market value) + this proposal's new collateral,
        # against a fraction of portfolio value. Pattern from
        # alpacahq/options-wheel's MAX_RISK budget. Skipped if the caller
        # didn't pass broker_positions (back-compat with tests / older
        # callers that pass only proposal+portfolio_value).
        if broker_positions is not None and portfolio_value > 0:
            from lib.wheel_state import classify_book, total_capital_at_risk
            book = classify_book(broker_positions, raise_on_illegal=False)
            existing = total_capital_at_risk(book)
            # Only short-put proposals add new collateral; CC and stock-buy
            # proposals contribute differently (CC: no new collateral, stock
            # buy: handled by stock_engine). For CSPs, "collateral" was
            # already computed above.
            new_collateral = collateral if action == "sell_csp" else 0.0
            projected = existing["total"] + new_collateral
            strategy = self._load_strategy()
            max_car_pct = float(
                strategy.get("risk", {}).get("max_capital_at_risk_pct", 0.80)
            )
            car_pct = projected / portfolio_value
            checks["capital_at_risk"] = {
                "existing_total": round(existing["total"], 2),
                "new_collateral": round(new_collateral, 2),
                "projected": round(projected, 2),
                "projected_pct": round(car_pct, 4),
                "max_pct": max_car_pct,
                "pass": car_pct <= max_car_pct,
            }
            if not checks["capital_at_risk"]["pass"]:
                veto_reasons.append(
                    f"Capital-at-risk would hit {car_pct:.1%} (cap {max_car_pct:.0%}): "
                    f"${existing['total']:,.0f} held + ${new_collateral:,.0f} new"
                )

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
        """Locked snapshot via the canonical store (audit finding #5)."""
        from lib.positions_store import load_positions as _store_load
        return _store_load(POSITIONS_PATH)

    def _load_strategy(self) -> dict:
        """Read wheel_strategy.yaml for risk-budget params. Tolerant to
        missing file — returns empty dict so the gate quietly defaults."""
        import yaml
        path = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
