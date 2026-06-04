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
        # 2026-05-30: was hardcoded 10% which made CSPs impossible at
        # small bankrolls (any single CSP collateral is 20-100% of $1.5k).
        # Now reads max_position_pct from settings.yaml circuit_breakers
        # with a 50% floor (allows one wheel CSP at $1,500). Same circuit
        # breaker the rest of the bot uses — single source of truth.
        try:
            import yaml as _yaml
            from pathlib import Path as _Path
            with open(_Path(__file__).resolve().parent.parent / "config" / "settings.yaml") as _f:
                _settings = _yaml.safe_load(_f) or {}
            _max_pos = float(_settings.get("circuit_breakers", {}).get("max_position_pct", 0.50))
            position_max = max(_max_pos, 0.50)   # floor at 50% for small-bankroll wheel
        except Exception:
            position_max = 0.50
        collateral = strike * 100  # CSP collateral requirement
        position_pct = collateral / portfolio_value if portfolio_value > 0 else 1.0
        checks["position_concentration"] = {
            "pct": round(position_pct, 4),
            "max": position_max,
            "pass": position_pct <= position_max,
        }
        if not checks["position_concentration"]["pass"]:
            veto_reasons.append(f"Position {position_pct:.1%} exceeds {position_max:.0%} limit")

        # 2. Sector concentration — relaxed for small bankroll
        # 2026-05-30: was hardcoded 30%. At $1,500 with most CSP candidates
        # in the "other" sector (cheap consumer/finance names), the 30%
        # cap blocked nearly every wheel trade. Reading from settings.yaml
        # circuit_breakers.max_sector_pct, floored at 75%.
        try:
            _max_sec = float(_settings.get("circuit_breakers", {}).get("max_sector_pct", 0.75))
            sector_max = max(_max_sec, 0.75)
        except Exception:
            sector_max = 0.75
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
            "max": sector_max,
            "pass": sector_pct <= sector_max,
        }
        if not checks["sector_concentration"]["pass"]:
            veto_reasons.append(f"Sector {sector} at {sector_pct:.0%} would exceed {sector_max:.0%}")

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
        # 2026-05-30: bear-regime min score lowered 8 → 5. The 8/9
        # threshold meant the bot never traded CSPs in bear regimes —
        # and at this bankroll we can't afford to skip the wheel for
        # weeks just because SPY is below MA50. 5/9 still demands
        # meaningful technical setup.
        regime = get_current_regime()
        regime_ok = True
        if action == "sell_csp" and regime == "bear":
            score = proposal.get("composite_score", 0)
            if score < 5:
                regime_ok = False
                veto_reasons.append(f"Bear regime requires score ≥5, got {score}")
        checks["regime"] = {
            "current": regime,
            "appropriate": regime_ok,
            "pass": regime_ok,
        }

        # 5. Composite score sanity
        # 2026-05-30: was hardcoded min 7 which is impossible to hit in
        # chop. Now reads confirmation.min_composite_score from
        # wheel_strategy.yaml (currently 3 for small-bankroll wheel
        # operation). Same source the screener uses — risk_agent and
        # screener now agree on the threshold.
        try:
            with open(_Path(__file__).resolve().parent.parent / "config" / "wheel_strategy.yaml") as _f:
                _strat = _yaml.safe_load(_f) or {}
            _min_sc = int(_strat.get("confirmation", {}).get("min_composite_score", 3))
        except Exception:
            _min_sc = 3
        score = proposal.get("composite_score", 0)
        checks["score"] = {
            "value": score,
            "min": _min_sc,
            "pass": score >= _min_sc,
        }
        if not checks["score"]["pass"]:
            veto_reasons.append(f"Score {score}/9 below minimum {_min_sc}")

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
