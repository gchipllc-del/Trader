"""
Fund Manager — portfolio-level review (TauricResearch Stage V analog).

Where the strategy/risk/bear/bull/compliance agents are PER-TRADE myopic,
the fund manager looks at the WHOLE BOOK once per cycle and answers:

  • Is the portfolio over-concentrated in one sector?
  • Are there too many correlated positions?
  • Is the cash buffer too thin / too deep for the active risk profile?
  • Does any single position need to be trimmed for portfolio-level reasons
    (not just position-level breakers)?

Output: a list of recommended actions, each one explainable. The fund
manager never executes directly — it produces recommendations that the
monitor loop reviews and acts on. This separation makes the manager's
decisions auditable independent of execution.

Inspired by the Fund Manager / Portfolio approval stage in
TauricResearch/TradingAgents — adapted to deterministic rules driven
by the active risk profile (lib/risk_profile.py) instead of LLM debate.
Zero LLM cost, fully deterministic, easy to test.

Action types:

  SECTOR_TRIM        — sector exposure exceeds max_sector_pct
  CORRELATION_TRIM   — too many positions in same correlation_group
  CASH_BUFFER_LOW    — cash < 5% of bankroll, must trim
  REBALANCE          — single position drifted past max_position_pct
                       (already handled by auto_trim_oversize_stocks
                       but FM logs it as a portfolio observation)
  HRP_REBALANCE      — position weight has drifted from the
                       Hierarchical Risk Parity target by more than
                       a configurable threshold. HRP allocates capital
                       across positions based on their covariance
                       structure without needing return forecasts —
                       robust against correlation-regime shifts. Pattern
                       from PyPortfolioOpt's HRPOpt; only fires when the
                       caller supplies a historical returns matrix.
  HOLD               — book is balanced, no action

The monitor calls fund_manager.review_portfolio() once per cycle. The
result feeds summary["alerts"] and may trigger a follow-up trim via
auto_trim_oversize_stocks (already wired in Wave 1 #3) plus optional
sector-level trims from this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lib.audit import log_event


@dataclass
class FundManagerAction:
    """One portfolio-level recommendation."""
    type: str                       # SECTOR_TRIM / CORRELATION_TRIM / etc.
    severity: str                   # "info" | "warn" | "critical"
    summary: str
    affected_tickers: list[str] = field(default_factory=list)
    suggested_amount: float = 0.0   # dollars to trim, if applicable


def _ticker_sector_map(positions: list[dict]) -> dict[str, str]:
    """Group positions by sector. Falls back to 'unknown' if not tagged.

    Sector data is set upstream by the screener (alpha_vantage_client). If
    a position lacks it (e.g. crypto, older entries pre-tagging), we
    bucket it as 'unknown' rather than guess.
    """
    return {p.get("ticker", "?"): str(p.get("sector", "unknown")).lower()
            for p in positions if p.get("ticker")}


class FundManager:
    """Reviews the entire open book once per cycle. Cannot execute; emits
    recommendations consumed by the monitor."""

    name = "fund_manager"

    def review_portfolio(
        self,
        *,
        positions: list[dict],
        bankroll: float,
        cash: float,
        max_sector_pct: float = 0.50,
        max_correlation_group_count: int = 3,
        min_cash_buffer_pct: float = 0.05,
        historical_returns: "pd.DataFrame | None" = None,
        hrp_max_drift_pct: float = 0.15,
    ) -> dict:
        """
        Run all portfolio-level checks. Returns a structured review dict.

        Args:
            positions: list of OPEN position dicts from positions_store
            bankroll: total portfolio value (cash + positions)
            cash: free cash balance
            max_sector_pct: alarm if any sector > this fraction of bankroll
            max_correlation_group_count: alarm if a single correlation_group
                holds more than N positions
            min_cash_buffer_pct: alarm if cash < this fraction of bankroll
            historical_returns: optional DataFrame of daily returns
                (columns = tickers held, rows = days). When provided AND
                ≥30 rows AND ≥3 tradeable positions, the fund manager
                computes Hierarchical Risk Parity target weights and
                emits HRP_REBALANCE actions for positions that have
                drifted >``hrp_max_drift_pct`` from their target weight.
                Skipped (no action, no error) when missing/insufficient.
            hrp_max_drift_pct: minimum absolute weight drift from HRP
                target that triggers an HRP_REBALANCE action.

        Returns:
            {
                "actions": [FundManagerAction, ...],
                "summary": str,
                "stats": {
                    "n_positions": int,
                    "cash_pct": float,
                    "sector_concentration": dict[sector, pct],
                    "correlation_clusters": dict[group, count],
                    "hrp_weights": dict[ticker, target_weight],  # if HRP ran
                },
            }
        """
        actions: list[FundManagerAction] = []
        # Core long-term holdings are excluded from portfolio-level review.
        # The fund manager only proposes trims/rebalances; for hold_forever
        # positions those proposals are by definition disallowed. Excluding
        # them here also keeps sector/correlation math from flagging an
        # intentional core exposure as a problem.
        opens = [p for p in positions
                 if p.get("status") == "open" and not p.get("hold_forever")]
        n = len(opens)

        if bankroll <= 0:
            log_event("fund_manager", "review_skipped",
                      {"reason": "non_positive_bankroll"}, result="degraded")
            return {
                "actions": [],
                "summary": "skipped: non-positive bankroll",
                "stats": {"n_positions": n},
            }

        # ── Sector concentration ────────────────────────────────────────
        sector_value: dict[str, float] = {}
        for p in opens:
            sector = str(p.get("sector", "unknown")).lower()
            mv = float(p.get("entry_price", 0) or 0) * float(p.get("shares", 0) or 0)
            # Prefer broker market value if present
            if p.get("market_value"):
                mv = float(p["market_value"])
            sector_value[sector] = sector_value.get(sector, 0.0) + mv

        sector_pct = {s: v / bankroll for s, v in sector_value.items()}
        for sector, pct in sector_pct.items():
            if sector == "unknown":
                continue  # don't act on bucketed-unknowns
            if pct > max_sector_pct:
                tickers = [p.get("ticker") for p in opens
                           if str(p.get("sector", "")).lower() == sector]
                excess = (pct - max_sector_pct) * bankroll
                actions.append(FundManagerAction(
                    type="SECTOR_TRIM",
                    severity="warn" if pct < max_sector_pct + 0.10 else "critical",
                    summary=(
                        f"Sector '{sector}' is {pct:.1%} of bankroll, "
                        f"over {max_sector_pct:.0%} limit — trim ~${excess:.0f}"
                    ),
                    affected_tickers=tickers,
                    suggested_amount=excess,
                ))

        # ── Correlation cluster ─────────────────────────────────────────
        cluster_count: dict[str, list[str]] = {}
        for p in opens:
            grp = str(p.get("correlation_group", "")).strip()
            if not grp:
                continue
            cluster_count.setdefault(grp, []).append(p.get("ticker", "?"))
        for grp, tickers in cluster_count.items():
            if len(tickers) > max_correlation_group_count:
                actions.append(FundManagerAction(
                    type="CORRELATION_TRIM",
                    severity="warn",
                    summary=(
                        f"Correlation group '{grp}' holds {len(tickers)} "
                        f"positions ({', '.join(tickers)}); cap is "
                        f"{max_correlation_group_count}"
                    ),
                    affected_tickers=tickers,
                ))

        # ── Cash buffer ─────────────────────────────────────────────────
        cash_pct = cash / bankroll if bankroll > 0 else 0.0
        if cash_pct < min_cash_buffer_pct:
            actions.append(FundManagerAction(
                type="CASH_BUFFER_LOW",
                severity="warn",
                summary=(
                    f"Cash buffer is {cash_pct:.1%} of bankroll, "
                    f"below {min_cash_buffer_pct:.0%} target — consider "
                    f"trimming a position to free liquidity"
                ),
            ))

        # ── Hierarchical Risk Parity rebalance ──────────────────────────
        # HRP allocates capital across positions by clustering on the
        # covariance structure — no return forecast needed, robust to
        # the kind of correlation-regime shifts that break mean-variance.
        # Only runs when the caller supplies a historical returns matrix
        # (the monitor fetches it; tests can pass synthetic data).
        hrp_weights: dict[str, float] = {}
        hrp_actions, hrp_weights = self._hrp_rebalance(
            opens=opens, bankroll=bankroll,
            historical_returns=historical_returns,
            max_drift_pct=hrp_max_drift_pct,
        )
        actions.extend(hrp_actions)

        # ── Compose summary ─────────────────────────────────────────────
        if not actions:
            summary = (
                f"Book balanced: {n} open, cash {cash_pct:.1%}, "
                f"top sector {max(sector_pct.items(), key=lambda kv: kv[1])[1]:.1%} "
                if sector_pct else
                f"Book balanced: {n} open, cash {cash_pct:.1%}"
            )
        else:
            summary = (
                f"{len(actions)} action(s): "
                + "; ".join(f"{a.type} ({a.severity})" for a in actions)
            )

        log_event("fund_manager", "review_complete", {
            "n_positions": n,
            "n_actions": len(actions),
            "action_types": [a.type for a in actions],
            "cash_pct": round(cash_pct, 4),
        })

        return {
            "actions": [a.__dict__ for a in actions],
            "summary": summary,
            "stats": {
                "n_positions": n,
                "cash_pct": round(cash_pct, 4),
                "sector_concentration": {k: round(v, 4) for k, v in sector_pct.items()},
                "correlation_clusters": {k: len(v) for k, v in cluster_count.items()},
                "hrp_weights": {k: round(v, 4) for k, v in hrp_weights.items()},
            },
        }

    def _hrp_rebalance(
        self,
        *,
        opens: list[dict],
        bankroll: float,
        historical_returns,
        max_drift_pct: float,
    ) -> tuple[list[FundManagerAction], dict[str, float]]:
        """Compute HRP target weights and flag drifted positions.

        Returns ``(actions, hrp_weights)``. Empty + empty dict when HRP
        can't run (no return data, too few positions, too few days).
        """
        # ── Pre-conditions ───────────────────────────────────────────
        if historical_returns is None:
            return [], {}
        try:
            import pandas as pd  # local import — avoid module-load cost
        except ImportError:
            return [], {}
        if not isinstance(historical_returns, pd.DataFrame):
            return [], {}

        # Need at least 3 stock positions with returns columns and ≥30 days
        stock_positions = [
            p for p in opens
            if p.get("type") == "stock" and p.get("ticker") in historical_returns.columns
        ]
        if len(stock_positions) < 3:
            return [], {}
        tickers = [p["ticker"] for p in stock_positions]
        rets = historical_returns[tickers].dropna()
        if len(rets) < 30:
            return [], {}

        # ── HRP optimization ─────────────────────────────────────────
        try:
            from pypfopt import HRPOpt
            opt = HRPOpt(returns=rets)
            target_raw = opt.optimize()
            target_weights = {t: float(w) for t, w in target_raw.items()}
        except Exception as e:
            log_event("fund_manager", "hrp_failed",
                      {"error": str(e)[:200], "n_tickers": len(tickers)},
                      result="degraded")
            return [], {}

        # ── Drift detection ──────────────────────────────────────────
        # Express actual weights as a fraction of the *invested* portion
        # (cash excluded) so the comparison is apples-to-apples with HRP,
        # which always sums to 1.0 across the asset set.
        invested_value = 0.0
        actual_market_values: dict[str, float] = {}
        for p in stock_positions:
            mv = float(p.get("market_value", 0) or 0)
            if not mv:
                mv = float(p.get("entry_price", 0) or 0) * float(p.get("shares", 0) or 0)
            actual_market_values[p["ticker"]] = mv
            invested_value += mv

        if invested_value <= 0:
            return [], target_weights

        actual_weights = {
            t: mv / invested_value for t, mv in actual_market_values.items()
        }

        actions: list[FundManagerAction] = []
        for ticker, target_w in target_weights.items():
            actual_w = actual_weights.get(ticker, 0.0)
            drift = actual_w - target_w
            if abs(drift) < max_drift_pct:
                continue
            direction = "trim" if drift > 0 else "add"
            # Dollar amount to move to bring weight in line. Bounded by
            # the existing position market value when trimming.
            dollar_delta = abs(drift) * invested_value
            sev = "warn" if abs(drift) < (max_drift_pct + 0.10) else "critical"
            actions.append(FundManagerAction(
                type="HRP_REBALANCE",
                severity=sev,
                summary=(
                    f"{ticker} weight {actual_w:.1%} vs HRP target "
                    f"{target_w:.1%} (drift {drift:+.1%}) — {direction} "
                    f"~${dollar_delta:.0f}"
                ),
                affected_tickers=[ticker],
                suggested_amount=dollar_delta,
            ))

        log_event("fund_manager", "hrp_evaluated", {
            "n_tickers": len(tickers),
            "n_days": len(rets),
            "target_weights": {k: round(v, 4) for k, v in target_weights.items()},
            "n_drift_actions": len(actions),
        })

        return actions, target_weights
