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

        Returns:
            {
                "actions": [FundManagerAction, ...],
                "summary": str,
                "stats": {
                    "n_positions": int,
                    "cash_pct": float,
                    "sector_concentration": dict[sector, pct],
                    "correlation_clusters": dict[group, count],
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
            },
        }
