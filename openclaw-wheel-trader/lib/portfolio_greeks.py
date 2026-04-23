"""
Portfolio Greeks — analytical Black-Scholes aggregation.

Computes delta / gamma / vega / theta at the position and portfolio
level so the risk agent can enforce vega-risk caps and so sizing
decisions respect portfolio-level exposure (not just per-trade
checks).

Why this matters for a wheel trader:
    A 10-CSP portfolio that looks "diversified" by ticker can still be
    stacked with $10k+ of vega risk — meaning a 1-point IV crush
    takes $10k off the book. Per-trade checks catch nothing; only
    portfolio aggregation does. Same logic for theta (the ENGINE of
    our returns — we want to know the daily theta burn we're
    collecting) and gamma (our blow-up risk on assignment day).

Library: py_vollib (installed), analytical closed-form Greeks.
    We deliberately avoid py_vollib_vectorized because it pulls
    pyarrow which fails to build on this Mac. Non-vectorized is fine
    for <50 positions — the wheel rarely exceeds that.

Position schema (from csp_engine.py + cc_engine.py):
    STOCK       — {type:"stock", shares, entry_price, [cc_strike, cc_expiration]}
    CSP (short) — {type:"csp",   strike, expiration, premium_collected}
    (CC is attached as cc_strike/cc_expiration on a stock position —
    we detect it and emit both the long-stock delta AND the short-call
    Greeks.)

Security / robustness:
    - ALL math is wrapped — any invalid input → zero contribution,
      event logged, portfolio keeps computing. A single weird position
      must not crash the risk agent.
    - Negative / zero / huge IVs are clamped to [0.05, 5.0].
    - Expired positions (DTE <= 0) → zero Greeks (option is dust).
    - Each position's result carries `valid=True/False` so the dashboard
      can flag suspicious rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

try:
    from py_vollib.black_scholes.greeks.analytical import (
        delta as _bs_delta,
        gamma as _bs_gamma,
        vega as _bs_vega,
        theta as _bs_theta,
    )
    _PY_VOLLIB_AVAILABLE = True
except ImportError:  # pragma: no cover — module is in requirements
    _PY_VOLLIB_AVAILABLE = False

from lib.audit import log_event

log = logging.getLogger(__name__)

# ── Clamps and defaults ──────────────────────────────────────────

DEFAULT_RISK_FREE_RATE = 0.045          # ~4.5% short-term treasuries 2026
DEFAULT_IV = 0.30                        # Fallback when chain IV unavailable
IV_MIN, IV_MAX = 0.05, 5.0               # 5% to 500% — beyond this it's bad data
SPOT_MIN = 0.01                          # Zero / negative spot → skip
STRIKE_MIN = 0.01
SHARES_PER_CONTRACT = 100

# ── Data structures ──────────────────────────────────────────────

@dataclass
class PositionGreeks:
    """Per-position Greek contribution (already scaled by quantity)."""
    ticker: str
    position_type: str          # "stock", "csp", "cc" (short call on owned stock)
    quantity: int               # contracts (options) or shares (stock)
    delta: float = 0.0          # Total delta (share-equivalents)
    gamma: float = 0.0          # Total gamma
    vega: float = 0.0           # Dollar P/L per 1 vol-point move
    theta: float = 0.0          # Dollar P/L per calendar day
    dte: int = 0                # Days to expiration (0 for stock)
    strike: float = 0.0         # 0 for stock
    spot: float = 0.0
    iv: float = 0.0
    valid: bool = True          # False if we skipped (bad input)
    reason: str = ""            # Why invalid, or descriptor


@dataclass
class PortfolioGreeks:
    """Aggregated exposure across all open positions."""
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0
    gross_delta: float = 0.0              # sum of abs(position delta)
    positions: list[PositionGreeks] = field(default_factory=list)
    invalid_count: int = 0                # positions we couldn't price
    ts: str = ""                          # ISO timestamp of computation

    def to_dict(self) -> dict:
        return {
            "total_delta": round(self.total_delta, 4),
            "total_gamma": round(self.total_gamma, 6),
            "total_vega": round(self.total_vega, 2),
            "total_theta": round(self.total_theta, 2),
            "gross_delta": round(self.gross_delta, 4),
            "invalid_count": self.invalid_count,
            "position_count": len(self.positions),
            "ts": self.ts,
            "positions": [asdict(p) for p in self.positions],
        }


# ── Helpers ──────────────────────────────────────────────────────

def _years_to_expiration(expiration_iso: str) -> float:
    """Convert YYYY-MM-DD → years from now (float). Returns 0 if past/invalid."""
    if not expiration_iso:
        return 0.0
    try:
        exp = datetime.fromisoformat(str(expiration_iso)).replace(
            tzinfo=timezone.utc) if "T" in str(expiration_iso) else \
            datetime.strptime(str(expiration_iso), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0.0
    delta_sec = (exp - datetime.now(timezone.utc)).total_seconds()
    if delta_sec <= 0:
        return 0.0
    return delta_sec / (365.25 * 24 * 3600)


def _dte_int(expiration_iso: str) -> int:
    yrs = _years_to_expiration(expiration_iso)
    return max(0, int(round(yrs * 365.25)))


def _clamp_iv(iv: Optional[float]) -> float:
    if iv is None or iv <= 0:
        return DEFAULT_IV
    try:
        f = float(iv)
    except (TypeError, ValueError):
        return DEFAULT_IV
    if f < IV_MIN:
        return IV_MIN
    if f > IV_MAX:
        return IV_MAX
    return f


def _safe_greek(
    fn: Callable[..., float],
    flag: str, S: float, K: float, t: float, r: float, sigma: float,
) -> float:
    """Wrap py_vollib call; any exception → 0.0, audit-logged."""
    try:
        return float(fn(flag, S, K, t, r, sigma))
    except Exception as e:
        log_event("portfolio_greeks", "greek_compute_error", {
            "fn": getattr(fn, "__name__", "?"),
            "flag": flag, "S": S, "K": K, "t": t, "r": r, "sigma": sigma,
            "error": f"{type(e).__name__}: {str(e)[:120]}",
        }, result="failed")
        return 0.0


# ── Per-position computations ────────────────────────────────────

def compute_stock_greeks(position: dict) -> PositionGreeks:
    """Long stock: delta = shares, gamma/vega/theta = 0.

    (The 'stock' position may carry a cc_strike/cc_expiration — caller
    is responsible for also invoking compute_cc_greeks on it.)
    """
    ticker = str(position.get("ticker", "?"))
    shares = int(position.get("shares", 0) or 0)
    entry = float(position.get("entry_price", 0) or 0)
    return PositionGreeks(
        ticker=ticker, position_type="stock", quantity=shares,
        delta=float(shares), gamma=0.0, vega=0.0, theta=0.0,
        spot=entry, valid=shares > 0,
        reason="stock" if shares > 0 else "zero shares",
    )


def compute_csp_greeks(
    position: dict, spot: float, iv: Optional[float] = None,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> PositionGreeks:
    """SHORT put — we sold it, so signs flip vs the long-put Greeks."""
    ticker = str(position.get("ticker", "?"))
    contracts = int(position.get("contracts", 1) or 1)
    strike = float(position.get("strike", 0) or 0)
    expiration = str(position.get("expiration", ""))
    t = _years_to_expiration(expiration)
    dte = _dte_int(expiration)
    sigma = _clamp_iv(iv)

    # Invalid-input guards
    if not _PY_VOLLIB_AVAILABLE:
        return PositionGreeks(
            ticker=ticker, position_type="csp", quantity=contracts,
            strike=strike, spot=spot, iv=sigma, dte=dte,
            valid=False, reason="py_vollib unavailable",
        )
    if spot < SPOT_MIN or strike < STRIKE_MIN or t <= 0:
        return PositionGreeks(
            ticker=ticker, position_type="csp", quantity=contracts,
            strike=strike, spot=spot, iv=sigma, dte=dte,
            valid=False,
            reason=("expired" if t <= 0 else "bad inputs"),
        )

    # Long-put Greeks from py_vollib (flag='p')
    long_d = _safe_greek(_bs_delta, 'p', spot, strike, t, r, sigma)
    long_g = _safe_greek(_bs_gamma, 'p', spot, strike, t, r, sigma)
    long_v = _safe_greek(_bs_vega,  'p', spot, strike, t, r, sigma)
    long_th = _safe_greek(_bs_theta, 'p', spot, strike, t, r, sigma)

    # SHORT put flips delta/gamma/theta signs (theta works for us),
    # vega is also flipped (short vol exposure = negative vega).
    # Then scale by contracts * 100 shares.
    mult = -contracts * SHARES_PER_CONTRACT
    return PositionGreeks(
        ticker=ticker, position_type="csp", quantity=contracts,
        strike=strike, spot=spot, iv=sigma, dte=dte,
        delta=long_d * mult,
        gamma=long_g * mult,
        vega=long_v * mult,
        theta=long_th * mult,
        valid=True, reason=f"short put {dte}DTE",
    )


def compute_cc_greeks(
    position: dict, spot: float, iv: Optional[float] = None,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> Optional[PositionGreeks]:
    """SHORT call attached to a stock position.

    Input is a STOCK position dict that has cc_strike + cc_expiration
    set. Returns None if no CC is attached.
    """
    cc_strike = position.get("cc_strike")
    cc_exp = position.get("cc_expiration")
    if not cc_strike or not cc_exp:
        return None

    ticker = str(position.get("ticker", "?"))
    shares = int(position.get("shares", 0) or 0)
    contracts = shares // SHARES_PER_CONTRACT  # 100 shares = 1 contract
    if contracts <= 0:
        return PositionGreeks(
            ticker=ticker, position_type="cc", quantity=0,
            valid=False, reason="insufficient shares for CC",
        )

    strike = float(cc_strike)
    t = _years_to_expiration(str(cc_exp))
    dte = _dte_int(str(cc_exp))
    sigma = _clamp_iv(iv)

    if not _PY_VOLLIB_AVAILABLE or spot < SPOT_MIN or strike < STRIKE_MIN or t <= 0:
        return PositionGreeks(
            ticker=ticker, position_type="cc", quantity=contracts,
            strike=strike, spot=spot, iv=sigma, dte=dte,
            valid=False,
            reason=("expired" if t <= 0 else "bad inputs / py_vollib missing"),
        )

    long_d = _safe_greek(_bs_delta, 'c', spot, strike, t, r, sigma)
    long_g = _safe_greek(_bs_gamma, 'c', spot, strike, t, r, sigma)
    long_v = _safe_greek(_bs_vega,  'c', spot, strike, t, r, sigma)
    long_th = _safe_greek(_bs_theta, 'c', spot, strike, t, r, sigma)

    mult = -contracts * SHARES_PER_CONTRACT  # short
    return PositionGreeks(
        ticker=ticker, position_type="cc", quantity=contracts,
        strike=strike, spot=spot, iv=sigma, dte=dte,
        delta=long_d * mult,
        gamma=long_g * mult,
        vega=long_v * mult,
        theta=long_th * mult,
        valid=True, reason=f"short call {dte}DTE",
    )


# ── Portfolio aggregation ────────────────────────────────────────

def compute_portfolio_greeks(
    positions: list[dict],
    spot_fetcher: Callable[[str], Optional[float]],
    iv_fetcher: Optional[Callable[[str, float, str], Optional[float]]] = None,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> PortfolioGreeks:
    """
    Aggregate Greeks across all OPEN positions.

    Args:
        positions: list of position dicts (as stored in data/positions.json).
        spot_fetcher: callable(ticker) -> current price. Called once per ticker.
        iv_fetcher: optional callable(ticker, strike, expiration) -> IV.
            If None or returns None, we fall back to DEFAULT_IV. (Traderbot
            has iv_rank.py which can seed this.)
        r: risk-free rate.

    Returns:
        PortfolioGreeks with per-position breakdown.
    """
    portfolio = PortfolioGreeks(ts=datetime.now(timezone.utc).isoformat())
    spot_cache: dict[str, Optional[float]] = {}

    def _get_spot(tkr: str) -> Optional[float]:
        if tkr not in spot_cache:
            try:
                spot_cache[tkr] = spot_fetcher(tkr)
            except Exception as e:
                log_event("portfolio_greeks", "spot_fetch_failed", {
                    "ticker": tkr,
                    "error": f"{type(e).__name__}: {str(e)[:120]}",
                }, result="failed")
                spot_cache[tkr] = None
        return spot_cache[tkr]

    def _get_iv(tkr: str, strike: float, exp: str) -> Optional[float]:
        if iv_fetcher is None:
            return None
        try:
            return iv_fetcher(tkr, strike, exp)
        except Exception as e:
            log_event("portfolio_greeks", "iv_fetch_failed", {
                "ticker": tkr, "strike": strike, "exp": exp,
                "error": f"{type(e).__name__}: {str(e)[:120]}",
            }, result="failed")
            return None

    for p in positions:
        if not isinstance(p, dict):
            continue
        if p.get("status") != "open":
            continue

        ptype = p.get("type", "")
        ticker = str(p.get("ticker", "?"))

        try:
            if ptype == "stock":
                # Long stock always contributes
                stock_pg = compute_stock_greeks(p)
                portfolio.positions.append(stock_pg)
                # Detect attached CC
                spot = _get_spot(ticker)
                if spot is None:
                    spot = float(p.get("entry_price", 0) or 0)
                iv_cc = _get_iv(ticker, float(p.get("cc_strike", 0) or 0),
                               str(p.get("cc_expiration", "")))
                cc_pg = compute_cc_greeks(p, spot=spot, iv=iv_cc, r=r)
                if cc_pg is not None:
                    portfolio.positions.append(cc_pg)

            elif ptype == "csp":
                spot = _get_spot(ticker)
                if spot is None or spot < SPOT_MIN:
                    # We can't price without a spot — record as invalid
                    portfolio.positions.append(PositionGreeks(
                        ticker=ticker, position_type="csp",
                        quantity=int(p.get("contracts", 1) or 1),
                        strike=float(p.get("strike", 0) or 0),
                        valid=False, reason="spot unavailable",
                    ))
                    continue
                iv = _get_iv(ticker, float(p.get("strike", 0) or 0),
                           str(p.get("expiration", "")))
                portfolio.positions.append(
                    compute_csp_greeks(p, spot=spot, iv=iv, r=r)
                )

            else:
                # Unknown type — log and skip (no speculative math)
                log_event("portfolio_greeks", "unknown_position_type", {
                    "ticker": ticker, "type": ptype,
                }, result="warning")

        except Exception as e:
            log.exception("Greeks compute failed for %s", ticker)
            log_event("portfolio_greeks", "position_compute_failed", {
                "ticker": ticker, "type": ptype,
                "error": f"{type(e).__name__}: {str(e)[:120]}",
            }, result="failed")
            portfolio.positions.append(PositionGreeks(
                ticker=ticker, position_type=ptype,
                quantity=0, valid=False,
                reason=f"compute_error: {type(e).__name__}",
            ))

    # Aggregate totals over VALID positions only
    for pg in portfolio.positions:
        if not pg.valid:
            portfolio.invalid_count += 1
            continue
        portfolio.total_delta += pg.delta
        portfolio.total_gamma += pg.gamma
        portfolio.total_vega += pg.vega
        portfolio.total_theta += pg.theta
        portfolio.gross_delta += abs(pg.delta)

    log_event("portfolio_greeks", "compute_complete", {
        "position_count": len(portfolio.positions),
        "invalid_count": portfolio.invalid_count,
        "total_delta": round(portfolio.total_delta, 2),
        "total_vega": round(portfolio.total_vega, 2),
        "total_theta": round(portfolio.total_theta, 2),
    }, result="success")

    return portfolio


# ── Risk-sizing helper ───────────────────────────────────────────

def vega_sized_contracts(
    portfolio_vega_limit: float,
    current_portfolio_vega: float,
    per_contract_vega: float,
    max_contracts: int = 10,
) -> int:
    """
    How many contracts can we add before breaching a vega-risk cap?

    Args:
        portfolio_vega_limit: max allowed |total_vega| for the portfolio (dollars
            P/L per 1 vol-point).
        current_portfolio_vega: current aggregated vega (signed).
        per_contract_vega: vega of ONE contract of the candidate trade (signed;
            short put is negative).
        max_contracts: hard cap.

    Returns:
        integer contracts in [0, max_contracts].

    Rationale:
        We cap |total_vega|. New position pushes the portfolio further in its
        own direction — compute remaining headroom on that side.
    """
    if per_contract_vega == 0 or portfolio_vega_limit <= 0:
        return 0
    # Direction the candidate will push total_vega
    direction = -1 if per_contract_vega < 0 else 1
    # Signed limit on that side: -limit if shorting vol, +limit if long
    target_limit = direction * portfolio_vega_limit
    # Headroom (could be negative → zero out)
    if direction > 0:
        headroom = target_limit - current_portfolio_vega
    else:
        headroom = current_portfolio_vega - target_limit  # room to go more negative
    if headroom <= 0:
        return 0
    raw = int(headroom / abs(per_contract_vega))
    return max(0, min(max_contracts, raw))
