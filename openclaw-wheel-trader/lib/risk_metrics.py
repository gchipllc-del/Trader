"""
Risk-adjusted performance metrics — pure-Python finance math, zero deps.

Why we wrote this instead of pulling empyrical: empyrical's last release
was October 2020 (Quantopian shutdown), and "finance-grade security"
forbids unmaintained deps in critical paths. The formulas here are
textbook (Sharpe 1966, Sortino 1991, Young 1991 for Calmar) and should
match empyrical's output to within rounding.

All functions accept either:
  - a list of per-trade returns (e.g. [0.012, -0.005, 0.030]), OR
  - a list of {"realized_pnl": $, "entry_value": $} dicts, OR
  - a list of equity curve floats (e.g. [1500, 1510, 1505, 1530])

Convention: returns are decimal (0.01 = 1%), not percent. Annualization
factor defaults to 252 (US trading days). Pass periods_per_year=365 for
daily crypto / prediction-market series, 12 for monthly aggregates.
"""

from __future__ import annotations

import math
from typing import Sequence


# ── Helpers ──────────────────────────────────────────────────────────


def _to_returns(series: Sequence) -> list[float]:
    """Coerce mixed inputs into a flat list of float returns.

    - If items are dicts with 'realized_pnl' + 'entry_value', compute return
      as pnl/entry_value.
    - If items are floats and look like an equity curve (all > 1.0 and
      monotonically larger than typical returns), convert to per-step
      pct changes.
    - Otherwise treat as already-returns.
    """
    if not series:
        return []
    out: list[float] = []
    if isinstance(series[0], dict):
        for t in series:
            entry = float(t.get("entry_value") or 0.0)
            pnl = float(t.get("realized_pnl") or t.get("net_profit") or 0.0)
            if entry > 0:
                out.append(pnl / entry)
            elif pnl != 0.0:
                # Fallback — treat absolute pnl as the return
                out.append(pnl)
        return out

    # Floats — could be returns or an equity curve.
    # Heuristic: if ANY value > 10, it's almost certainly an equity curve
    # (real returns are sub-1000% per period; realistic equity values are
    # in the hundreds-to-millions). Returns are typically in [-1, 1].
    floats = [float(x) for x in series]
    looks_like_curve = (
        len(floats) >= 2
        and max(abs(f) for f in floats) > 10.0
    )
    if looks_like_curve:
        for i in range(1, len(floats)):
            prev = floats[i - 1]
            if prev > 0:
                out.append((floats[i] - prev) / prev)
        return out
    return floats


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _stddev(xs: Sequence[float], ddof: int = 1) -> float:
    """Sample standard deviation. ddof=1 matches numpy.std(ddof=1)."""
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - ddof)
    return math.sqrt(var)


# ── Core metrics ─────────────────────────────────────────────────────


def sharpe_ratio(
    series: Sequence,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe = (mean_excess / stdev) * sqrt(periods_per_year).

    Excess return = mean(returns) - risk_free_per_period.
    Returns 0.0 for empty / constant series (rather than NaN/Inf, which
    poison downstream Hermes math).
    """
    rs = _to_returns(series)
    if not rs:
        return 0.0
    rf_per_period = risk_free / periods_per_year if risk_free else 0.0
    excess = [r - rf_per_period for r in rs]
    sd = _stddev(excess)
    if sd == 0.0:
        return 0.0
    return (_mean(excess) / sd) * math.sqrt(periods_per_year)


def sortino_ratio(
    series: Sequence,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sortino — like Sharpe but penalizes only downside variance.

    Downside deviation = sqrt(mean(min(0, r - target)^2)).
    Returns 0.0 if no downside variance (i.e., never lost) since the
    metric is mathematically infinite there — better to surface 0 and
    let callers interpret.
    """
    rs = _to_returns(series)
    if not rs:
        return 0.0
    target_per_period = target / periods_per_year if target else 0.0
    downside = [min(0.0, r - target_per_period) ** 2 for r in rs]
    dsd = math.sqrt(_mean(downside))
    if dsd == 0.0:
        return 0.0
    return ((_mean(rs) - target_per_period) / dsd) * math.sqrt(periods_per_year)


def max_drawdown(series: Sequence) -> float:
    """
    Maximum peak-to-trough drawdown of the cumulative-return series.

    Returned as a NEGATIVE decimal (-0.20 = -20% drawdown). 0.0 if no
    drawdown observed.
    """
    rs = _to_returns(series)
    if not rs:
        return 0.0
    # Cumulative-product equity curve, normalized to 1.0
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in rs:
        equity *= 1.0 + r
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak if peak > 0 else 0.0
        if dd < worst:
            worst = dd
    return worst


def calmar_ratio(
    series: Sequence,
    periods_per_year: int = 252,
) -> float:
    """
    Calmar = annualized return / |max drawdown|.

    Returns 0.0 if no drawdown (mathematically Inf, but 0 keeps Hermes
    sane). Annualized return uses geometric compounding of mean per-period.
    """
    rs = _to_returns(series)
    if not rs:
        return 0.0
    dd = max_drawdown(rs)
    if dd == 0.0:
        return 0.0
    annualized_return = (1.0 + _mean(rs)) ** periods_per_year - 1.0
    return annualized_return / abs(dd)


def win_rate(series: Sequence) -> float:
    """Fraction of returns that are strictly positive. 0.0 for empty."""
    rs = _to_returns(series)
    if not rs:
        return 0.0
    return sum(1 for r in rs if r > 0) / len(rs)


def profit_factor(series: Sequence) -> float:
    """Sum of wins / |sum of losses|. Inf-safe: returns 0.0 if no losses."""
    rs = _to_returns(series)
    if not rs:
        return 0.0
    wins = sum(r for r in rs if r > 0)
    losses = sum(r for r in rs if r < 0)
    if losses == 0:
        return 0.0  # never lost; profit-factor is undefined
    return wins / abs(losses)


def expectancy(series: Sequence) -> float:
    """Average return per trade — basic E(X). Useful as a sanity check
    against more complex metrics."""
    return _mean(_to_returns(series))


def summary(
    series: Sequence,
    *,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> dict:
    """
    One-shot bundle of all metrics. Convenient for dashboards + Hermes.

    Returns a dict with rounded floats (4 decimal places).
    """
    rs = _to_returns(series)
    return {
        "n_trades": len(rs),
        "mean_return": round(_mean(rs), 6) if rs else 0.0,
        "stdev": round(_stddev(rs), 6) if rs else 0.0,
        "sharpe": round(sharpe_ratio(rs, risk_free, periods_per_year), 4),
        "sortino": round(sortino_ratio(rs, risk_free, periods_per_year), 4),
        "calmar": round(calmar_ratio(rs, periods_per_year), 4),
        "max_drawdown": round(max_drawdown(rs), 4),
        "win_rate": round(win_rate(rs), 4),
        "profit_factor": round(profit_factor(rs), 4),
        "expectancy": round(expectancy(rs), 6),
    }
