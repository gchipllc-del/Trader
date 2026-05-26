"""
Sprint 6: Backtesting & Validation

Proves the Wheel Strategy works before risking real capital.
- Historical data pipeline (OHLCV + simulated options)
- Walk-forward validation (train 18mo, test 6mo)
- Monte Carlo simulation (1000+ permutations)
- Slippage & fee modeling
- Benchmark comparison (buy-and-hold SPY, etc.)

Source: Advanced Algorithmic Trading (Bayesian, time series, ML)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


@dataclass
class BacktestResult:
    """Results of a single backtest run."""
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    total_trades: int
    avg_trade_pnl: float
    total_premiums: float
    assignments: int
    wheel_cycles_completed: int


@dataclass
class MonteCarloResult:
    """Results of Monte Carlo simulation."""
    mean_return: float
    median_return: float
    std_return: float
    percentile_5: float     # 5th percentile (worst case)
    percentile_25: float
    percentile_75: float
    percentile_95: float    # 95th percentile (best case)
    probability_of_loss: float
    probability_of_ruin: float  # >50% drawdown
    n_simulations: int


def simulate_csp_outcome(
    stock_price: float,
    strike: float,
    premium: float,
    days_to_exp: int,
    daily_returns: np.ndarray,
) -> dict:
    """
    Simulate a single CSP from entry to expiration.

    Uses actual daily return distribution to generate price path.
    """
    # Generate price path
    path = [stock_price]
    for i in range(days_to_exp):
        idx = np.random.randint(0, len(daily_returns))
        path.append(path[-1] * (1 + daily_returns[idx]))

    final_price = path[-1]
    assigned = final_price < strike

    if assigned:
        # We buy shares at strike, but already collected premium
        cost_basis = strike - premium
        unrealized_pnl = (final_price - cost_basis) * 100
        return {
            "assigned": True,
            "premium_pnl": premium * 100,
            "cost_basis": cost_basis,
            "final_price": final_price,
            "unrealized_pnl": unrealized_pnl,
        }
    else:
        return {
            "assigned": False,
            "premium_pnl": premium * 100,
            "cost_basis": None,
            "final_price": final_price,
            "unrealized_pnl": 0,
        }


def simulate_cc_outcome(
    stock_price: float,
    strike: float,
    premium: float,
    days_to_exp: int,
    daily_returns: np.ndarray,
) -> dict:
    """Simulate a single covered call from entry to expiration."""
    path = [stock_price]
    for i in range(days_to_exp):
        idx = np.random.randint(0, len(daily_returns))
        path.append(path[-1] * (1 + daily_returns[idx]))

    final_price = path[-1]
    called_away = final_price > strike

    if called_away:
        capital_gain = (strike - stock_price) * 100
        return {
            "called_away": True,
            "premium_pnl": premium * 100,
            "capital_gain": capital_gain,
            "total_pnl": premium * 100 + capital_gain,
        }
    else:
        unrealized = (final_price - stock_price) * 100
        return {
            "called_away": False,
            "premium_pnl": premium * 100,
            "capital_gain": 0,
            "total_pnl": premium * 100 + unrealized,
        }


def _estimated_premium(
    underlying: float,
    dte: int,
    daily_returns: np.ndarray,
    *,
    delta_abs: float = 0.25,
    is_put: bool = True,
) -> float:
    """Approximate the premium of a moderately-OTM option.

    The simulation needs a premium that responds *correctly* to DTE so a
    DTE sweep produces meaningful relative results. The previous flat
    ``premium_rate = 0.015`` meant a 7-DTE put and a 35-DTE put had
    identical modeled premium — short-DTE variants looked artificially
    strong because they cycled 5× more often at the same modeled income.

    Black-Scholes intuition (deep-OTM approximation):
        premium ≈ S · σ · √T · f(|Δ|)

    where S is the underlying price, σ is annualized realized vol from
    the price series, T = DTE/365, and f(|Δ|) is a moneyness factor
    (~0.20 for a 0.25-delta contract). This gets the *shape* right:
    premium scales with √T and with vol, so longer-dated and
    higher-vol contracts pay more per contract while shorter-dated
    contracts pay more *per day* — which is the whole reason to compare
    DTE bands.

    Falls back to a flat-rate estimate (the historical default) if
    realized-vol can't be estimated from the input.
    """
    if len(daily_returns) < 10:
        return float(underlying) * 0.015  # back-compat fallback
    annualized_vol = float(np.std(daily_returns)) * np.sqrt(252)
    if annualized_vol <= 0 or not np.isfinite(annualized_vol):
        return float(underlying) * 0.015
    T = max(dte, 1) / 365.0
    # Moneyness factor — empirically calibrated against typical OTM
    # premiums for the 0.20-0.30 delta band Jesse trades. Calls (CC) are
    # priced slightly cheaper than the equivalent put for the same |Δ|
    # because of the cost-of-carry asymmetry; we ignore that and treat
    # both symmetrically — close enough for DTE-comparison purposes.
    f_delta = max(0.08, min(0.30, 0.20 + (0.25 - delta_abs) * 0.4))
    return float(underlying) * annualized_vol * np.sqrt(T) * f_delta


def run_wheel_backtest(
    daily_df: pd.DataFrame,
    initial_capital: float = 100000,
    put_delta: float = -0.25,
    call_delta: float = 0.25,
    dte: int = 35,
    premium_rate: float | None = None,  # deprecated; set non-None to force flat-rate (back-compat)
    slippage_per_contract: float = 5.0,
    assignment_fee: float = 0.0,
) -> BacktestResult:
    """
    Backtest the Wheel Strategy on historical data.

    Simplified simulation:
    - Every `dte` days, sell a CSP
    - If assigned, sell CCs until called away
    - Track cumulative P/L

    Premium model: by default, uses realized-vol × √DTE pricing (see
    ``_estimated_premium``). Pass ``premium_rate`` non-None to force
    the legacy flat-rate model (for back-compat with old callers).
    """
    daily_returns = daily_df["close"].pct_change().dropna().values
    returns = daily_returns
    prices = daily_df["close"].values

    capital = initial_capital
    peak_capital = capital
    max_drawdown = 0
    trades = []
    premiums_total = 0
    assignments = 0
    cycles = 0

    i = 0
    holding_shares = False
    cost_basis = 0

    # Daily mark-to-market equity for drawdown tracking. Without this,
    # held-share losses don't show up in max_drawdown — a CSP that
    # assigns into a stock that drops 30% looks like a "win" in raw
    # premium PnL. Equity = realized capital + unrealized P&L on any
    # held shares marked at the latest price.
    def _equity_at(price_idx: int) -> float:
        if not holding_shares:
            return capital
        unrealized = (prices[price_idx] - cost_basis) * 100
        return capital + unrealized

    while i < len(prices) - dte:
        current_price = prices[i]
        # Trade-level PnL for the about-to-be-recorded cycle. Used by the
        # win-rate calculation; we'll also adjust it for unrealized
        # share losses when a CSP gets assigned (otherwise an assignment
        # into a falling stock falsely counts as a "win").
        trade_pnl: float

        if not holding_shares:
            # Sell CSP
            strike = current_price * (1 + put_delta * 0.1)  # OTM put
            if premium_rate is not None:
                premium = strike * premium_rate
            else:
                premium = _estimated_premium(
                    strike, dte, daily_returns,
                    delta_abs=abs(put_delta), is_put=True,
                )
            exp_price = prices[min(i + dte, len(prices) - 1)]

            premium_pnl = premium * 100 - slippage_per_contract
            premiums_total += premium * 100

            if exp_price < strike:
                # Assigned — cost basis is strike less premium received
                cost_basis = strike - premium
                holding_shares = True
                assignments += 1
                # The full economic PnL of this trade is premium income
                # MINUS the immediate unrealized loss vs cost basis.
                # That makes "assigned into a falling stock" correctly
                # show as a losing trade.
                unrealized_at_assignment = (exp_price - cost_basis) * 100
                trade_pnl = premium_pnl + unrealized_at_assignment
                trades.append({
                    "type": "csp_assigned",
                    "pnl": trade_pnl,
                    "price": exp_price,
                    "premium_pnl": premium_pnl,
                    "unrealized_at_close": unrealized_at_assignment,
                })
            else:
                # Expired worthless — keep premium
                capital += premium_pnl
                trade_pnl = premium_pnl
                trades.append({"type": "csp_expired", "pnl": trade_pnl, "price": exp_price})
        else:
            # Sell CC
            strike = current_price * (1 - call_delta * 0.1)  # OTM call above
            strike = max(strike, cost_basis * 1.02)  # At least 2% above cost basis
            if premium_rate is not None:
                premium = current_price * premium_rate * 0.8  # CC premium slightly less
            else:
                # CCs are typically ~85% of the equivalent put premium for
                # the same delta — apply the same vol-based model with a
                # small discount.
                premium = _estimated_premium(
                    current_price, dte, daily_returns,
                    delta_abs=abs(call_delta), is_put=False,
                ) * 0.85
            exp_price = prices[min(i + dte, len(prices) - 1)]

            premium_pnl = premium * 100 - slippage_per_contract
            premiums_total += premium * 100

            if exp_price > strike:
                # Called away — realize cost-basis-to-strike gain plus premium.
                cap_gain = (strike - cost_basis) * 100
                trade_pnl = premium_pnl + cap_gain
                capital += trade_pnl
                holding_shares = False
                cycles += 1
                trades.append({"type": "cc_called", "pnl": trade_pnl, "price": exp_price})
            else:
                # Not called away — keep premium, hold shares to next cycle.
                # cost_basis drops by the premium received (running CC accounting).
                capital += premium_pnl
                cost_basis -= premium
                # Trade-level PnL for win-rate = premium income + the
                # price change over the cycle (since we still hold the
                # shares, the price move is the period's unrealized P&L).
                # Sign-correct: if the stock fell more than the premium,
                # the trade is correctly counted as a loss.
                trade_pnl = premium_pnl + (exp_price - current_price) * 100
                trades.append({"type": "cc_expired", "pnl": trade_pnl, "price": exp_price})

        # Daily mark-to-market for drawdown across the cycle window.
        # Walk the dte intermediate days, compute equity, update peak/dd.
        # This is the key fix — previously drawdown only saw realized
        # capital so held-share drops were invisible.
        cycle_end = min(i + dte, len(prices))
        for idx in range(i, cycle_end):
            eq = _equity_at(idx)
            peak_capital = max(peak_capital, eq)
            dd = (peak_capital - eq) / peak_capital if peak_capital > 0 else 0
            max_drawdown = max(max_drawdown, dd)

        i += dte

    # Calculate metrics
    # If we end while still holding shares, the *economic* return must
    # include the unrealized P&L on those shares (previously this was
    # only counted in drawdown, not the final figure).
    final_equity = capital
    if holding_shares and len(prices) > 0:
        final_equity = capital + (prices[-1] - cost_basis) * 100
    total_return = (final_equity - initial_capital) / initial_capital
    years = len(prices) / 252
    annualized = (1 + total_return) ** (1 / max(years, 0.1)) - 1

    trade_pnls = [t["pnl"] for t in trades]
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]

    win_rate = len(wins) / len(trades) if trades else 0
    avg_pnl = np.mean(trade_pnls) if trade_pnls else 0

    # Sharpe (simplified — excess return / vol)
    if trade_pnls:
        sharpe = np.mean(trade_pnls) / (np.std(trade_pnls) + 1e-10) * np.sqrt(252 / dte)
        downside = [p for p in trade_pnls if p < 0]
        sortino = np.mean(trade_pnls) / (np.std(downside) + 1e-10) * np.sqrt(252 / dte) if downside else sharpe
    else:
        sharpe = sortino = 0

    return BacktestResult(
        total_return=round(total_return, 4),
        annualized_return=round(annualized, 4),
        max_drawdown=round(max_drawdown, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        win_rate=round(win_rate, 4),
        total_trades=len(trades),
        avg_trade_pnl=round(avg_pnl, 2),
        total_premiums=round(premiums_total, 2),
        assignments=assignments,
        wheel_cycles_completed=cycles,
    )


def run_monte_carlo(
    daily_df: pd.DataFrame,
    n_simulations: int = 1000,
    initial_capital: float = 100000,
    **backtest_kwargs,
) -> MonteCarloResult:
    """
    Run N randomized backtests with shuffled return sequences.
    Gives probability distribution of outcomes.
    """
    results = []
    returns = daily_df["close"].pct_change().dropna().values

    for _ in range(n_simulations):
        # Shuffle returns to break temporal dependencies
        shuffled = daily_df.copy()
        shuffled_returns = np.random.choice(returns, size=len(daily_df), replace=True)
        base_price = daily_df["close"].iloc[0]
        shuffled["close"] = base_price * np.cumprod(1 + shuffled_returns)
        shuffled["high"] = shuffled["close"] * (1 + abs(np.random.randn(len(shuffled))) * 0.01)
        shuffled["low"] = shuffled["close"] * (1 - abs(np.random.randn(len(shuffled))) * 0.01)
        shuffled["open"] = shuffled["close"].shift(1).fillna(base_price)

        bt = run_wheel_backtest(shuffled, initial_capital, **backtest_kwargs)
        results.append(bt.total_return)

    results = np.array(results)

    return MonteCarloResult(
        mean_return=round(float(np.mean(results)), 4),
        median_return=round(float(np.median(results)), 4),
        std_return=round(float(np.std(results)), 4),
        percentile_5=round(float(np.percentile(results, 5)), 4),
        percentile_25=round(float(np.percentile(results, 25)), 4),
        percentile_75=round(float(np.percentile(results, 75)), 4),
        percentile_95=round(float(np.percentile(results, 95)), 4),
        probability_of_loss=round(float(np.mean(results < 0)), 4),
        probability_of_ruin=round(float(np.mean(results < -0.50)), 4),
        n_simulations=n_simulations,
    )


def compare_to_benchmark(
    daily_df: pd.DataFrame,
    wheel_result: BacktestResult,
) -> dict:
    """Compare Wheel returns to buy-and-hold."""
    prices = daily_df["close"].values
    bh_return = (prices[-1] - prices[0]) / prices[0]
    years = len(prices) / 252
    bh_annualized = (1 + bh_return) ** (1 / max(years, 0.1)) - 1

    # Buy and hold max drawdown
    peak = prices[0]
    bh_max_dd = 0
    for p in prices:
        peak = max(peak, p)
        dd = (peak - p) / peak
        bh_max_dd = max(bh_max_dd, dd)

    return {
        "wheel": {
            "total_return": wheel_result.total_return,
            "annualized": wheel_result.annualized_return,
            "max_drawdown": wheel_result.max_drawdown,
            "sharpe": wheel_result.sharpe_ratio,
        },
        "buy_and_hold": {
            "total_return": round(bh_return, 4),
            "annualized": round(bh_annualized, 4),
            "max_drawdown": round(bh_max_dd, 4),
        },
        "outperformance": round(wheel_result.annualized_return - bh_annualized, 4),
    }
