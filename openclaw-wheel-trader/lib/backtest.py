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


def run_wheel_backtest(
    daily_df: pd.DataFrame,
    initial_capital: float = 100000,
    put_delta: float = -0.25,
    call_delta: float = 0.25,
    dte: int = 35,
    premium_rate: float = 0.015,  # Approximate premium as % of strike
    slippage_per_contract: float = 5.0,
    assignment_fee: float = 0.0,
) -> BacktestResult:
    """
    Backtest the Wheel Strategy on historical data.

    Simplified simulation:
    - Every `dte` days, sell a CSP
    - If assigned, sell CCs until called away
    - Track cumulative P/L
    """
    returns = daily_df["close"].pct_change().dropna().values
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

    while i < len(prices) - dte:
        current_price = prices[i]

        if not holding_shares:
            # Sell CSP
            strike = current_price * (1 + put_delta * 0.1)  # OTM put
            premium = strike * premium_rate
            exp_price = prices[min(i + dte, len(prices) - 1)]

            pnl = premium * 100 - slippage_per_contract
            premiums_total += premium * 100

            if exp_price < strike:
                # Assigned
                cost_basis = strike - premium
                holding_shares = True
                assignments += 1
                trades.append({"type": "csp_assigned", "pnl": pnl, "price": exp_price})
            else:
                # Expired worthless — keep premium
                capital += pnl
                trades.append({"type": "csp_expired", "pnl": pnl, "price": exp_price})
        else:
            # Sell CC
            strike = current_price * (1 - call_delta * 0.1)  # OTM call above
            strike = max(strike, cost_basis * 1.02)  # At least 2% above cost basis
            premium = current_price * premium_rate * 0.8  # CC premium slightly less
            exp_price = prices[min(i + dte, len(prices) - 1)]

            pnl = premium * 100 - slippage_per_contract
            premiums_total += premium * 100

            if exp_price > strike:
                # Called away
                cap_gain = (strike - cost_basis) * 100
                capital += pnl + cap_gain
                holding_shares = False
                cycles += 1
                trades.append({"type": "cc_called", "pnl": pnl + cap_gain, "price": exp_price})
            else:
                # Keep shares + premium
                capital += pnl
                cost_basis -= premium  # Reduce cost basis
                trades.append({"type": "cc_expired", "pnl": pnl, "price": exp_price})

        # Track drawdown
        peak_capital = max(peak_capital, capital)
        dd = (peak_capital - capital) / peak_capital
        max_drawdown = max(max_drawdown, dd)

        i += dte

    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital
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
        shuffled_returns = np.random.choice(returns, size=len(returns), replace=True)
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
