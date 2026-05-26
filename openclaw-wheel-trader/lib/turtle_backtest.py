"""Turtle Trading backtest — verify the breakout strategy works on
equities (not just NQ futures from the source video).

Walks a historical bar stream day-by-day. On each bar:
  1. Compute regime (SMA200), Donchian high (40-bar), ATR(14).
  2. If no open position AND regime=long AND today's close > prior 40-bar
     high → ENTER long at today's close. Set stop at entry - 2×ATR.
  3. If open position:
     a. If today's close <= stop_price → EXIT loss, recompute next bar.
     b. If today's close >= 10-bar low (the canonical Turtle exit) →
        actually the Turtles exited longs on a 10-bar LOW break. So:
        if today's close < prior 10-bar low → EXIT.

Records every trade with entry, exit, PnL, days_held. Reports:
  - Win rate, avg win, avg loss, profit factor
  - Total return, max drawdown, Sharpe-ish
  - Trade frequency (trades/year)
  - Distribution check (does it match the asymmetric ~30% WR /
    +$4.8k:-$1.7k pattern from the source video?)

Pure Python — no pandas/numpy.

CLI:
  python main.py turtle-backtest --ticker SPY --lookback 1825   # 5y
  python main.py turtle-backtest --tickers SPY,AAPL,NVDA,MSFT,AMD
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Sequence

from lib.turtle_signal import (
    sma, donchian_break, atr, _fetch_alpaca_daily_ohlc,
    DEFAULT_REGIME_WINDOW, DEFAULT_BREAKOUT_WINDOW,
    DEFAULT_ATR_WINDOW, DEFAULT_ATR_MULTIPLIER,
)

DEFAULT_EXIT_WINDOW = 10   # Turtle long-exit Donchian (10-bar low)


def _rolling_high(values: Sequence[float], end_idx: int, window: int) -> float | None:
    """Max of values[end_idx-window:end_idx] (exclusive). None if not enough."""
    start = end_idx - window
    if start < 0:
        return None
    sl = values[start:end_idx]
    return max(sl) if sl else None


def _rolling_low(values: Sequence[float], end_idx: int, window: int) -> float | None:
    start = end_idx - window
    if start < 0:
        return None
    sl = values[start:end_idx]
    return min(sl) if sl else None


def _sma_at(values: Sequence[float], end_idx: int, window: int) -> float | None:
    start = end_idx - window
    if start < 0:
        return None
    sl = values[start:end_idx]
    return sum(sl) / window if len(sl) == window else None


def _atr_at(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
    end_idx: int, window: int = DEFAULT_ATR_WINDOW,
) -> float | None:
    """ATR over the window ending at end_idx (exclusive)."""
    if end_idx < window + 1:
        return None
    trs = []
    for i in range(end_idx - window, end_idx):
        if i <= 0:
            continue
        prev_close = closes[i - 1]
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        ))
    return sum(trs) / len(trs) if trs else None


def backtest_one(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
    regime_window: int = DEFAULT_REGIME_WINDOW,
    breakout_window: int = DEFAULT_BREAKOUT_WINDOW,
    exit_window: int = DEFAULT_EXIT_WINDOW,
    atr_window: int = DEFAULT_ATR_WINDOW,
    atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
) -> dict:
    """Run the Turtle backtest over one ticker's bar series.

    Returns a dict with the trade list + aggregate stats.
    """
    n = len(closes)
    trades = []
    open_trade: dict | None = None

    # Need warmup beyond all windows
    warmup = max(regime_window, breakout_window, atr_window) + 5
    if n <= warmup:
        return {
            "n_bars": n,
            "trades": [],
            "error": f"insufficient_history({n} bars, need > {warmup})",
        }

    for i in range(warmup, n):
        today_close = closes[i]

        # Position management — must come BEFORE entry check (Turtles
        # exit on the same bar that breaks the exit level)
        if open_trade is not None:
            # Stop loss (intraday — use today's low for realism)
            stop = open_trade["stop_price"]
            if lows[i] <= stop:
                # Exit at stop (conservative — assumes we got filled at stop)
                open_trade["exit_idx"] = i
                open_trade["exit_price"] = stop
                open_trade["exit_reason"] = "stop_loss"
                open_trade["bars_held"] = i - open_trade["entry_idx"]
                open_trade["pnl_pct"] = (
                    stop - open_trade["entry_price"]
                ) / open_trade["entry_price"]
                trades.append(open_trade)
                open_trade = None
            else:
                # 10-bar exit Donchian (canonical Turtle long-exit)
                exit_low = _rolling_low(closes, i, exit_window)
                if exit_low is not None and today_close < exit_low:
                    open_trade["exit_idx"] = i
                    open_trade["exit_price"] = today_close
                    open_trade["exit_reason"] = "donchian_exit"
                    open_trade["bars_held"] = i - open_trade["entry_idx"]
                    open_trade["pnl_pct"] = (
                        today_close - open_trade["entry_price"]
                    ) / open_trade["entry_price"]
                    trades.append(open_trade)
                    open_trade = None

        # Entry — only fire if no open position
        if open_trade is None:
            sma200 = _sma_at(closes, i, regime_window)
            if sma200 is None or today_close <= sma200:
                continue  # not in long regime
            prior_high = _rolling_high(closes, i, breakout_window)
            if prior_high is None or today_close <= prior_high:
                continue  # no breakout
            atr_val = _atr_at(highs, lows, closes, i, atr_window)
            if atr_val is None or atr_val <= 0:
                continue
            open_trade = {
                "entry_idx": i,
                "entry_price": today_close,
                "stop_price": today_close - atr_multiplier * atr_val,
                "atr_at_entry": atr_val,
            }

    # If still open at end, close at the last close (mark-to-market)
    if open_trade is not None:
        open_trade["exit_idx"] = n - 1
        open_trade["exit_price"] = closes[-1]
        open_trade["exit_reason"] = "end_of_data"
        open_trade["bars_held"] = (n - 1) - open_trade["entry_idx"]
        open_trade["pnl_pct"] = (
            closes[-1] - open_trade["entry_price"]
        ) / open_trade["entry_price"]
        trades.append(open_trade)

    # Aggregate stats
    if not trades:
        return {"n_bars": n, "trades": [], "n_trades": 0,
                "win_rate": None, "total_pnl_pct": 0.0}

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    n_w = len(wins)
    n_l = len(losses)
    avg_win = sum(t["pnl_pct"] for t in wins) / n_w if wins else 0.0
    avg_loss = sum(t["pnl_pct"] for t in losses) / n_l if losses else 0.0
    total_pnl = sum(t["pnl_pct"] for t in trades)
    # Compound (assume each trade uses full bankroll — simplified)
    eq = 1.0
    eq_curve = [1.0]
    peak = 1.0
    max_dd = 0.0
    for t in trades:
        eq *= (1 + t["pnl_pct"])
        eq_curve.append(eq)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    profit_factor = (
        abs(avg_win * n_w / (avg_loss * n_l))
        if avg_loss < 0 and n_l > 0 else float("inf")
    )

    # Bars-per-year estimate
    bars_per_year = 252
    years = max(n / bars_per_year, 0.1)
    trades_per_year = len(trades) / years

    return {
        "n_bars": n, "years": round(years, 2),
        "n_trades": len(trades),
        "wins": n_w, "losses": n_l,
        "win_rate": round(n_w / len(trades), 4) if trades else None,
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "total_pnl_pct": round(total_pnl, 4),
        "compounded_return": round(eq - 1.0, 4),
        "profit_factor": round(profit_factor, 2) if math.isfinite(profit_factor) else None,
        "max_drawdown": round(max_dd, 4),
        "trades_per_year": round(trades_per_year, 1),
        "rr_ratio": (round(abs(avg_win / avg_loss), 2) if avg_loss < 0 else None),
        "trades": trades,
    }


def backtest_ticker(ticker: str, lookback_days: int = 1825, **kwargs) -> dict:
    """Convenience wrapper — fetches OHLC then runs backtest_one."""
    highs, lows, closes = _fetch_alpaca_daily_ohlc(ticker, lookback_days)
    if not closes:
        return {"ticker": ticker, "error": "no_data"}
    result = backtest_one(highs, lows, closes, **kwargs)
    result["ticker"] = ticker
    result["lookback_days"] = lookback_days
    return result


def render_backtest(result: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"TURTLE BACKTEST — {result.get('ticker', '?')}  "
                 f"({result.get('lookback_days', '?')}d lookback)")
    lines.append("=" * 70)
    if "error" in result:
        lines.append(f"ERROR: {result['error']}")
        return "\n".join(lines)
    lines.append(f"Bars:           {result['n_bars']}  ({result['years']}y)")
    lines.append(f"Trades:         {result['n_trades']}  "
                 f"({result['trades_per_year']}/year)")
    lines.append(f"Wins/Losses:    {result['wins']} / {result['losses']}")
    wr = result.get("win_rate")
    if wr is not None:
        lines.append(f"Win rate:       {wr:.1%}")
    lines.append(f"Avg win/loss:   {result['avg_win_pct']:+.2%} / "
                 f"{result['avg_loss_pct']:+.2%}")
    rr = result.get("rr_ratio")
    if rr is not None:
        lines.append(f"R:R ratio:      {rr:.2f}  (win/abs(loss))")
    pf = result.get("profit_factor")
    if pf is not None:
        lines.append(f"Profit factor:  {pf:.2f}")
    lines.append(f"Total PnL%:     {result['total_pnl_pct']:+.2%}")
    lines.append(f"Compounded:     {result['compounded_return']:+.2%}")
    lines.append(f"Max drawdown:   {result['max_drawdown']:.2%}")
    lines.append("")
    # Sample of last 5 trades
    trades = result.get("trades") or []
    if trades:
        lines.append(f"Last {min(5, len(trades))} trades:")
        for t in trades[-5:]:
            lines.append(
                f"  entry ${t['entry_price']:.2f} → exit ${t['exit_price']:.2f}  "
                f"pnl {t['pnl_pct']:+.2%}  {t['bars_held']}d  {t['exit_reason']}"
            )
    lines.append("")
    return "\n".join(lines)


def universe_backtest(tickers: list[str], lookback_days: int = 1825) -> list[dict]:
    out = []
    for t in tickers:
        try:
            out.append(backtest_ticker(t, lookback_days=lookback_days))
        except Exception as e:
            out.append({"ticker": t, "error": str(e)[:120]})
    return out


__all__ = ["backtest_one", "backtest_ticker", "universe_backtest", "render_backtest"]


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    lookback = int(sys.argv[2]) if len(sys.argv) > 2 else 1825
    print(render_backtest(backtest_ticker(ticker, lookback_days=lookback)))
