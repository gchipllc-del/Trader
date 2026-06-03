"""
Stock-strategy Backtester — Tier S #1 (2026-04-25).

Replays the stock-trading strategy against historical OHLCV data so we can
A/B test parameter changes BEFORE flipping them in production. Without this
every parameter tweak costs 2 weeks of empirical drift.

Reuses the live screening machinery where possible:
    - lib.quant_screener.score_ticker (pure function on bars)
    - lib.zones.detect_zones (pure)
    - lib.candlestick.get_latest_signal (pure)
    - lib.momentum.analyze_momentum (pure)
    - lib.kelly.kelly_position_size (pure)

Skipped (live-data signals; defer to live trading or future work):
    - Bayesian forecaster (calibrated on live trade history)
    - Kronos forecaster (could be added; needs cache rebuild on historical bars)
    - News sentiment (no historical news cache)
    - Earnings proximity (no historical earnings calendar)
    - LLM analyst (live API calls, not deterministic on history)

Output:
    - BacktestReport with total return, win rate, avg trade, max drawdown,
      sharpe, equity curve, full trade list
    - Compares to buy-and-hold SPY (passive baseline)

Usage:
    from lib.stock_backtest import run_backtest
    report = run_backtest(
        tickers=["F", "BAC", ...],
        days_back=180,
        starting_capital=1500,
        params={"min_composite_score": 7, "max_concurrent": 5, "kelly_fraction": 0.5},
    )
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd

from lib.alpaca_client import AlpacaClient
from lib.audit import log_event


# ── Result types ─────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """Single closed trade."""
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    realized_pnl: float
    pnl_pct: float
    composite_score: int
    close_reason: str  # "target_hit", "stop_loss", "trailing_stop", "end_of_window"


@dataclass
class BacktestReport:
    """Aggregated backtest results."""
    starting_capital: float
    ending_capital: float
    total_return: float            # Decimal, e.g. 0.15 = +15%
    annualized_return: float
    max_drawdown: float            # Most negative peak-to-trough decimal
    sharpe_ratio: float            # Annualized, risk-free rate 0
    win_rate: float                # Fraction of trades closed positive
    total_trades: int
    avg_trade_pct: float           # Mean trade pnl_pct
    avg_winner_pct: float
    avg_loser_pct: float
    profit_factor: float           # Sum gains / abs(sum losses)
    days: int
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    spy_buy_hold_return: float = 0.0
    params_used: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Backtest: {self.days} days, starting ${self.starting_capital:.0f}",
            f"  Final equity:     ${self.ending_capital:.2f}",
            f"  Total return:     {self.total_return*100:+.2f}%",
            f"  Annualized:       {self.annualized_return*100:+.2f}%",
            f"  Max drawdown:     {self.max_drawdown*100:.2f}%",
            f"  Sharpe (ann):     {self.sharpe_ratio:.2f}",
            f"  Trades closed:    {self.total_trades}",
            f"  Win rate:         {self.win_rate*100:.1f}%",
            f"  Avg trade:        {self.avg_trade_pct*100:+.2f}%",
            f"  Avg winner:       {self.avg_winner_pct*100:+.2f}%",
            f"  Avg loser:        {self.avg_loser_pct*100:+.2f}%",
            f"  Profit factor:    {self.profit_factor:.2f}",
            f"  vs SPY buy-hold:  {self.spy_buy_hold_return*100:+.2f}%",
        ]
        return "\n".join(lines)


# ── Position state during simulation ─────────────────────────────

@dataclass
class _SimPosition:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    target_price: float
    stop_loss: float
    trailing_stop_pct: float
    composite_score: int
    high_water_mark: float
    partial_exited: bool = False


# ── Core simulation ──────────────────────────────────────────────

def _slice_to_date(df: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    """Return rows of df with index <= end_date. Tolerates non-tz indexes."""
    if df.empty:
        return df
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is None:
        end = end_date.tz_localize(None) if hasattr(end_date, "tz_localize") else end_date
    else:
        end = end_date
    return df[df.index <= end]


def _score_candidate(
    ticker: str,
    daily_slice: pd.DataFrame,
    params: dict,
    sim_date: pd.Timestamp | None = None,
    enable_kronos: bool = False,
    enable_news: bool = False,
    enable_llm: bool = False,
    enable_bayesian: bool = True,
) -> dict | None:
    """Run the full screening pipeline for a single ticker as-of a sim date.

    Mirrors lib.stock_engine.score_stock_buy but:
      - Reads parameters from `params` (override-friendly for A/B)
      - Operates on the pre-sliced DataFrame (no live data)
      - Optionally enriches with Kronos / news / LLM / Bayesian signals
        (all gated by enable_* flags so the simple backtest stays fast)
    """
    try:
        from lib.zones import detect_zones, get_nearest_support, get_nearest_resistance
        from lib.candlestick import get_latest_signal
        from lib.momentum import analyze_momentum
        from lib.screener import _load_strategy_config  # for trend helper
    except Exception:
        return None

    if len(daily_slice) < 50:
        return None

    closes = daily_slice["close"].values
    current_price = float(closes[-1])

    # ── Turtle entry pre-filter (2026-05-26) ───────────────────────────
    # Adapted from lib/turtle_signal. When ``require_turtle_entry`` is
    # set in params (default OFF for back-compat), a candidate must
    # ALSO clear:
    #   - long regime: today's close > 200-period SMA
    #   - donchian:    today's close > prior 40-bar high
    # Otherwise return None (signal silenced). ATR-based stop overlay
    # is left to caller; we keep the existing stop_pct flow below.
    #
    # The cron's existing momentum / composite scoring runs UNCHANGED
    # afterward — Turtle is a quality gate, not a replacement.
    # entry_kind tracks which strategy fired ("breakout" | "dipbuy").
    # Stop / target are overridden later if entry_kind == "dipbuy".
    entry_kind: str | None = None
    dipbuy_data: dict | None = None
    if params.get("require_turtle_entry", False):
        regime_window = params.get("turtle_regime_window", 200)
        breakout_window = params.get("turtle_breakout_window", 40)
        if len(closes) <= max(regime_window, breakout_window):
            return None
        sma_n = float(pd.Series(closes).rolling(regime_window).mean().iloc[-1])
        long_regime = not pd.isna(sma_n) and current_price > sma_n
        prior_high = float(
            pd.Series(closes[:-1]).rolling(breakout_window).max().iloc[-1]
        )
        breakout_up = not pd.isna(prior_high) and current_price > prior_high
        turtle_passed = long_regime and breakout_up

        if turtle_passed:
            entry_kind = "breakout"
        else:
            # 2026-05-27: dip-buy MEAN-REVERSION fallback (mirror of
            # stock_engine.score_stock_buy). Try dipbuy on the same bar
            # if Turtle didn't fire. Mutually exclusive with Turtle
            # (a bar can't both break a 40-bar high AND be RSI≤30).
            if params.get("enable_dipbuy", True):
                try:
                    from lib.dipbuy_signal import dipbuy_signal
                    highs_l = daily_slice["high"].astype(float).tolist()
                    lows_l = daily_slice["low"].astype(float).tolist()
                    vols_l = (daily_slice["volume"].astype(float).tolist()
                              if "volume" in daily_slice.columns else None)
                    closes_l = closes.tolist() if hasattr(closes, "tolist") else list(closes)
                    dip = dipbuy_signal(
                        closes_l, highs_l, lows_l, vols_l,
                        rsi_threshold=float(params.get("dipbuy_rsi_threshold", 30)),
                        ma_touch_pct=float(params.get("dipbuy_ma_touch_pct", 0.02)),
                        regime_window=regime_window,
                    )
                    if dip["fire"]:
                        entry_kind = "dipbuy"
                        dipbuy_data = dip
                except Exception:
                    pass
            if entry_kind is None:
                return None

    # ── Volume confirmation gate (2026-05-27) ──────────────────────────
    # Mirror of stock_engine.score_stock_buy volume check. Turtle breakouts
    # on dead volume usually fail — require today's volume >=
    # ``volume_confirmation_multiplier`` × N-day average (default 1.0 →
    # at least average). Set to 1.5 to demand a true volume thrust.
    if params.get("enable_volume_confirmation", True):
        try:
            vol_window = int(params.get("volume_confirmation_window", 20))
            vol_mult = float(params.get("volume_confirmation_multiplier", 1.0))
            if "volume" in daily_slice.columns and len(daily_slice) > vol_window:
                vols = daily_slice["volume"].astype(float).values
                today_vol = float(vols[-1])
                avg_vol = float(sum(vols[-(vol_window + 1):-1]) / vol_window)
                if avg_vol > 0 and today_vol < avg_vol * vol_mult:
                    return None
        except Exception:
            pass  # fail open

    # Build a weekly-ish frame by resampling daily to 5-day buckets
    weekly = daily_slice["close"].resample("W").last().to_frame("close")
    weekly["high"] = daily_slice["high"].resample("W").max()
    weekly["low"] = daily_slice["low"].resample("W").min()
    weekly["open"] = daily_slice["open"].resample("W").first()
    weekly["volume"] = daily_slice["volume"].resample("W").sum()
    weekly = weekly.dropna()
    if len(weekly) < 10:
        return None

    # ── Trend score (0-3) — simplified version ──
    # Use 20d/50d MA cross + price vs 50d
    ma20 = pd.Series(closes).rolling(20).mean().iloc[-1]
    ma50 = pd.Series(closes).rolling(50).mean().iloc[-1] if len(closes) >= 50 else ma20
    trend_score = 0
    if current_price > ma20:
        trend_score += 1
    if ma20 > ma50:
        trend_score += 1
    if pd.Series(closes).iloc[-1] > pd.Series(closes).iloc[-20] * 1.0:
        trend_score += 1

    # ── Level score (0-3) — proximity to nearest support ──
    # Also capture nearest resistance for the profit-target snap below
    # (matches live behavior in stock_engine.py: target = resistance level
    # when available, falls back to current_price × (1 + default_target_pct)).
    nearest_resistance = None
    try:
        zones = detect_zones(daily_slice, current_price)
        support = get_nearest_support(zones, current_price)
        nearest_resistance = get_nearest_resistance(zones, current_price)
        level_score = 0
        if support:
            dist = abs(current_price - support.level) / support.level
            tiers = params.get("support_distance_tiers", [0.04, 0.07, 0.10])
            if dist < tiers[0]:
                level_score = 3
            elif dist < tiers[1]:
                level_score = 2
            elif dist < tiers[2]:
                level_score = 1
    except Exception:
        level_score = 0

    # ── Signal score (0-3) — bullish candlestick on recent bars ──
    bullish = ["hammer", "bullish_engulfing", "morning_star",
               "dragonfly_doji", "bullish_harami", "tweezers_bottom"]
    signal_score = 0
    try:
        sig = get_latest_signal(daily_slice, "bullish", bullish)
        if sig:
            signal_score = sig.strength
    except Exception:
        pass

    # ── Momentum score (0-4) — RSI + MACD + volume ratio + ROC ──
    momentum_score = 0
    try:
        mom = analyze_momentum(daily_slice)
        if mom:
            momentum_score = mom.momentum_score
    except Exception:
        pass

    composite = trend_score + level_score + signal_score + momentum_score
    min_score = params.get("min_composite_score", 7)

    # 2026-05-30: parity-fix with lib/stock_engine.score_stock_buy.
    # Previously the backtest only checked composite >= min_score, which
    # let through high-composite candidates with NO candlestick signal
    # AND in downtrends. Live engine requires BOTH:
    #   - weekly_trend.direction != "downtrend"
    #   - has_signal (signal_score >= 1) OR has_strong_momentum
    # Without these the backtest fires 6x more trades than live, most of
    # them in adverse setups. This was the cause of the -27% backtest
    # result on the reverted config — backtest was punishing live for
    # trades live wouldn't take.
    momentum_only_min = params.get("momentum_only_min_score", 3)
    has_signal = signal_score >= 1
    has_strong_momentum = momentum_score >= momentum_only_min
    allow_momentum_only = params.get("allow_momentum_only", True)

    # Downtrend detection — live uses lib/trend.multi_timeframe_analysis;
    # we approximate with the same MA-stack rule the live trend module
    # uses internally: price < MA20 < MA50 → confirmed downtrend.
    not_downtrend = True
    try:
        wma20 = pd.Series(closes).rolling(20).mean().iloc[-1]
        wma50 = pd.Series(closes).rolling(50).mean().iloc[-1] if len(closes) >= 50 else wma20
        if current_price < wma20 and current_price < wma50 and wma20 < wma50:
            not_downtrend = False
    except Exception:
        pass

    if not not_downtrend:
        return None  # live wouldn't trade in a downtrend, neither should backtest

    if composite < min_score:
        # Allow momentum-only path when score is decent on momentum alone
        if not (allow_momentum_only and has_strong_momentum):
            return None

    # Even at composite >= min_score, live requires has_signal OR strong momentum
    if allow_momentum_only:
        if not (has_signal or has_strong_momentum):
            return None
    else:
        if not has_signal:
            return None

    # Confluence gate moved to AFTER bayesian_data computation below so
    # the Bayesian signal can actually vote — see end of function.

    stop_pct = params.get("stop_loss_pct", 0.035)
    target_pct = params.get("default_target_pct", 0.10)
    # Live snaps target to nearest resistance when available; fallback is
    # current_price × (1 + target_pct). Previously the backtest used the
    # flat fallback unconditionally, which silently inflated avg_winner
    # for any variant that widened default_target_pct (e.g. wider_target).
    if nearest_resistance is not None and nearest_resistance.level > current_price:
        target_price = float(nearest_resistance.level)
    else:
        target_price = current_price * (1 + target_pct)

    # 2026-05-27: dip-buy stop/target override (mirror of stock_engine).
    # Mean-reversion entries inside a pullback need a wider ATR-based
    # stop and don't aim at the (above-price) resistance zone — they
    # aim 1.5R above current. This OVERRIDES the breakout-style stop
    # and target_price set above for the single trade.
    if entry_kind == "dipbuy" and dipbuy_data is not None:
        stop_pct = float(dipbuy_data["stop_pct"])
        target_pct_override = float(dipbuy_data["target_pct"])
        target_price = current_price * (1 + target_pct_override)

    # ── Optional enrichment signals (Tier S — all cached on first call) ──
    kronos_data = None
    news_data = None
    llm_data = None
    bayesian_data = None

    if enable_kronos and sim_date is not None:
        try:
            from lib.historical_kronos import get_historical_kronos
            kronos_data = get_historical_kronos(ticker, sim_date, daily_slice)
            if kronos_data:
                veto = params.get("kronos_veto_return", -0.02)
                if kronos_data.get("expected_return", 0) < veto:
                    return None  # Kronos says strongly bearish — skip
        except Exception:
            pass

    if enable_news and sim_date is not None:
        try:
            from lib.historical_news import get_historical_sentiment
            news_data = get_historical_sentiment(ticker, sim_date)
            if news_data and news_data.get("sentiment", 0.5) < 0.25 \
                    and news_data.get("confidence", 0) > 0.3:
                return None  # Strongly bearish news with confidence — skip
        except Exception:
            pass

    if enable_llm and sim_date is not None:
        try:
            from lib.historical_llm import get_historical_llm
            llm_data = get_historical_llm(
                ticker, sim_date, daily_slice,
                target_pct=target_pct, horizon_days=21,
            )
        except Exception:
            pass

    if enable_bayesian:
        try:
            from lib.bayesian_forecaster import forecast_stock
            bayesian_data = forecast_stock(
                ticker=ticker,
                composite_score=composite,
                trend_score=trend_score,
                level_score=level_score,
                signal_score=signal_score,
                momentum_score=momentum_score,
                kronos_expected_return=kronos_data.get("expected_return") if kronos_data else None,
                kronos_confidence=kronos_data.get("confidence", 0.5) if kronos_data else 0.5,
                news_sentiment=news_data.get("sentiment") if news_data else None,
                news_confidence=news_data.get("confidence", 0.5) if news_data else 0.5,
            )
        except Exception:
            pass

    # ── Confluence gate (2026-05-26) ───────────────────────────────────
    # Now that bayesian_data, kronos, news, and (in live mode) PEAD are
    # all populated, run the multi-signal agreement check.
    # confluence_filter handles signal availability gracefully — missing
    # signals count as "skipped", not as votes against.
    if params.get("require_confluence", False):
        from lib.confluence_filter import confluence_check, should_fire
        # Normalize bayesian_data for the confluence check (object → dict)
        bayes_dict = None
        if bayesian_data is not None:
            wp = getattr(bayesian_data, "win_probability", None)
            if wp is not None:
                bayes_dict = {"win_prob": float(wp)}
        candidate_for_conf = {
            "ticker": ticker,
            "current_price": current_price,
        }
        conf = confluence_check(
            ticker=ticker, daily_slice=daily_slice,
            candidate=candidate_for_conf, params=params,
            bayesian_data=bayes_dict,
            sim_date=sim_date,  # routes PEAD to historical_pead.get_historical_pead
        )
        fire, size_mult, conf_reason = should_fire(conf, params)
        if not fire:
            return None

    return {
        "ticker": ticker,
        "entry_kind": entry_kind or "breakout",   # NEW — breakout | dipbuy
        "current_price": current_price,
        "composite_score": composite,
        "trend_score": trend_score,
        "level_score": level_score,
        "signal_score": signal_score,
        "momentum_score": momentum_score,
        "target_price": target_price,
        "stop_loss": current_price * (1 - stop_pct),
        "trailing_stop_pct": params.get("trailing_stop_pct", 0.02),
        "kronos_expected_return": kronos_data.get("expected_return") if kronos_data else None,
        "news_sentiment": news_data.get("sentiment") if news_data else None,
        "llm_probability": llm_data.get("probability") if llm_data else None,
        "bayesian_win_prob": (
            getattr(bayesian_data, "win_probability", None) if bayesian_data else None
        ),
    }


def _kelly_size(
    candidate: dict,
    portfolio_value: float,
    cash: float,
    params: dict,
) -> int:
    """Kelly-sized share count, clamped by max_position_pct and cash."""
    from lib.kelly import kelly_position_size

    max_pos_pct = params.get("max_position_pct", 0.30)
    fraction = params.get("kelly_fraction", 0.50)

    sizing = kelly_position_size(
        portfolio_value=portfolio_value,
        current_price=candidate["current_price"],
        target_price=candidate["target_price"],
        stop_loss=candidate["stop_loss"],
        composite_score=candidate["composite_score"],
        max_position_pct=max_pos_pct,
        fraction=fraction,
    )
    shares = int(sizing.get("shares", 0))
    if shares <= 0:
        return 0
    # Don't overspend cash
    cost = shares * candidate["current_price"]
    if cost > cash:
        shares = int(cash / candidate["current_price"])
    return max(0, shares)


def _check_exits(
    pos: _SimPosition,
    bar: pd.Series,
    params: dict,
) -> tuple[float | None, str, int | None]:
    """
    Return (exit_price, reason, partial_shares).

    `partial_shares` is None for full exits and an int when only some shares
    are sold (scale-out). The caller reduces pos.shares and keeps the
    position open when partial_shares is set.

    Logic order (mirrors live monitor):
      1. Scale-out (partial) at partial_exit_threshold if enable_scale_out
         and not yet partial_exited
      2. Stop loss hit on the day's low → exit at stop_loss price (worst case)
      3. Target hit on the day's high → exit at target_price
      4. Update high-water; trailing stop hit → exit at trailing price
         (tiered widths from trailing_stop_tiered when enabled)

    2026-05-26: paper-to-live realism additions:
      - exit_slippage_pct (default 0.0005, 5 bps): live SELL market orders
        fill at BID, not mid/close. Applied to every exit price.
      - gap_open_stops (default True): if today's OPEN gapped below the
        stop_loss, exit at the OPEN price (worst-case real outcome),
        not at stop_loss.

    2026-05-27: live-parity additions to close out the backtest-vs-live
    divergence found while validating the wider_target variant:
      - scale_out: partial exit at partial_exit_threshold (live default +7%)
      - trailing_stop_tiered: trail tightens as PnL grows (lib/stock_engine
        applies these in the live monitor; previously backtest only used
        the flat trailing_stop_pct, so any winners-let-run effect was
        invisible).
    """
    high = float(bar["high"])
    low = float(bar["low"])
    open_p = float(bar["open"]) if "open" in bar else float(bar["close"])
    close = float(bar["close"])

    exit_slip = float(params.get("exit_slippage_pct", 0.0005))
    gap_aware = bool(params.get("gap_open_stops", True))

    def _slip(price: float, side: str = "sell") -> float:
        # SELL fills at BID → price × (1 - slippage)
        return price * (1.0 - exit_slip) if side == "sell" else price * (1.0 + exit_slip)

    # ── Scale-out (partial exit) ───────────────────────────────────
    # Mirror live: sell partial_exit_fraction of remaining shares the
    # first time price reaches partial_exit_threshold above entry.
    # Caller is responsible for reducing pos.shares and setting
    # pos.partial_exited = True. Need ≥2 shares so we can split.
    if (params.get("enable_scale_out", False)
            and not pos.partial_exited
            and pos.shares >= 2):
        threshold = float(params.get("partial_exit_threshold", 0.07))
        fraction = float(params.get("partial_exit_fraction", 0.5))
        partial_price = pos.entry_price * (1.0 + threshold)
        if high >= partial_price:
            shares_to_sell = max(1, int(pos.shares * fraction))
            # Cap at shares-1 so the position remains alive for normal
            # stop/target/trail evaluation on subsequent bars.
            shares_to_sell = min(shares_to_sell, pos.shares - 1)
            return _slip(partial_price), "scale_out", shares_to_sell

    # Stop loss — with gap-through detection
    if low <= pos.stop_loss:
        if gap_aware and open_p < pos.stop_loss:
            return _slip(open_p), "stop_loss_gap_through", None
        return _slip(pos.stop_loss), "stop_loss", None

    # Target
    if high >= pos.target_price:
        if gap_aware and open_p > pos.target_price:
            return _slip(open_p), "target_hit_gap_up", None
        return _slip(pos.target_price), "target_hit", None

    # Trailing — with optional tiered widths
    if close > pos.high_water_mark:
        pos.high_water_mark = close

    # Pick trail width: tier ladder (live default) or flat fallback
    tiered_cfg = params.get("trailing_stop_tiered", {})
    chosen_trail = pos.trailing_stop_pct
    if tiered_cfg.get("enabled", False) and pos.high_water_mark > pos.entry_price:
        pnl_at_peak = (pos.high_water_mark - pos.entry_price) / pos.entry_price
        tiers = tiered_cfg.get("tiers", [
            {"min_pnl": 0.20, "trail": 0.005},
            {"min_pnl": 0.12, "trail": 0.008},
            {"min_pnl": 0.07, "trail": 0.012},
            {"min_pnl": 0.03, "trail": 0.018},
            {"min_pnl": 0.00, "trail": 0.025},
        ])
        for t in sorted(tiers, key=lambda x: -x["min_pnl"]):
            if pnl_at_peak >= t["min_pnl"]:
                chosen_trail = float(t["trail"])
                break
        # RSI-exhaustion tightening from live is skipped here: it needs a
        # live momentum lookup we don't recompute per-bar in backtest.

    trail_price = pos.high_water_mark * (1 - chosen_trail)
    if low <= trail_price and pos.high_water_mark > pos.entry_price * 1.02:
        if gap_aware and open_p < trail_price:
            return _slip(open_p), "trailing_stop_gap_through", None
        return _slip(trail_price), "trailing_stop", None

    return None, "", None


def run_backtest(
    tickers: list[str],
    days_back: int = 180,
    starting_capital: float = 1500.0,
    params: dict | None = None,
    spy_baseline: bool = True,
    enable_kronos: bool = False,
    enable_news: bool = False,
    enable_llm: bool = False,
    enable_bayesian: bool = True,
) -> BacktestReport:
    """
    Run a stock-strategy backtest over `days_back` days on the given universe.

    Default params come from current wheel_strategy.yaml; pass `params` to
    override individual values for A/B testing.
    """
    import yaml
    cfg_path = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"
    with open(cfg_path) as f:
        strategy = yaml.safe_load(f)
    base = strategy.get("stock_params", {}).copy()
    base["kelly_fraction"] = strategy.get("kelly", {}).get("fraction", 0.50)
    if params:
        base.update(params)
    params = base

    log_event("backtest", "run_started", {
        "tickers": tickers, "days_back": days_back,
        "starting_capital": starting_capital, "params": params,
    })

    # Fetch historical bars + WARMUP for indicator windows.
    #
    # 2026-05-27: warmup bumped from 60 → 250 days. The Turtle pre-filter
    # needs the 200-bar SMA + 40-bar Donchian high — that's 240 bars
    # minimum before the first sim date can even produce a signal. Previously
    # warmup=60 meant a 60d backtest fetched only 120d total bars, so every
    # Turtle gate call returned None (insufficient data) and the backtest
    # produced 0 trades regardless of the rest of the strategy. The 250
    # ceiling covers the longest indicator we currently use; if we ever add
    # a 500-MA, bump this again.
    #
    # Retry on empty response: Alpaca occasionally returns an empty dict
    # right after the laptop wakes from sleep, even though the same call
    # would succeed seconds later. Without this, multi-variant overnight
    # runs lose hours of work to one transient miss.
    WARMUP_DAYS = 250
    client = AlpacaClient()
    universe = list(set(tickers + (["SPY"] if spy_baseline else [])))
    fetch_days = days_back + WARMUP_DAYS
    print(f"  Fetching {fetch_days}d bars for {len(universe)} tickers...")
    bars = None
    for attempt, delay in enumerate((0, 5, 15, 45), start=1):
        if delay:
            print(f"    bars empty, retrying in {delay}s (attempt {attempt}/4)...")
            time.sleep(delay)
        bars = client.get_bars(universe, timeframe="1Day", limit=fetch_days)
        if bars:
            break
    if not bars:
        raise RuntimeError("No bars returned from Alpaca after 4 attempts")

    # Build the union of all trading dates from any ticker (most-traded universe)
    all_dates = sorted({d for df in bars.values() for d in df.index})
    sim_dates = [d for d in all_dates if d >= all_dates[-1] - pd.Timedelta(days=days_back)]
    if not sim_dates:
        raise RuntimeError("No simulation dates")

    # State
    cash = starting_capital
    positions: list[_SimPosition] = []
    closed_trades: list[BacktestTrade] = []
    equity_curve: list[tuple[pd.Timestamp, float]] = []

    max_concurrent = params.get("max_concurrent_positions", 5)
    max_per_scan = params.get("max_trades_per_scan", 3)

    print(f"  Simulating {len(sim_dates)} trading days...")

    for sim_date in sim_dates:
        # ── 1: Mark-to-market existing positions ──
        portfolio_value = cash
        for pos in positions:
            df = bars.get(pos.ticker)
            if df is None or sim_date not in df.index:
                portfolio_value += pos.shares * pos.entry_price  # stale fallback
                continue
            bar = df.loc[sim_date]
            portfolio_value += pos.shares * float(bar["close"])

        # ── 2: Check exits on existing positions ──
        still_open: list[_SimPosition] = []
        for pos in positions:
            df = bars.get(pos.ticker)
            if df is None or sim_date not in df.index:
                still_open.append(pos)
                continue
            bar = df.loc[sim_date]
            exit_price, reason, partial_shares = _check_exits(pos, bar, params)
            if exit_price is not None and partial_shares is not None:
                # Partial scale-out: book a trade on the sold shares,
                # reduce the position, leave it open for further evolution.
                sold = min(partial_shares, pos.shares - 1) if pos.shares > 1 else 0
                if sold > 0:
                    partial_pnl = (exit_price - pos.entry_price) * sold
                    partial_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                    cash += sold * exit_price
                    closed_trades.append(BacktestTrade(
                        ticker=pos.ticker,
                        entry_date=pos.entry_date,
                        entry_price=pos.entry_price,
                        exit_date=sim_date,
                        exit_price=exit_price,
                        shares=sold,
                        realized_pnl=partial_pnl,
                        pnl_pct=partial_pnl_pct,
                        composite_score=pos.composite_score,
                        close_reason=reason,
                    ))
                    pos.shares -= sold
                    pos.partial_exited = True
                still_open.append(pos)
            elif exit_price is not None:
                pnl = (exit_price - pos.entry_price) * pos.shares
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                cash += pos.shares * exit_price
                closed_trades.append(BacktestTrade(
                    ticker=pos.ticker,
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    exit_date=sim_date,
                    exit_price=exit_price,
                    shares=pos.shares,
                    realized_pnl=pnl,
                    pnl_pct=pnl_pct,
                    composite_score=pos.composite_score,
                    close_reason=reason,
                ))
            else:
                still_open.append(pos)
        positions = still_open

        # ── 3: Scan for new entries (if slots available) ──
        slots = max_concurrent - len(positions)
        if slots > 0:
            held_tickers = {p.ticker for p in positions}
            candidates = []
            for ticker in tickers:
                if ticker in held_tickers:
                    continue
                df = bars.get(ticker)
                if df is None:
                    continue
                sliced = _slice_to_date(df, sim_date)
                cand = _score_candidate(
                    ticker, sliced, params,
                    sim_date=sim_date,
                    enable_kronos=enable_kronos,
                    enable_news=enable_news,
                    enable_llm=enable_llm,
                    enable_bayesian=enable_bayesian,
                )
                if cand:
                    candidates.append(cand)

            # Rank by composite score and take top N
            candidates.sort(key=lambda c: c["composite_score"], reverse=True)
            # 2026-05-26: paper-to-live realism. Live market-order buys
            # pay the ASK, not the mid/close — built-in slippage roughly
            # equal to half the bid-ask spread plus a few bps of market
            # impact. We model this as a single ``slippage_pct`` knob
            # applied to entries (and a matching one on exits in
            # _check_exits). Default 0.0005 (5 bps) is conservative for
            # liquid large-caps; widen for small/mid caps via per-ticker
            # overrides if needed later.
            entry_slip = float(params.get("entry_slippage_pct", 0.0005))
            for cand in candidates[: min(max_per_scan, slots)]:
                shares = _kelly_size(cand, portfolio_value, cash, params)
                if shares < 1:
                    continue
                effective_entry = cand["current_price"] * (1.0 + entry_slip)
                cost = shares * effective_entry
                cash -= cost
                positions.append(_SimPosition(
                    ticker=cand["ticker"],
                    entry_date=sim_date,
                    entry_price=effective_entry,
                    shares=shares,
                    target_price=cand["target_price"],
                    stop_loss=cand["stop_loss"],
                    trailing_stop_pct=cand["trailing_stop_pct"],
                    composite_score=cand["composite_score"],
                    high_water_mark=effective_entry,
                ))

        # ── 4: Record equity ──
        equity_curve.append((sim_date, portfolio_value))

    # ── End of window: mark any remaining positions to last close ──
    for pos in positions:
        df = bars.get(pos.ticker)
        if df is None or df.empty:
            continue
        last_close = float(df["close"].iloc[-1])
        last_date = df.index[-1]
        pnl = (last_close - pos.entry_price) * pos.shares
        pnl_pct = (last_close - pos.entry_price) / pos.entry_price
        cash += pos.shares * last_close
        closed_trades.append(BacktestTrade(
            ticker=pos.ticker, entry_date=pos.entry_date, entry_price=pos.entry_price,
            exit_date=last_date, exit_price=last_close,
            shares=pos.shares, realized_pnl=pnl, pnl_pct=pnl_pct,
            composite_score=pos.composite_score, close_reason="end_of_window",
        ))

    # ── Compute metrics ──
    ending = cash
    total_return = (ending - starting_capital) / starting_capital
    days = (sim_dates[-1] - sim_dates[0]).days or 1
    annualized = (1 + total_return) ** (365.0 / days) - 1 if (1 + total_return) > 0 else -1.0

    # Max drawdown from equity curve
    peak = starting_capital
    max_dd = 0.0
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd

    # Sharpe (annualized, daily returns, rf=0)
    if len(equity_curve) >= 2:
        eq_series = pd.Series([e for _, e in equity_curve])
        daily_returns = eq_series.pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    winners = [t for t in closed_trades if t.realized_pnl > 0]
    losers = [t for t in closed_trades if t.realized_pnl <= 0]
    win_rate = len(winners) / len(closed_trades) if closed_trades else 0
    avg_pct = np.mean([t.pnl_pct for t in closed_trades]) if closed_trades else 0
    avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0
    gains = sum(t.realized_pnl for t in winners)
    abs_losses = abs(sum(t.realized_pnl for t in losers))
    profit_factor = gains / abs_losses if abs_losses > 0 else (math.inf if gains > 0 else 0)

    spy_bh = 0.0
    if spy_baseline and "SPY" in bars and not bars["SPY"].empty:
        spy_df = _slice_to_date(bars["SPY"], sim_dates[0])
        if not spy_df.empty:
            spy_start = float(spy_df["close"].iloc[-1])
        else:
            spy_start = float(bars["SPY"]["close"].iloc[0])
        spy_end = float(bars["SPY"]["close"].iloc[-1])
        spy_bh = (spy_end - spy_start) / spy_start

    report = BacktestReport(
        starting_capital=starting_capital,
        ending_capital=round(ending, 2),
        total_return=round(total_return, 4),
        annualized_return=round(annualized, 4),
        max_drawdown=round(max_dd, 4),
        sharpe_ratio=round(sharpe, 2),
        win_rate=round(win_rate, 4),
        total_trades=len(closed_trades),
        avg_trade_pct=round(float(avg_pct), 4),
        avg_winner_pct=round(float(avg_win), 4),
        avg_loser_pct=round(float(avg_loss), 4),
        profit_factor=round(profit_factor, 2) if profit_factor != math.inf else 999.0,
        days=days,
        trades=closed_trades,
        equity_curve=equity_curve,
        spy_buy_hold_return=round(spy_bh, 4),
        params_used=params,
    )

    log_event("backtest", "run_complete", {
        "starting_capital": starting_capital,
        "ending_capital": ending,
        "total_return": total_return,
        "trades": len(closed_trades),
        "win_rate": win_rate,
        "max_dd": max_dd,
    })

    return report


# ── A/B compare helper ───────────────────────────────────────────

def compare_params(
    tickers: list[str],
    days_back: int,
    starting_capital: float,
    variants: dict[str, dict],
) -> dict[str, BacktestReport]:
    """
    Run multiple parameter variants on the same data and return reports.
    `variants` is {label: param_overrides_dict}.
    """
    results = {}
    for label, params in variants.items():
        print(f"\n=== Variant: {label} ===")
        results[label] = run_backtest(
            tickers=tickers,
            days_back=days_back,
            starting_capital=starting_capital,
            params=params,
        )
        print(results[label].summary())
    return results
