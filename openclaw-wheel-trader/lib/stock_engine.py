"""
Phase 1: Stock Trading Engine — for building portfolio before options.

Uses the same analysis framework (trend + zones + candlestick signals)
to buy stocks at support with bullish confirmation. Sells at resistance
or trailing stop. Graduates to options (Phase 2) once portfolio reaches
the CSP collateral threshold.

This is the stepping stone to the full Wheel Strategy.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from lib.audit import log_event
from lib.alpaca_client import AlpacaClient
from lib.screener import _load_strategy_config
from lib.zones import detect_zones, get_nearest_support, get_nearest_resistance
from lib.trend import analyze_trend, multi_timeframe_analysis
from lib.candlestick import get_latest_signal, scan_patterns
from lib.iv_rank import calculate_historical_volatility
from lib.memory_palace import diary_write, remember_trade_decision, get_current_regime
from lib.circuit_breaker import check_paper_mode, check_daily_loss, CircuitBreakerTripped
from lib.quant_screener import screen_universe, print_screening_report
from lib.momentum import analyze_momentum

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
STRATEGY_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"

# Phase thresholds
PHASE_2_THRESHOLD = 5000   # Start selling CSPs on cheap stocks
PHASE_3_THRESHOLD = 10000  # Full Wheel on bigger tickers


def _load_settings() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_strategy() -> dict:
    with open(STRATEGY_PATH) as f:
        return yaml.safe_load(f)


def _load_positions() -> list[dict]:
    if not POSITIONS_PATH.exists():
        return []
    with open(POSITIONS_PATH) as f:
        return json.load(f)


def _save_positions(positions: list[dict]):
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)


def get_current_phase(portfolio_value: float) -> int:
    """Determine trading phase based on portfolio size."""
    if portfolio_value >= PHASE_3_THRESHOLD:
        return 3
    elif portfolio_value >= PHASE_2_THRESHOLD:
        return 2
    return 1


def score_stock_buy(
    ticker: str,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    current_price: float,
    max_position_value: float,
) -> dict | None:
    """
    Score a stock as a buy candidate using trend + level + signal + momentum.

    Composite score 0-13:
      - Trend alignment (0-3)
      - Support level quality (0-3)
      - Candlestick signal (0-3)
      - Momentum (0-4): RSI, MACD, volume surge, ROC

    Parameters are read from wheel_strategy.yaml so Hermes can tune them.
    """
    if len(daily_df) < 30 or len(weekly_df) < 10:
        return None

    strategy = _load_strategy()
    stock_cfg = strategy.get("stock_params", {})
    min_score = stock_cfg.get("min_composite_score", 3)
    stop_loss_pct = stock_cfg.get("stop_loss_pct", 0.05)
    default_target_pct = stock_cfg.get("default_target_pct", 0.10)
    allow_momentum_only = stock_cfg.get("allow_momentum_only", False)
    support_distance_tiers = stock_cfg.get("support_distance_tiers", [0.03, 0.05, 0.08])

    # --- Trend Score (0-3) ---
    mtf = multi_timeframe_analysis(weekly_df, daily_df)
    trend_score = mtf["alignment_score"]

    weekly_trend = mtf["weekly"]
    # Don't buy into strong downtrends (strength 3 = confirmed)
    if weekly_trend.direction == "downtrend" and weekly_trend.strength >= 3:
        return None

    # --- Level Score (0-3) ---
    zones = detect_zones(daily_df, current_price)
    nearest_support = get_nearest_support(zones, current_price)

    level_score = 0
    zone_level = 0.0
    zone_touches = 0

    if nearest_support:
        distance_to_zone = abs(current_price - nearest_support.level) / nearest_support.level
        tiers = support_distance_tiers
        if distance_to_zone < tiers[0]:
            level_score = 3
        elif distance_to_zone < tiers[1]:
            level_score = 2
        elif distance_to_zone < tiers[2]:
            level_score = 1

        zone_level = nearest_support.level
        zone_touches = nearest_support.touches

    # --- Signal Score (0-3) ---
    signal_score = 0
    pattern_name = None

    bullish_patterns = [
        "hammer", "bullish_engulfing", "morning_star",
        "dragonfly_doji", "bullish_harami", "tweezers_bottom",
    ]
    signal = get_latest_signal(daily_df, "bullish", bullish_patterns)
    if signal:
        signal_score = signal.strength
        pattern_name = signal.pattern

    # --- Momentum Score (0-4) ---
    momentum = analyze_momentum(daily_df)
    momentum_score = momentum.momentum_score if momentum else 0
    momentum_details = {}
    if momentum:
        momentum_details = {
            "rsi": momentum.rsi,
            "rsi_signal": momentum.rsi_signal,
            "macd_cross": momentum.macd_cross,
            "volume_surge": momentum.volume_surge,
            "roc_5d": momentum.roc_5d,
        }

    # --- Composite ---
    composite = trend_score + level_score + signal_score + momentum_score

    # Entry logic — more permissive in growth mode
    not_downtrend = weekly_trend.direction != "downtrend"
    has_signal = signal_score >= 1
    has_momentum = momentum_score >= 2

    if allow_momentum_only:
        # Growth mode: momentum alone (score >= 2) can trigger entry
        tradeable = composite >= min_score and not_downtrend and (has_signal or has_momentum)
    else:
        # Conservative: require candlestick signal
        tradeable = composite >= min_score and not_downtrend and has_signal

    # Calculate position size
    shares = calculate_position_size(current_price, max_position_value)
    if shares < 1:
        return None

    # Nearest resistance for profit target
    nearest_resistance = get_nearest_resistance(zones, current_price)
    target_price = nearest_resistance.level if nearest_resistance else current_price * (1 + default_target_pct)

    # Dynamic stop: tighter when momentum is strong, wider when weak
    if momentum_score >= 3:
        actual_stop_pct = stop_loss_pct * 0.8  # Tighter stop, trust the momentum
    else:
        actual_stop_pct = stop_loss_pct

    return {
        "ticker": ticker,
        "trade_type": "stock_buy",
        "current_price": round(current_price, 2),
        "shares": shares,
        "position_value": round(current_price * shares, 2),
        "trend_score": trend_score,
        "level_score": level_score,
        "signal_score": signal_score,
        "momentum_score": momentum_score,
        "composite_score": composite,
        "zone_level": zone_level,
        "zone_touches": zone_touches,
        "pattern": pattern_name,
        "momentum": momentum_details,
        "target_price": round(target_price, 2),
        "stop_loss": round(current_price * (1 - actual_stop_pct), 2),
        "tradeable": tradeable,
        "weekly_trend": weekly_trend.direction,
    }


def calculate_position_size(
    price: float,
    max_position_value: float,
    min_shares: int = 1,
) -> int:
    """
    Calculate how many shares to buy.
    Never exceed max_position_value per position.
    """
    if price <= 0:
        return 0
    shares = int(max_position_value / price)
    return max(shares, 0)


def scan_for_stocks(
    client: AlpacaClient,
    daily_data: dict[str, pd.DataFrame],
    weekly_data: dict[str, pd.DataFrame],
    portfolio_value: float,
) -> list[dict]:
    """
    Scan tickers for stock buy candidates.

    Three-gate system:
      Gate 1: Quantitative screen (Sharpe, drawdown, volatility)
      Gate 2: Technical screen (trend + zones + candlestick signals)
      Gate 3: Momentum screen (RSI, MACD, volume, ROC)

    Returns ranked list of tradeable candidates.
    """
    settings = _load_settings()
    strategy = _load_strategy()
    max_position_pct = settings["circuit_breakers"]["max_position_pct"]
    max_position_value = portfolio_value * max_position_pct
    max_concurrent = strategy.get("stock_params", {}).get("max_concurrent_positions", 5)

    # Gate 1: Quantitative screening
    quant_scores = screen_universe(
        daily_data,
        max_price=max_position_value,
        exclude_avoid=True,
    )

    print("\n  --- Quant Screening ---")
    print_screening_report(quant_scores)

    quant_passed = {s.ticker for s in quant_scores if s.verdict != "AVOID"}

    # Check existing positions — respect max concurrent
    positions = _load_positions()
    held_tickers = set(
        p["ticker"] for p in positions
        if p.get("status") in ("open", "assigned") and p.get("type") == "stock"
    )
    open_count = len(held_tickers)

    if open_count >= max_concurrent:
        log_event("stock_engine", "max_positions_reached", {"open": open_count, "max": max_concurrent})
        print(f"  Max concurrent positions reached ({open_count}/{max_concurrent})")
        return []

    slots_available = max_concurrent - open_count

    # Gate 2+3: Technical + Momentum screening (only on quant-passed tickers)
    candidates = []
    for ticker in quant_passed:
        if ticker in held_tickers:
            continue

        daily_df = daily_data.get(ticker)
        if daily_df is None or len(daily_df) < 30:
            continue

        current_price = daily_df["close"].iloc[-1]
        weekly_df = weekly_data.get(ticker)
        if weekly_df is None or len(weekly_df) < 10:
            continue

        candidate = score_stock_buy(
            ticker, daily_df, weekly_df, current_price, max_position_value,
        )
        if candidate and candidate["tradeable"]:
            # Attach quant score for ranking
            qs = next((s for s in quant_scores if s.ticker == ticker), None)
            if qs:
                candidate["quant_score"] = qs.quant_score
                candidate["sharpe"] = qs.sharpe_ratio
                candidate["max_drawdown"] = qs.max_drawdown
            candidates.append(candidate)

    # Sort by: composite score (includes momentum), then quant score
    candidates.sort(
        key=lambda c: (c["composite_score"], c.get("quant_score", 0)),
        reverse=True,
    )

    # Limit to available slots
    candidates = candidates[:slots_available]

    log_event("stock_engine", "scan_complete", {
        "quant_passed": len(quant_passed),
        "technical_passed": len(candidates),
        "slots_available": slots_available,
        "top": candidates[0]["ticker"] if candidates else "none",
    })

    return candidates


def execute_stock_buy(
    candidate: dict,
    client: AlpacaClient,
    portfolio_value: float,
    current_daily_pnl: float = 0,
) -> dict | None:
    """
    Execute a stock purchase through circuit breakers.

    Uses market orders for simplicity (stocks are liquid).
    """
    ticker = candidate["ticker"]
    shares = candidate["shares"]
    price = candidate["current_price"]

    # Safety checks
    try:
        check_paper_mode()
        check_daily_loss(current_daily_pnl)
    except CircuitBreakerTripped as e:
        log_event("stock_engine", "blocked", {"ticker": ticker, "error": str(e)})
        diary_write("strategy_agent", f"{ticker}|STOCK_BLOCKED|{e}")
        return None

    # Position size check
    position_value = price * shares
    settings = _load_settings()
    max_pct = settings["circuit_breakers"]["max_position_pct"]
    if position_value / portfolio_value > max_pct:
        shares = int(portfolio_value * max_pct / price)
        if shares < 1:
            log_event("stock_engine", "too_expensive", {"ticker": ticker, "price": price})
            return None

    reasoning = (
        f"Buying {shares} shares of {ticker} at ${price:.2f}. "
        f"Composite score {candidate['composite_score']}/9 "
        f"(trend:{candidate['trend_score']} level:{candidate['level_score']} "
        f"signal:{candidate['signal_score']}). "
        f"Support zone at {candidate['zone_level']} ({candidate['zone_touches']} touches). "
        f"Pattern: {candidate['pattern'] or 'none'}. "
        f"Target: ${candidate['target_price']:.2f}  Stop: ${candidate['stop_loss']:.2f}."
    )

    log_event("stock_engine", "executing", {
        "ticker": ticker, "shares": shares, "price": price,
        "score": candidate["composite_score"],
    }, result="pending")

    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        trading_client = client._get_trading_client()
        client.limiter.wait_if_needed()

        order = trading_client.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))

        response = {
            "id": str(order.id),
            "status": str(order.status),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": str(order.side),
        }

        log_event("stock_engine", "executed", {
            "ticker": ticker, "order_id": response["id"],
            "status": response["status"],
        }, result="success")

    except Exception as e:
        log_event("stock_engine", "failed", {"ticker": ticker, "error": str(e)}, result="failed")
        diary_write("strategy_agent", f"{ticker}|STOCK_FAILED|{e}")
        return None

    # Track position
    positions = _load_positions()
    positions.append({
        "ticker": ticker,
        "type": "stock",
        "status": "open",
        "shares": shares,
        "entry_price": price,
        "target_price": candidate["target_price"],
        "stop_loss": candidate["stop_loss"],
        "order_id": response["id"],
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "composite_score": candidate["composite_score"],
    })
    _save_positions(positions)

    # Memory
    remember_trade_decision(
        ticker=ticker, trade_type="stock_buy",
        details={"shares": shares, "price": price, "score": candidate["composite_score"]},
        reasoning=reasoning,
    )
    diary_write("strategy_agent",
        f"{ticker}|STOCK_BUY|{shares}sh@${price:.2f}|"
        f"score_{candidate['composite_score']}/9|{candidate['pattern'] or 'no_pattern'}")

    return response


def check_stock_exits(
    client: AlpacaClient,
    daily_data: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    Check open stock positions for exit signals:
    - Hit target price (take profit at resistance)
    - Hit stop loss (cut losses)
    - Trailing stop (lock in profits as price rises)
    - Bearish reversal signal at resistance
    - Momentum death (MACD bearish cross + RSI overbought)
    """
    positions = _load_positions()
    stock_positions = [p for p in positions if p.get("type") == "stock" and p.get("status") == "open"]

    if not stock_positions:
        return []

    strategy = _load_strategy()
    stock_cfg = strategy.get("stock_params", {})
    trailing_stop_pct = stock_cfg.get("trailing_stop_pct", 0.0)

    broker_positions = client.get_positions()
    broker_map = {p["symbol"]: p for p in broker_positions}
    positions_changed = False

    exits = []
    for pos in stock_positions:
        ticker = pos.get("ticker", "")
        target = pos.get("target_price", 0)
        stop = pos.get("stop_loss", 0)

        bp = broker_map.get(ticker)
        if not bp:
            continue

        current_price = float(bp["current_price"])
        entry_price = pos.get("entry_price", 0)
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        # Trailing stop: ratchet stop up as price rises
        if trailing_stop_pct > 0 and current_price > entry_price:
            trailing_stop = current_price * (1 - trailing_stop_pct)
            if trailing_stop > stop:
                pos["stop_loss"] = round(trailing_stop, 2)
                stop = pos["stop_loss"]
                positions_changed = True

        exit_signal = None

        # Stop loss hit
        if current_price <= stop:
            exit_signal = {
                "ticker": ticker,
                "action": "stop_loss",
                "reason": f"Stop loss hit: ${current_price:.2f} <= ${stop:.2f}",
                "pnl_pct": pnl_pct,
            }

        # Target hit
        elif current_price >= target:
            exit_signal = {
                "ticker": ticker,
                "action": "take_profit",
                "reason": f"Target reached: ${current_price:.2f} >= ${target:.2f}",
                "pnl_pct": pnl_pct,
            }

        # Momentum death: bearish MACD cross + RSI overbought
        elif ticker in daily_data and pnl_pct > 0.02:
            df = daily_data[ticker]
            mom = analyze_momentum(df)
            if mom and mom.macd_cross == "bearish_cross" and mom.rsi > 70:
                exit_signal = {
                    "ticker": ticker,
                    "action": "momentum_exit",
                    "reason": f"Momentum death: bearish MACD cross + RSI {mom.rsi:.0f}",
                    "pnl_pct": pnl_pct,
                }

        # Bearish reversal at resistance
        if exit_signal is None and ticker in daily_data:
            df = daily_data[ticker]
            bearish_signal = get_latest_signal(df, "bearish", [
                "shooting_star", "bearish_engulfing", "evening_star",
                "gravestone_doji", "bearish_harami",
            ])
            zones_list = detect_zones(df, current_price)
            resistance = get_nearest_resistance(zones_list, entry_price)

            if bearish_signal and resistance and abs(current_price - resistance.level) / resistance.level < 0.03:
                exit_signal = {
                    "ticker": ticker,
                    "action": "bearish_reversal",
                    "reason": f"Bearish {bearish_signal.pattern} at resistance {resistance.level:.2f}",
                    "pnl_pct": pnl_pct,
                }

        if exit_signal:
            exits.append(exit_signal)
            log_event("stock_engine", "exit_signal", exit_signal)

    # Save trailing stop updates
    if positions_changed:
        _save_positions(positions)

    return exits


def execute_stock_sell(
    ticker: str,
    client: AlpacaClient,
    reason: str = "signal",
) -> dict | None:
    """Sell all shares of a stock position."""
    positions = _load_positions()
    pos = None
    for p in positions:
        if p.get("ticker") == ticker and p.get("type") == "stock" and p.get("status") == "open":
            pos = p
            break

    if not pos:
        return None

    shares = pos.get("shares", 0)
    if shares < 1:
        return None

    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        trading_client = client._get_trading_client()
        client.limiter.wait_if_needed()

        order = trading_client.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))

        response = {
            "id": str(order.id),
            "status": str(order.status),
            "symbol": order.symbol,
            "qty": str(order.qty),
        }

        # Update position
        pos["status"] = "closed"
        pos["closed_at"] = datetime.now(timezone.utc).isoformat()
        pos["close_reason"] = reason
        _save_positions(positions)

        log_event("stock_engine", "sold", {
            "ticker": ticker, "shares": shares, "reason": reason,
        }, result="success")

        diary_write("strategy_agent", f"{ticker}|STOCK_SOLD|{shares}sh|{reason}")
        return response

    except Exception as e:
        log_event("stock_engine", "sell_failed", {"ticker": ticker, "error": str(e)}, result="failed")
        return None


def run_stock_scan_and_execute(
    client: AlpacaClient,
    daily_data: dict[str, pd.DataFrame],
    weekly_data: dict[str, pd.DataFrame],
    portfolio_value: float,
    max_trades: int = 3,
) -> list[dict]:
    """
    Full stock trading pipeline: scan → score → execute.
    Also checks existing positions for exit signals.
    """
    strategy = _load_strategy()
    max_trades = strategy.get("stock_params", {}).get("max_trades_per_scan", max_trades)

    log_event("stock_engine", "pipeline_started", {"portfolio": portfolio_value, "max_trades": max_trades})

    results = []

    # Check exits first — free up capital for new trades
    exits = check_stock_exits(client, daily_data)
    for exit_signal in exits:
        ticker = exit_signal["ticker"]
        print(f"  🔔 Exit signal: {ticker} — {exit_signal['reason']}")
        resp = execute_stock_sell(ticker, client, exit_signal["action"])
        if resp:
            results.append({"action": "sell", **exit_signal, "order": resp})

    # Scan for new buys
    candidates = scan_for_stocks(client, daily_data, weekly_data, portfolio_value)

    if not candidates:
        log_event("stock_engine", "no_candidates", {})
        return results

    # Print momentum details for top candidates
    for c in candidates[:5]:
        mom = c.get("momentum", {})
        mom_str = ""
        if mom:
            mom_str = (f" RSI:{mom.get('rsi', '-')} MACD:{mom.get('macd_cross', '-')} "
                      f"Vol:{mom.get('volume_surge', '-')}x ROC5d:{mom.get('roc_5d', 0):+.1%}")
        print(f"  📊 {c['ticker']:5s} score={c['composite_score']}/13 "
              f"(T:{c['trend_score']} L:{c['level_score']} S:{c['signal_score']} M:{c['momentum_score']})"
              f"{mom_str}")

    # Execute top candidates
    executed = 0
    for candidate in candidates:
        if executed >= max_trades:
            break

        resp = execute_stock_buy(candidate, client, portfolio_value)
        if resp:
            results.append({"action": "buy", "candidate": candidate, "order": resp})
            executed += 1

    log_event("stock_engine", "pipeline_complete", {
        "exits": len(exits), "buys": executed,
    })

    return results
