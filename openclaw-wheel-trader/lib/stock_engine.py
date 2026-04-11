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

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"

# Phase thresholds
PHASE_2_THRESHOLD = 5000   # Start selling CSPs on cheap stocks
PHASE_3_THRESHOLD = 10000  # Full Wheel on bigger tickers


def _load_settings() -> dict:
    with open(CONFIG_PATH) as f:
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
    Score a stock as a buy candidate using the same framework
    as the options screener (trend + level + signal = 0-9).

    Returns scored candidate dict or None if it doesn't pass.
    """
    if len(daily_df) < 50 or len(weekly_df) < 20:
        return None

    # --- Trend Score (0-3) ---
    mtf = multi_timeframe_analysis(weekly_df, daily_df)
    trend_score = mtf["alignment_score"]

    weekly_trend = mtf["weekly"]
    # Don't buy into strong downtrends
    if weekly_trend.direction == "downtrend" and weekly_trend.strength >= 2:
        return None

    # --- Level Score (0-3) ---
    zones = detect_zones(daily_df, current_price)
    nearest_support = get_nearest_support(zones, current_price)

    level_score = 0
    zone_level = 0.0
    zone_touches = 0

    if nearest_support:
        distance_to_zone = abs(current_price - nearest_support.level) / nearest_support.level
        if distance_to_zone < 0.03:  # Within 3% of support
            level_score = 3
        elif distance_to_zone < 0.05:
            level_score = 2
        elif distance_to_zone < 0.08:
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

    # --- Composite ---
    composite = trend_score + level_score + signal_score

    # Phase 1 stocks: more permissive than options
    # Require a candlestick signal (signal_score >= 1) and not strong downtrend
    # Allow ranging markets — stocks are less risky than selling puts into chop
    # Minimum score 3 (must have at least a signal + one other factor)
    min_score = 3
    not_downtrend = weekly_trend.direction != "downtrend"
    has_signal = signal_score >= 1
    tradeable = composite >= min_score and not_downtrend and has_signal

    # Calculate position size
    shares = calculate_position_size(current_price, max_position_value)
    if shares < 1:
        return None

    # Nearest resistance for profit target
    nearest_resistance = get_nearest_resistance(zones, current_price)
    target_price = nearest_resistance.level if nearest_resistance else current_price * 1.10

    return {
        "ticker": ticker,
        "trade_type": "stock_buy",
        "current_price": round(current_price, 2),
        "shares": shares,
        "position_value": round(current_price * shares, 2),
        "trend_score": trend_score,
        "level_score": level_score,
        "signal_score": signal_score,
        "composite_score": composite,
        "zone_level": zone_level,
        "zone_touches": zone_touches,
        "pattern": pattern_name,
        "target_price": round(target_price, 2),
        "stop_loss": round(current_price * 0.95, 2),  # 5% stop loss
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

    Returns ranked list of tradeable candidates.
    """
    settings = _load_settings()
    max_position_pct = settings["circuit_breakers"]["max_position_pct"]
    max_position_value = portfolio_value * max_position_pct

    candidates = []

    # Check existing positions to avoid doubling up
    positions = _load_positions()
    held_tickers = set(
        p["ticker"] for p in positions
        if p.get("status") in ("open", "assigned") and p.get("type") == "stock"
    )

    for ticker, daily_df in daily_data.items():
        if ticker in held_tickers:
            log_event("stock_engine", "skip_held", {"ticker": ticker})
            continue

        if len(daily_df) < 50:
            continue

        current_price = daily_df["close"].iloc[-1]
        weekly_df = weekly_data.get(ticker)
        if weekly_df is None or len(weekly_df) < 20:
            continue

        candidate = score_stock_buy(
            ticker, daily_df, weekly_df, current_price, max_position_value,
        )
        if candidate and candidate["tradeable"]:
            candidates.append(candidate)

    # Sort by composite score, then by proximity to support
    candidates.sort(
        key=lambda c: (c["composite_score"], c["level_score"]),
        reverse=True,
    )

    log_event("stock_engine", "scan_complete", {
        "candidates": len(candidates),
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
    - Bearish reversal signal at resistance
    """
    positions = _load_positions()
    stock_positions = [p for p in positions if p.get("type") == "stock" and p.get("status") == "open"]

    if not stock_positions:
        return []

    broker_positions = client.get_positions()
    broker_map = {p["symbol"]: p for p in broker_positions}

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

        # Bearish reversal at resistance
        elif ticker in daily_data:
            df = daily_data[ticker]
            bearish_signal = get_latest_signal(df, "bearish", [
                "shooting_star", "bearish_engulfing", "evening_star",
                "gravestone_doji", "bearish_harami",
            ])
            zones = detect_zones(df, current_price)
            resistance = get_nearest_resistance(zones, entry_price)

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
    max_trades: int = 2,
) -> list[dict]:
    """
    Full stock trading pipeline: scan → score → execute.
    Also checks existing positions for exit signals.
    """
    log_event("stock_engine", "pipeline_started", {"portfolio": portfolio_value})

    results = []

    # Check exits first
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
