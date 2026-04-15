"""
Dashboard Data Layer — aggregates all data sources for both web and terminal dashboards.

Every function returns a plain dict (JSON-serializable) and handles errors
gracefully so the presentation layer never crashes.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lib.audit import get_recent_events
from lib.alpaca_client import AlpacaClient

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"
TRADE_HISTORY_PATH = Path(__file__).parent.parent / "data" / "trade_history.json"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
STRATEGY_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"

# Quant score cache (expensive to compute)
_quant_cache = {"data": None, "timestamp": 0}
QUANT_CACHE_TTL = 900  # 15 minutes (fetching 17 tickers x 252 days is slow)


def _get_client() -> AlpacaClient:
    from dotenv import load_dotenv
    load_dotenv()
    return AlpacaClient()


def get_portfolio_summary() -> dict:
    """Portfolio value, cash, phase, regime, daily P/L."""
    try:
        client = _get_client()
        account = client.get_account()
        positions = client.get_positions()

        daily_pl = sum(float(p.get("unrealized_pl", 0)) for p in positions)
        portfolio = account["portfolio_value"]

        from lib.stock_engine import get_current_phase, PHASE_2_THRESHOLD, PHASE_3_THRESHOLD
        phase = get_current_phase(portfolio)
        phase_labels = {
            1: f"Stock Trading (CSPs at ${PHASE_2_THRESHOLD:,})",
            2: f"Stocks + CSPs (full Wheel at ${PHASE_3_THRESHOLD:,})",
            3: "Full Wheel Strategy",
        }

        from lib.memory_palace import get_current_regime
        regime = get_current_regime() or "unknown"

        with open(CONFIG_PATH) as f:
            settings = yaml.safe_load(f)
        mode = settings.get("mode", "paper")

        return {
            "portfolio_value": portfolio,
            "cash": account["cash"],
            "buying_power": account["buying_power"],
            "equity": account["equity"],
            "daily_pl": round(daily_pl, 2),
            "phase": phase,
            "phase_label": phase_labels.get(phase, ""),
            "regime": regime,
            "mode": mode,
            "status": str(account["status"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}


def get_positions_table() -> list[dict]:
    """Merge broker positions with local tracking data."""
    try:
        client = _get_client()
        broker_positions = client.get_positions()
        broker_map = {p["symbol"]: p for p in broker_positions}

        local_positions = _load_json(POSITIONS_PATH, [])
        open_local = [p for p in local_positions if p.get("status") in ("open", "assigned")]

        result = []
        seen_tickers = set()

        # Start from broker positions (live data)
        for bp in broker_positions:
            ticker = bp["symbol"]
            seen_tickers.add(ticker)

            # Find matching local position
            local = next((p for p in open_local if p.get("ticker") == ticker), {})

            entry = float(bp.get("avg_entry_price", 0))
            current = float(bp.get("current_price", 0))
            pnl = float(bp.get("unrealized_pl", 0))
            pnl_pct = (current - entry) / entry if entry > 0 else 0

            result.append({
                "ticker": ticker,
                "type": local.get("type", "stock"),
                "shares": int(float(bp.get("qty", 0))),
                "entry_price": entry,
                "current_price": current,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
                "market_value": float(bp.get("market_value", 0)),
                "target": local.get("target_price", 0),
                "stop": local.get("stop_loss", 0),
                "score": local.get("composite_score", 0),
                "opened_at": local.get("opened_at", ""),
            })

        # Add local-only positions (pending fill, timing gaps)
        for local in open_local:
            ticker = local.get("ticker", "")
            if ticker not in seen_tickers:
                result.append({
                    "ticker": ticker,
                    "type": local.get("type", "stock"),
                    "shares": local.get("shares", 0),
                    "entry_price": local.get("entry_price", 0),
                    "current_price": 0,
                    "pnl": 0,
                    "pnl_pct": 0,
                    "market_value": 0,
                    "target": local.get("target_price", 0),
                    "stop": local.get("stop_loss", 0),
                    "score": local.get("composite_score", 0),
                    "opened_at": local.get("opened_at", ""),
                    "pending": True,
                })

        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_open_orders() -> list[dict]:
    """Pending orders from Alpaca."""
    try:
        client = _get_client()
        return client.get_open_orders()
    except Exception as e:
        return [{"error": str(e)}]


def get_quant_scores() -> list[dict]:
    """Quant scores with 5-minute cache."""
    global _quant_cache

    now = time.time()
    if _quant_cache["data"] and (now - _quant_cache["timestamp"]) < QUANT_CACHE_TTL:
        return _quant_cache["data"]

    try:
        client = _get_client()

        with open(STRATEGY_PATH) as f:
            strategy = yaml.safe_load(f)
        tickers = strategy.get("tickers_phase1", strategy.get("tickers", []))

        bars = client.get_bars(tickers, timeframe="1Day", limit=252)

        from lib.quant_screener import screen_universe
        scores = screen_universe(bars, exclude_avoid=False)

        result = []
        for s in scores:
            result.append({
                "ticker": s.ticker,
                "price": s.price,
                "return_1y": s.total_return_1y,
                "max_drawdown": s.max_drawdown,
                "sharpe": s.sharpe_ratio,
                "sortino": s.sortino_ratio,
                "volatility": s.volatility,
                "quant_score": s.quant_score,
                "verdict": s.verdict,
            })

        _quant_cache = {"data": result, "timestamp": now}
        return result
    except Exception as e:
        if _quant_cache["data"]:
            return _quant_cache["data"]  # Return stale cache
        return [{"error": str(e)}]


def get_events(n: int = 30) -> list[dict]:
    """Recent audit events."""
    events = get_recent_events(n)
    for e in events:
        details = e.get("details", {})
        summary_parts = []
        for key in ["ticker", "shares", "qty", "price", "score", "error", "reason"]:
            if key in details:
                summary_parts.append(f"{key}={details[key]}")
        e["summary"] = ", ".join(summary_parts) if summary_parts else ""
    return events


def get_trade_history() -> dict:
    """Completed trades with cumulative P/L."""
    trades = _load_json(TRADE_HISTORY_PATH, [])

    if not trades:
        return {"trades": [], "total_pl": 0, "win_rate": 0, "total_trades": 0, "pl_series": []}

    total_pl = sum(t.get("total_pnl", 0) for t in trades)
    wins = sum(1 for t in trades if t.get("total_pnl", 0) > 0)
    win_rate = wins / len(trades) if trades else 0

    cumulative = 0
    pl_series = []
    for t in trades:
        cumulative += t.get("total_pnl", 0)
        pl_series.append({
            "date": t.get("completed_at", "")[:10],
            "pl": round(cumulative, 2),
        })

    return {
        "trades": trades,
        "total_pl": round(total_pl, 2),
        "win_rate": round(win_rate, 4),
        "total_trades": len(trades),
        "pl_series": pl_series,
    }


def get_circuit_breaker_status() -> dict:
    """Circuit breaker limits vs current values."""
    try:
        with open(CONFIG_PATH) as f:
            settings = yaml.safe_load(f)

        cb = settings.get("circuit_breakers", {})
        client = _get_client()
        account = client.get_account()
        positions = client.get_positions()
        orders = client.get_open_orders()

        portfolio = account["portfolio_value"]
        daily_pl = sum(float(p.get("unrealized_pl", 0)) for p in positions)

        # Largest position as % of portfolio
        max_pos_pct = 0
        if portfolio > 0:
            for p in positions:
                pct = float(p.get("market_value", 0)) / portfolio
                max_pos_pct = max(max_pos_pct, pct)

        breakers = {
            "daily_loss": {
                "limit": cb.get("max_daily_loss", -500),
                "current": round(daily_pl, 2),
                "pct_used": round(abs(daily_pl / cb.get("max_daily_loss", -500)), 2) if daily_pl < 0 else 0,
                "tripped": daily_pl <= cb.get("max_daily_loss", -500),
            },
            "position_size": {
                "limit": cb.get("max_position_pct", 0.10),
                "current": round(max_pos_pct, 4),
                "pct_used": round(max_pos_pct / cb.get("max_position_pct", 0.10), 2) if cb.get("max_position_pct") else 0,
                "tripped": max_pos_pct > cb.get("max_position_pct", 0.10),
            },
            "open_orders": {
                "limit": cb.get("max_open_orders", 5),
                "current": len(orders),
                "pct_used": round(len(orders) / cb.get("max_open_orders", 5), 2),
                "tripped": len(orders) >= cb.get("max_open_orders", 5),
            },
        }

        return {
            "breakers": breakers,
            "paper_mode": settings.get("mode") == "paper",
            "live_approved": settings.get("live_migration_approved", False),
        }
    except Exception as e:
        return {"error": str(e)}


def get_full_dashboard_state() -> dict:
    """All dashboard data in one call (minus quant scores — too slow)."""
    return {
        "portfolio": get_portfolio_summary(),
        "positions": get_positions_table(),
        "orders": get_open_orders(),
        "events": get_events(20),
        "history": get_trade_history(),
        "breakers": get_circuit_breaker_status(),
    }


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)
