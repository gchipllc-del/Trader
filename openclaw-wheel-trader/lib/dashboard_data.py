"""
Dashboard Data Layer — aggregates all data sources for both web and terminal dashboards.

Every function returns a plain dict (JSON-serializable) and handles errors
gracefully so the presentation layer never crashes.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lib.audit import get_recent_events
from lib.alpaca_client import AlpacaClient

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"
TRADE_HISTORY_PATH = Path(__file__).parent.parent / "data" / "trade_history.json"
BASELINE_PATH = Path(__file__).parent.parent / "data" / "baseline_equity.json"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
STRATEGY_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"

# Quant score cache (expensive to compute)
_quant_cache = {"data": None, "timestamp": 0}
QUANT_CACHE_TTL = 900  # 15 minutes (fetching 17 tickers x 252 days is slow)


def _get_client() -> AlpacaClient:
    from dotenv import load_dotenv
    load_dotenv()
    return AlpacaClient()


def _fetch_account_starting_equity() -> tuple[float | None, str | None]:
    """
    Query Alpaca portfolio history for the earliest recorded equity value
    (i.e. the account's actual starting capital). Returns (equity, iso_date)
    or (None, None) if unavailable.
    """
    try:
        import os
        import requests
        from dotenv import load_dotenv
        load_dotenv()
        headers = {
            "APCA-API-KEY-ID": os.environ.get("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
        }
        base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not headers["APCA-API-KEY-ID"] or not headers["APCA-API-SECRET-KEY"]:
            return None, None
        r = requests.get(
            f"{base}/v2/account/portfolio/history",
            params={"period": "all", "timeframe": "1D"},
            headers=headers,
            timeout=10,
        )
        if r.status_code != 200:
            return None, None
        data = r.json()
        eq = data.get("equity") or []
        ts = data.get("timestamp") or []
        if not eq or not ts:
            return None, None
        # The very first equity entry = the account's starting capital.
        first_equity = float(eq[0])
        first_ts = datetime.fromtimestamp(ts[0], tz=timezone.utc)
        if first_equity <= 0:
            return None, None
        return first_equity, first_ts.isoformat()
    except Exception:
        return None, None


def _get_baseline(current_equity: float) -> tuple[float, str]:
    """
    Return (baseline_equity, set_at_iso) for the %-gain-to-date metric.

    Two-tier baseline: the file may carry both ``start_baseline`` (the
    true account starting capital, set once and never changed — anchors
    the long-term % growth headline) and ``baseline_equity`` (the
    operating baseline used by the self-audit's P&L-drift check, which
    can be periodically rebaselined after a reconciliation event).
    The dashboard prefers ``start_baseline`` when present so long-term
    growth doesn't reset every time we reconcile.

    On first call (no baseline file yet), we prefer Alpaca portfolio
    history's first recorded equity — that's the account's actual
    starting capital. If unavailable, fall back to current equity.
    Users can override via ``set_baseline(amount)``.
    """
    try:
        if BASELINE_PATH.exists():
            with open(BASELINE_PATH) as f:
                data = json.load(f)
            # Prefer the immutable long-term start; fall back to the
            # operating baseline; final fallback is current equity.
            if "start_baseline" in data:
                baseline = float(data.get("start_baseline", current_equity))
                set_at = str(data.get("start_baseline_date") or data.get("set_at", ""))
            else:
                baseline = float(data.get("baseline_equity", current_equity))
                set_at = str(data.get("set_at", ""))
            if baseline <= 0:
                baseline = float(current_equity)
            return baseline, set_at
    except Exception:
        pass  # Fall through to fresh snapshot

    # First run — try to use the account's actual starting equity.
    source = "current_equity"
    starting_equity, starting_date = _fetch_account_starting_equity()
    if starting_equity is not None and starting_date is not None:
        baseline_value = starting_equity
        set_at = starting_date
        source = "alpaca_portfolio_history"
    else:
        baseline_value = float(current_equity)
        set_at = datetime.now(timezone.utc).isoformat()

    data = {
        "baseline_equity": float(baseline_value),
        "set_at": set_at,
        "source": source,
        "note": (
            "Auto-detected from Alpaca portfolio history (account's first "
            "recorded equity). Edit baseline_equity to set a different starting "
            "capital, or delete this file to re-detect on next load."
            if source == "alpaca_portfolio_history"
            else "Auto-snapshotted from current equity (Alpaca history "
                 "unavailable). Edit baseline_equity to set a specific starting "
                 "capital, or delete this file to re-snapshot on next load."
        ),
    }
    try:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        try:
            from lib.audit import log_event
            log_event("dashboard", "baseline_snapshotted",
                      {"baseline": float(baseline_value), "source": source})
        except Exception:
            pass
    except Exception:
        pass
    return float(baseline_value), data["set_at"]


def set_baseline(amount: float | None = None) -> dict:
    """
    Manually set (or reset) the %-gain-to-date baseline.

    Pass `amount=None` to snapshot the *current* portfolio value.  Returns
    the new baseline dict so callers can report it to the user.
    """
    if amount is None:
        client = _get_client()
        account = client.get_account()
        amount = float(account["portfolio_value"])

    data = {
        "baseline_equity": float(amount),
        "set_at": datetime.now(timezone.utc).isoformat(),
        "note": "Manually set via set_baseline().",
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    try:
        from lib.audit import log_event
        log_event("dashboard", "baseline_manual_set", {"baseline": float(amount)})
    except Exception:
        pass
    return data


def get_portfolio_summary() -> dict:
    """Portfolio value, cash, phase, regime, daily P/L."""
    try:
        client = _get_client()
        account = client.get_account()
        positions = client.get_positions()

        daily_pl = sum(float(p.get("unrealized_pl", 0)) for p in positions)
        portfolio = account["portfolio_value"]

        # %-gain-to-date vs baseline equity (auto-snapshots on first call)
        baseline, baseline_set_at = _get_baseline(float(portfolio))
        dollar_gain = float(portfolio) - baseline
        pct_gain = (dollar_gain / baseline) if baseline > 0 else 0.0

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
            "baseline_equity": round(baseline, 2),
            "baseline_set_at": baseline_set_at,
            "dollar_gain_to_date": round(dollar_gain, 2),
            "pct_gain_to_date": round(pct_gain, 6),
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


def get_agent_thinking(limit_per_agent: int = 6) -> dict:
    """Surface recent diary entries from each governance agent.

    Returns the last N entries from strategy_agent, risk_agent, bull_agent,
    bear_agent, compliance_agent — the bot's actual reasoning trail. Powers
    the dashboard's 'Bot Thinking' panel so the operator can see what each
    agent has been saying about recent trade decisions.

    Each entry has a compressed format like:
      strategy_agent: "AAPL|CSP_170P|score_8/9|zone_168|hammer"
      bull_agent:     "NVDA|BULL_BOOST|score_10/10|strong_composite_score,..."
      bear_agent:     "AAPL|BEAR_DOWNSIZE|score_3/10|kronos_bearish,..."
      risk_agent:     "VETO|AAPL|sell_cc|pos_17%|sector_tech_17%"
    """
    diaries_dir = Path(__file__).parent.parent / "data" / "palace" / "diaries"
    agents = ["strategy_agent", "risk_agent", "compliance_agent",
              "bull_agent", "bear_agent"]
    result: dict = {}
    for agent in agents:
        path = diaries_dir / f"{agent}.jsonl"
        if not path.exists():
            result[agent] = []
            continue
        try:
            rows: list[dict] = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            # Newest last → reverse to newest first, cap
            rows = list(reversed(rows))[:limit_per_agent]
            # Parse the entry string into structured pieces for the UI
            parsed = []
            for r in rows:
                entry = r.get("entry", "")
                parts = entry.split("|") if entry else []
                # First "verdict-ish" token if it looks like one
                verdict = "info"
                if parts:
                    first = parts[0].upper()
                    if any(k in first for k in ("VETO", "BLOCK", "REJECT", "FAIL")):
                        verdict = "block"
                    elif any(k in first for k in ("APPROVE", "CLEAR", "BOOST", "ENTER", "PASS")):
                        verdict = "good"
                    elif "DOWNSIZE" in first or "WARN" in first or "CAUTION" in first:
                        verdict = "warn"
                # Also pick up keywords from any segment
                for seg in parts:
                    su = seg.upper()
                    if "BLOCKED" in su or "VETO" in su:
                        verdict = "block"
                        break
                    if "BOOST" in su or "APPROVE" in su:
                        if verdict == "info":
                            verdict = "good"
                    if "DOWNSIZE" in su:
                        if verdict == "info":
                            verdict = "warn"
                parsed.append({
                    "timestamp": r.get("timestamp", ""),
                    "entry": entry,
                    "ticker": parts[0] if parts and len(parts[0]) <= 8 else None,
                    "verdict": verdict,
                    "tokens": parts[:8],
                })
            result[agent] = parsed
        except OSError:
            result[agent] = []
    return result


def get_hermes_state() -> dict:
    """Surface the scientific-Hermes loop state for the dashboard:
      - current goal-distance / velocity / drawdown
      - mode (review|live)
      - ledger stats + recent experiments
      - last weekly review file path
    Lightweight — all reads from local JSONL / YAML, no network.
    """
    out: dict = {}
    try:
        from lib.hermes_goal_score import compute_goal_metrics
        out["goal"] = compute_goal_metrics()
    except Exception as e:
        out["goal_error"] = str(e)[:200]

    try:
        from lib.hermes_scientific import get_mode
        out["mode"] = get_mode()
    except Exception:
        out["mode"] = "unknown"

    try:
        from lib.hermes_ledger import history, stats
        out["ledger_stats"] = stats()
        out["recent_experiments"] = history(limit=10)
    except Exception:
        out["recent_experiments"] = []
        out["ledger_stats"] = {"counts": {}, "total": 0, "keep_rate": None}

    # Last weekly review
    try:
        reviews_dir = Path(__file__).parent.parent / "data" / "hermes_reviews"
        if reviews_dir.exists():
            md_files = sorted(reviews_dir.glob("weekly_*.md"), reverse=True)
            if md_files:
                out["last_review"] = {
                    "path": str(md_files[0]),
                    "modified_at": datetime.fromtimestamp(
                        md_files[0].stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
    except OSError:
        pass

    return out


def get_markov_summary(ticker: str = "SPY", refresh: bool = False) -> dict:
    """Return the most recent Markov-regime summary for the dashboard panel.

    Reads data/markov_latest.json (populated by ``main.py markov``); on
    miss or when stale (>6h) and ``refresh=True``, recomputes inline. We
    don't auto-refresh on every dashboard hit because the Alpaca history
    pull adds 2-5s.
    """
    cache = Path(__file__).parent.parent / "data" / "markov_latest.json"
    fresh = False
    data: dict | None = None
    if cache.exists():
        try:
            with open(cache) as f:
                data = json.load(f)
            try:
                mtime = datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc)
                age_h = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0
                fresh = age_h < 6
                if data is not None:
                    data["_cache_age_hours"] = round(age_h, 2)
            except OSError:
                pass
        except (OSError, json.JSONDecodeError):
            data = None

    if (data is None or refresh) and not fresh:
        try:
            from lib.markov_regime import markov_summary
            data = markov_summary(ticker)
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "w") as f:
                json.dump(data, f, indent=2, default=str)
            data["_cache_age_hours"] = 0.0
        except Exception as e:
            return {"error": str(e)[:200], "ticker": ticker}

    if data is None:
        return {
            "error": "no markov data — run `python main.py markov`",
            "ticker": ticker,
        }
    return data


def get_goal_progress() -> dict:
    """Unified-goals progress payload for the dashboard milestone bar.

    Returns current equity, anchor, target, hard floor, halt state, and
    the milestone schedule. Hides cleanly when tradingcore isn't on path.
    """
    try:
        from tradingcore.unified_goals import load_goals, get_progress
        goals = load_goals()
        tb = get_progress("traderbot")
        return {
            "current": tb.get("current"),
            "anchor": tb.get("anchor"),
            "target": tb.get("target"),
            "hard_floor": goals.get("traderbot", {}).get("hard_floor"),
            "hard_floor_buffer": goals.get("traderbot", {}).get("hard_floor_buffer"),
            "halt_state": tb.get("halt_state"),
            "halt_reason": tb.get("halt_reason"),
            "pct_growth_from_anchor": tb.get("pct_growth_from_anchor"),
            "pct_to_target": tb.get("pct_to_target"),
            "milestones": tb.get("milestones", []),
            "stock_buys_gate": goals.get("traderbot", {}).get("stock_buys_gate", {}),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


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

        # Effective daily-loss limit = the more restrictive of (dollar floor)
        # and (pct of current equity). Mirrors lib/circuit_breaker.check_daily_loss.
        dollar_floor = cb.get("max_daily_loss", -500)
        pct_limit = cb.get("max_daily_loss_pct")
        if pct_limit is not None and portfolio > 0:
            effective_limit = max(dollar_floor, portfolio * pct_limit)  # both negative; max = stricter
        else:
            effective_limit = dollar_floor

        breakers = {
            "daily_loss": {
                "limit": round(effective_limit, 2),
                "dollar_floor": dollar_floor,
                "pct_limit": pct_limit,
                "current": round(daily_pl, 2),
                "pct_used": round(abs(daily_pl / effective_limit), 2) if daily_pl < 0 and effective_limit < 0 else 0,
                "tripped": daily_pl <= effective_limit,
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
