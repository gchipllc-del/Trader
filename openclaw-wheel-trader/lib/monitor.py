"""
Sprint 3: Position Monitoring & Management

Runs every 5 minutes during market hours. Checks all open positions for:
- P/L updates
- Approaching expiration (roll candidates)
- Early close opportunities (>80% profit with >14 DTE)
- Assignment events
- Circuit breaker conditions

Source: Video pattern (cron monitoring), Wheel Strategy management rules
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

from lib.audit import log_event
from lib.memory_palace import (
    diary_write, kg_add, kg_invalidate, remember_trade_decision,
    remember_regime_change,
)
from lib.alpaca_client import AlpacaClient

CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
STRATEGY_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"

# File-locked positions store (Wave 3 #15).
from lib.positions_store import (
    POSITIONS_PATH,
    load_positions as _store_load,
    save_positions as _store_save,
    mutate_positions as _store_mutate,
)


def mutate_positions():
    return _store_mutate(POSITIONS_PATH)


def _load_positions() -> list[dict]:
    return _store_load(POSITIONS_PATH)


def _save_positions(positions: list[dict]):
    _store_save(positions, POSITIONS_PATH)


# Module-level mtime caches — settings/strategy YAMLs are read many times
# per monitor cycle (per-position helpers + heartbeat watchdog + Hermes).
# Re-parse only when the file actually changes on disk. Tests that
# monkeypatch _load_settings replace the function entirely, bypassing the
# cache — that path is unaffected.
_settings_cache: tuple[float, dict] | None = None
_strategy_cache: tuple[float, dict] | None = None


def _load_settings() -> dict:
    global _settings_cache
    mtime = CONFIG_PATH.stat().st_mtime
    if _settings_cache is not None and _settings_cache[0] == mtime:
        return _settings_cache[1]
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    _settings_cache = (mtime, cfg)
    return cfg


def _load_strategy() -> dict:
    global _strategy_cache
    mtime = STRATEGY_PATH.stat().st_mtime
    if _strategy_cache is not None and _strategy_cache[0] == mtime:
        return _strategy_cache[1]
    with open(STRATEGY_PATH) as f:
        cfg = yaml.safe_load(f)
    _strategy_cache = (mtime, cfg)
    return cfg


# ============================================================
# HEARTBEAT TRACKER
# ============================================================

_missed_checks = 0
HEARTBEAT_PATH = Path(__file__).parent.parent / "data" / "heartbeat.json"


def _record_heartbeat():
    """Persist a timestamp marking the start of this monitor cycle so
    the next firing can detect missed cycles in between."""
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_PATH.write_text(json.dumps({
        "last_check_at": datetime.now(timezone.utc).isoformat(),
    }))


def _check_for_missed_cycles():
    """
    Wave 3 #17: actually drive the kill-switch watchdog. Compares the
    timestamp from the last cycle to now; if the gap exceeds 1.5×
    the configured interval, count it as one or more missed checks.

    Mac asleep, launchd outage, machine reboot, etc. are exactly what
    this is for. A genuinely-on-time cycle records 0 misses.
    """
    if not HEARTBEAT_PATH.exists():
        return  # First cycle ever; nothing to compare against.
    try:
        prev = json.loads(HEARTBEAT_PATH.read_text())
        last_iso = prev.get("last_check_at", "")
        if not last_iso:
            return
        last = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return

    settings = _load_settings()
    interval = max(60, int(settings.get("monitoring", {})
                                  .get("check_interval_seconds", 180)))
    now = datetime.now(timezone.utc)
    gap = (now - last).total_seconds()
    misses = int(gap / interval) - 1
    if misses <= 0:
        return

    log_event("monitor", "missed_cycles_detected", {
        "gap_seconds": round(gap, 1),
        "interval_seconds": interval,
        "missed_cycles": misses,
        "last_check_at": last_iso,
    })
    for _ in range(misses):
        # record_missed_check() escalates internally to alert/kill
        # at the configured thresholds and stops counting once the
        # kill threshold is hit (Wave 3 #14).
        if record_missed_check() == "KILLED":
            break


def reset_heartbeat():
    global _missed_checks
    _missed_checks = 0


def record_missed_check():
    global _missed_checks
    _missed_checks += 1
    settings = _load_settings()
    mon = settings.get("monitoring", {})

    if _missed_checks >= mon.get("missed_check_kill", 10):
        log_event("monitor", "kill_threshold_reached", {"missed": _missed_checks})
        from lib.kill_switch import activate_kill_switch
        activate_kill_switch(reason=f"missed_{_missed_checks}_consecutive_checks")
        # Wave 3 #14: reset the counter so a re-armed monitor (operator
        # restarts trading after a kill) doesn't immediately re-trip on the
        # very next missed check. The kill_switch itself is the durable
        # halt — this counter is just the trigger.
        _missed_checks = 0
        return "KILLED"

    if _missed_checks >= mon.get("missed_check_alert", 3):
        log_event("monitor", "missed_check_alert", {"missed": _missed_checks})
        send_alert(f"⚠️ Monitoring missed {_missed_checks} consecutive checks")
        return "ALERTED"

    return "OK"


# ============================================================
# TELEGRAM ALERTS
# ============================================================

def send_alert(message: str):
    """Send alert via Telegram. Fails silently if not configured."""
    import os
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        log_event("telegram", "not_configured", {"message": message[:100]})
        return

    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
    except Exception as e:
        log_event("telegram", "send_failed", {"error": str(e)})


# ============================================================
# POSITION CHECKS
# ============================================================

def check_early_close(position: dict, current_price: float) -> dict | None:
    """
    Check if a position should be closed early.
    Close at 80% of max profit with >14 DTE remaining.
    """
    strategy = _load_strategy()
    mgmt = strategy.get("management", {})
    threshold = mgmt.get("early_close_threshold", 0.80)
    min_dte = mgmt.get("early_close_min_dte", 14)

    premium = position.get("premium_collected", 0)
    max_profit = premium * 100  # Per contract

    # Calculate current profit (for short options, profit = premium - current_value)
    # current_price here is the option's current market price
    current_value = current_price * 100
    current_profit = max_profit - current_value

    # Calculate DTE
    exp = position.get("expiration", "")
    if exp:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - datetime.now(timezone.utc).date()).days
        except ValueError:
            dte = 0
    else:
        dte = 0

    profit_pct = current_profit / max_profit if max_profit > 0 else 0

    if profit_pct >= threshold and dte > min_dte:
        return {
            "action": "early_close",
            "reason": f"Captured {profit_pct:.0%} of max profit with {dte} DTE remaining",
            "profit_pct": profit_pct,
            "dte": dte,
        }

    return None


def check_roll_candidate(position: dict, current_price: float) -> dict | None:
    """
    Check if a position should be rolled.
    Roll when: ITM with <7 DTE, if we can collect additional credit.
    """
    strategy = _load_strategy()
    mgmt = strategy.get("management", {})
    roll_dte = mgmt.get("roll_trigger_dte", 7)

    exp = position.get("expiration", "")
    strike = position.get("strike", 0)

    if exp:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - datetime.now(timezone.utc).date()).days
        except ValueError:
            dte = 999
    else:
        dte = 999

    # For puts: ITM means stock price < strike
    # current_price here is the STOCK price
    is_itm = current_price < strike if position.get("type") == "csp" else current_price > strike

    if is_itm and dte <= roll_dte:
        return {
            "action": "roll",
            "reason": f"ITM with {dte} DTE. Consider rolling down and out for additional credit.",
            "dte": dte,
            "itm_amount": abs(current_price - strike),
        }

    return None


def check_assignment(position: dict, broker_positions: list[dict]) -> dict | None:
    """
    Detect if a CSP has been assigned (we now hold shares).
    Check broker positions for unexpected share ownership.
    """
    ticker = position.get("ticker", "")

    # Look for shares of the ticker in broker positions
    for bp in broker_positions:
        if bp.get("symbol") == ticker and float(bp.get("qty", 0)) >= 100:
            # Check if this is a new position (wasn't there before)
            if position.get("type") == "csp" and position.get("status") == "open":
                return {
                    "action": "assigned",
                    "reason": f"CSP assigned — now holding {bp['qty']} shares of {ticker}",
                    "shares": int(float(bp["qty"])),
                    "avg_price": float(bp.get("avg_entry_price", 0)),
                }

    return None


# ============================================================
# MAIN MONITORING LOOP
# ============================================================

def run_monitoring_check(client: AlpacaClient) -> dict:
    """
    Single monitoring check. Called every 3 minutes by launchd cron.

    Returns summary dict of actions taken.
    """
    # Wave 3 #17: detect missed cycles from the previous heartbeat BEFORE
    # resetting in-process state. If the gap was big enough, this can
    # trip the kill switch.
    _check_for_missed_cycles()
    _record_heartbeat()
    reset_heartbeat()
    log_event("monitor", "check_started", {})

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "positions_checked": 0,
        "actions": [],
        "alerts": [],
        "degraded": [],  # tickers we couldn't fetch data for this cycle
    }

    try:
        account = client.get_account()
        broker_positions = client.get_positions()
        positions = _load_positions()
        open_positions = [p for p in positions if p.get("status") == "open"]

        summary["portfolio_value"] = account["portfolio_value"]
        summary["cash"] = account["cash"]
        summary["positions_checked"] = len(open_positions)

        # Batch-fetch current option prices for early close checks
        option_prices = {}
        option_price_fetch_failed = False
        try:
            from lib.data_pipeline import fetch_option_prices_for_positions
            option_prices = fetch_option_prices_for_positions(client, open_positions)
        except Exception as e:
            option_price_fetch_failed = True
            log_event("monitor", "option_price_fetch_failed", {"error": str(e)},
                      result="degraded")
            # Surface loudly: option exit checks (early close, stop) are now blind.
            uncheckable = [
                p.get("ticker", "?") for p in open_positions
                if p.get("type") in ("csp", "cc")
            ]
            if uncheckable:
                msg = (
                    f"⚠️ DATA OUTAGE: option price fetch failed — "
                    f"{len(uncheckable)} option position(s) NOT checked this cycle: "
                    f"{', '.join(uncheckable[:6])}"
                )
                summary["alerts"].append(msg)
                summary["degraded"].extend(
                    {"ticker": t, "type": "option", "reason": "option_price_fetch_failed"}
                    for t in uncheckable
                )
                send_alert(msg)

        # --- Auto-trim oversize positions ---
        # Position-size circuit breaker is entry-only; this enforces it on
        # held positions every monitor cycle so winners running or duplicate
        # buys can't leave us above max_position_pct indefinitely.
        try:
            from lib.stock_engine import auto_trim_oversize_stocks
            trims = auto_trim_oversize_stocks(client)
            for t in trims:
                summary["actions"].append({
                    "action": "auto_trim", "ticker": t["ticker"],
                    "shares_sold": t["shares_sold"],
                    "from_pct": t["from_pct"],
                })
                summary["alerts"].append(
                    f"⚖️  {t['ticker']}: AUTO-TRIM {t['shares_sold']}sh "
                    f"({t['from_pct']:.1%} → ≤{t['to_pct_target']:.0%})"
                )
        except Exception as e:
            log_event("monitor", "auto_trim_failed", {"error": str(e)[:200]})

        # --- Stock position monitoring (Phase 1) ---
        stock_positions = [p for p in open_positions if p.get("type") == "stock"]
        if stock_positions:
            try:
                from lib.stock_engine import (
                    check_stock_exits, execute_stock_sell, execute_partial_stock_sell,
                )
                from lib.data_pipeline import fetch_all_data
                import yaml as _yaml

                # Fetch daily data for stock exit checks
                stock_tickers = list(set(p.get("ticker") for p in stock_positions if p.get("ticker")))
                stock_daily = client.get_bars(stock_tickers, timeframe="1Day", limit=50)

                # Surface any ticker we can't run exit checks against —
                # missing bars = stops/targets/exits silently skipped.
                missing_bars = [t for t in stock_tickers if not (
                    isinstance(stock_daily, dict) and stock_daily.get(t) is not None
                    and len(stock_daily[t]) > 0
                )]
                if missing_bars:
                    msg = (
                        f"⚠️ DATA OUTAGE: stock bars missing for "
                        f"{len(missing_bars)} ticker(s) — exit checks NOT run: "
                        f"{', '.join(missing_bars)}"
                    )
                    summary["alerts"].append(msg)
                    summary["degraded"].extend(
                        {"ticker": t, "type": "stock", "reason": "bars_fetch_failed"}
                        for t in missing_bars
                    )
                    log_event("monitor", "stock_bars_missing",
                              {"tickers": missing_bars}, result="degraded")
                    send_alert(msg)

                exits = check_stock_exits(client, stock_daily)
                for exit_signal in exits:
                    t = exit_signal["ticker"]
                    reason = exit_signal["reason"]
                    pnl_pct = exit_signal.get("pnl_pct", 0)
                    action = exit_signal.get("action", "")

                    # Scale-out: partial sell (keeps position open with fewer shares)
                    if action == "scale_out":
                        shares = exit_signal.get("partial_shares", 0)
                        resp = execute_partial_stock_sell(t, shares, client, "scale_out")
                        if resp:
                            msg = f"💰 {t}: SCALE-OUT {shares}sh @ +{pnl_pct:.1%} (runner continues)"
                            summary["actions"].append(exit_signal)
                            summary["alerts"].append(msg)
                            diary_write("strategy_agent",
                                f"{t}|SCALE_OUT|{shares}sh|pnl_{pnl_pct:+.1%}|runner_continues")
                    else:
                        # Full close (stop loss, target, momentum death, bearish reversal)
                        resp = execute_stock_sell(t, client, action)
                        if resp:
                            emoji = "💰" if pnl_pct > 0 else "🔴"
                            msg = f"{emoji} {t}: {reason} (P/L: {pnl_pct:+.1%})"
                            summary["actions"].append(exit_signal)
                            summary["alerts"].append(msg)
                            diary_write("strategy_agent",
                                f"{t}|STOCK_EXIT|{action}|pnl_{pnl_pct:+.1%}")
            except Exception as e:
                log_event("monitor", "stock_check_failed", {"error": str(e)})

        # --- Options position monitoring ---
        option_positions = [p for p in open_positions if p.get("type") in ("csp", "cc")]
        for pos in option_positions:
            ticker = pos.get("ticker", "")

            # Check assignment
            assignment = check_assignment(pos, broker_positions)
            if assignment:
                # Update position
                pos["status"] = "assigned"
                pos["assigned_at"] = datetime.now(timezone.utc).isoformat()
                pos["assigned_shares"] = assignment["shares"]
                pos["cost_basis"] = assignment["avg_price"] - pos.get("premium_collected", 0)

                summary["actions"].append(assignment)
                summary["alerts"].append(f"📋 {ticker}: ASSIGNED at {pos.get('strike')}")

                # Memory
                kg_invalidate(ticker, "entered_csp",
                              f"{pos.get('strike')}P_{pos.get('expiration')}")
                kg_add(ticker, "assigned", f"{assignment['shares']}_shares",
                       metadata={"cost_basis": pos["cost_basis"]})
                diary_write("strategy_agent",
                    f"{ticker}|ASSIGNED|{pos.get('strike')}|shares_{assignment['shares']}|"
                    f"cost_basis_{pos['cost_basis']:.2f}")

                send_alert(f"📋 {ticker} CSP ASSIGNED\n"
                          f"Strike: {pos.get('strike')}\n"
                          f"Shares: {assignment['shares']}\n"
                          f"Cost basis: ${pos['cost_basis']:.2f}")
                continue

            # Check early close using live option prices
            if option_prices and ticker in option_prices:
                early = check_early_close(pos, option_prices[ticker])
                if early:
                    summary["actions"].append(early)
                    summary["alerts"].append(
                        f"💰 {ticker}: {early['reason']}"
                    )
                    diary_write("strategy_agent",
                        f"{ticker}|EARLY_CLOSE_CANDIDATE|"
                        f"profit_{early['profit_pct']:.0%}|{early['dte']}DTE")

            # Check roll candidate (stock price needed)
            stock_price = None
            for bp in broker_positions:
                if bp.get("symbol") == ticker:
                    stock_price = float(bp.get("current_price", 0))
                    break

            if not stock_price:
                # Quietly skipping roll/early-close decisions when we have no
                # stock price was the bite at finding #5. Make it visible so a
                # broker outage is at least audible.
                summary["degraded"].append({
                    "ticker": ticker, "type": "option",
                    "reason": "no_underlying_price",
                })
                log_event("monitor", "option_check_degraded",
                          {"ticker": ticker, "reason": "no_underlying_price"},
                          result="degraded")

            if stock_price:
                roll = check_roll_candidate(pos, stock_price)
                if roll:
                    summary["actions"].append(roll)
                    summary["alerts"].append(f"🔄 {ticker}: {roll['reason']}")
                    diary_write("strategy_agent",
                        f"{ticker}|ROLL_CANDIDATE|{roll['dte']}DTE|ITM_{roll['itm_amount']:.2f}")

        # Save updated positions
        _save_positions(positions)

        # Send alerts
        for alert in summary["alerts"]:
            send_alert(alert)

        # Daily summary (if near market close — 3:50 PM ET)
        now = datetime.now(timezone.utc)
        if now.hour == 19 and now.minute >= 50:  # ~3:50 PM ET
            daily_summary = (
                f"📊 Daily Summary\n"
                f"Portfolio: ${account['portfolio_value']:,.2f}\n"
                f"Cash: ${account['cash']:,.2f}\n"
                f"Open positions: {len(open_positions)}\n"
                f"Actions today: {len(summary['actions'])}"
            )
            send_alert(daily_summary)

        # Hermes self-optimization (run after close — 4:10 PM ET)
        if now.hour == 20 and 10 <= now.minute <= 15:
            try:
                # Load settings here — was a NameError on every Hermes window
                # firing because run_monitoring_check never bound `settings`.
                # Caught during 2026-05-01 verification: monitor.hermes_failed
                # event fired with `"error": "name 'settings' is not defined"`.
                settings = _load_settings()
                hermes_cfg = settings.get("hermes", {})
                if hermes_cfg.get("enabled") and hermes_cfg.get("run_after_close"):
                    from agents.hermes_optimizer import run_optimization
                    lookback = hermes_cfg.get("lookback_days", 14)
                    min_trades = hermes_cfg.get("min_trades_to_optimize", 3)
                    report = run_optimization(lookback_days=lookback)
                    changes = report.get("changes", {})
                    if changes:
                        change_str = ", ".join(f"{k}: {v['old']}→{v['new']}" for k, v in changes.items())
                        send_alert(f"🔮 Hermes optimized: {change_str}")
                        summary["alerts"].append(f"Hermes: {len(changes)} params adjusted")
            except Exception as e:
                log_event("monitor", "hermes_failed", {"error": str(e)})

        # Fund manager portfolio-level review (TauricResearch Stage V analog).
        # Runs every cycle but only alerts when actions are needed — keeps
        # the operator looped in on sector / correlation / cash drift.
        try:
            from agents.fund_manager import FundManager
            fm_review = FundManager().review_portfolio(
                positions=open_positions,
                bankroll=float(account.get("portfolio_value", 0) or 0),
                cash=float(account.get("cash", 0) or 0),
            )
            if fm_review["actions"]:
                for a in fm_review["actions"]:
                    severity_emoji = {"info": "ℹ️", "warn": "⚠️", "critical": "🔴"}
                    e = severity_emoji.get(a["severity"], "•")
                    summary["alerts"].append(f"{e} FundMgr: {a['summary']}")
                summary["actions"].append({
                    "action": "fund_manager_review",
                    "n_actions": len(fm_review["actions"]),
                    "types": [a["type"] for a in fm_review["actions"]],
                })
        except Exception as e:
            log_event("monitor", "fund_manager_failed", {"error": str(e)[:200]},
                      result="degraded")

    except Exception as e:
        log_event("monitor", "check_failed", {"error": str(e)}, result="failed")
        summary["error"] = str(e)
        send_alert(f"❌ Monitoring check failed: {e}")

    log_event("monitor", "check_complete", {
        "positions": summary["positions_checked"],
        "actions": len(summary["actions"]),
    }, result="success")

    return summary


def start_monitoring_loop(client: AlpacaClient):
    """
    Continuous monitoring loop. Runs until killed.
    In production, this is launched as a background process or cron.
    """
    settings = _load_settings()
    interval = settings.get("monitoring", {}).get("check_interval_seconds", 300)

    log_event("monitor", "loop_started", {"interval": interval})
    send_alert("🟢 Monitoring loop started")

    print(f"🟢 Monitoring started — checking every {interval}s (Ctrl+C to stop)")
    check_num = 0

    while True:
        check_num += 1
        try:
            summary = run_monitoring_check(client)
            now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            pv = summary.get("portfolio_value", 0)
            actions = len(summary.get("actions", []))
            positions = summary.get("positions_checked", 0)
            print(f"  [{now}] Check #{check_num}: "
                  f"${pv:,.2f} portfolio, {positions} positions, {actions} actions")

            for alert in summary.get("alerts", []):
                print(f"    {alert}")

        except Exception as e:
            record_missed_check()
            log_event("monitor", "loop_error", {"error": str(e)})
            print(f"  ❌ Check #{check_num} failed: {e}")

        time.sleep(interval)
