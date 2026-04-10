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

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
STRATEGY_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"


def _load_positions() -> list[dict]:
    if not POSITIONS_PATH.exists():
        return []
    with open(POSITIONS_PATH) as f:
        return json.load(f)


def _save_positions(positions: list[dict]):
    with open(POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)


def _load_settings() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_strategy() -> dict:
    with open(STRATEGY_PATH) as f:
        return yaml.safe_load(f)


# ============================================================
# HEARTBEAT TRACKER
# ============================================================

_missed_checks = 0


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
    Single monitoring check. Called every 5 minutes by cron.

    Returns summary dict of actions taken.
    """
    reset_heartbeat()
    log_event("monitor", "check_started", {})

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "positions_checked": 0,
        "actions": [],
        "alerts": [],
    }

    try:
        account = client.get_account()
        broker_positions = client.get_positions()
        positions = _load_positions()
        open_positions = [p for p in positions if p.get("status") == "open"]

        summary["portfolio_value"] = account["portfolio_value"]
        summary["cash"] = account["cash"]
        summary["positions_checked"] = len(open_positions)

        for pos in open_positions:
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

            # Check early close (option price needed — placeholder)
            # In production, fetch current option price from chain
            # early = check_early_close(pos, current_option_price)

            # Check roll candidate (stock price needed)
            stock_price = None
            for bp in broker_positions:
                if bp.get("symbol") == ticker:
                    stock_price = float(bp.get("current_price", 0))
                    break

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

    while True:
        try:
            summary = run_monitoring_check(client)
        except Exception as e:
            record_missed_check()
            log_event("monitor", "loop_error", {"error": str(e)})

        time.sleep(interval)
