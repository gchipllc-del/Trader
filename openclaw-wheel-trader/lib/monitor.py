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
    record_trade_outcome, reflect_on_outcome, search_memory,
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


def _resolve_position_outcome(
    pos: dict,
    *,
    exit_reason: str,
    final_price: float | None = None,
    final_pnl_dollars: float | None = None,
) -> None:
    """Bridge from a position-close event to the MemPalace learning loop.

    Looks up the original decision_drawer_id on the position, computes
    realized return + holding days, records an outcome drawer, and
    fires an LLM reflection. All best-effort: failure here never blocks
    trading or breaks the close detection that called us.
    """
    try:
        drawer_id = pos.get("decision_drawer_id") or pos.get("cc_decision_drawer_id")
        if not drawer_id:
            return  # legacy position from before the learning loop landed

        ticker = pos.get("ticker") or "?"
        opened_at = pos.get("opened_at") or ""
        holding_days = 0
        if opened_at:
            try:
                opened = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
                holding_days = max(0, (datetime.now(timezone.utc) - opened).days)
            except (ValueError, TypeError):
                pass

        # Realized return: try the explicit P&L the caller supplied, else
        # infer from strike + premium + final_price for typical exits.
        realized = 0.0
        if final_pnl_dollars is not None:
            collateral = float(pos.get("strike", 0) or 0) * 100
            realized = (final_pnl_dollars / collateral) if collateral > 0 else 0.0
        elif exit_reason == "csp_assigned" and final_price is not None:
            strike = float(pos.get("strike", 0) or 0)
            premium = float(pos.get("premium_collected", 0) or 0)
            cost_basis = strike - premium
            if cost_basis > 0:
                realized = (final_price - cost_basis) / cost_basis
        elif exit_reason in ("csp_expired", "cc_expired"):
            premium = float(pos.get("premium_collected", 0) or pos.get("cc_premium", 0) or 0)
            strike = float(pos.get("strike", 0) or 0)
            collateral = max(strike * 100, 1.0)
            realized = (premium * 100) / collateral
        elif exit_reason == "cc_called_away" and final_price is not None:
            cost_basis = float(pos.get("cost_basis", 0) or 0)
            premium = float(pos.get("cc_premium", 0) or 0)
            strike = float(pos.get("cc_strike", 0) or final_price)
            entry = cost_basis + premium  # original cost before CC adjust
            if entry > 0:
                realized = (strike - entry) / entry

        outcome_id = record_trade_outcome(
            ticker=ticker,
            decision_drawer_id=drawer_id,
            realized_return_pct=realized,
            holding_days=holding_days,
            exit_reason=exit_reason,
            final_pnl_dollars=final_pnl_dollars,
        )

        # Best-effort LLM reflection — pull the original reasoning back
        # from the palace and ask the LLM to write a short lesson.
        try:
            original = search_memory(ticker, wing=f"wing_{ticker.lower()}",
                                     hall="hall_facts", n_results=20)
            decision_content = next(
                (o.get("content", "") for o in original
                 if o.get("drawer_id") == drawer_id),
                "",
            )
            if decision_content:
                outcome_summary = (
                    f"{exit_reason} after {holding_days} days, "
                    f"realized return {realized:+.2%}"
                )
                reflect_on_outcome(
                    decision_drawer_id=drawer_id,
                    outcome_drawer_id=outcome_id,
                    ticker=ticker,
                    decision_reasoning=decision_content,
                    outcome_summary=outcome_summary,
                )
        except Exception as e:
            log_event("monitor", "reflection_failed",
                      {"ticker": ticker, "error": str(e)[:200]}, result="degraded")
    except Exception as e:
        log_event("monitor", "resolve_outcome_failed",
                  {"ticker": pos.get("ticker"), "error": str(e)[:200]},
                  result="degraded")


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

def run_monitoring_check(client: AlpacaClient, *, timeout_seconds: int = 120) -> dict:
    """
    Single monitoring check. Called every 3 minutes by launchd cron.

    Returns summary dict of actions taken.

    Wall-clock timeout (default 120s) defends against the 2026-05-12
    incident where an Alpaca HTTPS connection stayed "established" with
    no response, hanging the monitor for 11 hours and blocking launchd
    from firing fresh cycles. SIGALRM interrupts a blocking syscall on
    the next return-from-kernel — works against SSL socket reads where
    the alpaca-py SDK has no built-in timeout.
    """
    import signal as _signal

    class _MonitorTimeout(Exception):
        pass

    def _on_alarm(signum, frame):  # noqa: ARG001
        raise _MonitorTimeout(
            f"monitor cycle exceeded {timeout_seconds}s wallclock — "
            f"likely Alpaca SDK hang on an unresponsive HTTPS connection"
        )

    # signal.SIGALRM is main-thread-only on macOS / Linux. The monitor
    # cron runs as a one-shot process so this is the main thread.
    _prev_handler = _signal.signal(_signal.SIGALRM, _on_alarm)
    _signal.alarm(int(timeout_seconds))

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
                # original_purchase_price is the unadjusted broker fill price —
                # we use it as the HARD CC strike floor so future CCs can't
                # ratchet below the actual capital we put in, even after
                # premiums "net out" the running cost_basis. Wave-D
                # alpacahq/options-wheel insight: cost_basis math is good
                # for tax/P&L reporting, bad for assignment-risk decisions.
                pos["original_purchase_price"] = float(assignment["avg_price"])

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

                # Learning loop: resolve the CSP decision with its outcome.
                # CSP assignment = strike was reached → outcome is the
                # premium collected minus the immediate unrealized loss
                # vs the assignment price. Future agents will see this
                # outcome attached to the original reasoning.
                _resolve_position_outcome(
                    pos, exit_reason="csp_assigned",
                    final_price=float(assignment["avg_price"]),
                )

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
            # Fetch ~90 days of daily returns for the long-stock book so
            # the HRP rebalance check can run. Best-effort: if the fetch
            # fails or there are <3 stock positions, HRP just doesn't fire.
            hist_returns = None
            stock_tickers_for_hrp = list({
                p.get("ticker") for p in open_positions
                if p.get("type") == "stock" and p.get("ticker")
            })
            if len(stock_tickers_for_hrp) >= 3:
                try:
                    import pandas as pd
                    bars = client.get_bars(stock_tickers_for_hrp,
                                           timeframe="1Day", limit=120)
                    series_by_ticker = {}
                    for t in stock_tickers_for_hrp:
                        if isinstance(bars, dict) and t in bars and len(bars[t]) > 30:
                            series_by_ticker[t] = bars[t]["close"].pct_change().dropna()
                    if len(series_by_ticker) >= 3:
                        hist_returns = pd.DataFrame(series_by_ticker).dropna()
                except Exception as _hrp_e:
                    log_event("monitor", "hrp_returns_fetch_failed",
                              {"error": str(_hrp_e)[:200]}, result="degraded")

            fm_review = FundManager().review_portfolio(
                positions=open_positions,
                bankroll=float(account.get("portfolio_value", 0) or 0),
                cash=float(account.get("cash", 0) or 0),
                historical_returns=hist_returns,
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

    except _MonitorTimeout as e:
        # Hang detected — abort the cycle cleanly so launchd can re-fire.
        # Without this branch the entire process stays blocked in
        # whatever SSL recv() was hung, and the cron never advances.
        log_event("monitor", "check_timeout", {
            "timeout_seconds": int(timeout_seconds),
            "error": str(e)[:200],
        }, result="failed")
        summary["error"] = f"timeout after {timeout_seconds}s"
        send_alert(f"⏱️  Monitor cycle aborted after {timeout_seconds}s — "
                   f"likely Alpaca connection hang. Next cycle will retry.")
    except Exception as e:
        log_event("monitor", "check_failed", {"error": str(e)}, result="failed")
        summary["error"] = str(e)
        send_alert(f"❌ Monitoring check failed: {e}")
    finally:
        # Disarm the alarm and restore the previous SIGALRM handler so
        # callers (tests, REPL) aren't left with a dangling timer.
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, _prev_handler)

    # Self-audit: scan the last 4h of the bot's own audit log for
    # pipeline-funnel anomalies (e.g. "many attempts, 0 executed" —
    # the pattern that hid the 2026-05-13 contracts/shares conflation
    # bug for half a day). Read-only, never blocks. Critical findings
    # ride the same Telegram alert channel. Pass the broker client so
    # state-reconciliation + P&L-reconciliation can also fire.
    try:
        from lib.self_audit import run_self_audit
        audit_result = run_self_audit(hours=4, broker_client=client)
        for a in audit_result.get("alerts", []):
            sev = a.get("severity", "info")
            summary["alerts"].append(
                f"{'🔴' if sev == 'critical' else '⚠️'} SELF-AUDIT [{a['code']}]: {a['summary']}"
            )
            if sev == "critical":
                send_alert(
                    f"🔴 SELF-AUDIT {a['code']}: {a['summary']}"
                )
    except Exception as e:
        log_event("monitor", "self_audit_failed",
                  {"error": str(e)[:200]}, result="degraded")

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
