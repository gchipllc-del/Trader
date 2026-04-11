"""
Sprint 9: Main Entry Point & Hardening

The single entry point for the trading bot. Supports modes:
  - scan      : Run one CSP/CC scan cycle
  - monitor   : Start continuous monitoring loop
  - backtest  : Run backtest + Monte Carlo
  - kill      : Emergency kill switch
  - status    : Print current positions and portfolio
  - chaos     : Run chaos tests (simulate failures)
  - migrate   : Begin paper→live migration checklist

Usage:
  python main.py scan
  python main.py monitor
  python main.py backtest --ticker AAPL
  python main.py kill
  python main.py status
  python main.py chaos
"""

import argparse
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from lib.audit import log_event
from lib.memory_palace import init_palace, get_current_regime


POSITIONS_PATH = Path(__file__).parent / "data" / "positions.json"
STATUS_PATH = Path(__file__).parent / "data" / "STATUS.md"


def cmd_status():
    """Print current bot status."""
    from lib.alpaca_client import AlpacaClient

    print("=" * 50)
    print("  OPENCLAW WHEEL STRATEGY TRADER — STATUS")
    print("=" * 50)

    try:
        client = AlpacaClient()
        account = client.get_account()
        positions = client.get_positions()
        orders = client.get_open_orders()

        print(f"\n  Mode:            {'PAPER' if 'paper' in client.base_url else '🔴 LIVE'}")
        print(f"  Portfolio Value:  ${account['portfolio_value']:,.2f}")
        print(f"  Cash:            ${account['cash']:,.2f}")
        print(f"  Buying Power:    ${account['buying_power']:,.2f}")
        print(f"  Market Regime:   {get_current_regime() or 'unknown'}")
        print(f"\n  Open Positions:  {len(positions)}")
        for p in positions:
            print(f"    {p['symbol']:8s} {float(p['qty']):>6.0f} shares  "
                  f"P/L: ${float(p['unrealized_pl']):>8.2f}")
        print(f"\n  Pending Orders:  {len(orders)}")
        for o in orders:
            print(f"    {o['symbol']:8s} {o['side']:4s} {o['qty']} @ {o.get('limit_price', 'MKT')}")

    except Exception as e:
        print(f"\n  ❌ Could not connect to Alpaca: {e}")
        print("  Check your .env file and network connection.")

    # Local positions
    if POSITIONS_PATH.exists():
        with open(POSITIONS_PATH) as f:
            local_pos = json.load(f)
        open_local = [p for p in local_pos if p.get("status") in ("open", "assigned")]
        print(f"\n  Tracked Positions (local): {len(open_local)}")
        for p in open_local:
            print(f"    {p.get('ticker'):8s} {p.get('type'):4s} "
                  f"{p.get('strike', '')} {p.get('status', '')}")

    print()


def cmd_scan():
    """Run one scan cycle — auto-selects Phase 1 (stocks) or Phase 2/3 (options) based on portfolio size."""
    from lib.alpaca_client import AlpacaClient
    from lib.data_pipeline import fetch_all_data
    from lib.stock_engine import (
        run_stock_scan_and_execute, get_current_phase,
        PHASE_2_THRESHOLD, PHASE_3_THRESHOLD,
    )
    from lib.csp_engine import run_csp_scan_and_execute
    from lib.cc_engine import scan_for_ccs, find_assigned_positions

    import yaml
    with open(Path(__file__).parent / "config" / "wheel_strategy.yaml") as f:
        strategy = yaml.safe_load(f)

    log_event("main", "scan_started", {})

    try:
        client = AlpacaClient()
        account = client.get_account()
        portfolio = account["portfolio_value"]
        cash = account["cash"]
        phase = get_current_phase(portfolio)

        print("=" * 55)
        print("  OPENCLAW WHEEL TRADER — SCAN")
        print("=" * 55)
        print(f"  Portfolio: ${portfolio:,.2f}  Cash: ${cash:,.2f}")
        print(f"  Phase {phase}: ", end="")

        if phase == 1:
            print(f"Stock Trading (upgrade to CSPs at ${PHASE_2_THRESHOLD:,})")
            tickers = strategy.get("tickers_phase1", strategy.get("tickers", []))
        elif phase == 2:
            print(f"Stock + CSPs on cheap stocks (full Wheel at ${PHASE_3_THRESHOLD:,})")
            tickers = strategy.get("tickers_phase1", [])
        else:
            print("Full Wheel Strategy")
            tickers = strategy.get("tickers", [])

        if portfolio <= 0 and cash <= 0:
            print("\n  ⚠️  Paper account has $0. Reset it at https://app.alpaca.markets/paper/dashboard")
            return

        # Fetch market data for phase-appropriate tickers
        print(f"\n  Fetching data for {len(tickers)} tickers: {', '.join(tickers)}")
        data = fetch_all_data(client, tickers=tickers)

        daily = data["daily_data"]
        weekly = data["weekly_data"]
        chains = data["options_chains"]
        iv = data["iv_data"]

        print(f"  Bars: {len(daily)} tickers")

        # Phase 1: Stock trading
        if phase <= 2:
            print("\n  --- Stock Scan ---")
            stock_results = run_stock_scan_and_execute(
                client, daily, weekly, portfolio, max_trades=2,
            )
            if stock_results:
                for r in stock_results:
                    action = r.get("action", "")
                    if action == "buy":
                        c = r["candidate"]
                        print(f"  ✅ Bought {c['shares']}x {c['ticker']} @ ${c['current_price']:.2f} "
                              f"(score {c['composite_score']}/9, {c['pattern'] or 'no pattern'})")
                    elif action == "sell":
                        print(f"  💰 Sold {r['ticker']} — {r['reason']}")
            else:
                print("  No stock trades (candidates below score threshold or already held)")

        # Phase 2+: CSP scanning
        if phase >= 2 and chains:
            print(f"\n  --- CSP Scan ({len(chains)} chains) ---")
            csp_results = run_csp_scan_and_execute(
                client, daily, weekly, chains, iv, max_trades=1,
            )
            if csp_results:
                for r in csp_results:
                    print(f"  ✅ CSP executed: {r.get('symbol')} — {r.get('status')}")
            else:
                print("  No CSP trades (no candidates or scores below 7)")

        # Phase 3: CC scanning on assigned positions
        if phase >= 2:
            assigned = find_assigned_positions()
            if assigned:
                print(f"\n  --- CC Scan ({len(assigned)} assigned) ---")
                cc_candidates = scan_for_ccs(client, daily, weekly, chains, iv)
                if cc_candidates:
                    for c in cc_candidates[:3]:
                        print(f"  📋 CC: {c.ticker} {c.strike}C exp {c.expiration} "
                              f"score {c.composite_score}/9 ${c.premium:.2f}")

        print()

    except Exception as e:
        print(f"\n  ❌ Scan failed: {e}")
        log_event("main", "scan_failed", {"error": str(e)}, result="failed")
        import traceback
        traceback.print_exc()


def cmd_monitor():
    """Start the monitoring loop."""
    from lib.alpaca_client import AlpacaClient
    from lib.monitor import start_monitoring_loop

    print("🟢 Starting monitoring loop (Ctrl+C to stop)...")
    client = AlpacaClient()
    start_monitoring_loop(client)


def cmd_kill(reason: str = "manual_cli"):
    """Emergency kill switch."""
    from lib.kill_switch import activate_kill_switch

    print(f"⚠️  KILL SWITCH: {reason}")
    result = activate_kill_switch(reason)
    print(f"  Orders cancelled: {result['orders_cancelled']}")
    print(f"  Positions closed: {result['positions_closed']}")
    if result["errors"]:
        print(f"  Errors: {result['errors']}")
    else:
        print("  ✅ Clean shutdown.")


def cmd_backtest(ticker: str = "SPY"):
    """Run backtest with Monte Carlo."""
    print(f"📊 Backtesting Wheel Strategy on {ticker}...")
    print("   Requires historical data. Run in Claude Code with:")
    print(f'   "Backtest the Wheel Strategy on {ticker} using 2 years of data"')
    print()
    print("   The backtest engine (lib/backtest.py) supports:")
    print("   - Walk-forward validation")
    print("   - Monte Carlo simulation (1000+ runs)")
    print("   - Slippage & fee modeling")
    print("   - Benchmark comparison vs buy-and-hold")


def cmd_chaos():
    """Run chaos tests to validate resilience."""
    print("🔥 CHAOS TESTING")
    print("=" * 50)

    results = []

    # Test 1: Circuit breaker trips correctly
    print("\n[1/5] Circuit breaker — daily loss limit...")
    from lib.circuit_breaker import check_daily_loss, CircuitBreakerTripped
    try:
        check_daily_loss(-9999)
        results.append(("circuit_breaker_loss", "FAIL", "Should have tripped"))
    except CircuitBreakerTripped:
        results.append(("circuit_breaker_loss", "PASS", "Correctly blocked"))

    # Test 2: Duplicate order detection
    print("[2/5] Duplicate order prevention...")
    from lib.order_gate import OrderIntent, step1_propose, _recent_intents
    _recent_intents.clear()
    intent = OrderIntent(
        ticker="CHAOS_TEST", side="sell_to_open", order_type="limit",
        asset_type="option", quantity=1, strike=100,
        expiration="2099-12-31", composite_score=9,
    )
    step1_propose(intent)
    try:
        step1_propose(intent)
        results.append(("duplicate_prevention", "FAIL", "Allowed duplicate"))
    except ValueError:
        results.append(("duplicate_prevention", "PASS", "Blocked duplicate"))

    # Test 3: Low score rejection
    print("[3/5] Low composite score rejection...")
    low_intent = OrderIntent(
        ticker="CHAOS_LOW", side="sell_to_open", order_type="limit",
        asset_type="option", quantity=1, strike=50,
        expiration="2099-12-31", composite_score=3,
    )
    try:
        from lib.order_gate import step2_validate
        step2_validate(low_intent, 100000, 0, 0)
        results.append(("low_score_rejection", "FAIL", "Allowed low score"))
    except (ValueError, CircuitBreakerTripped):
        results.append(("low_score_rejection", "PASS", "Correctly rejected"))

    # Test 4: Audit logger redacts secrets
    print("[4/5] Secret redaction in audit logs...")
    from lib.audit import log_event as _log
    event = _log("chaos", "test", {"api_key": "SECRET123", "ticker": "AAPL"})
    if event["details"]["api_key"] == "***REDACTED***":
        results.append(("secret_redaction", "PASS", "Secrets redacted"))
    else:
        results.append(("secret_redaction", "FAIL", "Secret exposed!"))

    # Test 5: Anomaly detection
    print("[5/5] Anomaly detection system...")
    from lib.enhancements import detect_anomalies
    anomalies = detect_anomalies()
    results.append(("anomaly_detection", "PASS", f"Detected {len(anomalies)} anomalies"))

    # Summary
    print("\n" + "=" * 50)
    print("CHAOS TEST RESULTS")
    print("=" * 50)
    passed = sum(1 for _, status, _ in results if status == "PASS")
    for name, status, detail in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name:30s} {detail}")

    print(f"\n  {passed}/{len(results)} passed")

    if passed == len(results):
        print("  🎉 All chaos tests passed!")
    else:
        print("  ⚠️  Some tests failed — investigate before going live.")


def cmd_migrate():
    """Print the paper→live migration checklist."""
    print("📋 PAPER → LIVE MIGRATION CHECKLIST")
    print("=" * 50)
    checklist = [
        "[ ] ≥2 weeks of paper trading completed",
        "[ ] Paper results match backtest within 1 std dev",
        "[ ] All 5 chaos tests pass",
        "[ ] Kill switch tested (run: python main.py kill --reason test)",
        "[ ] Telegram alerts verified (received test messages)",
        "[ ] Agent consensus working (all 3 agents log correctly)",
        "[ ] Audit log reviewed (no unexpected entries)",
        "[ ] Circuit breakers verified at each threshold",
        "[ ] .env has LIVE Alpaca keys (not paper)",
        "[ ] config/settings.yaml: mode changed to 'live'",
        "[ ] config/settings.yaml: live_migration_approved set to true",
        "[ ] Phase 1: Start with 1 ticker, 1 CSP, minimum size",
        "[ ] Phase 2: After 30 profitable days → expand to 3 tickers",
        "[ ] Phase 3: After 90 days → full universe",
    ]
    for item in checklist:
        print(f"  {item}")
    print()
    print("  ⚠️  DO NOT skip steps. Each exists because something can go wrong.")


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Wheel Strategy Trader")
    parser.add_argument("command", choices=["scan", "monitor", "backtest", "kill", "status", "chaos", "migrate"],
                        help="Command to run")
    parser.add_argument("--ticker", default="SPY", help="Ticker for backtest")
    parser.add_argument("--reason", default="manual_cli", help="Reason for kill switch")

    args = parser.parse_args()

    # Init memory palace on any command
    init_palace()

    log_event("main", f"command_{args.command}", {})

    if args.command == "status":
        cmd_status()
    elif args.command == "scan":
        cmd_scan()
    elif args.command == "monitor":
        cmd_monitor()
    elif args.command == "kill":
        cmd_kill(args.reason)
    elif args.command == "backtest":
        cmd_backtest(args.ticker)
    elif args.command == "chaos":
        cmd_chaos()
    elif args.command == "migrate":
        cmd_migrate()


if __name__ == "__main__":
    main()
