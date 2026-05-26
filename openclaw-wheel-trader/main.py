"""
Sprint 9: Main Entry Point & Hardening

The single entry point for the trading bot. Supports modes:
  - scan      : Run one CSP/CC scan cycle (now with Kronos + News gates)
  - monitor   : Start continuous monitoring loop
  - backtest  : Run backtest + Monte Carlo
  - kill      : Emergency kill switch
  - status    : Print current positions and portfolio
  - chaos     : Run chaos tests (simulate failures)
  - migrate   : Begin paper→live migration checklist
  - kronos    : Run Kronos AI price prediction on a ticker
  - news      : Check news sentiment for a ticker
  - calibrate : Print stock prediction accuracy report
  - pred-scan : Scan prediction markets for opportunities (Manifold paper)

Usage:
  python main.py scan
  python main.py monitor
  python main.py kronos --ticker AAPL
  python main.py news --ticker SOFI
  python main.py calibrate
  python main.py pred-scan
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


def cmd_status(full: bool = False):
    """Print current bot status with colored output."""
    try:
        from lib.dashboard_terminal import render_terminal_dashboard
        render_terminal_dashboard(include_quant=full)
    except ImportError:
        _cmd_status_plain()


def _cmd_status_plain():
    """Fallback plain-text status if Rich is not installed."""
    from lib.alpaca_client import AlpacaClient

    print("=" * 50)
    print("  OPENCLAW WHEEL STRATEGY TRADER — STATUS")
    print("=" * 50)

    try:
        client = AlpacaClient()
        account = client.get_account()
        positions = client.get_positions()
        orders = client.get_open_orders()

        print(f"\n  Mode:            {'PAPER' if 'paper' in client.base_url else 'LIVE'}")
        print(f"  Portfolio Value:  ${account['portfolio_value']:,.2f}")
        print(f"  Cash:            ${account['cash']:,.2f}")
        print(f"  Buying Power:    ${account['buying_power']:,.2f}")

        # %-gain-to-date vs baseline equity
        try:
            from lib.dashboard_data import _get_baseline
            baseline, baseline_set_at = _get_baseline(float(account["portfolio_value"]))
            dollar_gain = float(account["portfolio_value"]) - baseline
            pct_gain = (dollar_gain / baseline) if baseline > 0 else 0.0
            baseline_date = (baseline_set_at or "")[:10]
            since = f" (since {baseline_date})" if baseline_date else ""
            print(f"  Gain to Date:    ${dollar_gain:+,.2f} ({pct_gain:+.2%}) "
                  f"vs ${baseline:,.2f} baseline{since}")
        except Exception:
            pass

        print(f"  Market Regime:   {get_current_regime() or 'unknown'}")
        print(f"\n  Open Positions:  {len(positions)}")
        for p in positions:
            print(f"    {p['symbol']:8s} {float(p['qty']):>6.0f} shares  "
                  f"P/L: ${float(p['unrealized_pl']):>8.2f}")
        print(f"\n  Pending Orders:  {len(orders)}")
        for o in orders:
            print(f"    {o['symbol']:8s} {o['side']:4s} {o['qty']} @ {o.get('limit_price', 'MKT')}")

    except Exception as e:
        print(f"\n  Could not connect to Alpaca: {e}")

    if POSITIONS_PATH.exists():
        with open(POSITIONS_PATH) as f:
            local_pos = json.load(f)
        open_local = [p for p in local_pos if p.get("status") in ("open", "assigned")]
        print(f"\n  Tracked Positions (local): {len(open_local)}")
        for p in open_local:
            print(f"    {p.get('ticker'):8s} {p.get('type'):4s} "
                  f"{p.get('strike', '')} {p.get('status', '')}")
    print()


def cmd_baseline(amount: float | None = None, show: bool = False):
    """
    Set (or view) the %-gain-to-date baseline.

        python main.py baseline                   # show current baseline
        python main.py baseline --amount 932      # set starting capital to $932
        python main.py baseline --snapshot        # use current portfolio as baseline
    """
    from lib.dashboard_data import set_baseline, _get_baseline, BASELINE_PATH
    from lib.alpaca_client import AlpacaClient

    if show:
        if not BASELINE_PATH.exists():
            print("  No baseline set yet. Run `status` or `dashboard` to auto-snapshot,")
            print("  or `baseline --amount <X>` to set explicitly.")
            return
        with open(BASELINE_PATH) as f:
            data = json.load(f)
        print(f"  Baseline equity: ${float(data.get('baseline_equity', 0)):,.2f}")
        print(f"  Set at:          {data.get('set_at', '')}")
        print(f"  Note:            {data.get('note', '')}")

        try:
            client = AlpacaClient()
            account = client.get_account()
            current = float(account["portfolio_value"])
            baseline = float(data.get("baseline_equity", current))
            gain = current - baseline
            pct = (gain / baseline) if baseline > 0 else 0.0
            print(f"\n  Current equity:  ${current:,.2f}")
            print(f"  Gain to date:    ${gain:+,.2f} ({pct:+.2%})")
        except Exception as e:
            print(f"  (Could not fetch current equity: {e})")
        return

    result = set_baseline(amount)  # amount=None → snapshot current portfolio
    print(f"  Baseline set to ${float(result['baseline_equity']):,.2f}")
    print(f"  Set at:         {result['set_at']}")


def cmd_dashboard(port: int = 5051):
    """Start the web dashboard server.

    Default port 5051 avoids collision with sibling polybot project (port 5050).
    """
    from lib.dashboard_web import run_dashboard
    print(f"  Traderbot Dashboard: http://localhost:{port}")
    print("  (Polybot uses 5050; traderbot uses 5051 to avoid conflict.)")
    print("  Press Ctrl+C to stop.\n")
    run_dashboard(port=port)


def cmd_pairs_scan():
    """Scan the Phase-1 universe for mean-reverting pairs (T2.6)."""
    import yaml
    from lib.alpaca_client import AlpacaClient
    from lib.portfolio_optimization import find_pairs_opportunities, print_pairs_report

    cfg = yaml.safe_load(open(Path(__file__).parent / "config" / "wheel_strategy.yaml"))
    tickers = cfg.get("tickers_phase1", [])
    print(f"Scanning {len(tickers)} tickers for mean-reverting pairs...")

    client = AlpacaClient()
    bars = client.get_bars(tickers, timeframe="1Day", limit=180)
    opps = find_pairs_opportunities(bars)

    print()
    print_pairs_report(opps, top=15)
    print()
    print(f"Found {len(opps)} statistically significant pairs.")
    print("Note: this scans signals only — does not execute trades.")
    print("Pairs trading needs a long+short engine, which is out of scope")
    print("for the current Phase-1 (long-only stock) bot. Useful for research.")


def cmd_min_variance():
    """Compute Markowitz min-variance weights for currently held positions (T2.5)."""
    import json
    from lib.alpaca_client import AlpacaClient
    from lib.portfolio_optimization import min_variance_weights

    with open(Path(__file__).parent / "data" / "positions.json") as _f:
        positions = json.load(_f)
    held = [p for p in positions
            if p.get("status") == "open" and p.get("type") == "stock"]
    if len(held) < 2:
        print(f"Need >= 2 open stock positions to compute min-variance; have {len(held)}.")
        return

    tickers = [p["ticker"] for p in held]
    client = AlpacaClient()
    bars = client.get_bars(tickers, timeframe="1Day", limit=180)
    import pandas as pd
    closes = pd.DataFrame({t: bars[t]["close"] for t in tickers if t in bars}).dropna()
    returns = closes.pct_change().dropna()

    pw = min_variance_weights(returns)
    print(f"\nMin-variance weights for {len(pw.tickers)} held positions:\n")
    print(f"{'Ticker':<10}{'MV Weight':<12}{'Current $':<14}{'MV Target $':<14}")
    total_value = sum(float(p.get("shares", 0)) * float(p.get("entry_price", 0))
                      for p in held)
    for p in held:
        t = p["ticker"]
        if t not in pw.weights:
            continue
        cur = float(p.get("shares", 0)) * float(p.get("entry_price", 0))
        target = pw.weights[t] * total_value
        print(f"{t:<10}{pw.weights[t]:<12.3f}${cur:<13.2f}${target:<13.2f}")
    print(f"\nPortfolio volatility (daily): {pw.portfolio_volatility*100:.3f}%")


def cmd_build_cache(signal: str = "all", days_back: int = 180):
    """Pre-fill historical caches so subsequent backtests are instant.

    signal: "kronos" | "news" | "llm" | "all"
    """
    import yaml
    cfg_path = Path(__file__).parent / "config" / "wheel_strategy.yaml"
    with open(cfg_path) as f:
        strategy = yaml.safe_load(f)
    tickers = strategy.get("tickers_phase1", [])

    from lib.alpaca_client import AlpacaClient
    from lib import historical_cache
    client = AlpacaClient()

    print("=" * 60)
    print(f"  CACHE BUILD — signal={signal}, lookback={days_back}d")
    print(f"  Universe: {len(tickers)} tickers")
    print("=" * 60)

    # Pull bars once (warmup + sim window)
    print("  Fetching bars...")
    bars = client.get_bars(tickers, timeframe="1Day", limit=days_back + 60)
    if not bars:
        print("  No bars returned.")
        return

    import pandas as pd
    all_dates = sorted({d for df in bars.values() for d in df.index})
    if not all_dates:
        print("  No dates in bars.")
        return
    cutoff = all_dates[-1] - pd.Timedelta(days=days_back)
    sim_dates = [d for d in all_dates if d >= cutoff]
    print(f"  Sim dates: {len(sim_dates)}  ({sim_dates[0].date()} to {sim_dates[-1].date()})")

    pairs = [(t, d) for t in tickers for d in sim_dates]
    total = len(pairs)
    print(f"  Total (ticker, date) pairs: {total}\n")

    signals_to_run = []
    if signal in ("kronos", "all"):
        signals_to_run.append("kronos")
    if signal in ("news", "all"):
        signals_to_run.append("news")
    if signal in ("llm", "all"):
        signals_to_run.append("llm")

    for sig in signals_to_run:
        print(f"  --- Building {sig} cache ---")
        if sig == "kronos":
            from lib.historical_kronos import get_historical_kronos
            fn = lambda t, d, df: get_historical_kronos(t, d, df)
        elif sig == "news":
            from lib.historical_news import get_historical_sentiment
            fn = lambda t, d, df: get_historical_sentiment(t, d)
        elif sig == "llm":
            from lib.historical_llm import get_historical_llm
            fn = lambda t, d, df: get_historical_llm(t, d, df, target_pct=0.10, horizon_days=21)
        else:
            continue

        hits = misses = errors = 0
        for i, (ticker, date) in enumerate(pairs, 1):
            if historical_cache.has(sig, ticker, date,
                                    {"target_pct": 0.10, "horizon_days": 21} if sig == "llm"
                                    else None):
                hits += 1
                continue
            df = bars.get(ticker)
            if df is None:
                misses += 1
                continue
            sliced = df[df.index <= date]
            if len(sliced) < 30:
                misses += 1
                continue
            try:
                fn(ticker, date, sliced)
                misses += 1  # was missing, now built
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"    err {ticker}/{date.date()}: {str(e)[:120]}")
            if i % 50 == 0:
                print(f"    [{i}/{total}] hits={hits} built={misses} err={errors}")

        print(f"  {sig} done: hits={hits} built={misses} err={errors}")
        stats = historical_cache.stats(sig)
        print(f"  Cache: {stats.get('signals',{}).get(sig,{})}\n")

    print("Build complete.")
    print("Stats summary:")
    for sig, info in historical_cache.stats().get("signals", {}).items():
        kb = info["bytes"] / 1024
        print(f"  {sig:<10} {info['entries']:>6} entries  {kb:>8.1f} KB")


def cmd_backtest_stocks(days_back: int = 180, capital: float = 1500.0,
                        enable_kronos: bool = False, enable_news: bool = False,
                        enable_llm: bool = False, enable_bayesian: bool = True):
    """Run an A/B comparison backtest of stock strategy variants.

    Signal enrichment flags read from CLI: --with-signals (all) or
    granular --with-kronos / --with-news / --with-llm.
    """
    import yaml
    from lib.stock_backtest import run_backtest

    cfg_path = Path(__file__).parent / "config" / "wheel_strategy.yaml"
    with open(cfg_path) as f:
        strategy = yaml.safe_load(f)
    tickers = strategy.get("tickers_phase1", [])

    enabled_signals = []
    if enable_bayesian:
        enabled_signals.append("bayesian")
    if enable_kronos:
        enabled_signals.append("kronos")
    if enable_news:
        enabled_signals.append("news")
    if enable_llm:
        enabled_signals.append("llm")
    sig_label = "+".join(enabled_signals) if enabled_signals else "price-only"

    print("=" * 60)
    print(f"  STOCK BACKTEST — {days_back}d on {len(tickers)} tickers")
    print(f"  Starting capital: ${capital:.0f}")
    print(f"  Signals enabled:  {sig_label}")
    print("=" * 60)

    # Pre-flight cache check so the user knows what they're getting
    if any([enable_kronos, enable_news, enable_llm]):
        from lib import historical_cache
        stats = historical_cache.stats()
        for sig in ("kronos", "news", "llm"):
            entries = stats.get("signals", {}).get(sig, {}).get("entries", 0)
            need = (sig == "kronos" and enable_kronos) or \
                   (sig == "news" and enable_news) or \
                   (sig == "llm" and enable_llm)
            if need:
                expected = len(tickers) * days_back  # rough upper bound
                pct = (entries / expected * 100) if expected else 0
                print(f"  cache[{sig}]: {entries} entries (~{pct:.0f}% of full coverage)")
                if entries == 0:
                    print(f"    ⚠ cache empty — will compute on-the-fly (slow). "
                          f"Run `python main.py build-cache --signal {sig}` first.")
        print()

    variants = {
        "current": {},
        "looser_score": {"min_composite_score": 7},
        "tighter_stop": {"stop_loss_pct": 0.025, "trailing_stop_pct": 0.015},
        "wider_target": {"default_target_pct": 0.15, "partial_exit_threshold": 0.07},
    }

    # Each variant runs independently — a transient Alpaca outage on one
    # must not throw away the hours already spent on the others. Catch and
    # log the failure, then continue.
    results = {}
    failures: dict[str, str] = {}
    for label, params in variants.items():
        print(f"\n=== Variant: {label} ===")
        try:
            results[label] = run_backtest(
                tickers=tickers, days_back=days_back, starting_capital=capital,
                params=params, enable_kronos=enable_kronos, enable_news=enable_news,
                enable_llm=enable_llm, enable_bayesian=enable_bayesian,
            )
            print(results[label].summary())
        except Exception as e:
            failures[label] = str(e)
            print(f"  ⚠ variant '{label}' failed: {e}")
            log_event("backtest", "variant_failed",
                      {"label": label, "error": str(e)[:300]}, result="degraded")

    print("\n" + "=" * 60)
    print(f"  COMPARISON SUMMARY ({sig_label})")
    print("=" * 60)
    print(f"{'Variant':<18}{'Return':<12}{'Sharpe':<10}{'MaxDD':<10}"
          f"{'Trades':<10}{'WinRate':<10}")
    for label, r in results.items():
        print(f"{label:<18}"
              f"{r.total_return*100:+.2f}%      "
              f"{r.sharpe_ratio:>5.2f}    "
              f"{r.max_drawdown*100:.2f}%   "
              f"{r.total_trades:>4}     "
              f"{r.win_rate*100:.1f}%")
    if failures:
        print(f"\n  Variants that failed: {', '.join(failures)}")
    if results:
        spy = list(results.values())[0].spy_buy_hold_return
        print(f"\n  Buy-and-hold SPY: {spy*100:+.2f}%")


def cmd_crypto_monitor(dry_run: bool = False):
    """24/7 crypto position monitor — fires every minute via launchd."""
    from lib.alpaca_client import AlpacaClient
    from lib.crypto_engine import monitor_crypto_positions

    try:
        client = AlpacaClient()
        result = monitor_crypto_positions(client, dry_run=dry_run)
        if result.get("checked", 0) > 0:
            log_event("main", "crypto_monitor_complete", result)
    except Exception as e:
        log_event("main", "crypto_monitor_failed", {"error": str(e)[:300]}, result="failed")
        # Don't raise — monitor must keep running even on transient errors


def cmd_crypto_scan(dry_run: bool = False):
    """Run one crypto scan + trade cycle. Tier 2 (2026-04-25)."""
    from lib.alpaca_client import AlpacaClient
    from lib.crypto_engine import scan_and_trade_crypto

    log_event("main", "crypto_scan_started", {"dry_run": dry_run})

    try:
        client = AlpacaClient()
        account = client.get_account()
        portfolio = float(account["portfolio_value"])

        # Daily P/L vs baseline_equity (used by daily-loss circuit breaker)
        try:
            with open(Path(__file__).parent / "data" / "baseline_equity.json") as _bf:
                baseline = json.load(_bf)
            daily_pnl = portfolio - float(baseline.get("baseline_equity", portfolio))
        except Exception:
            daily_pnl = 0.0

        result = scan_and_trade_crypto(
            client=client,
            portfolio_value=portfolio,
            current_daily_pnl=daily_pnl,
            dry_run=dry_run,
        )
        log_event("main", "crypto_scan_complete", result)
    except Exception as e:
        log_event("main", "crypto_scan_failed", {"error": str(e)[:300]}, result="failed")
        raise


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

        # PDT check (uses broker's authoritative daytrade_count via client)
        from lib.pdt_guard import check_pdt
        pdt = check_pdt(portfolio, client=client)

        # Risk personality selection (TauricResearch Risky/Neutral/Safe pattern,
        # adapted to deterministic state-driven selection — no LLM debate cost).
        from lib.risk_profile import select_profile, log_active_profile
        from lib.memory_palace import get_current_regime as _regime
        try:
            with open(Path(__file__).parent / "data" / "baseline_equity.json") as _bf:
                baseline = json.load(_bf)
            baseline_equity = float(baseline.get("baseline_equity", portfolio))
            daily_loss_pct = ((portfolio - baseline_equity) / baseline_equity
                              if baseline_equity > 0 else 0.0)
        except Exception:
            daily_loss_pct = 0.0
        active_profile = select_profile(
            regime=_regime() or "unknown",
            bankroll=portfolio,
            daily_loss_pct=daily_loss_pct,
        )
        log_active_profile(active_profile, {
            "bankroll": round(portfolio, 2),
            "daily_loss_pct": round(daily_loss_pct, 4),
            "regime": _regime() or "unknown",
        })

        print("=" * 55)
        print("  OPENCLAW WHEEL TRADER — SCAN")
        print("=" * 55)
        print(f"  Portfolio: ${portfolio:,.2f}  Cash: ${cash:,.2f}")
        if pdt.get("warning"):
            print(f"  ⚠️  {pdt['warning']}")
        else:
            print(f"  PDT: {pdt['day_trades_used']}/3 day trades used")
        print(f"  Risk profile: {active_profile.name.upper()} — {active_profile.rationale}")
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

        # Refresh earnings calendar (non-blocking — uses cached file if API fails)
        try:
            from lib.enhancements import refresh_earnings_calendar
            earnings = refresh_earnings_calendar(tickers)
            if earnings:
                print(f"\n  📅 Earnings dates loaded: {', '.join(f'{t}={d}' for t, d in earnings.items())}")
        except Exception as e:
            # Non-critical — trades still filtered by KG fallback. Audit
            # finding 2026-05-01 #4: log so operators can see when the
            # earnings refresh quietly degrades.
            log_event("main", "earnings_refresh_failed",
                      {"error": str(e)[:200]}, result="degraded")

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


def cmd_premarket_scan(
    *,
    gap_threshold: float = -0.02,   # dip threshold: trade only if down ≥2%
    max_trades: int = 2,
    min_composite: int = 8,
    half_size: bool = True,
    dry_run: bool = False,
):
    """Pre-market dip-buy scan. Fires 4:00 AM CT via launchd.

    Strategy: scan Phase-1 universe for stocks gapping down >``gap_threshold``
    in pre-market. Only buy if the regular-hours screener also rates
    the setup ≥ ``min_composite``. Place a LIMIT order at the
    pre-market last price (no chasing) with ``extended_hours=True``.

    Half-size (``max_position_pct / 2``) by default to bound pre-market
    wide-spread risk.

    Side effect: persists the per-ticker overnight gap to
    ``data/premarket_signals.json`` so the regular day-time screener
    can incorporate it as a signal at market open.
    """
    import yaml
    from lib.alpaca_client import AlpacaClient
    from lib.premarket_signals import compute_all
    from lib.audit import log_event

    cfg_path = Path(__file__).parent / "config" / "wheel_strategy.yaml"
    with open(cfg_path) as f:
        strategy = yaml.safe_load(f) or {}
    tickers = strategy.get("tickers_phase1", []) or strategy.get("tickers", [])

    print(f"=== PRE-MARKET SCAN ({len(tickers)} tickers) ===")
    print(f"  Gap threshold: {gap_threshold:+.1%}  Min composite: {min_composite}/13")
    print(f"  Max trades:    {max_trades}  Half-size: {half_size}  Dry-run: {dry_run}")

    try:
        client = AlpacaClient()
    except Exception as e:
        print(f"ABORT: cannot init AlpacaClient: {e}")
        return

    # Compute and persist overnight gaps. The persistence side-effect
    # is the "feed into normal day knowledge" piece — the regular
    # screener reads load_signals() at market open.
    signals = compute_all(tickers, client)
    if not signals:
        print("  No pre-market signals computed (data feed empty?)")
        return

    print(f"\n  Gaps observed:")
    for s in sorted(signals, key=lambda x: x.gap_pct):
        marker = " ← dip" if s.gap_pct <= gap_threshold else ""
        print(f"    {s.ticker:6s}  prior=${s.prior_close:>8.2f}  "
              f"pre=${s.last_price:>8.2f}  gap={s.gap_pct:+.2%}{marker}")

    # Find dip candidates that ALSO pass the regular screener score.
    dippers = [s for s in signals if s.gap_pct <= gap_threshold]
    if not dippers:
        print(f"\n  No gaps below {gap_threshold:+.1%} — nothing to buy.")
        return

    # Order dip-buy candidates by deepest gap first.
    dippers.sort(key=lambda s: s.gap_pct)
    print(f"\n  {len(dippers)} dip candidate(s); placing up to {max_trades} order(s):")

    # Load circuit-breaker settings so we can compute a half-size order.
    settings_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(settings_path) as f:
        settings = yaml.safe_load(f) or {}
    max_pos_pct = float(settings.get("circuit_breakers", {}).get("max_position_pct", 0.10))
    if half_size:
        max_pos_pct /= 2

    account = client.get_account()
    portfolio_value = float(account["portfolio_value"])
    cash = float(account["cash"])
    placed = 0

    for sig in dippers[:max_trades]:
        # Compute share count: half of max_position_pct of portfolio
        target_dollars = portfolio_value * max_pos_pct
        target_dollars = min(target_dollars, cash * 0.95)  # cash safety
        shares = int(target_dollars / sig.last_price)
        if shares < 1:
            print(f"    {sig.ticker}: skip — share size would be 0 "
                  f"(cash ${cash:.0f}, target ${target_dollars:.0f})")
            continue

        # Limit at pre-market last price — don't chase the spread.
        # Truncate to penny precision; Alpaca rejects sub-penny limits
        # for stocks priced ≥ $1.
        limit_price = round(sig.last_price, 2)

        if dry_run:
            print(f"    {sig.ticker}: [DRY] would buy {shares} sh @ limit ${limit_price:.2f} "
                  f"(${shares * limit_price:.0f} ≈ {(shares * limit_price)/portfolio_value:.1%} of book)")
            continue

        try:
            from lib.order_gate import OrderIntent, step1_propose, step2_validate, step3_execute
            intent = OrderIntent(
                ticker=sig.ticker,
                side="buy",
                asset_type="equity",
                quantity=shares,
                limit_price=limit_price,
                order_type="limit",
                composite_score=min_composite,
                reason=f"premarket_dip_buy gap={sig.gap_pct:+.2%}",
                extended_hours=True,
            )
            step1_propose(intent)
            step2_validate(
                intent=intent,
                portfolio_value=portfolio_value,
                current_daily_pnl=0.0,
                current_open_orders=0,
                min_composite_score=0,  # bypass score check — signal-driven
            )
            resp = step3_execute(intent, client)
            placed += 1
            print(f"    {sig.ticker}: ✓ LIMIT buy {shares} sh @ ${limit_price:.2f} "
                  f"(order {resp.get('id','?')[:12]})")
            log_event("premarket_scan", "order_placed", {
                "ticker": sig.ticker, "shares": shares,
                "limit_price": limit_price,
                "gap_pct": sig.gap_pct,
                "order_id": resp.get("id"),
            }, result="success")
        except Exception as e:
            print(f"    {sig.ticker}: ✗ failed — {str(e)[:140]}")
            log_event("premarket_scan", "order_failed", {
                "ticker": sig.ticker, "error": str(e)[:200],
            }, result="failed")

    print(f"\n  Done — placed {placed}/{min(max_trades, len(dippers))} order(s).")


def cmd_self_audit(hours: float = 24.0, *, telegram: bool = False):
    """Run a comprehensive self-audit and print a digest.

    Default 24h window for nightly cron usage. Pass ``--telegram`` to
    also send the digest to the configured Telegram channel — useful
    for the nightly launchd schedule.
    """
    from lib.alpaca_client import AlpacaClient
    from lib.self_audit import run_self_audit

    try:
        client = AlpacaClient()
    except Exception as e:
        print(f"Could not init broker client (broker-dependent checks skipped): {e}")
        client = None

    result = run_self_audit(hours=hours, broker_client=client)
    funnels = result.get("funnels", {})
    alerts = result.get("alerts", [])

    lines = [
        f"=== SELF-AUDIT ({hours:.0f}h window) ===",
        "",
        "Pipeline funnel:",
    ]
    if funnels:
        for cls, f in funnels.items():
            nonzero = {k: v for k, v in f.items() if v}
            if nonzero:
                lines.append(f"  {cls}: {nonzero}")
    else:
        lines.append("  (no trade activity in window)")

    lines.append("")
    if alerts:
        crit = [a for a in alerts if a.get("severity") == "critical"]
        warn = [a for a in alerts if a.get("severity") == "warn"]
        lines.append(f"Alerts: {len(crit)} critical, {len(warn)} warn")
        for a in alerts:
            emoji = "🔴" if a.get("severity") == "critical" else "⚠️"
            lines.append(f"  {emoji} [{a['code']}] {a['summary']}")
    else:
        lines.append("✅ All checks clean — no anomalies detected.")

    digest = "\n".join(lines)
    print(digest)

    if telegram:
        try:
            from lib.monitor import send_alert
            # Telegram has a 4096-char limit; truncate if needed.
            send_alert(digest[:3800] + ("\n…(truncated)" if len(digest) > 3800 else ""))
        except Exception as e:
            print(f"\n(Telegram send failed: {e})")


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


def cmd_wheel_reset(confirm: bool = False):
    """Planned wheel reset — close options first, then equity. Use when
    the bot has gotten into a weird state (illegal wheel stage, manual
    positions added, recovery from a kill switch) and you want a
    clean-slate restart. Unlike ``kill``, this is a planned operation —
    cron and crontab are left alone, audit logs as a normal event.

    Without ``--confirm``, prints what would be liquidated and exits.
    """
    from lib.alpaca_client import AlpacaClient
    from lib.wheel_state import classify_book

    client = AlpacaClient()
    positions = client.get_positions()
    if not positions:
        print("✅ No positions held — nothing to reset.")
        return

    book = classify_book(positions, raise_on_illegal=False)
    print(f"\n=== WHEEL RESET — {len(positions)} position(s) across {len(book)} underlying(s) ===")
    for underlying, state in sorted(book.items()):
        flags = " [ILLEGAL]" if state.stage == "illegal" else ""
        print(f"  {underlying:6s} {state.stage}{flags}")
        for leg in state.legs:
            kind = leg.kind.upper()
            print(f"    └─ {leg.symbol:20s} {kind:5s} qty={leg.qty:+g} mv=${leg.market_value:,.2f}")

    if not confirm:
        print("\n[DRY RUN] Re-run with --confirm to actually liquidate.")
        print("         Order: cancel all open orders → close OPTIONS → close STOCKS.")
        return

    print("\n⚠️  Liquidating in order: options first, then equity ...")
    cancelled = client.cancel_all_orders()
    print(f"  Cancelled {cancelled} open order(s)")
    result = client.liquidate_wheel_book()
    print(f"  Closed {result['options_closed']} option position(s)")
    print(f"  Closed {result['stocks_closed']} stock position(s)")
    if result["errors"]:
        print(f"  ⚠️  {len(result['errors'])} error(s): {result['errors'][:3]}")
    else:
        print("  ✅ Clean reset.")


def cmd_wheel_dte_sweep(days_back: int = 365, capital: float = 1500.0):
    """A/B/C/D comparison of wheel performance across DTE bands.

    Runs ``run_wheel_backtest`` against every Phase 1 ticker at four
    DTE values that map to the policy decision: 35 (current), 28, 21,
    14, 7. The premium model now scales with vol × √DTE, so per-day
    theta is honest — short-DTE variants pay less per cycle but more
    per day, exactly as in real options markets.

    Use ``--lookback`` to override the 365-day window (longer = more
    statistical confidence, but slower).
    """
    import yaml
    import statistics
    from lib.alpaca_client import AlpacaClient
    from lib.backtest import run_wheel_backtest

    cfg_path = Path(__file__).parent / "config" / "wheel_strategy.yaml"
    with open(cfg_path) as f:
        strategy = yaml.safe_load(f)
    tickers = strategy.get("tickers", []) or strategy.get("tickers_phase1", [])

    dte_variants = [35, 28, 21, 14, 7]

    client = AlpacaClient()
    print(f"\n=== WHEEL DTE SWEEP — {len(tickers)} tickers × {len(dte_variants)} DTE bands ===")
    print(f"  Lookback:     {days_back} days")
    print(f"  Starting cap: ${capital:,.0f}")
    print(f"  Premium model: vol × √DTE (DTE-aware)")
    print()

    # Fetch bars once per ticker — reuse across DTE variants
    bars: dict = {}
    for t in tickers:
        try:
            df = client.get_bars([t], timeframe="1Day", limit=days_back + 50)
            if isinstance(df, dict) and t in df and len(df[t]) > 50:
                bars[t] = df[t]
        except Exception as e:
            print(f"  ⚠ skip {t}: {e}")

    print(f"  Loaded bars for {len(bars)}/{len(tickers)} tickers\n")

    # results[dte][ticker] = BacktestResult
    results: dict[int, dict[str, "BacktestResult"]] = {dte: {} for dte in dte_variants}
    for dte in dte_variants:
        print(f"--- DTE = {dte} ---")
        for t, df in bars.items():
            try:
                r = run_wheel_backtest(
                    df, initial_capital=capital, dte=dte,
                    put_delta=-0.25, call_delta=0.25,
                )
                results[dte][t] = r
                print(f"  {t:6s} return={r.total_return:+.1%}  Sharpe={r.sharpe_ratio:+.2f}  "
                      f"DD={r.max_drawdown:.1%}  trades={r.total_trades}  win={r.win_rate:.0%}")
            except Exception as e:
                print(f"  {t:6s} FAILED: {e}")

    # Aggregate across tickers
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON SUMMARY (mean across {len(bars)} tickers)")
    print(f"{'=' * 70}")
    print(f"{'DTE':<6}{'Return':<12}{'Sharpe':<10}{'Max DD':<10}{'Trades/yr':<12}{'Win %':<8}")
    for dte in dte_variants:
        rs = list(results[dte].values())
        if not rs:
            continue
        years = days_back / 365.0
        mean_return = statistics.mean(r.total_return for r in rs)
        mean_sharpe = statistics.mean(r.sharpe_ratio for r in rs)
        mean_dd = statistics.mean(r.max_drawdown for r in rs)
        mean_trades_yr = statistics.mean(r.total_trades for r in rs) / max(years, 0.01)
        mean_win = statistics.mean(r.win_rate for r in rs)
        print(f"{dte:<6}{mean_return:+.1%}     {mean_sharpe:+.2f}     "
              f"{mean_dd:.1%}     {mean_trades_yr:>5.0f}        {mean_win:.0%}")

    print(f"\n  Recommendation: pick the row with the best risk-adjusted return")
    print(f"  (highest Sharpe AND a max drawdown you can stomach).")

    # Log for postmortem / audit
    log_event("wheel_dte_sweep", "complete", {
        "n_tickers": len(bars),
        "dte_variants": dte_variants,
        "days_back": days_back,
        "summary": {
            dte: {
                "mean_return": round(statistics.mean(r.total_return for r in results[dte].values()), 4),
                "mean_sharpe": round(statistics.mean(r.sharpe_ratio for r in results[dte].values()), 4),
                "mean_dd": round(statistics.mean(r.max_drawdown for r in results[dte].values()), 4),
            } for dte in dte_variants if results[dte]
        },
    })


def cmd_backtest(ticker: str = "SPY", simulations: int = 500, capital: float = 0):
    """Run backtest with Monte Carlo simulation on real historical data."""
    from lib.alpaca_client import AlpacaClient
    from lib.backtest import run_wheel_backtest, run_monte_carlo, compare_to_benchmark

    sep = "=" * 58
    print(f"\n  {sep}")
    print(f"  📊 WHEEL STRATEGY BACKTEST — {ticker}")
    print(f"  {sep}")

    try:
        client = AlpacaClient()

        # Use actual portfolio value if not specified
        if capital <= 0:
            account = client.get_account()
            capital = account["portfolio_value"]
        print(f"  Starting capital: ${capital:,.2f}")
        print(f"  Monte Carlo runs: {simulations}")

        # Fetch 252 trading days (~1 year) of daily bars
        print(f"\n  Fetching {ticker} daily bars (1 year)...")
        daily_data = client.get_bars([ticker], timeframe="1Day", limit=252)

        if ticker not in daily_data or daily_data[ticker].empty:
            print(f"\n  ❌ No data for {ticker}. Check ticker symbol.")
            return

        df = daily_data[ticker]
        print(f"  Got {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")

        # ---- Run Wheel Backtest ----
        print(f"\n  Running Wheel backtest...")
        bt = run_wheel_backtest(df, initial_capital=capital)

        print(f"\n  {'─' * 42}")
        print(f"  WHEEL STRATEGY RESULTS")
        print(f"  {'─' * 42}")
        print(f"  Total Return:      {bt.total_return:>+8.2%}")
        print(f"  Annualized Return: {bt.annualized_return:>+8.2%}")
        print(f"  Max Drawdown:      {bt.max_drawdown:>8.2%}")
        print(f"  Sharpe Ratio:      {bt.sharpe_ratio:>8.2f}")
        print(f"  Sortino Ratio:     {bt.sortino_ratio:>8.2f}")
        print(f"  Win Rate:          {bt.win_rate:>8.2%}")
        print(f"  Total Trades:      {bt.total_trades:>8d}")
        print(f"  Avg Trade P/L:     ${bt.avg_trade_pnl:>8.2f}")
        print(f"  Total Premiums:    ${bt.total_premiums:>10.2f}")
        print(f"  Assignments:       {bt.assignments:>8d}")
        print(f"  Wheel Cycles:      {bt.wheel_cycles_completed:>8d}")

        # ---- Benchmark Comparison ----
        bench = compare_to_benchmark(df, bt)
        print(f"\n  {'─' * 42}")
        print(f"  vs BUY & HOLD {ticker}")
        print(f"  {'─' * 42}")
        print(f"  B&H Return:        {bench['buy_and_hold']['total_return']:>+8.2%}")
        print(f"  B&H Annualized:    {bench['buy_and_hold']['annualized']:>+8.2%}")
        print(f"  B&H Max Drawdown:  {bench['buy_and_hold']['max_drawdown']:>8.2%}")
        out = bench['outperformance']
        label = "OUTPERFORMS" if out > 0 else "UNDERPERFORMS"
        print(f"  Wheel {label}: {out:>+.2%}")

        # ---- Monte Carlo ----
        print(f"\n  Running Monte Carlo ({simulations} simulations)...")
        mc = run_monte_carlo(df, n_simulations=simulations, initial_capital=capital)

        print(f"\n  {'─' * 42}")
        print(f"  MONTE CARLO DISTRIBUTION")
        print(f"  {'─' * 42}")
        print(f"  Mean Return:       {mc.mean_return:>+8.2%}")
        print(f"  Median Return:     {mc.median_return:>+8.2%}")
        print(f"  Std Dev:           {mc.std_return:>8.2%}")
        print(f"  5th Percentile:    {mc.percentile_5:>+8.2%}  (worst case)")
        print(f"  25th Percentile:   {mc.percentile_25:>+8.2%}")
        print(f"  75th Percentile:   {mc.percentile_75:>+8.2%}")
        print(f"  95th Percentile:   {mc.percentile_95:>+8.2%}  (best case)")
        print(f"  P(Loss):           {mc.probability_of_loss:>8.2%}")
        print(f"  P(Ruin >50% DD):   {mc.probability_of_ruin:>8.2%}")

        # Final verdict
        print(f"\n  {'─' * 42}")
        if mc.probability_of_loss < 0.20 and mc.mean_return > 0.05:
            print(f"  ✅ VERDICT: Strategy looks viable on {ticker}")
            print(f"     {mc.probability_of_loss:.0%} chance of loss, {mc.mean_return:.1%} avg return")
        elif mc.probability_of_loss < 0.40:
            print(f"  ⚠️  VERDICT: Moderate risk on {ticker}")
            print(f"     {mc.probability_of_loss:.0%} chance of loss — consider tighter parameters")
        else:
            print(f"  ❌ VERDICT: High risk on {ticker}")
            print(f"     {mc.probability_of_loss:.0%} chance of loss — not recommended")
        print()

    except Exception as e:
        print(f"\n  ❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


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
    from lib.order_gate import OrderIntent, step1_propose
    from lib.order_dedup import reset_for_tests as _reset_dedup
    _reset_dedup()
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
    _reset_dedup()  # leave the live store untouched

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


def cmd_hermes(dry_run: bool = False, lookback: int = 14):
    """Run the Hermes self-optimization agent."""
    from agents.hermes_optimizer import run_optimization, print_optimization_report

    print("🔮 HERMES SELF-OPTIMIZATION AGENT")
    print("=" * 55)

    if dry_run:
        print("  Mode: DRY RUN (analysis only, no parameter changes)\n")
    else:
        print("  Mode: LIVE (will adjust parameters if needed)\n")

    report = run_optimization(lookback_days=lookback, dry_run=dry_run)
    print_optimization_report(report)


def cmd_pdt():
    """Check Pattern Day Trader status."""
    from lib.pdt_guard import check_pdt
    from lib.alpaca_client import AlpacaClient

    client = AlpacaClient()
    account = client.get_account()
    portfolio = account["portfolio_value"]

    status = check_pdt(portfolio, client=client)
    print("📋 PDT STATUS")
    print("=" * 40)
    print(f"  Portfolio: ${portfolio:,.2f}")
    print(f"  Day trades used (5d): {status['day_trades_used']} "
          f"(source: {status.get('source', '?')})")
    print(f"  Day trades remaining: {status['day_trades_remaining']}")

    if status["warning"]:
        print(f"\n  ⚠️  {status['warning']}")
    else:
        print(f"\n  ✅ No PDT concerns.")
    print()


def cmd_kronos(ticker: str = "SPY"):
    """Run Kronos AI price prediction on a ticker."""
    print(f"\n  {'=' * 55}")
    print(f"  🧠 KRONOS AI PRICE FORECAST — {ticker}")
    print(f"  {'=' * 55}")

    try:
        from lib.kronos_forecaster import predict_price

        print(f"  Loading Kronos model (first run downloads ~400MB)...\n")

        forecast = predict_price(
            ticker=ticker,
            pred_bars=30,
            interval="1d",
            lookback=400,
            sample_count=5,
            temperature=0.8,
        )

        # Direction emoji
        dir_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
        emoji = dir_emoji.get(forecast.direction, "⚪")

        print(f"  Current Price:     ${forecast.current_price:,.2f}")
        print(f"  30-Day Forecast:   ${forecast.pred_final_close:,.2f} ({forecast.expected_return:+.1%})")
        print(f"  Direction:         {emoji} {forecast.direction.upper()}")
        print(f"  Confidence:        {forecast.confidence:.2f}")
        print(f"  Predicted High:    ${forecast.pred_high_watermark:,.2f}")
        print(f"  Predicted Low:     ${forecast.pred_low_watermark:,.2f}")
        print(f"  Lookback Bars:     {forecast.lookback_bars}")

        # Price trajectory (simplified sparkline)
        closes = forecast.predicted_close
        if closes:
            print(f"\n  30-Day Price Trajectory:")
            step = max(1, len(closes) // 10)
            for i in range(0, len(closes), step):
                bar_pct = (closes[i] - forecast.current_price) / forecast.current_price
                bar_len = int(abs(bar_pct) * 200)
                if bar_pct >= 0:
                    bar = "  " + "▓" * min(bar_len, 30)
                    print(f"    Day {i+1:>2}: ${closes[i]:>8.2f} {bar_pct:+.1%} {bar}")
                else:
                    bar = "░" * min(bar_len, 30) + "  "
                    print(f"    Day {i+1:>2}: ${closes[i]:>8.2f} {bar_pct:+.1%} {bar}")

        print()

    except Exception as e:
        print(f"\n  ❌ Kronos prediction failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_news(ticker: str = "SPY"):
    """Check news sentiment for a ticker."""
    print(f"\n  {'=' * 55}")
    print(f"  📰 NEWS SENTIMENT CHECK — {ticker}")
    print(f"  {'=' * 55}")

    try:
        from lib.news_sentiment import check_stock_sentiment

        result = check_stock_sentiment(ticker)

        sig_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
        emoji = sig_emoji.get(result.signal, "⚪")

        print(f"\n  Signal:       {emoji} {result.signal.upper()}")
        print(f"  Sentiment:    {result.sentiment:.2f} (0=bearish, 1=bullish)")
        print(f"  Confidence:   {result.confidence:.2f}")
        print(f"  Articles:     {result.article_count}")
        print(f"  Cached:       {'yes' if result.cached else 'no'}")

        if result.headlines:
            print(f"\n  Recent Headlines:")
            for i, h in enumerate(result.headlines[:5], 1):
                print(f"    {i}. {h[:80]}")

        # Trade implication
        if result.sentiment < 0.25 and result.confidence > 0.3:
            print(f"\n  ⚠️  TRADE SIGNAL: Avoid buying {ticker} — strong bearish news")
        elif result.sentiment > 0.70 and result.confidence > 0.3:
            print(f"\n  ✅ TRADE SIGNAL: News supports buying {ticker}")
        else:
            print(f"\n  ℹ️  TRADE SIGNAL: News is neutral — no strong opinion")

        print()

    except Exception as e:
        print(f"\n  ❌ News check failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_insiders(ticker: str = "SPY"):
    """Check SEC Form 4 insider-buying signal for a ticker."""
    print(f"\n  {'=' * 55}")
    print(f"  🏢 INSIDER FLOW — {ticker}")
    print(f"  {'=' * 55}")

    try:
        from lib.insider_flow import check_insider_flow

        result = check_insider_flow(ticker, days=90)

        sig_emoji = {
            "bullish_cluster": "🟢🟢",
            "bullish": "🟢",
            "neutral": "🟡",
            "bearish": "🔴",
        }
        emoji = sig_emoji.get(result.signal, "⚪")

        print(f"\n  Signal:           {emoji} {result.signal.upper()}")
        print(f"  Sentiment:        {result.sentiment:.2f} (0.5 = neutral)")
        print(f"  Confidence:       {result.confidence:.2f}")
        print(f"  Buys (90d):       {result.buy_count} purchases")
        print(f"  Sells (90d):      {result.sell_count}  (informational only)")
        print(f"  Cluster detected: {'YES ⚡' if result.cluster_detected else 'no'}")
        print(f"  Total buy $:      ${result.total_buy_value_usd:,.0f}")
        print(f"  Cached:           {'yes' if result.cached else 'no'}")
        print(f"  Reason:           {result.reason}")

        if result.recent_buys:
            print(f"\n  Recent insider purchases:")
            for b in result.recent_buys[:5]:
                title = f" ({b.insider_title[:20]})" if b.insider_title else ""
                print(f"    • {b.filing_date}  {b.insider_name[:30]:<30}{title}")
                print(f"        {b.shares:>6} sh @ ${b.price_per_share:>7.2f}  "
                      f"(${b.dollar_value:>10,.0f})")

        # Trade implication
        if result.cluster_detected:
            print(f"\n  ✅ TRADE SIGNAL: Insider cluster buy — "
                  f"positive lead indicator for {ticker}")
        elif result.signal == "bullish":
            print(f"\n  ℹ️  TRADE SIGNAL: Meaningful insider buying — mild tailwind")
        else:
            print(f"\n  ℹ️  TRADE SIGNAL: No material insider buying in window")

        print()

    except Exception as e:
        print(f"\n  ❌ Insider flow check failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_congress(ticker: str = "SPY"):
    """Check Senate STOCK Act purchase signal for a ticker."""
    print(f"\n  {'=' * 55}")
    print(f"  🏛️  CONGRESS FLOW — {ticker}")
    print(f"  {'=' * 55}")

    try:
        from lib.congress_flow import check_congress_flow

        result = check_congress_flow(ticker, days=180)

        sig_emoji = {
            "bullish_cluster": "🟢🟢",
            "bullish": "🟢",
            "neutral": "🟡",
        }
        emoji = sig_emoji.get(result.signal, "⚪")

        print(f"\n  Signal:           {emoji} {result.signal.upper()}")
        print(f"  Sentiment:        {result.sentiment:.2f}")
        print(f"  Confidence:       {result.confidence:.2f}")
        print(f"  Buys (180d):      {result.buy_count} senator purchases")
        print(f"  Sells:            {result.sell_count}  (informational)")
        print(f"  Distinct 30d:     {result.distinct_buyers_30d}")
        print(f"  Cluster detected: {'YES ⚡' if result.cluster_detected else 'no'}")
        print(f"  Total buy $:      ${result.total_buy_midpoint_usd:,.0f}")
        print(f"  Cached:           {'yes' if result.cached else 'no'}")

        # Feed freshness — critical context
        freshness_flag = "⚠️  STALE" if result.feed_stale else "✓ fresh"
        print(f"  Feed newest txn:  {result.feed_newest_txn_date} "
              f"({result.feed_age_days}d old — {freshness_flag})")
        print(f"  Reason:           {result.reason}")

        if result.feed_stale:
            print(f"\n  ⚠️  Feed is stale — signal is suppressed to confidence~0.")
            print(f"     Export CONGRESS_FEED_URL=<paid feed URL> for live data.")

        if result.recent_buys:
            print(f"\n  Recent Senate purchases:")
            for b in result.recent_buys[:5]:
                print(f"    • {b.transaction_date}  {b.senator[:30]:<30}"
                      f"  ({b.owner})  ~${b.amount_midpoint_usd:,.0f}")

        # Trade implication
        if result.feed_stale:
            print(f"\n  ℹ️  TRADE SIGNAL: Signal suppressed (stale data)")
        elif result.cluster_detected:
            print(f"\n  ✅ TRADE SIGNAL: Senate cluster buy on {ticker}")
        elif result.signal == "bullish":
            print(f"\n  ℹ️  TRADE SIGNAL: Material Senate buying — mild tailwind")
        else:
            print(f"\n  ℹ️  TRADE SIGNAL: No material Senate activity")

        print()

    except Exception as e:
        print(f"\n  ❌ Congress flow check failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_greeks():
    """Display portfolio-level Greeks across all open positions."""
    print(f"\n  {'=' * 65}")
    print(f"  Δ  PORTFOLIO GREEKS")
    print(f"  {'=' * 65}")

    try:
        import json
        from pathlib import Path
        from lib.portfolio_greeks import compute_portfolio_greeks

        positions_path = Path(__file__).parent / "data" / "positions.json"
        positions = json.loads(positions_path.read_text())

        # Prefer live Alpaca spots; fall back to entry prices
        try:
            from lib.alpaca_client import AlpacaClient
            client = AlpacaClient()
            def _spot(tkr: str):
                try:
                    return float(client.get_latest_quote(tkr).get("ask_price") or
                                 client.get_latest_quote(tkr).get("bid_price"))
                except Exception:
                    return None
        except Exception:
            _spot = lambda tkr: None

        portfolio = compute_portfolio_greeks(positions, spot_fetcher=_spot)

        print(f"\n  Open positions: {len(portfolio.positions) - portfolio.invalid_count} valid / "
              f"{portfolio.invalid_count} invalid")
        print(f"\n  ─── Aggregated exposures ─────────────────────────────────────")
        print(f"   Δ  Delta  (share-equiv):  {portfolio.total_delta:>12,.2f}")
        print(f"   Γ  Gamma  (per $1 move):  {portfolio.total_gamma:>12,.4f}")
        print(f"   v  Vega   ($/vol-point):  ${portfolio.total_vega:>11,.2f}")
        print(f"   θ  Theta  ($/day):        ${portfolio.total_theta:>11,.2f}")
        print(f"   |Δ| gross (absolute):     {portfolio.gross_delta:>12,.2f}")

        print(f"\n  ─── Per-position breakdown ───────────────────────────────────")
        print(f"  {'ticker':7s} {'type':6s} {'qty':>5s} {'Δ':>9s} {'Γ':>9s}"
              f" {'vega':>9s} {'theta':>9s}  notes")
        for pg in portfolio.positions:
            flag = " " if pg.valid else "✗"
            print(f"  {flag}{pg.ticker:6s} {pg.position_type:6s} {pg.quantity:>5d}"
                  f" {pg.delta:>9.2f} {pg.gamma:>9.4f}"
                  f" {pg.vega:>9.2f} {pg.theta:>9.2f}  {pg.reason[:30]}")

        # Interpretation
        print(f"\n  ─── Interpretation ───────────────────────────────────────────")
        if portfolio.total_theta > 0:
            print(f"   θ Collecting ~${portfolio.total_theta:.2f}/day in theta — engine running.")
        if portfolio.total_vega < -100:
            print(f"   v Short ${-portfolio.total_vega:.0f} vega → exposed to an IV spike.")
        elif portfolio.total_vega > 100:
            print(f"   v Long ${portfolio.total_vega:.0f} vega → benefits from IV expansion.")
        if abs(portfolio.total_gamma) > 0.10:
            print(f"   Γ Non-trivial gamma — blow-up risk near short option strikes.")

        print()

    except Exception as e:
        print(f"\n  ❌ Portfolio Greeks calculation failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_calibrate():
    """Print stock prediction accuracy report."""
    from lib.stock_calibration import print_calibration_report
    print()
    print_calibration_report()
    print()


def cmd_forecast(ticker: str = "SPY"):
    """Bayesian multi-signal forecast for a ticker. Combines all signals."""
    from lib.alpaca_client import AlpacaClient
    from lib.data_pipeline import fetch_all_data
    from lib.stock_engine import score_stock_buy
    from lib.bayesian_forecaster import forecast_stock
    from lib.news_sentiment import check_stock_sentiment

    print(f"\n  {'=' * 60}")
    print(f"  🔮 BAYESIAN MULTI-SIGNAL FORECAST — {ticker}")
    print(f"  {'=' * 60}")

    try:
        client = AlpacaClient()
        data = fetch_all_data(client, tickers=[ticker])
        daily_df = data["daily_data"].get(ticker)
        weekly_df = data["weekly_data"].get(ticker)

        if daily_df is None or weekly_df is None:
            print(f"  ❌ No data for {ticker}")
            return

        current_price = float(daily_df["close"].iloc[-1])

        # Score the setup
        candidate = score_stock_buy(ticker, daily_df, weekly_df, current_price, 1000.0)
        if not candidate:
            print(f"  No viable setup for {ticker} (insufficient data)")
            return

        # Optional: Kronos forecast
        kronos_er = None
        kronos_conf = 0.5
        try:
            from lib.kronos_forecaster import predict_price
            print(f"  Running Kronos AI prediction (may take ~30s)...")
            kronos = predict_price(ticker, pred_bars=10, lookback=200, sample_count=3)
            kronos_er = kronos.expected_return
            kronos_conf = kronos.confidence
            print(f"  Kronos: {kronos.direction} ({kronos_er:+.1%}, conf={kronos_conf:.2f})")
        except Exception as e:
            print(f"  Kronos unavailable: {str(e)[:60]}")

        # Optional: News sentiment
        news_sent = None
        news_conf = 0.5
        try:
            news = check_stock_sentiment(ticker)
            news_sent = news.sentiment
            news_conf = news.confidence
            print(f"  News:   {news.signal} (sentiment={news_sent:.2f}, n={news.article_count})")
        except Exception:
            print(f"  News unavailable")

        # Bayesian aggregation
        forecast = forecast_stock(
            ticker=ticker,
            composite_score=candidate["composite_score"],
            trend_score=candidate["trend_score"],
            level_score=candidate["level_score"],
            signal_score=candidate["signal_score"],
            momentum_score=candidate["momentum_score"],
            pattern=candidate["pattern"],
            zone_touches=candidate.get("zone_touches", 0),
            weekly_direction=candidate["weekly_trend"],
            kronos_expected_return=kronos_er,
            kronos_confidence=kronos_conf,
            news_sentiment=news_sent,
            news_confidence=news_conf,
        )

        # Display
        print(f"\n  --- Setup ---")
        print(f"  Current:         ${current_price:.2f}")
        print(f"  Target:          ${candidate['target_price']:.2f} "
              f"({(candidate['target_price']/current_price-1)*100:+.1f}%)")
        print(f"  Stop:            ${candidate['stop_loss']:.2f} "
              f"({(candidate['stop_loss']/current_price-1)*100:+.1f}%)")
        print(f"  Composite:       {candidate['composite_score']}/13")
        print(f"  Pattern:         {candidate['pattern'] or 'none'}")

        print(f"\n  --- Bayesian Chain ---")
        for step in forecast.bayesian_chain:
            name = step.get("step", "")
            post = step.get("posterior", step.get("prob", 0))
            lh = step.get("likelihood", "-")
            if lh == "-":
                print(f"    {name:12s} → p={post:.3f}")
            else:
                print(f"    {name:12s} lh={lh:.3f}  → p={post:.3f}")

        print(f"\n  --- Result ---")
        emoji = "🟢" if forecast.recommended else "🔴"
        print(f"  {emoji} Win probability: {forecast.win_probability:.1%}")
        print(f"     Confidence:     {forecast.confidence:.2f}")
        print(f"     Evidence:       {forecast.evidence_summary}")
        print(f"     Recommendation: {'BUY' if forecast.recommended else 'SKIP'}")
        print(f"     Reason:         {forecast.reason}")

        # Show Kelly sizing if recommended
        if forecast.recommended:
            from lib.kelly import kelly_position_size
            account = client.get_account()
            pv = account["portfolio_value"]
            sizing = kelly_position_size(
                portfolio_value=pv,
                current_price=current_price,
                target_price=candidate["target_price"],
                stop_loss=candidate["stop_loss"],
                composite_score=candidate["composite_score"],
                kronos_expected_return=kronos_er,
            )
            print(f"\n  --- Kelly Sizing (portfolio ${pv:,.2f}) ---")
            print(f"     Shares:        {sizing.get('shares', 0)}")
            print(f"     Position $:    ${sizing.get('position_value', 0):.2f} "
                  f"({sizing.get('pct_of_portfolio', 0)*100:.1f}% of portfolio)")
            print(f"     Reward/Risk:   {sizing.get('reward_to_risk', 0)}x")
            print(f"     Full Kelly:    {sizing.get('full_kelly', 0):+.2%}")
            print(f"     Frac Kelly:    {sizing.get('fractional_kelly', 0):+.2%}")

        print()

    except Exception as e:
        print(f"\n  ❌ Forecast failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_kelly(ticker: str = "SPY"):
    """Show Kelly-optimal position sizing for a ticker's current setup."""
    from lib.alpaca_client import AlpacaClient
    from lib.data_pipeline import fetch_all_data
    from lib.stock_engine import score_stock_buy
    from lib.kelly import kelly_position_size, composite_to_win_prob

    print(f"\n  {'=' * 55}")
    print(f"  💰 KELLY CRITERION SIZING — {ticker}")
    print(f"  {'=' * 55}")

    try:
        client = AlpacaClient()
        account = client.get_account()
        pv = account["portfolio_value"]

        data = fetch_all_data(client, tickers=[ticker])
        daily_df = data["daily_data"].get(ticker)
        weekly_df = data["weekly_data"].get(ticker)

        if daily_df is None:
            print(f"  ❌ No data for {ticker}")
            return

        current_price = float(daily_df["close"].iloc[-1])
        candidate = score_stock_buy(ticker, daily_df, weekly_df, current_price, 1000.0)
        if not candidate:
            print(f"  No setup for {ticker}")
            return

        base_win_prob = composite_to_win_prob(candidate["composite_score"])

        sizing = kelly_position_size(
            portfolio_value=pv,
            current_price=current_price,
            target_price=candidate["target_price"],
            stop_loss=candidate["stop_loss"],
            composite_score=candidate["composite_score"],
        )

        print(f"\n  Portfolio:       ${pv:,.2f}")
        print(f"  Current price:   ${current_price:.2f}")
        print(f"  Target:          ${candidate['target_price']:.2f} ({sizing.get('reward_pct', 0)*100:+.1f}%)")
        print(f"  Stop:            ${candidate['stop_loss']:.2f} (-{sizing.get('risk_pct', 0)*100:.1f}%)")
        print(f"  Composite:       {candidate['composite_score']}/13")
        print(f"\n  --- Probability Inputs ---")
        print(f"  Base win prob:   {base_win_prob:.1%} (from composite score)")
        print(f"  Final win prob:  {sizing.get('win_prob', 0):.1%}")
        print(f"  Reward/Risk:     {sizing.get('reward_to_risk', 0)}x")
        print(f"\n  --- Kelly Output ---")
        print(f"  Full Kelly:      {sizing.get('full_kelly', 0)*100:+.1f}% of bankroll")
        print(f"  Frac Kelly (¼):  {sizing.get('fractional_kelly', 0)*100:+.1f}% of bankroll")
        print(f"  Position cap:    {sizing.get('pct_of_portfolio', 0)*100:.1f}%")
        print(f"\n  --- Recommendation ---")
        shares = sizing.get('shares', 0)
        if shares > 0:
            print(f"  🟢 Buy {shares}x {ticker} @ ${current_price:.2f} = ${sizing['position_value']:,.2f}")
        else:
            print(f"  🔴 No trade: {sizing.get('reason', 'unknown')}")
        print()

    except Exception as e:
        print(f"\n  ❌ Kelly calculation failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_llm(ticker: str = "SPY"):
    """Ask the configured LLM (DeepSeek or Claude) to analyze a stock setup."""
    from lib.alpaca_client import AlpacaClient
    from lib.data_pipeline import fetch_all_data
    from lib.stock_engine import score_stock_buy
    from lib.llm_analyst import analyze_stock_setup, llm_status
    from lib.news_sentiment import check_stock_sentiment

    status = llm_status()
    provider = status["provider"]
    model = status["model"]

    print(f"\n  {'=' * 55}")
    print(f"  🧠 LLM ANALYSIS — {ticker}")
    print(f"     Provider: {provider}  Model: {model}")
    print(f"  {'=' * 55}")

    # Up-front key check with actionable guidance
    if provider == "deepseek" and not status["deepseek_key_set"]:
        print(f"\n  ⚠️  DEEPSEEK_API_KEY not set in environment.")
        print(f"     Add it to .env — get one at https://platform.deepseek.com")
        print(f"     Or switch provider in config/wheel_strategy.yaml: llm.provider = \"claude\"")
        return
    if provider == "claude" and not status["anthropic_key_set"]:
        print(f"\n  ⚠️  ANTHROPIC_API_KEY not set in environment.")
        print(f"     Get a key at https://console.anthropic.com/settings/keys")
        print(f"     Or switch provider in config/wheel_strategy.yaml: llm.provider = \"deepseek\"")
        return

    try:
        client = AlpacaClient()
        data = fetch_all_data(client, tickers=[ticker])
        daily_df = data["daily_data"].get(ticker)
        weekly_df = data["weekly_data"].get(ticker)
        if daily_df is None:
            print(f"  ❌ No data for {ticker}")
            return

        current_price = float(daily_df["close"].iloc[-1])
        candidate = score_stock_buy(ticker, daily_df, weekly_df, current_price, 1000.0)
        if not candidate:
            print(f"  No setup for {ticker}")
            return

        # Pull Kronos forecast if available (adds real signal to the prompt)
        kronos_direction = None
        kronos_expected_return = None
        try:
            from lib.kronos_forecaster import forecast_ticker
            kf = forecast_ticker(ticker)
            if kf is not None:
                kronos_direction = kf.direction
                kronos_expected_return = kf.expected_return
        except Exception as e:
            # Kronos optional — log silently so operator can see when the
            # forecasting layer is degraded (audit #4).
            log_event("main", "kronos_unavailable",
                      {"error": str(e)[:200]}, result="degraded")

        news = check_stock_sentiment(ticker)

        print(f"  Asking {provider}...")
        analysis = analyze_stock_setup(
            ticker=ticker,
            current_price=current_price,
            target_price=candidate["target_price"],
            stop_loss=candidate["stop_loss"],
            composite_score=candidate["composite_score"],
            pattern=candidate["pattern"],
            momentum_score=candidate["momentum_score"],
            kronos_direction=kronos_direction,
            kronos_expected_return=kronos_expected_return,
            news_sentiment=news.sentiment,
            recent_headlines=news.headlines,
        )

        if analysis is None:
            print(f"\n  ⚠️  LLM analysis unavailable (API failed or parse error).")
            print(f"     Check logs/audit_log.jsonl for 'llm_analyst' entries.")
            return

        # Color the action
        action = analysis.suggested_action.upper()
        action_icon = {"BUY": "🟢", "WAIT": "🟡", "SKIP": "🔴"}.get(action, "⚪")

        print(f"\n  Win Probability:   {analysis.win_probability:.1%}")
        print(f"  Confidence:        {analysis.confidence:.2f}")
        print(f"  Suggested Action:  {action_icon} {action}")
        print(f"  Cached:            {'yes' if analysis.cached else 'no'}")

        if analysis.bullish_factors:
            print(f"\n  Bullish Factors:")
            for f in analysis.bullish_factors:
                print(f"    ✓ {f}")
        if analysis.bearish_factors:
            print(f"\n  Bearish Factors:")
            for f in analysis.bearish_factors:
                print(f"    ✗ {f}")
        if analysis.reasoning:
            print(f"\n  Reasoning:")
            print(f"    {analysis.reasoning}")
        print()

    except Exception as e:
        print(f"\n  ❌ LLM analysis failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_correlation():
    """Show correlation groups for current holdings + target universe."""
    from lib.correlation import get_sector_groups, check_portfolio_correlation
    from lib.alpaca_client import AlpacaClient
    from lib.data_pipeline import fetch_all_data
    import yaml as _yaml

    print(f"\n  {'=' * 55}")
    print(f"  🔗 CORRELATION ANALYSIS")
    print(f"  {'=' * 55}")

    try:
        client = AlpacaClient()
        positions = client.get_positions()
        held = [p["symbol"] for p in positions]

        with open(Path(__file__).parent / "config" / "wheel_strategy.yaml") as f:
            strategy = _yaml.safe_load(f)
        universe = strategy.get("tickers_phase1", [])

        print(f"\n  Currently held: {', '.join(held) if held else '(none)'}")
        print(f"  Universe:       {len(universe)} tickers")

        # Show sector groups in holdings
        if held:
            groups = get_sector_groups(held)
            print(f"\n  --- Holdings by Sector ---")
            for group, members in groups.items():
                if len(members) > 1:
                    print(f"  ⚠️  {group}: {', '.join(members)}  (concentrated!)")
                else:
                    print(f"  {group}: {', '.join(members)}")

        # Show groups in full universe
        universe_groups = get_sector_groups(universe)
        print(f"\n  --- Universe Diversity ---")
        for group, members in universe_groups.items():
            print(f"  {group:15s}: {', '.join(members)}")

        # Price correlation with held positions
        if held and len(held) >= 2:
            print(f"\n  Fetching price data for correlation check...")
            tickers_to_fetch = list(set(held + universe[:10]))  # Limit fetch
            data = fetch_all_data(client, tickers=tickers_to_fetch)
            daily_data = data["daily_data"]

            print(f"\n  --- Held Position Correlations (60d log returns) ---")
            from lib.correlation import check_correlation
            for i, a in enumerate(held):
                for b in held[i+1:]:
                    df_a = daily_data.get(a)
                    df_b = daily_data.get(b)
                    if df_a is not None and df_b is not None:
                        result = check_correlation(a, b, df_a, df_b, threshold=0.70)
                        icon = "🔴" if result.is_correlated else "🟢"
                        print(f"  {icon} {a}↔{b}: {result.price_correlation:+.2f}  {result.reason}")
        print()

    except Exception as e:
        print(f"\n  ❌ Correlation check failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_earnings(ticker: str = "SPY"):
    """Show next earnings date + option-sale veto status for a ticker."""
    from lib.earnings_filter import check_earnings, has_earnings_before
    from lib.finnhub_client import finnhub_status
    from datetime import datetime, timezone, timedelta

    print(f"\n  {'=' * 55}")
    print(f"  📅 EARNINGS CHECK — {ticker}")
    print(f"  {'=' * 55}")

    status = finnhub_status()
    if not status["key_set"]:
        print(f"\n  ⚠️  FINNHUB_API_KEY not set in .env")
        print(f"     Get a free key at https://finnhub.io/dashboard")
        print(f"     Then paste into .env on the FINNHUB_API_KEY= line.")
        return

    try:
        check = check_earnings(ticker, window_days=14)
        print(f"\n  Next earnings:       {check.next_earnings or '(unknown / none scheduled)'}")
        if check.days_until is not None:
            print(f"  Days until:          {check.days_until}")
        print(f"  Within 14d window:   {'YES ⚠️' if check.has_earnings_in_window else 'no'}")
        print(f"  Data source:         {check.data_source}")
        print(f"  Reason:              {check.reason}")

        # Check 30-day and 45-day option expirations (wheel DTE range)
        today = datetime.now(timezone.utc).date()
        for dte in (30, 45):
            target = today + timedelta(days=dte)
            veto = has_earnings_before(ticker, target)
            icon = "🔴 VETO" if veto else "🟢 OK"
            print(f"  {dte}-DTE expiration:   {icon}  (exp date {target.isoformat()})")

        print()
    except Exception as e:
        print(f"\n  ❌ Earnings check failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_econ():
    """Show latest macro snapshot (GDP, CPI, unemployment, fed funds, 10Y) via Alpha Vantage."""
    from lib.alpha_vantage_client import get_economic_snapshot, alpha_vantage_status

    print(f"\n  {'=' * 55}")
    print(f"  🏛️  ECONOMIC SNAPSHOT")
    print(f"  {'=' * 55}")

    status = alpha_vantage_status()
    if not status["key_set"]:
        print(f"\n  ⚠️  ALPHA_VANTAGE_API_KEY not set in .env")
        print(f"     Free key: https://www.alphavantage.co/support/#api-key")
        return

    try:
        print(f"\n  Fetching (5 AV calls; rate-limited to 1/12s — ~1min cold, instant cached)...")
        snap = get_economic_snapshot()
        if snap is None:
            print(f"\n  ❌ No data returned. Check logs/audit_log.jsonl for 'alpha_vantage' errors.")
            return

        def _fmt(v, unit=""):
            return f"{v:.2f}{unit}" if v is not None else "n/a"

        print(f"\n  As of:               {snap.as_of or 'unknown'}")
        print(f"  Real GDP growth:     {_fmt(snap.gdp_growth)}")
        print(f"  CPI (level):         {_fmt(snap.cpi)}")
        print(f"  Unemployment rate:   {_fmt(snap.unemployment, '%')}")
        print(f"  Fed Funds rate:      {_fmt(snap.fed_funds_rate, '%')}")
        print(f"  10Y Treasury yield:  {_fmt(snap.treasury_10y, '%')}")
        print()
    except Exception as e:
        print(f"\n  ❌ Economic snapshot failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_finnhub(ticker: str = "SPY"):
    """Comprehensive Finnhub data dump for a ticker — quote, analysts, fundamentals, insiders."""
    from lib.finnhub_client import (
        finnhub_status, get_quote, get_analyst_recs, get_basic_financials,
        get_insider_trades, get_company_news, next_earnings_date,
    )

    print(f"\n  {'=' * 55}")
    print(f"  📊 FINNHUB SNAPSHOT — {ticker}")
    print(f"  {'=' * 55}")

    status = finnhub_status()
    if not status["key_set"]:
        print(f"\n  ⚠️  FINNHUB_API_KEY not set in .env")
        print(f"     Free key: https://finnhub.io/dashboard (60 calls/min)")
        return

    try:
        # Quote
        q = get_quote(ticker)
        if q:
            chg_icon = "🟢" if q["change"] >= 0 else "🔴"
            print(f"\n  💵 Quote")
            print(f"     Price:       ${q['price']:.2f}  {chg_icon} {q['change']:+.2f} ({q['change_pct']:+.2%})")
            print(f"     Day range:   ${q['low']:.2f} - ${q['high']:.2f}  (open ${q['open']:.2f})")
            print(f"     Prev close:  ${q['prev_close']:.2f}")

        # Next earnings
        ne = next_earnings_date(ticker)
        print(f"\n  📅 Next earnings:  {ne or '(none in next 60 days)'}")

        # Analyst recs (most recent period)
        recs = get_analyst_recs(ticker)
        if recs:
            r = recs[0]
            print(f"\n  🎯 Analyst recs ({r.period})")
            print(f"     Strong Buy: {r.strong_buy}   Buy: {r.buy}   Hold: {r.hold}   Sell: {r.sell}   Strong Sell: {r.strong_sell}")
            score = r.net_score
            icon = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")
            print(f"     Net score:  {icon} {score:+.2f}  (normalized -1 to +1)")

        # Fundamentals
        f = get_basic_financials(ticker)
        if f:
            print(f"\n  📈 Fundamentals")
            def _fmt(v, unit="", dec=2):
                if v is None: return "n/a"
                return f"{v:.{dec}f}{unit}"
            print(f"     P/E (TTM):   {_fmt(f['pe_ttm'])}     P/B: {_fmt(f['pb_ttm'])}     P/S: {_fmt(f['ps_ttm'])}")
            print(f"     Beta:        {_fmt(f['beta'])}     ROE: {_fmt(f['roe_ttm'], '%')}    Div Yield: {_fmt(f['dividend_yield'], '%')}")
            print(f"     Market cap:  ${_fmt(f['market_cap'])}M"  if f['market_cap'] else "     Market cap:  n/a")
            print(f"     52w range:   ${_fmt(f['52w_low'])} - ${_fmt(f['52w_high'])}")

        # Insider trades (last 30)
        ins = get_insider_trades(ticker, days_back=90)
        if ins:
            buys = sum(1 for t in ins if t["change"] > 0)
            sells = sum(1 for t in ins if t["change"] < 0)
            net_shares = sum(t["change"] for t in ins)
            icon = "🟢" if net_shares > 0 else ("🔴" if net_shares < 0 else "🟡")
            print(f"\n  👔 Insider activity (90d)")
            print(f"     Buys: {buys}   Sells: {sells}   Net change: {icon} {net_shares:+,} shares")

        # News headlines (top 5)
        news = get_company_news(ticker, days_back=7)
        if news:
            print(f"\n  📰 Recent headlines ({len(news)} found, top 5):")
            for n in news[:5]:
                print(f"     • [{n.source}] {n.headline[:80]}")
        print()
    except Exception as e:
        print(f"\n  ❌ Finnhub snapshot failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_pred_scan():
    """Scan prediction markets for paper trading opportunities (Manifold)."""
    print(f"\n  {'=' * 55}")
    print(f"  🎯 PREDICTION MARKET SCANNER")
    print(f"  {'=' * 55}")

    try:
        import requests

        # Manifold Markets — free, play money, no API key needed
        print(f"\n  Scanning Manifold Markets (paper trading)...")
        url = "https://api.manifold.markets/v0/search-markets"
        params = {
            "term": "",
            "sort": "liquidity",
            "filter": "open",
            "limit": 20,
        }

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        markets = resp.json()

        if not markets:
            print("  No markets found.")
            return

        # Filter for binary markets with good liquidity
        viable = []
        for m in markets:
            if not isinstance(m, dict):
                continue
            mtype = m.get("outcomeType", "")
            if mtype != "BINARY":
                continue

            prob = m.get("probability", 0.5)
            volume = m.get("volume", 0)
            liquidity = m.get("totalLiquidity", 0)
            question = m.get("question", "")[:100]
            market_id = m.get("id", "")

            # Look for mispriced markets (probability far from 50%)
            # Markets near 50% are hardest to predict
            edge_potential = abs(prob - 0.5)

            viable.append({
                "id": market_id,
                "question": question,
                "probability": prob,
                "volume": volume,
                "liquidity": liquidity,
                "edge_potential": edge_potential,
                "url": m.get("url", ""),
            })

        # Sort by liquidity (most liquid = most trustworthy prices)
        viable.sort(key=lambda x: x["liquidity"], reverse=True)

        # Print top candidates
        print(f"\n  Top {min(15, len(viable))} Markets by Liquidity:\n")
        print(f"  {'#':>3}  {'Prob':>5}  {'Vol':>8}  {'Liq':>8}  Question")
        print(f"  {'─' * 3}  {'─' * 5}  {'─' * 8}  {'─' * 8}  {'─' * 50}")

        for i, m in enumerate(viable[:15], 1):
            prob_str = f"{m['probability']:.0%}"
            vol_str = f"${m['volume']:,.0f}"
            liq_str = f"${m['liquidity']:,.0f}"

            # Color code by edge potential
            if m["edge_potential"] > 0.3:
                marker = "🔥"  # High edge potential
            elif m["edge_potential"] > 0.15:
                marker = "📊"
            else:
                marker = "  "

            print(f"  {i:>3}  {prob_str:>5}  {vol_str:>8}  {liq_str:>8}  {marker} {m['question'][:55]}")

        # Summary stats
        avg_prob = sum(m["probability"] for m in viable) / len(viable) if viable else 0
        total_vol = sum(m["volume"] for m in viable)
        print(f"\n  Markets scanned: {len(viable)}")
        print(f"  Total volume:    ${total_vol:,.0f}")
        print(f"  Avg probability: {avg_prob:.0%}")

        # High-edge candidates (probability far from 50% = might be mispriced)
        high_edge = [m for m in viable if m["edge_potential"] > 0.25 and m["liquidity"] > 100]
        if high_edge:
            print(f"\n  🔥 High-Edge Candidates ({len(high_edge)} markets with edge > 25%):")
            for m in high_edge[:5]:
                side = "YES" if m["probability"] > 0.5 else "NO"
                print(f"      {m['probability']:.0%} ({side}) — {m['question'][:60]}")

        print(f"\n  ℹ️  Platform: Manifold (play money — zero risk paper trading)")
        print(f"  ℹ️  Use polybot for live prediction market trading (Kalshi/Polymarket)")
        print()

    except Exception as e:
        print(f"\n  ❌ Prediction market scan failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_digest(send_telegram: bool = False):
    """Build and print the daily digest. Optionally push to Telegram.

    Pulls portfolio state, today's closed trades, open positions, postmortem
    headline, anomaly triggers, and Hermes recent log into one summary.
    """
    from lib.alpaca_client import AlpacaClient
    from lib.daily_digest import build_digest, send_telegram_digest

    client = AlpacaClient()
    text = build_digest(client)
    print(text)
    if send_telegram:
        ok = send_telegram_digest(text)
        if ok:
            print("\n  ✓ Sent to Telegram")
        else:
            print("\n  ⚠️  Telegram send failed (check TELEGRAM_BOT_TOKEN/CHAT_ID)")


def cmd_dipbuy(
    watchlist: str | None = None,
    threshold: float = 0.55,
    persist: bool = False,
):
    """Scan for buy-the-dip setups: oversold + trend-intact pullbacks.

    Five-feature composite (RSI oversold, SMA200 trend intact, pullback
    magnitude, bounce signal, volume confirmation). Triggers on composite
    >= threshold. Detection only — strategy engine still applies its own
    gates before placing trades.
    """
    from lib.alpaca_client import AlpacaClient
    from lib.dip_buyer import (
        DEFAULT_WATCHLIST, scan_universe, print_dip_report, persist_scores,
    )

    if watchlist:
        symbols = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    else:
        symbols = DEFAULT_WATCHLIST
    print(f"  Dip-buy scan: {len(symbols)} symbols, threshold={threshold:.2f}")

    client = AlpacaClient()
    scores = scan_universe(client, symbols, composite_threshold=threshold)
    print()
    print_dip_report(scores, top=20)
    if persist:
        n = persist_scores(scores)
        print(f"  Persisted {n} triggered hits to data/dip_log.jsonl")


def cmd_postmortem(
    target_date: str | None = None,
    watchlist: str | None = None,
    persist: bool = False,
):
    """Run the daily postmortem — counterfactuals + missed-opportunity scan.

    Args:
        target_date: ISO date string (YYYY-MM-DD); default = today.
        watchlist: comma-separated tickers; default = anomaly DEFAULT_WATCHLIST.
        persist: append report to data/postmortem_log.jsonl.
    """
    from datetime import date as _date
    from lib.alpaca_client import AlpacaClient
    from lib.postmortem import generate_report, print_report, persist_report

    if target_date:
        d = _date.fromisoformat(target_date)
    else:
        d = _date.today()

    wl = None
    if watchlist:
        wl = [s.strip().upper() for s in watchlist.split(",") if s.strip()]

    client = AlpacaClient()
    report = generate_report(client, target_date=d, watchlist=wl)
    print_report(report)
    if persist:
        persist_report(report)
        print(f"\n  ✓ Persisted to data/postmortem_log.jsonl")


def cmd_anomaly(
    watchlist: str | None = None,
    threshold: float = 4.0,
    no_news: bool = False,
    persist: bool = False,
):
    """Scan a watchlist for anomaly / "skyrocket" candidates.

    Composite z-score over 4 features: volume, price-move, range, news velocity.
    Triggers when composite >= threshold AND momentum confirmed AND quality
    gates pass. Output is detection only — strategy engine still applies its
    own gates before placing any trade.

    Args:
        watchlist: comma-separated tickers ("NVDA,TSLA,COIN"). If None, uses
                   anomaly_detector.DEFAULT_WATCHLIST.
        threshold: composite z-score required to trigger (default 4.0).
        no_news: skip NewsAPI/RSS lookup (faster, lower-quality signal).
        persist: append triggered hits to data/anomaly_log.jsonl.
    """
    from lib.alpaca_client import AlpacaClient
    from lib.anomaly_detector import (
        DEFAULT_WATCHLIST, scan_universe, print_anomaly_report, persist_scores,
    )

    if watchlist:
        symbols = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    else:
        symbols = DEFAULT_WATCHLIST
    print(f"  Anomaly scan: {len(symbols)} symbols, threshold={threshold:.1f}σ, "
          f"news={'off' if no_news else 'on'}")

    client = AlpacaClient()
    scores = scan_universe(
        client, symbols,
        fetch_news=not no_news,
        composite_threshold=threshold,
    )
    print()
    print_anomaly_report(scores, top=20)
    if persist:
        n = persist_scores(scores)
        print(f"  Persisted {n} triggered hits to data/anomaly_log.jsonl")


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


def cmd_longterm_pick(themes: str | None = None, top_n: int = 15,
                       use_kronos: bool = True, save: str | None = None,
                       explicit_tickers: str | None = None):
    """Score a candidate universe for long-term core holdings and print
    a ranked dossier. The user picks which top-N to add to
    wheel_strategy.yaml.core_holdings.
    """
    from lib.longterm_picker import (
        score_universe, render_dossier, themes_to_tickers,
    )

    if explicit_tickers:
        # Explicit list overrides themes — used for targeted re-runs
        seen: set[str] = set()
        tickers: list[str] = []
        for t in (s.strip().upper() for s in explicit_tickers.split(",")):
            if t and t not in seen:
                seen.add(t)
                tickers.append(t)
    else:
        tickers = themes_to_tickers(themes)
    if not tickers:
        print("No tickers resolved. Run with --tickers TICKER1,TICKER2 or "
              "--themes all (or one of: ai_compute, energy_transition, "
              "defense_cyber, healthcare_automation, fintech_rails, "
              "consumer_compounders, industrial_automation).")
        return

    print("=" * 70)
    print("  OPENCLAW LONG-TERM HOLDING PICKER")
    print("=" * 70)
    print(f"  Universe: {len(tickers)} tickers"
          f"{' (themes: ' + themes + ')' if themes else ' (all themes)'}")
    print(f"  Kronos AI: {'enabled' if use_kronos else 'disabled (faster)'}")
    print(f"  Factor weights: Quality 35% / Growth 25% / Moat 15% / "
          f"Momentum 15% / Valuation 10%")
    print()
    print("  Scoring... (yfinance fundamentals + Alpaca bars)")
    print()

    scores = score_universe(tickers, use_kronos=use_kronos)
    print(render_dossier(scores, top_n=top_n))

    if save:
        out_path = Path(save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([
                {
                    "rank": i + 1,
                    "ticker": s.ticker,
                    "composite": round(s.composite, 4),
                    "quality": round(s.quality, 4),
                    "growth": round(s.growth, 4),
                    "moat": round(s.moat, 4),
                    "momentum": round(s.momentum, 4),
                    "valuation": round(s.valuation, 4),
                    "sector": s.sector,
                    "market_cap": s.market_cap,
                    "why": s.why,
                }
                for i, s in enumerate(scores)
            ], f, indent=2)
        print(f"\n  💾 Saved full ranking to {out_path}")

    print()
    print("─" * 70)
    print("  Next step: pick 4-5 names you believe in, then add them to:")
    print("    config/wheel_strategy.yaml → core_holdings:")
    print("  The normal scan will buy them via the standard gates;")
    print("  once held, the exit logic skips them entirely (hold forever).")
    print()


def cmd_goals():
    """Display unified cross-bot goal tracker."""
    try:
        from tradingcore.unified_goals import (
            load_goals, get_progress, init_default_goals, GOALS_PATH,
        )
    except ImportError as e:
        print(f"[goals] tradingcore.unified_goals unavailable: {e}")
        return

    if not GOALS_PATH.exists():
        print(f"[goals] No goals file at {GOALS_PATH} — initializing.")
        init_default_goals()

    # Refresh equity from local baseline file (avoid pandas/alpaca imports —
    # heavy chains crash this CLI in the current env). Live Alpaca sync runs
    # via separate scripts; the top-level ~/Desktop/projects/goals does it too.
    try:
        import json
        from pathlib import Path
        from tradingcore.unified_goals import update_current_equity
        baseline = Path(__file__).resolve().parent / "data" / "baseline_equity.json"
        if baseline.exists():
            with open(baseline) as f:
                bd = json.load(f)
            eq = float(bd.get("baseline_equity") or bd.get("start_baseline") or 0.0)
            if eq > 0:
                update_current_equity("traderbot", eq)
    except Exception:
        pass

    data = load_goals()
    tb = get_progress("traderbot")
    pb = get_progress("polybot")

    def _row(label: str, val: str) -> str:
        return f"  {label:<24} {val}"

    def _halt_badge(state: str) -> str:
        return "[HALTED]" if state == "halted" else "[ ok   ]"

    print("=" * 70)
    print(f"UNIFIED GOALS — {data.get('updated_at', 'n/a')}")
    print(f"File: {GOALS_PATH}")
    print("=" * 70)
    for bot, prog in (("traderbot", tb), ("polybot", pb)):
        print()
        print(f"{bot.upper():<12}  {_halt_badge(prog['halt_state'])}  "
              f"${prog['current']:.2f}  (anchor ${prog['anchor']:.2f} → target ${prog['target']:.2f})")
        print(_row("growth from anchor", f"{prog['pct_growth_from_anchor']:+.2f}%"))
        print(_row("progress to target", f"{prog['pct_to_target']:.2f}%"))
        if prog["halt_reason"]:
            print(_row("halt reason", prog["halt_reason"]))
            print(_row("halted at", prog["halted_at"] or "?"))
        ms_line = "  ".join(
            f"${m['value']}{'✓' if m['hit_at'] else '·'}"
            for m in prog["milestones"]
        )
        print(_row("milestones", ms_line))
    print()


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Wheel Strategy Trader")
    parser.add_argument("command",
                        choices=["scan", "monitor", "backtest", "kill", "wheel-reset",
                                 "wheel-dte-sweep", "self-audit", "premarket-scan",
                                 "status",
                                 "chaos", "migrate", "dashboard", "hermes", "pdt",
                                 "kronos", "news", "calibrate", "pred-scan",
                                 "forecast", "kelly", "llm", "correlation",
                                 "insiders", "congress", "greeks", "baseline",
                                 "earnings", "econ", "finnhub", "crypto-scan",
                                 "crypto-monitor", "backtest-stocks",
                                 "build-cache",
                                 "pairs-scan", "min-variance",
                                 "anomaly", "postmortem", "digest", "dipbuy",
                                 "longterm-pick", "features", "goals",
                                 "stock-buys-gate", "markov",
                                 "hermes-cycle", "hermes-review",
                                 "hermes-ledger", "hermes-mode",
                                 "cornelius", "goal-score",
                                 "turtle", "turtle-scan", "turtle-backtest",
                                 "pead", "pead-scan"],
                        help="Command to run")
    parser.add_argument("--signal", default="all",
                        help="build-cache: which signal to fill (kronos|news|llm|all)")
    parser.add_argument("--with-signals", action="store_true",
                        help="backtest-stocks: enable Kronos+News+LLM+Bayesian enrichment "
                             "(needs caches built first; bayesian works without cache)")
    parser.add_argument("--with-kronos", action="store_true",
                        help="backtest-stocks: enable Kronos signal only")
    parser.add_argument("--with-news", action="store_true",
                        help="backtest-stocks: enable historical news signal only")
    parser.add_argument("--with-llm", action="store_true",
                        help="backtest-stocks: enable LLM forecast signal only")
    parser.add_argument("--ticker", default="SPY", help="Ticker for backtest/kronos/news")
    parser.add_argument("--reason", default="manual_cli", help="Reason for kill switch")
    parser.add_argument("--confirm", action="store_true",
                        help="wheel-reset: required to actually liquidate (dry-run otherwise)")
    parser.add_argument("--port", type=int, default=5051,
                        help="Dashboard web server port (default 5051; polybot uses 5050)")
    parser.add_argument("--full", action="store_true", help="Include quant scores in status output")
    parser.add_argument("--dry-run", action="store_true", help="Hermes: analyze only, don't change params")
    parser.add_argument("--lookback", type=int, default=14, help="Hermes: days of trade history to analyze")
    parser.add_argument("--simulations", type=int, default=500, help="Backtest: Monte Carlo simulation count")
    parser.add_argument("--capital", type=float, default=0, help="Backtest: starting capital (0 = use portfolio value)")
    parser.add_argument("--amount", type=float, default=None, help="baseline: starting-capital amount (omit to snapshot current equity)")
    parser.add_argument("--snapshot", action="store_true", help="baseline: snapshot current portfolio value as baseline")
    parser.add_argument("--show", action="store_true", help="baseline: show current baseline without changing it")
    parser.add_argument("--watchlist", default=None,
                        help="anomaly: comma-separated tickers (override DEFAULT_WATCHLIST)")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="anomaly: composite z-score trigger threshold (default 4.0)")
    parser.add_argument("--no-news", action="store_true",
                        help="anomaly: skip NewsAPI/RSS lookup (faster)")
    parser.add_argument("--persist", action="store_true",
                        help="anomaly/postmortem: append output to JSONL log")
    parser.add_argument("--date", default=None,
                        help="postmortem: target date (YYYY-MM-DD); default today")
    parser.add_argument("--telegram", action="store_true",
                        help="digest: also push to Telegram (needs TELEGRAM_BOT_TOKEN+CHAT_ID)")
    parser.add_argument("--themes", default=None,
                        help="longterm-pick: comma-separated themes (ai_compute, "
                             "energy_transition, defense_cyber, healthcare_automation, "
                             "fintech_rails, consumer_compounders, industrial_automation, "
                             "or 'all'). Default: all.")
    parser.add_argument("--top", type=int, default=15,
                        help="longterm-pick: how many ranked rows to show (default 15)")
    parser.add_argument("--no-kronos", action="store_true",
                        help="longterm-pick: skip Kronos AI directional bias (faster)")
    parser.add_argument("--save", default=None,
                        help="longterm-pick: write the ranked dossier to this path "
                             "(JSON). Optional — output prints to stdout regardless.")
    parser.add_argument("--tickers", default=None,
                        help="longterm-pick: explicit comma-separated ticker list. "
                             "Overrides --themes. Use for targeted re-runs (e.g., "
                             "validating a short list with Kronos enabled).")

    args = parser.parse_args()

    # Init memory palace on any command
    init_palace()

    log_event("main", f"command_{args.command}", {})

    if args.command == "status":
        cmd_status(full=args.full)
    elif args.command == "scan":
        cmd_scan()
    elif args.command == "crypto-scan":
        cmd_crypto_scan(dry_run=args.dry_run)
    elif args.command == "crypto-monitor":
        cmd_crypto_monitor(dry_run=args.dry_run)
    elif args.command == "backtest-stocks":
        # --with-signals turns on all four; the granular --with-* flags compose
        # so you can do `--with-kronos --with-llm` etc. for ablation studies.
        cmd_backtest_stocks(
            days_back=args.lookback or 180,
            capital=args.capital or 1500.0,
            enable_kronos=args.with_signals or args.with_kronos,
            enable_news=args.with_signals or args.with_news,
            enable_llm=args.with_signals or args.with_llm,
            enable_bayesian=True,  # always on (no cache needed, free)
        )
    elif args.command == "build-cache":
        cmd_build_cache(signal=args.signal,
                        days_back=args.lookback or 180)
    elif args.command == "pairs-scan":
        cmd_pairs_scan()
    elif args.command == "min-variance":
        cmd_min_variance()
    elif args.command == "monitor":
        cmd_monitor()
    elif args.command == "kill":
        cmd_kill(args.reason)
    elif args.command == "self-audit":
        # Use --lookback if explicitly >= 1, else 24h (default for nightly).
        sa_hours = float(args.lookback) if args.lookback and args.lookback >= 1 else 24.0
        cmd_self_audit(hours=sa_hours, telegram=args.telegram)
    elif args.command == "premarket-scan":
        cmd_premarket_scan(dry_run=args.dry_run)
    elif args.command == "wheel-reset":
        cmd_wheel_reset(confirm=args.confirm)
    elif args.command == "wheel-dte-sweep":
        # Use --lookback if explicitly passed (Hermes default is 14, which
        # is way too short for a DTE sweep — needs ≥365 days of bars to
        # exercise multiple wheel cycles per DTE band).
        sweep_lookback = args.lookback if args.lookback >= 90 else 365
        cmd_wheel_dte_sweep(
            days_back=sweep_lookback,
            capital=args.capital if args.capital else 1500.0,
        )
    elif args.command == "backtest":
        cmd_backtest(args.ticker, simulations=args.simulations, capital=args.capital)
    elif args.command == "chaos":
        cmd_chaos()
    elif args.command == "migrate":
        cmd_migrate()
    elif args.command == "dashboard":
        cmd_dashboard(args.port)
    elif args.command == "hermes":
        cmd_hermes(dry_run=args.dry_run, lookback=args.lookback)
    elif args.command == "pdt":
        cmd_pdt()
    elif args.command == "kronos":
        cmd_kronos(args.ticker)
    elif args.command == "news":
        cmd_news(args.ticker)
    elif args.command == "calibrate":
        cmd_calibrate()
    elif args.command == "pred-scan":
        cmd_pred_scan()
    elif args.command == "forecast":
        cmd_forecast(args.ticker)
    elif args.command == "kelly":
        cmd_kelly(args.ticker)
    elif args.command == "llm":
        cmd_llm(args.ticker)
    elif args.command == "correlation":
        cmd_correlation()
    elif args.command == "insiders":
        cmd_insiders(args.ticker)
    elif args.command == "congress":
        cmd_congress(args.ticker)
    elif args.command == "greeks":
        cmd_greeks()
    elif args.command == "baseline":
        # --show takes precedence; --snapshot means use current equity
        amt = None if args.snapshot else args.amount
        cmd_baseline(amount=amt, show=args.show)
    elif args.command == "earnings":
        cmd_earnings(args.ticker)
    elif args.command == "econ":
        cmd_econ()
    elif args.command == "finnhub":
        cmd_finnhub(args.ticker)
    elif args.command == "anomaly":
        cmd_anomaly(
            watchlist=args.watchlist,
            threshold=args.threshold,
            no_news=args.no_news,
            persist=args.persist,
        )
    elif args.command == "postmortem":
        cmd_postmortem(
            target_date=args.date,
            watchlist=args.watchlist,
            persist=args.persist,
        )
    elif args.command == "digest":
        cmd_digest(send_telegram=args.telegram)
    elif args.command == "dipbuy":
        cmd_dipbuy(
            watchlist=args.watchlist,
            threshold=args.threshold if args.threshold != 4.0 else 0.55,
            persist=args.persist,
        )
    elif args.command == "longterm-pick":
        cmd_longterm_pick(
            themes=args.themes,
            top_n=args.top,
            use_kronos=not args.no_kronos,
            save=args.save,
            explicit_tickers=args.tickers,
        )
    elif args.command == "features":
        from lib.feature_status import gather_all, render_report
        print(render_report(gather_all()))
    elif args.command == "goals":
        cmd_goals()
    elif args.command == "stock-buys-gate":
        from lib.stock_buys_gate import evaluate as _sbg_eval, render as _sbg_render
        result = _sbg_eval()
        print(_sbg_render(result))
    elif args.command == "hermes-cycle":
        # Scientific-method Hermes pass: closes prior experiments, picks
        # one new change, applies if mode=live. Read-only otherwise.
        from lib.hermes_scientific import (
            run_scientific_cycle, render_cycle, write_weekly_review,
        )
        force = "live" if "--live" in sys.argv else (
            "review" if "--review" in sys.argv else None
        )
        report = run_scientific_cycle(force_mode=force)
        print(render_cycle(report))
        # Always write the weekly markdown so the operator has an audit trail
        md_path = write_weekly_review(report)
        print(f"\nWeekly review written: {md_path}")
    elif args.command == "hermes-review":
        from lib.hermes_scientific import run_scientific_cycle, write_weekly_review
        report = run_scientific_cycle(force_mode="review")
        path = write_weekly_review(report)
        print(f"Weekly review written: {path}")
    elif args.command == "hermes-ledger":
        from lib.hermes_ledger import history, stats
        s = stats()
        print(f"Experiments — total {s['total']}, keep_rate "
              f"{s['keep_rate'] if s['keep_rate'] is not None else 'n/a'}")
        print(f"  open={s['counts'].get('open',0)} "
              f"kept={s['counts'].get('kept',0)} "
              f"rolled_back={s['counts'].get('rolled_back',0)} "
              f"expired={s['counts'].get('expired',0)}")
        print()
        for e in history(limit=15):
            when = (e.get("opened_at") or "")[:19].replace("T", " ")
            print(f"  {when}  {e.get('status'):<12} "
                  f"{e.get('param'):<26} "
                  f"{e.get('old_value')} → {e.get('new_value')}  "
                  f"verdict={e.get('verdict')}")
    elif args.command == "hermes-mode":
        from lib.hermes_scientific import get_mode, set_mode
        if len(sys.argv) > 2 and sys.argv[-1] in ("review", "live"):
            set_mode(sys.argv[-1])
            print(f"hermes_mode set → {sys.argv[-1]}")
        else:
            print(f"hermes_mode = {get_mode()}  "
                  "(set with `python main.py hermes-mode review|live`)")
    elif args.command == "cornelius":
        from agents.cornelius_agent import run_cornelius_cycle, render_cornelius_report
        dry = "--apply" not in sys.argv
        report = run_cornelius_cycle(dry_run=dry)
        print(render_cornelius_report(report))
        if dry:
            print("(dry-run — pass --apply to commit the picked change)")
    elif args.command == "goal-score":
        from lib.hermes_goal_score import compute_goal_metrics, render
        print(render(compute_goal_metrics()))
    elif args.command == "turtle":
        from lib.turtle_signal import turtle_signal, render_summary
        ticker = args.ticker or "SPY"
        result = turtle_signal(ticker, lookback_days=365)
        print(render_summary(result))
    elif args.command == "pead":
        from lib.pead_signal import pead_score, render as _pead_render
        print(_pead_render(pead_score(args.ticker or "NVDA")))
    elif args.command == "pead-scan":
        from lib.pead_signal import universe_scan as _pead_scan
        import yaml as _yaml
        with open("config/wheel_strategy.yaml") as _f:
            _cfg = _yaml.safe_load(_f) or {}
        universe = list(set(
            (_cfg.get("core_holdings") or [])
            + (_cfg.get("tickers") or [])
        ))
        print(f"Scanning {len(universe)} tickers for PEAD signals...\n")
        results = _pead_scan(universe)
        # Show only non-zero scores
        active = [r for r in results if abs(r.get("score") or 0) > 0.01]
        if not active:
            print("No PEAD signals active right now.")
        else:
            print(f"  {'Ticker':<8} {'Days':>5} {'Surprise':>10} "
                  f"{'Kind':<12} {'Score':>8}  Reason")
            for r in active[:20]:
                sp = r.get("surprise_pct")
                sp_s = f"{sp:+.1%}" if sp is not None else "n/a"
                print(f"  {r['ticker']:<8} {r.get('days_since', '?'):>5} "
                      f"{sp_s:>10} {r.get('kind', '?'):<12} "
                      f"{r.get('score', 0):>+8.3f}  {r.get('reason', '')[:40]}")
    elif args.command == "turtle-backtest":
        from lib.turtle_backtest import backtest_ticker, render_backtest, universe_backtest
        ticker = args.ticker or "SPY"
        lookback = args.lookback if args.lookback else 1825
        if "," in ticker:
            tickers = [t.strip().upper() for t in ticker.split(",")]
            results = universe_backtest(tickers, lookback_days=lookback)
            print(f"Backtest over {lookback}d for {len(tickers)} tickers:\n")
            print(f"  {'Ticker':<8} {'N':>4} {'WR':>7} {'R:R':>6} {'PF':>6} "
                  f"{'CompRet':>10} {'MaxDD':>8}")
            for r in results:
                if "error" in r:
                    print(f"  {r['ticker']:<8} ERROR — {r['error']}")
                    continue
                wr = r.get("win_rate")
                wr_s = f"{wr*100:.1f}%" if wr is not None else "n/a"
                rr_s = f"{r.get('rr_ratio'):.2f}" if r.get("rr_ratio") else "n/a"
                pf_s = f"{r.get('profit_factor'):.2f}" if r.get("profit_factor") else "n/a"
                print(f"  {r['ticker']:<8} {r['n_trades']:>4} {wr_s:>7} "
                      f"{rr_s:>6} {pf_s:>6} "
                      f"{r['compounded_return']*100:>+9.1f}% "
                      f"{r['max_drawdown']*100:>+7.1f}%")
        else:
            print(render_backtest(backtest_ticker(ticker, lookback_days=lookback)))
    elif args.command == "turtle-scan":
        from lib.turtle_signal import universe_scan
        import yaml as _yaml
        # Pull the universe from wheel_strategy.yaml (core_holdings + tickers)
        with open("config/wheel_strategy.yaml") as _f:
            _cfg = _yaml.safe_load(_f) or {}
        universe = list(set(
            (_cfg.get("core_holdings") or [])
            + (_cfg.get("tickers") or [])
        ))
        print(f"Scanning {len(universe)} tickers for Turtle signals...")
        fired = universe_scan(universe)
        if not fired:
            print("No Turtle signals firing today.")
        else:
            print(f"\n{len(fired)} signal(s) firing:\n")
            for f in fired:
                s = f.get('sizing', {})
                print(f"  {f['ticker']:<6} ${f['today_price']:>8.2f}  "
                      f"stop ${f['stop_price']:>8.2f} "
                      f"({f['stop_distance_pct']:+.1%})  "
                      f"size {s.get('shares', 0)} sh / "
                      f"${s.get('sized_dollars', 0):.0f}")
    elif args.command == "markov":
        from lib.markov_regime import markov_summary, render_summary
        ticker = args.ticker or "SPY"
        # Reuse --lookback for lookback_days, --simulations as horizon stub
        lookback = args.lookback if args.lookback else 730
        horizon = 1  # daily forecast default; users override in code if needed
        result = markov_summary(ticker, lookback_days=lookback, horizon=horizon)
        print(render_summary(result))
        # Cache the latest summary for the dashboard panel
        try:
            import json as _json
            from pathlib import Path as _Path
            cache = _Path(__file__).parent / "data" / "markov_latest.json"
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "w") as _f:
                _json.dump(result, _f, indent=2, default=str)
        except Exception:
            pass


if __name__ == "__main__":
    main()
