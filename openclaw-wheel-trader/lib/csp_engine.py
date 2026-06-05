"""
Sprint 2: Cash-Secured Put Engine

The first leg of the Wheel. This module:
1. Pulls options chain for target tickers
2. Runs screener to find best CSP candidates
3. Confirms with candlestick patterns at support zones
4. Proposes trade through the 3-step order gate
5. Stores decision in memory palace

Source: Wheel Strategy + Candlestick Bible + Naked Forex
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lib.audit import log_event
from lib.order_gate import OrderIntent, step1_propose, step2_validate, step3_execute
from lib.circuit_breaker import CircuitBreakerTripped
from lib.screener import score_csp_candidate, rank_candidates, WheelCandidate, _load_strategy_config
from lib.memory_palace import (
    remember_trade_decision, recall_ticker_history,
    get_current_regime, search_memory, diary_write, kg_query,
)
from lib.alpaca_client import AlpacaClient
from lib.earnings_filter import earnings_veto, next_earnings_date

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


def scan_for_csps(
    client: AlpacaClient,
    daily_data: dict[str, "pd.DataFrame"],
    weekly_data: dict[str, "pd.DataFrame"],
    options_chains: dict[str, list[dict]],
    iv_data: dict[str, dict],
) -> list[WheelCandidate]:
    """
    Scan all target tickers for CSP candidates.

    Args:
        client: Alpaca API client
        daily_data: {ticker: daily OHLCV DataFrame}
        weekly_data: {ticker: weekly OHLCV DataFrame}
        options_chains: {ticker: [list of put option dicts]}
        iv_data: {ticker: iv evaluation dict from evaluate_premium_environment()}

    Returns:
        Ranked list of tradeable CSP candidates
    """
    config = _load_strategy_config()
    candidates = []

    # Quant screen: only sell puts on fundamentally sound underlyings
    from lib.quant_screener import screen_universe
    quant_scores = screen_universe(daily_data, exclude_avoid=True)
    quant_approved = {s.ticker for s in quant_scores if s.verdict in ("STRONG", "OK")}

    # 2026-05-30: was iterating config["tickers"] (the SHORT expensive
    # list — AAPL, MSFT, NVDA, etc.). At $1,500 bankroll none of those
    # CSPs are affordable, so the whole CSP scan was a silent no-op.
    # Switched to iterating the actual options_chains keys — those are
    # the tickers main.py just fetched data for, which during Phase 2
    # are tickers_phase1 (NIO, GRAB, NU, AAL, etc.) — the cheap names
    # we CAN sell puts on. The CSP scoring still applies the standard
    # delta/DTE/OI/spread/score filters per option.
    csp_universe = list(options_chains.keys()) or config.get("tickers", [])
    for ticker in csp_universe:
        if ticker not in daily_data or ticker not in options_chains:
            continue
        if ticker not in quant_approved:
            log_event("csp_engine", "skip_quant_fail", {"ticker": ticker})
            continue

        # Check memory — any reason to skip this ticker?
        history = recall_ticker_history(ticker)
        regime = get_current_regime()

        # Skip if we already have an open CSP on this ticker
        positions = _load_positions()
        existing_csps = [
            p for p in positions
            if p.get("ticker") == ticker and p.get("type") == "csp" and p.get("status") == "open"
        ]
        if existing_csps:
            log_event("csp_engine", "skip_existing", {
                "ticker": ticker, "reason": "already_has_open_csp",
            })
            continue

        # Score each put in the chain
        for option in options_chains[ticker]:
            candidate = score_csp_candidate(
                ticker=ticker,
                option=option,
                daily_df=daily_data[ticker],
                weekly_df=weekly_data[ticker],
                iv_data=iv_data.get(ticker, {}),
                config=config,
            )
            if candidate:
                # CLAUDE.md rule: never sell options expiring through an earnings date
                if earnings_veto(ticker, candidate.expiration):
                    log_event("csp_engine", "skip_earnings", {
                        "ticker": ticker,
                        "strike": candidate.strike,
                        "expiration": candidate.expiration,
                    })
                    continue
                # 2026-05-30: AFFORDABILITY pre-filter. The CSP collateral
                # (strike × 100) must fit within max_position_pct of the
                # current bankroll. Without this, the ranker often surfaces
                # high-score but unaffordable candidates (e.g. F $15P at
                # $1,500 collateral on a $1,507 account → 99% — gets
                # vetoed by risk_agent and the next-best candidate never
                # gets tried because max_trades=1.
                try:
                    portfolio = float(client.get_account().get("portfolio_value", 0) or 0)
                    if portfolio > 0:
                        # mirror risk_agent's position_max read
                        import yaml as _y
                        from pathlib import Path as _P
                        with open(_P(__file__).resolve().parent.parent / "config" / "settings.yaml") as _f:
                            _s = _y.safe_load(_f) or {}
                        _max_pos = float(_s.get("circuit_breakers", {}).get("max_position_pct", 0.50))
                        max_collateral = portfolio * max(_max_pos, 0.50)
                        coll = candidate.strike * 100
                        if coll > max_collateral:
                            log_event("csp_engine", "skip_unaffordable", {
                                "ticker": ticker, "strike": candidate.strike,
                                "collateral": coll, "max_allowed": round(max_collateral, 2),
                                "portfolio": portfolio,
                            })
                            continue
                except Exception:
                    pass  # if affordability check fails, defer to risk_agent
                candidates.append(candidate)

    return rank_candidates(candidates)


def execute_csp(
    candidate: WheelCandidate,
    client: AlpacaClient,
    portfolio_value: float,
    current_daily_pnl: float,
    current_open_orders: int,
    last_loss_time: datetime | None = None,
) -> dict | None:
    """
    Execute a CSP trade through the full 3-step order gate.

    Returns:
        Order response dict on success, None if blocked/failed
    """
    ticker = candidate.ticker

    # Defense-in-depth: re-check earnings at execute time (handles stale scan cache,
    # last-minute earnings schedule changes, etc). CLAUDE.md: never sell through earnings.
    if earnings_veto(ticker, candidate.expiration):
        log_event("csp_engine", "skip_earnings_at_execute", {
            "ticker": ticker, "strike": candidate.strike, "expiration": candidate.expiration,
        })
        diary_write("strategy_agent",
            f"{ticker}|CSP_SKIP|earnings|exp_{candidate.expiration}")
        return None

    # Last-mile broker reconciliation — same fix pattern stock_engine got after
    # the 2026-04-28 BAC double-buy. positions.json can be 5-30s stale across
    # concurrent scan processes; the broker is the single source of truth.
    # Reject if the broker already shows an OPEN option position with this
    # exact OCC symbol (same strike + expiration + side).
    # Hoisted out of the try so the global capital-at-risk gate below can
    # still see the snapshot even if the dup-check raises mid-loop.
    broker_positions: list[dict] | None = None
    try:
        target_symbol = client._build_option_symbol(
            ticker, candidate.expiration, "put", candidate.strike,
        )
        broker_positions = client.get_positions() or []
        for bp in broker_positions:
            sym = str(bp.get("symbol") or bp.get("ticker") or "").upper()
            qty = float(bp.get("qty", 0) or 0)
            if sym == target_symbol.upper() and qty != 0:
                log_event("csp_engine", "duplicate_csp_blocked", {
                    "ticker": ticker, "strike": candidate.strike,
                    "expiration": candidate.expiration,
                    "reason": "broker_position_exists",
                })
                diary_write("strategy_agent",
                    f"{ticker}|CSP_BLOCKED|duplicate_at_broker|"
                    f"{candidate.strike}P_{candidate.expiration}")
                return None
    except Exception as e:
        # Don't gate trades on broker connectivity issues — order_dedup's
        # file-locked hash store is still in front of step3 so a same-cycle
        # dup still gets caught. Just log and continue.
        log_event("csp_engine", "broker_reconcile_failed",
                  {"ticker": ticker, "error": str(e)[:200]}, result="degraded")

    # Build reasoning string for memory
    reasoning = (
        f"Selling {ticker} {candidate.strike}P exp {candidate.expiration} "
        f"for ${candidate.premium} premium. "
        f"Composite score {candidate.composite_score}/9 "
        f"(trend:{candidate.trend_score} level:{candidate.level_score} signal:{candidate.signal_score}). "
        f"Zone at {candidate.zone_level} with {candidate.zone_touches} touches. "
        f"IV rank {candidate.iv_rank:.0%}. "
        f"Annualized return {candidate.annualized_return:.1%}. "
        f"Pattern: {candidate.candlestick_pattern or 'none'}."
    )

    # Agent consensus — strategy proposes, risk + compliance review.
    # Pass `broker_positions` so risk_agent can run the global
    # capital-at-risk gate. Reuse the list we already fetched above for
    # broker reconciliation — saves a duplicate API call.
    from agents.consensus import seek_consensus
    consensus = seek_consensus(
        candidate, portfolio_value, broker_positions=broker_positions,
    )
    if not consensus["approved"]:
        log_event("csp_engine", "consensus_rejected", {
            "ticker": ticker,
            "decision": consensus["decision"],
            "blocking_agent": consensus.get("blocking_agent"),
            "reason": consensus.get("reason", ""),
        })
        diary_write("strategy_agent",
            f"{ticker}|CSP_{consensus['decision']}|{consensus.get('blocking_agent', '?')}|"
            f"{consensus.get('reason', '')[:80]}")
        return None

    # Step 1: PROPOSE
    intent = OrderIntent(
        ticker=ticker,
        side="sell_to_open",
        order_type="limit",
        asset_type="option",
        quantity=1,
        limit_price=candidate.premium,
        option_type="put",
        strike=candidate.strike,
        expiration=candidate.expiration,
        reason=reasoning,
        composite_score=candidate.composite_score,
    )

    try:
        intent = step1_propose(intent)
    except ValueError as e:
        log_event("csp_engine", "propose_failed", {"ticker": ticker, "error": str(e)})
        diary_write("strategy_agent", f"{ticker}|CSP_BLOCKED|duplicate|{e}")
        return None

    # Step 2: VALIDATE
    try:
        step2_validate(
            intent=intent,
            portfolio_value=portfolio_value,
            current_daily_pnl=current_daily_pnl,
            current_open_orders=current_open_orders,
            last_loss_time=last_loss_time,
        )
    except (CircuitBreakerTripped, ValueError) as e:
        log_event("csp_engine", "validate_failed", {"ticker": ticker, "error": str(e)})
        diary_write("strategy_agent", f"{ticker}|CSP_BLOCKED|validation|{e}")
        return None

    # Step 3: EXECUTE
    try:
        response = step3_execute(intent, client)
    except Exception as e:
        log_event("csp_engine", "execute_failed", {"ticker": ticker, "error": str(e)})
        diary_write("strategy_agent", f"{ticker}|CSP_FAILED|execution|{e}")
        return None

    # Success — remember everything. Capture the drawer_id so the
    # outcome can later be linked back to the original reasoning
    # (learning-loop pattern from TradingAgents v0.2.4).
    decision_drawer_id = remember_trade_decision(
        ticker=ticker,
        trade_type="csp",
        details={
            "strike": candidate.strike,
            "expiration": candidate.expiration,
            "premium": candidate.premium,
            "delta": candidate.delta,
            "composite_score": candidate.composite_score,
            "order_id": response.get("id", ""),
        },
        reasoning=reasoning,
    )

    diary_write("strategy_agent",
        f"{ticker}|CSP_EXECUTED|{candidate.strike}P|${candidate.premium}|"
        f"score_{candidate.composite_score}/9|"
        f"zone_{candidate.zone_level}|{candidate.candlestick_pattern or 'no_pattern'}"
    )

    # Track position — store decision_drawer_id so the outcome can be
    # resolved against this specific decision at close time.
    positions = _load_positions()
    positions.append({
        "ticker": ticker,
        "type": "csp",
        "status": "open",
        "strike": candidate.strike,
        "expiration": candidate.expiration,
        "premium_collected": candidate.premium,
        "order_id": response.get("id", ""),
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "composite_score": candidate.composite_score,
        "decision_drawer_id": decision_drawer_id,
    })
    _save_positions(positions)

    return response


def run_csp_scan_and_execute(
    client: AlpacaClient,
    daily_data: dict,
    weekly_data: dict,
    options_chains: dict,
    iv_data: dict,
    max_trades: int = 1,
) -> list[dict]:
    """
    Full pipeline: scan → rank → execute top N candidates.
    Called by the monitoring cron or manually.
    """
    log_event("csp_engine", "scan_started", {"tickers": list(daily_data.keys())})

    # Pre-flight: derive the wheel state from broker positions and refuse
    # to scan if anything is illegal (short equity, long options, uncovered
    # short calls, etc.). Catching state drift here prevents the
    # 2026-04-28 BAC double-buy class of bug — better to halt one scan
    # than to compound the drift with another trade.
    from lib.wheel_state import classify_book, IllegalWheelState
    try:
        classify_book(client.get_positions())
    except IllegalWheelState as e:
        log_event("csp_engine", "preflight_halt", {
            "reason": str(e)[:300],
            "illegal_underlyings": [
                u for u, s in e.per_underlying.items() if s.stage == "illegal"
            ],
        }, result="failed")
        diary_write("strategy_agent", f"CSP_SCAN_HALT|illegal_wheel_state|{str(e)[:120]}")
        return []

    candidates = scan_for_csps(client, daily_data, weekly_data, options_chains, iv_data)

    if not candidates:
        log_event("csp_engine", "no_candidates", {})
        diary_write("strategy_agent", "SCAN|no_tradeable_candidates_found")
        return []

    log_event("csp_engine", "candidates_found", {
        "count": len(candidates),
        "top": f"{candidates[0].ticker} {candidates[0].strike}P score={candidates[0].composite_score}",
    })

    # Get account info for validation
    account = client.get_account()
    open_orders = client.get_open_orders()

    # 2026-05-20: was hardcoded `current_daily_pnl=0` with a TODO,
    # which silently disabled the daily-loss circuit breaker on the
    # CSP path. Now derives from broker equity vs morning baseline.
    daily_pnl_dollars = 0.0
    try:
        with open(Path(__file__).parent.parent / "data" / "baseline_equity.json") as _bf:
            baseline = json.load(_bf)
        portfolio_now = float(account.get("portfolio_value", 0) or 0)
        baseline_eq = float(baseline.get("baseline_equity", portfolio_now))
        daily_pnl_dollars = portfolio_now - baseline_eq
    except Exception:
        # If baseline file missing, fall through with 0 (same as before
        # but at least we tried). Log this so it's discoverable.
        log_event("csp_engine", "daily_pnl_baseline_missing",
                  {}, result="degraded")

    # 2026-06-05: was `candidates[:max_trades]` — pre-slicing meant if the
    # top-ranked candidate got vetoed by compliance (earnings filter, etc),
    # the bot never tried the #2 candidate even when it was perfectly
    # tradeable. With NIO #1 blocked daily by upcoming earnings, the bot
    # was effectively wheel-paralyzed. Now we iterate ALL ranked
    # candidates and break only when max_trades have actually executed.
    # Caps inspection at min(len(candidates), max_trades * 5) so a
    # systemic broker failure can't run away through 50 candidates.
    executed = []
    inspection_cap = min(len(candidates), max(max_trades * 5, 10))
    for candidate in candidates[:inspection_cap]:
        if len(executed) >= max_trades:
            break
        result = execute_csp(
            candidate=candidate,
            client=client,
            portfolio_value=account["portfolio_value"],
            current_daily_pnl=daily_pnl_dollars,
            current_open_orders=len(open_orders),
        )
        if result:
            executed.append(result)

    log_event("csp_engine", "scan_complete", {
        "executed": len(executed),
        "candidates_inspected": min(len(candidates), inspection_cap),
        "candidates_total": len(candidates),
    })
    return executed
