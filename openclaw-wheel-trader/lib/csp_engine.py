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

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"


def _load_positions() -> list[dict]:
    if not POSITIONS_PATH.exists():
        return []
    with open(POSITIONS_PATH) as f:
        return json.load(f)


def _save_positions(positions: list[dict]):
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)


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

    for ticker in config.get("tickers", []):
        if ticker not in daily_data or ticker not in options_chains:
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

    # Success — remember everything
    remember_trade_decision(
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

    # Track position
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

    executed = []
    for candidate in candidates[:max_trades]:
        result = execute_csp(
            candidate=candidate,
            client=client,
            portfolio_value=account["portfolio_value"],
            current_daily_pnl=0,  # TODO: calculate from today's trades
            current_open_orders=len(open_orders),
        )
        if result:
            executed.append(result)

    log_event("csp_engine", "scan_complete", {"executed": len(executed)})
    return executed
