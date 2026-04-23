"""
Sprint 4: Covered Call Engine

After CSP assignment, we hold 100 shares. This module:
1. Identifies assigned positions needing covered calls
2. Scores call options at resistance zones (bearish patterns)
3. Ensures strike is above cost basis
4. Executes through the order gate
5. Tracks cost basis reduction across CC cycles
6. Handles call assignment (shares called away → back to CSPs)

Source: Wheel Strategy + Candlestick Bible (bearish patterns) + Naked Forex (resistance zones)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from lib.audit import log_event
from lib.order_gate import OrderIntent, step1_propose, step2_validate, step3_execute
from lib.circuit_breaker import CircuitBreakerTripped
from lib.screener import score_cc_candidate, rank_candidates, WheelCandidate, _load_strategy_config
from lib.memory_palace import (
    remember_trade_decision, diary_write, kg_add, kg_invalidate, kg_query,
)
from lib.alpaca_client import AlpacaClient

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"
TRADE_HISTORY_PATH = Path(__file__).parent.parent / "data" / "trade_history.json"


def _load_positions() -> list[dict]:
    if not POSITIONS_PATH.exists():
        return []
    with open(POSITIONS_PATH) as f:
        return json.load(f)


def _save_positions(positions: list[dict]):
    with open(POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)


def _load_trade_history() -> list[dict]:
    if not TRADE_HISTORY_PATH.exists():
        return []
    with open(TRADE_HISTORY_PATH) as f:
        return json.load(f)


def _save_trade_history(history: list[dict]):
    TRADE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def find_assigned_positions() -> list[dict]:
    """Find positions that were assigned and need covered calls."""
    positions = _load_positions()
    return [
        p for p in positions
        if p.get("status") == "assigned"
        and p.get("assigned_shares", 0) >= 100
        and not p.get("cc_active")  # No active covered call yet
    ]


def scan_for_ccs(
    client: AlpacaClient,
    daily_data: dict,
    weekly_data: dict,
    options_chains: dict,
    iv_data: dict,
) -> list[WheelCandidate]:
    """
    Scan assigned positions for covered call candidates.
    """
    config = _load_strategy_config()
    assigned = find_assigned_positions()
    candidates = []

    for pos in assigned:
        ticker = pos.get("ticker", "")
        cost_basis = pos.get("cost_basis", 0)

        if ticker not in daily_data or ticker not in options_chains:
            continue

        for option in options_chains.get(ticker, []):
            candidate = score_cc_candidate(
                ticker=ticker,
                option=option,
                daily_df=daily_data[ticker],
                weekly_df=weekly_data[ticker],
                iv_data=iv_data.get(ticker, {}),
                cost_basis=cost_basis,
                config=config,
            )
            if candidate:
                candidates.append(candidate)

    return rank_candidates(candidates)


def check_dividend_conflict(ticker: str, expiration: str) -> bool:
    """
    Check if selling a call through an ex-dividend date.
    Returns True if there IS a conflict (skip this expiration).
    
    TODO: Integrate with earnings/dividend calendar API.
    For now, returns False (no conflict detected).
    """
    # Placeholder — wire to dividend calendar in Sprint 8
    return False


def execute_cc(
    candidate: WheelCandidate,
    position: dict,
    client: AlpacaClient,
    portfolio_value: float,
    current_daily_pnl: float,
    current_open_orders: int,
) -> dict | None:
    """Execute a covered call through the order gate."""
    ticker = candidate.ticker

    # Check dividend conflict
    if check_dividend_conflict(ticker, candidate.expiration):
        log_event("cc_engine", "dividend_conflict", {
            "ticker": ticker, "expiration": candidate.expiration,
        })
        diary_write("strategy_agent",
            f"{ticker}|CC_SKIP|dividend_conflict|exp_{candidate.expiration}")
        return None

    reasoning = (
        f"Selling {ticker} {candidate.strike}C exp {candidate.expiration} "
        f"for ${candidate.premium} premium. "
        f"Cost basis: ${position.get('cost_basis', 0):.2f}. "
        f"Strike above cost basis by ${candidate.strike - position.get('cost_basis', 0):.2f}. "
        f"Composite score {candidate.composite_score}/9. "
        f"Resistance zone at {candidate.zone_level}. "
        f"Pattern: {candidate.candlestick_pattern or 'none'}."
    )

    # Agent consensus — strategy proposes, risk + compliance review
    from agents.consensus import seek_consensus
    consensus = seek_consensus(candidate, portfolio_value, cost_basis=position.get("cost_basis"))
    if not consensus["approved"]:
        log_event("cc_engine", "consensus_rejected", {
            "ticker": ticker,
            "decision": consensus["decision"],
            "blocking_agent": consensus.get("blocking_agent"),
            "reason": consensus.get("reason", ""),
        })
        diary_write("strategy_agent",
            f"{ticker}|CC_{consensus['decision']}|{consensus.get('blocking_agent', '?')}")
        return None

    intent = OrderIntent(
        ticker=ticker,
        side="sell_to_open",
        order_type="limit",
        asset_type="option",
        quantity=1,
        limit_price=candidate.premium,
        option_type="call",
        strike=candidate.strike,
        expiration=candidate.expiration,
        reason=reasoning,
        composite_score=candidate.composite_score,
    )

    try:
        intent = step1_propose(intent)
        step2_validate(intent, portfolio_value, current_daily_pnl, current_open_orders)
        response = step3_execute(intent, client)
    except (CircuitBreakerTripped, ValueError, Exception) as e:
        log_event("cc_engine", "execute_failed", {"ticker": ticker, "error": str(e)})
        diary_write("strategy_agent", f"{ticker}|CC_FAILED|{e}")
        return None

    # Success — update position and memory
    remember_trade_decision(
        ticker=ticker,
        trade_type="cc",
        details={
            "strike": candidate.strike,
            "expiration": candidate.expiration,
            "premium": candidate.premium,
            "cost_basis": position.get("cost_basis", 0),
        },
        reasoning=reasoning,
    )

    # Update position tracking
    positions = _load_positions()
    for p in positions:
        if p.get("ticker") == ticker and p.get("status") == "assigned" and not p.get("cc_active"):
            p["cc_active"] = True
            p["cc_strike"] = candidate.strike
            p["cc_expiration"] = candidate.expiration
            p["cc_premium"] = candidate.premium
            p["cc_order_id"] = response.get("id", "")
            # Reduce cost basis by CC premium
            p["cost_basis"] = p.get("cost_basis", 0) - candidate.premium
            break
    _save_positions(positions)

    diary_write("strategy_agent",
        f"{ticker}|CC_EXECUTED|{candidate.strike}C|${candidate.premium}|"
        f"new_cost_basis_{positions[-1].get('cost_basis', 0):.2f}")

    return response


def handle_call_assignment(ticker: str, position: dict):
    """
    Shares called away. Complete the Wheel cycle.
    Log final P/L, free capital, return to CSP scanning.
    """
    original_premium_put = position.get("premium_collected", 0)
    cc_premiums = position.get("cc_premium", 0)
    strike = position.get("strike", 0)  # Original put strike (entry price)
    cc_strike = position.get("cc_strike", 0)  # Call strike (exit price)

    # P/L = (call strike - put strike) * 100 + all premiums collected
    capital_gain = (cc_strike - strike) * 100
    total_premiums = (original_premium_put + cc_premiums) * 100
    total_pnl = capital_gain + total_premiums

    # Update position
    positions = _load_positions()
    for p in positions:
        if p.get("ticker") == ticker and p.get("status") == "assigned":
            p["status"] = "completed"
            p["completed_at"] = datetime.now(timezone.utc).isoformat()
            p["total_pnl"] = total_pnl
            break
    _save_positions(positions)

    # Add to trade history
    history = _load_trade_history()
    history.append({
        "ticker": ticker,
        "type": "wheel_cycle",
        "put_strike": strike,
        "put_premium": original_premium_put,
        "call_strike": cc_strike,
        "call_premium": cc_premiums,
        "capital_gain": capital_gain,
        "total_pnl": total_pnl,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_trade_history(history)

    # Knowledge graph
    kg_invalidate(ticker, "assigned", f"{position.get('assigned_shares', 100)}_shares")
    kg_add(ticker, "wheel_completed", f"pnl_{total_pnl:.2f}",
           metadata={"put_strike": strike, "call_strike": cc_strike})

    diary_write("strategy_agent",
        f"{ticker}|WHEEL_COMPLETE|put_{strike}|call_{cc_strike}|"
        f"pnl_${total_pnl:.2f}|premiums_${total_premiums:.2f}")

    send_msg = (
        f"🎡 WHEEL COMPLETE: {ticker}\n"
        f"Put: {strike} → Assigned → Call: {cc_strike}\n"
        f"Capital gain: ${capital_gain:.2f}\n"
        f"Premiums: ${total_premiums:.2f}\n"
        f"Total P/L: ${total_pnl:.2f}"
    )

    from lib.monitor import send_alert
    send_alert(send_msg)
    log_event("cc_engine", "wheel_completed", {
        "ticker": ticker, "total_pnl": total_pnl,
    }, result="success")
