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
from __future__ import annotations

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
from lib.earnings_filter import earnings_veto

TRADE_HISTORY_PATH = Path(__file__).parent.parent / "data" / "trade_history.json"

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


def _load_trade_history() -> list[dict]:
    if not TRADE_HISTORY_PATH.exists():
        return []
    with open(TRADE_HISTORY_PATH) as f:
        return json.load(f)


def _save_trade_history(history: list[dict]):
    TRADE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def _load_core_holdings() -> set[str]:
    """Load the wheel_strategy.yaml :: core_holdings forever-hold set.
    Returns uppercase tickers. Returns empty set on failure.
    """
    import yaml
    from pathlib import Path
    try:
        path = Path(__file__).resolve().parent.parent / "config" / "wheel_strategy.yaml"
        with open(path) as f:
            strategy = yaml.safe_load(f) or {}
        return {str(t).upper() for t in (strategy.get("core_holdings") or [])}
    except Exception:
        return set()


def find_assigned_positions() -> list[dict]:
    """Find positions that were assigned and need covered calls.

    2026-05-26: forever-hold protection. Tickers in
    wheel_strategy.yaml.core_holdings (or any position explicitly
    stamped hold_forever) are SKIPPED — the bot owns those long-term
    and refuses to write CCs that could call them away. The skip is
    logged so the operator sees that protection fired.
    """
    positions = _load_positions()
    core_holdings = _load_core_holdings()
    eligible: list[dict] = []
    skipped_core: list[str] = []
    for p in positions:
        if p.get("status") != "assigned":
            continue
        if p.get("assigned_shares", 0) < 100:
            continue
        if p.get("cc_active"):
            continue
        ticker = str(p.get("ticker", "")).upper()
        if p.get("hold_forever") or ticker in core_holdings:
            skipped_core.append(ticker)
            continue
        eligible.append(p)
    if skipped_core:
        log_event(
            "cc_engine", "core_holdings_skipped_from_cc",
            {"tickers": skipped_core, "count": len(skipped_core)},
            result="success",
        )
    return eligible


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
    # Pre-flight: refuse to scan if broker state is illegal. An uncovered
    # short call (the most likely CC-related drift) would otherwise let
    # us layer a SECOND CC on top, doubling the risk.
    from lib.wheel_state import classify_book, IllegalWheelState
    try:
        classify_book(client.get_positions())
    except IllegalWheelState as e:
        log_event("cc_engine", "preflight_halt", {
            "reason": str(e)[:300],
            "illegal_underlyings": [
                u for u, s in e.per_underlying.items() if s.stage == "illegal"
            ],
        }, result="failed")
        diary_write("strategy_agent", f"CC_SCAN_HALT|illegal_wheel_state|{str(e)[:120]}")
        return []

    config = _load_strategy_config()
    assigned = find_assigned_positions()
    candidates = []

    for pos in assigned:
        ticker = pos.get("ticker", "")
        cost_basis = pos.get("cost_basis", 0)
        # Hard floor: prevent CC strikes from drifting below the original
        # broker fill price, even after multiple CC cycles compound the
        # cost-basis-adjusted-by-premium accounting. Back-compat: legacy
        # positions without this field fall back to the current (adjusted)
        # cost_basis — same behavior as before.
        original_purchase_price = pos.get("original_purchase_price", cost_basis)

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
                original_purchase_price=original_purchase_price,
            )
            if candidate:
                # CLAUDE.md rule: never sell options expiring through an earnings date
                if earnings_veto(ticker, candidate.expiration):
                    log_event("cc_engine", "skip_earnings", {
                        "ticker": ticker,
                        "strike": candidate.strike,
                        "expiration": candidate.expiration,
                    })
                    continue
                candidates.append(candidate)

    return rank_candidates(candidates)


def check_dividend_conflict(ticker: str, expiration: str) -> bool:
    """
    Check if selling a call through an ex-dividend date.

    ITM calls get early-exercised the day before ex-dividend to capture the
    dividend — which would lose us both the dividend AND any remaining
    extrinsic value. Conservative rule: if an ex-div date falls between
    today and expiration, flag the conflict and skip this expiration.

    Returns True if there IS a conflict (skip this expiration).
    Fails open (returns False) when Finnhub unavailable.
    """
    try:
        from datetime import datetime, timezone
        from lib.finnhub_client import next_ex_dividend_date

        try:
            exp_date = datetime.fromisoformat(expiration).date()
        except (ValueError, TypeError):
            # Can't parse expiration — can't enforce
            return False

        today = datetime.now(timezone.utc).date()
        lookahead = max(0, (exp_date - today).days) + 1

        ex_div_iso = next_ex_dividend_date(ticker, days_ahead=lookahead)
        if not ex_div_iso:
            return False

        try:
            ex_div_date = datetime.fromisoformat(ex_div_iso).date()
        except (ValueError, TypeError):
            return False

        # Conflict if ex-div between today (inclusive) and expiration (inclusive)
        if today <= ex_div_date <= exp_date:
            log_event("cc_engine", "dividend_ex_date_in_window", {
                "ticker": ticker,
                "ex_div": ex_div_iso,
                "expiration": expiration,
            })
            return True
        return False

    except Exception as e:
        log_event("cc_engine", "dividend_check_error", {
            "ticker": ticker, "expiration": expiration, "error": str(e)[:200],
        })
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

    # Defense-in-depth: re-check earnings at execute time (handles stale scan cache,
    # last-minute earnings schedule changes, etc). CLAUDE.md: never sell through earnings.
    if earnings_veto(ticker, candidate.expiration):
        log_event("cc_engine", "skip_earnings_at_execute", {
            "ticker": ticker, "strike": candidate.strike, "expiration": candidate.expiration,
        })
        diary_write("strategy_agent",
            f"{ticker}|CC_SKIP|earnings|exp_{candidate.expiration}")
        return None

    # Check dividend conflict
    if check_dividend_conflict(ticker, candidate.expiration):
        log_event("cc_engine", "dividend_conflict", {
            "ticker": ticker, "expiration": candidate.expiration,
        })
        diary_write("strategy_agent",
            f"{ticker}|CC_SKIP|dividend_conflict|exp_{candidate.expiration}")
        return None

    # Last-mile broker reconciliation — same fix pattern stock_engine got after
    # the 2026-04-28 BAC double-buy. positions.json can be stale across
    # concurrent scan processes; reject if the broker already shows an OPEN
    # option position with this exact OCC symbol (same strike + expiration).
    # Hoisted so the global capital-at-risk gate below can reuse the snapshot.
    broker_positions: list[dict] | None = None
    try:
        target_symbol = client._build_option_symbol(
            ticker, candidate.expiration, "call", candidate.strike,
        )
        broker_positions = client.get_positions() or []
        for bp in broker_positions:
            sym = str(bp.get("symbol") or bp.get("ticker") or "").upper()
            qty = float(bp.get("qty", 0) or 0)
            if sym == target_symbol.upper() and qty != 0:
                log_event("cc_engine", "duplicate_cc_blocked", {
                    "ticker": ticker, "strike": candidate.strike,
                    "expiration": candidate.expiration,
                    "reason": "broker_position_exists",
                })
                diary_write("strategy_agent",
                    f"{ticker}|CC_BLOCKED|duplicate_at_broker|"
                    f"{candidate.strike}C_{candidate.expiration}")
                return None
    except Exception as e:
        log_event("cc_engine", "broker_reconcile_failed",
                  {"ticker": ticker, "error": str(e)[:200]}, result="degraded")

    reasoning = (
        f"Selling {ticker} {candidate.strike}C exp {candidate.expiration} "
        f"for ${candidate.premium} premium. "
        f"Cost basis: ${position.get('cost_basis', 0):.2f}. "
        f"Strike above cost basis by ${candidate.strike - position.get('cost_basis', 0):.2f}. "
        f"Composite score {candidate.composite_score}/9. "
        f"Resistance zone at {candidate.zone_level}. "
        f"Pattern: {candidate.candlestick_pattern or 'none'}."
    )

    # Agent consensus — strategy proposes, risk + compliance review.
    # Pass broker_positions through for the global capital-at-risk gate.
    from agents.consensus import seek_consensus
    consensus = seek_consensus(
        candidate, portfolio_value,
        cost_basis=position.get("cost_basis"),
        broker_positions=broker_positions,
    )
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

    # Success — update position and memory. Capture the drawer_id so
    # the outcome can later be linked back to the decision reasoning.
    decision_drawer_id = remember_trade_decision(
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
            # Linkage for the learning loop
            p["cc_decision_drawer_id"] = decision_drawer_id
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

    # Learning loop: resolve BOTH the original CSP decision (if linked)
    # and the CC decision with their respective outcomes. Best-effort:
    # any failure here is logged degraded and never blocks the cycle.
    try:
        from lib.memory_palace import (
            record_trade_outcome, reflect_on_outcome, search_memory,
        )
        # CC outcome — called away, full capital gain realized
        cc_drawer = position.get("cc_decision_drawer_id")
        if cc_drawer and cc_strike > 0:
            cc_return = capital_gain / (strike * 100) if strike > 0 else 0.0
            cc_outcome_id = record_trade_outcome(
                ticker=ticker,
                decision_drawer_id=cc_drawer,
                realized_return_pct=cc_return,
                holding_days=0,  # filled in below if opened_at available
                exit_reason="cc_called_away",
                final_pnl_dollars=capital_gain + cc_premiums * 100,
            )
            # Reflection on the CC decision
            originals = search_memory(ticker, wing=f"wing_{ticker.lower()}",
                                       hall="hall_facts", n_results=20)
            content = next((o.get("content", "") for o in originals
                            if o.get("drawer_id") == cc_drawer), "")
            if content:
                reflect_on_outcome(
                    decision_drawer_id=cc_drawer,
                    outcome_drawer_id=cc_outcome_id,
                    ticker=ticker,
                    decision_reasoning=content,
                    outcome_summary=(
                        f"Called away at ${cc_strike} from cost basis ${strike}. "
                        f"Cap gain ${capital_gain:.2f}, total cycle P&L ${total_pnl:.2f}."
                    ),
                )
        # CSP outcome — the original entry that started the whole cycle.
        # Marks it as "csp_assigned_then_called_away" so the learning
        # signal is "this assignment was good — got called away profitably".
        csp_drawer = position.get("decision_drawer_id")
        if csp_drawer:
            csp_return = total_pnl / (strike * 100) if strike > 0 else 0.0
            record_trade_outcome(
                ticker=ticker,
                decision_drawer_id=csp_drawer,
                realized_return_pct=csp_return,
                holding_days=0,
                exit_reason="csp_assigned_then_called_away",
                final_pnl_dollars=total_pnl,
            )
    except Exception as e:
        log_event("cc_engine", "outcome_resolution_failed",
                  {"ticker": ticker, "error": str(e)[:200]}, result="degraded")

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
