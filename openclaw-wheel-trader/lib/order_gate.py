"""
Order Gate — 3-step pipeline that every order must pass through.

Step 1: PROPOSE — Generate order intent, log it
Step 2: VALIDATE — Run circuit breakers + agent consensus
Step 3: EXECUTE — Only after steps 1 & 2 pass, send to Alpaca

No single function call can place an order. This is by design.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Literal

from lib.audit import log_event
from lib.circuit_breaker import run_all_checks, CircuitBreakerTripped
from lib.order_dedup import check_and_record


@dataclass
class OrderIntent:
    """Immutable order proposal. Created in Step 1, validated in Step 2."""
    ticker: str
    side: Literal["sell_to_open", "buy_to_close", "buy", "sell"]
    order_type: Literal["limit", "market"]
    asset_type: Literal["option", "equity", "crypto"]
    quantity: float                       # int for stock/option, float for crypto
    limit_price: float | None = None
    option_type: str | None = None       # "put" or "call"
    strike: float | None = None
    expiration: str | None = None        # YYYY-MM-DD
    notional: float | None = None         # crypto buys: dollar amount instead of qty
    reason: str = ""
    composite_score: int = 0             # Trend + Level + Signal (0-9)
    extended_hours: bool = False         # True to route through pre/post-market
    created_at: str = ""
    intent_hash: str = ""
    _validated: bool = False             # Set by step2_validate

    def __post_init__(self):
        # Validate quantity / notional BEFORE anything else (security audit
        # finding 2026-05-01 #1+#2). A NaN, Inf, or absurdly large value
        # would silently produce garbage in step2_validate's order_value
        # arithmetic and bypass circuit-breaker checks.
        try:
            qty = float(self.quantity)
        except (TypeError, ValueError):
            raise ValueError(
                f"OrderIntent.quantity must be numeric, got {self.quantity!r}"
            )
        if not (qty == qty):  # NaN check
            raise ValueError("OrderIntent.quantity is NaN")
        if qty <= 0:
            raise ValueError(
                f"OrderIntent.quantity must be > 0, got {qty}"
            )
        # Hard upper bound — no legitimate strategy needs > 1M units of
        # anything; this catches accidental garbage (Inf, encoding errors).
        if qty > 1_000_000:
            raise ValueError(
                f"OrderIntent.quantity exceeds sanity limit of 1e6, got {qty}"
            )

        if self.notional is not None:
            try:
                n = float(self.notional)
            except (TypeError, ValueError):
                raise ValueError(
                    f"OrderIntent.notional must be numeric, got {self.notional!r}"
                )
            if not (n == n):
                raise ValueError("OrderIntent.notional is NaN")
            if n <= 0:
                raise ValueError(
                    f"OrderIntent.notional must be > 0 if set, got {n}"
                )
            if n > 10_000_000:
                raise ValueError(
                    f"OrderIntent.notional exceeds sanity limit of $10M, got {n}"
                )

        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.intent_hash:
            # Hash to detect duplicate orders. Includes notional so a buy and
            # an immediately-following close (different qty) get distinct hashes.
            hash_input = (
                f"{self.ticker}:{self.side}:{self.strike}:{self.expiration}:"
                f"{self.quantity}:{self.notional}"
            )
            self.intent_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]


DUPLICATE_WINDOW_SECONDS = 60


def step1_propose(intent: OrderIntent) -> OrderIntent:
    """
    Step 1: PROPOSE — Log the intent. No execution happens here.
    Returns the intent for Step 2.

    Dedup uses a file-locked store (lib/order_dedup) so concurrent scan
    and monitor processes share the same view. The previous in-RAM dict
    only protected within one process and missed the BAC double-buy on
    2026-04-28.
    """
    is_dup, age = check_and_record(intent.intent_hash, DUPLICATE_WINDOW_SECONDS)
    if is_dup:
        log_event("order_gate", "duplicate_blocked", {
            "hash": intent.intent_hash,
            "ticker": intent.ticker,
            "seconds_since_last": round(age, 1),
        }, result="blocked")
        raise ValueError(
            f"Duplicate order detected for {intent.ticker} within {DUPLICATE_WINDOW_SECONDS}s"
        )

    log_event("order_gate", "step1_proposed", {
        "intent": asdict(intent),
    }, result="pending")

    return intent


def step2_validate(
    intent: OrderIntent,
    portfolio_value: float,
    current_daily_pnl: float,
    current_open_orders: int,
    last_loss_time: datetime | None = None,
    min_composite_score: int | None = None,
) -> bool:
    # 2026-06-08: was hardcoded 7 — same restriction as risk_agent + the
    # confirmation YAML key. THIRD copy of the same gate. With the
    # confirmation min_composite_score lowered to 3 for small-bankroll
    # wheel operation, this gate was silently re-imposing the 7 floor
    # and blocking every CSP (NIO scored 3 today, would have fired
    # without this gate). Now reads from wheel_strategy.yaml so all
    # three gates stay synchronized.
    if min_composite_score is None:
        try:
            import yaml as _y
            from pathlib import Path as _P
            with open(_P(__file__).resolve().parent.parent / "config" / "wheel_strategy.yaml") as _f:
                _s = _y.safe_load(_f) or {}
            min_composite_score = int(_s.get("confirmation", {}).get("min_composite_score", 3))
        except Exception:
            min_composite_score = 3
    """
    Step 2: VALIDATE — Run circuit breakers and score check.
    Raises on failure. Returns True on pass.
    """
    # Calculate order value
    if intent.asset_type == "option":
        order_value = (intent.strike or 0) * 100 * intent.quantity  # CSP collateral
    elif intent.asset_type == "crypto":
        # Crypto buys use notional (dollars). Sells use qty * limit_price (or 0).
        order_value = float(intent.notional) if intent.notional else (intent.limit_price or 0) * intent.quantity
    else:
        order_value = (intent.limit_price or 0) * intent.quantity

    # The `max_contracts_per_order` breaker applies ONLY to option orders
    # — for stocks/crypto, ``intent.quantity`` counts shares/units, not
    # contracts, and accidentally tripping the option ceiling was blocking
    # legitimate stock buys (e.g. 22 shares of AAL hitting the "max 2
    # contracts" gate on 2026-05-13). Pass 0 for non-option asset types
    # so the breaker becomes a no-op for them.
    contracts_arg = intent.quantity if intent.asset_type == "option" else 0

    # Run all circuit breaker checks
    try:
        run_all_checks(
            order_value=order_value,
            portfolio_value=portfolio_value,
            current_daily_pnl=current_daily_pnl,
            current_open_orders=current_open_orders,
            contracts=contracts_arg,
            last_loss_time=last_loss_time,
        )
    except CircuitBreakerTripped as e:
        log_event("order_gate", "step2_breaker_tripped", {
            "hash": intent.intent_hash,
            "reason": str(e),
        }, result="blocked")
        raise

    # Check composite score threshold (caller can lower for non-equity strategies
    # like crypto where the score scale differs).
    if intent.composite_score < min_composite_score:
        log_event("order_gate", "step2_low_score", {
            "hash": intent.intent_hash,
            "score": intent.composite_score,
            "required": min_composite_score,
        }, result="blocked")
        raise ValueError(
            f"Composite score {intent.composite_score} below minimum {min_composite_score}"
        )

    log_event("order_gate", "step2_validated", {
        "hash": intent.intent_hash,
        "order_value": order_value,
    }, result="success")

    intent._validated = True
    return True


def submit_close(intent: OrderIntent, alpaca_client) -> dict:
    """
    Submit a closing order (stop-loss exit, target hit, partial scale-out).

    Closes go through propose (dedup + log) and execute, but skip breaker
    validation: we never want a daily-loss or open-orders cap to block an
    exit. Dedup is preserved so a duplicate close request inside the
    DUPLICATE_WINDOW_SECONDS window still gets blocked.
    """
    if intent.side not in ("sell", "buy_to_close"):
        raise ValueError(
            f"submit_close only handles closing sides; got {intent.side}"
        )
    intent = step1_propose(intent)
    log_event("order_gate", "step2_skipped_close", {
        "hash": intent.intent_hash,
        "ticker": intent.ticker,
        "side": intent.side,
    }, result="success")
    intent._validated = True
    return step3_execute(intent, alpaca_client)


def step3_execute(intent: OrderIntent, alpaca_client) -> dict:
    """
    Step 3: EXECUTE — Send the validated order to Alpaca.
    This is the ONLY function that calls the broker.

    Args:
        intent: The validated OrderIntent (must have passed step2_validate)
        alpaca_client: The Alpaca API client instance

    Returns:
        Order response from Alpaca

    Raises:
        RuntimeError: If step2_validate was not called on this intent.
    """
    if not intent._validated:
        log_event("order_gate", "step3_not_validated", {
            "hash": intent.intent_hash,
            "ticker": intent.ticker,
        }, result="blocked")
        raise RuntimeError(
            "Cannot execute: OrderIntent was not validated by step2_validate. "
            "All orders must pass through the full propose → validate → execute pipeline."
        )

    log_event("order_gate", "step3_executing", {
        "hash": intent.intent_hash,
        "ticker": intent.ticker,
        "side": intent.side,
        "quantity": intent.quantity,
    }, result="pending")

    try:
        response = alpaca_client.submit_order(intent)

        log_event("order_gate", "step3_executed", {
            "hash": intent.intent_hash,
            "order_id": response.get("id", "unknown"),
            "status": response.get("status", "unknown"),
        }, result="success")

        return response

    except Exception as e:
        log_event("order_gate", "step3_failed", {
            "hash": intent.intent_hash,
            "error": str(e),
        }, result="failed")
        raise
