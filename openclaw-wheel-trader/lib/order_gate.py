"""
Order Gate — 3-step pipeline that every order must pass through.

Step 1: PROPOSE — Generate order intent, log it
Step 2: VALIDATE — Run circuit breakers + agent consensus
Step 3: EXECUTE — Only after steps 1 & 2 pass, send to Alpaca

No single function call can place an order. This is by design.
"""

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Literal

from lib.audit import log_event
from lib.circuit_breaker import run_all_checks, CircuitBreakerTripped


@dataclass
class OrderIntent:
    """Immutable order proposal. Created in Step 1, validated in Step 2."""
    ticker: str
    side: Literal["sell_to_open", "buy_to_close", "buy", "sell"]
    order_type: Literal["limit", "market"]
    asset_type: Literal["option", "equity"]
    quantity: int
    limit_price: float | None = None
    option_type: str | None = None       # "put" or "call"
    strike: float | None = None
    expiration: str | None = None        # YYYY-MM-DD
    reason: str = ""
    composite_score: int = 0             # Trend + Level + Signal (0-9)
    created_at: str = ""
    intent_hash: str = ""
    _validated: bool = False             # Set by step2_validate

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.intent_hash:
            # Hash to detect duplicate orders
            hash_input = f"{self.ticker}:{self.side}:{self.strike}:{self.expiration}:{self.quantity}"
            self.intent_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]


# Track recent intent hashes to prevent duplicates
_recent_intents: dict[str, float] = {}  # hash -> timestamp
DUPLICATE_WINDOW_SECONDS = 60


def step1_propose(intent: OrderIntent) -> OrderIntent:
    """
    Step 1: PROPOSE — Log the intent. No execution happens here.
    Returns the intent for Step 2.
    """
    # Check for duplicate within window
    now = time.time()
    if intent.intent_hash in _recent_intents:
        last_time = _recent_intents[intent.intent_hash]
        if now - last_time < DUPLICATE_WINDOW_SECONDS:
            log_event("order_gate", "duplicate_blocked", {
                "hash": intent.intent_hash,
                "ticker": intent.ticker,
                "seconds_since_last": round(now - last_time, 1),
            }, result="blocked")
            raise ValueError(
                f"Duplicate order detected for {intent.ticker} within {DUPLICATE_WINDOW_SECONDS}s"
            )

    _recent_intents[intent.intent_hash] = now

    # Clean up old entries
    cutoff = now - DUPLICATE_WINDOW_SECONDS * 2
    expired = [h for h, t in _recent_intents.items() if t < cutoff]
    for h in expired:
        del _recent_intents[h]

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
) -> bool:
    """
    Step 2: VALIDATE — Run circuit breakers and score check.
    Raises on failure. Returns True on pass.
    """
    # Calculate order value
    if intent.asset_type == "option":
        order_value = (intent.strike or 0) * 100 * intent.quantity  # CSP collateral
    else:
        order_value = (intent.limit_price or 0) * intent.quantity

    # Run all circuit breaker checks
    try:
        run_all_checks(
            order_value=order_value,
            portfolio_value=portfolio_value,
            current_daily_pnl=current_daily_pnl,
            current_open_orders=current_open_orders,
            contracts=intent.quantity,
            last_loss_time=last_loss_time,
        )
    except CircuitBreakerTripped as e:
        log_event("order_gate", "step2_breaker_tripped", {
            "hash": intent.intent_hash,
            "reason": str(e),
        }, result="blocked")
        raise

    # Check composite score threshold
    if intent.composite_score < 7:
        log_event("order_gate", "step2_low_score", {
            "hash": intent.intent_hash,
            "score": intent.composite_score,
            "required": 7,
        }, result="blocked")
        raise ValueError(
            f"Composite score {intent.composite_score}/9 below minimum 7/9"
        )

    log_event("order_gate", "step2_validated", {
        "hash": intent.intent_hash,
        "order_value": order_value,
    }, result="success")

    intent._validated = True
    return True


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
