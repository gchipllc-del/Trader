"""
KILL SWITCH — Emergency full liquidation.
Callable from Telegram, CLI, or cron failsafe.

Actions (in order):
1. Cancel all pending orders
2. Close all open positions at market
3. Halt all monitoring crons
4. Send emergency alert to Telegram
5. Log everything
"""

from lib.audit import log_event
from lib.alpaca_client import AlpacaClient


def activate_kill_switch(reason: str = "manual") -> dict:
    """
    Nuclear option. Closes everything. Use when:
    - Monitoring cron missed 10+ checks
    - Unrecoverable error detected
    - Manual emergency from Telegram
    - Anomalous trading activity detected
    """
    log_event("kill_switch", "activated", {
        "reason": reason,
    }, result="pending")

    results = {
        "reason": reason,
        "orders_cancelled": 0,
        "positions_closed": 0,
        "errors": [],
    }

    try:
        client = AlpacaClient()

        # Step 1: Cancel all pending orders
        try:
            results["orders_cancelled"] = client.cancel_all_orders()
        except Exception as e:
            results["errors"].append(f"cancel_orders: {e}")

        # Step 2: Close all positions at market
        try:
            results["positions_closed"] = client.close_all_positions()
        except Exception as e:
            results["errors"].append(f"close_positions: {e}")

    except Exception as e:
        results["errors"].append(f"client_init: {e}")

    # Step 3: Log final result
    status = "success" if not results["errors"] else "partial"
    log_event("kill_switch", "completed", results, result=status)

    # Step 4: TODO — Send Telegram alert (Sprint 3)
    # Step 5: TODO — Halt cron jobs (Sprint 3)

    return results


if __name__ == "__main__":
    import sys
    reason = sys.argv[1] if len(sys.argv) > 1 else "manual_cli"
    print(f"⚠️  ACTIVATING KILL SWITCH: {reason}")
    result = activate_kill_switch(reason)
    print(f"Orders cancelled: {result['orders_cancelled']}")
    print(f"Positions closed: {result['positions_closed']}")
    if result["errors"]:
        print(f"Errors: {result['errors']}")
    else:
        print("✅ Clean shutdown complete.")
