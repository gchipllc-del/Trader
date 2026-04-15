"""
PDT Guard — Pattern Day Trader protection.

FINRA rule: Accounts under $25,000 are limited to 3 day trades
in any rolling 5-business-day period. A "day trade" is buying and
selling the same security on the same day.

This module tracks round trips and blocks trades that would
violate the PDT rule, preventing account restriction.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

from lib.audit import log_event

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


class PDTViolation(Exception):
    """Raised when a trade would violate the PDT rule."""
    pass


def _load_settings() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_positions() -> list[dict]:
    if not POSITIONS_PATH.exists():
        return []
    with open(POSITIONS_PATH) as f:
        return json.load(f)


def count_day_trades(lookback_days: int = 5) -> int:
    """
    Count day trades in the last N business days.

    A day trade = opened and closed on the same calendar day.
    """
    positions = _load_positions()
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days + 2)  # +2 for weekends

    day_trades = 0
    for p in positions:
        if p.get("status") != "closed":
            continue

        opened = p.get("opened_at", "")
        closed = p.get("closed_at", "")
        if not opened or not closed:
            continue

        try:
            open_dt = datetime.fromisoformat(opened.replace("Z", "+00:00"))
            close_dt = datetime.fromisoformat(closed.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        if open_dt < cutoff:
            continue

        # Same calendar day = day trade
        if open_dt.date() == close_dt.date():
            day_trades += 1

    return day_trades


def check_pdt(portfolio_value: float = 0) -> dict:
    """
    Check PDT status.

    Returns: {
        "day_trades_used": int,
        "day_trades_remaining": int,
        "can_day_trade": bool,
        "warning": str | None,
    }
    """
    settings = _load_settings()
    pdt_config = settings.get("pdt", {})

    if not pdt_config.get("enabled", True):
        return {"day_trades_used": 0, "day_trades_remaining": 999, "can_day_trade": True, "warning": None}

    # PDT doesn't apply to accounts >= $25k
    if portfolio_value >= 25000:
        return {"day_trades_used": 0, "day_trades_remaining": 999, "can_day_trade": True, "warning": None}

    max_trades = pdt_config.get("max_day_trades_5d", 3)
    warning_at = pdt_config.get("warning_at", 2)

    used = count_day_trades()
    remaining = max(0, max_trades - used)
    can_trade = remaining > 0

    warning = None
    if used >= warning_at and remaining > 0:
        warning = f"PDT Warning: {used}/{max_trades} day trades used in 5 days. {remaining} remaining."
    elif not can_trade:
        warning = f"PDT LIMIT REACHED: {used}/{max_trades} day trades. Hold positions overnight."

    return {
        "day_trades_used": used,
        "day_trades_remaining": remaining,
        "can_day_trade": can_trade,
        "warning": warning,
    }


def guard_day_trade(ticker: str, portfolio_value: float = 0):
    """
    Call before executing a same-day sell.
    Raises PDTViolation if the trade would breach the limit.
    """
    status = check_pdt(portfolio_value)

    if not status["can_day_trade"]:
        log_event("pdt_guard", "blocked", {
            "ticker": ticker,
            "day_trades_used": status["day_trades_used"],
        }, result="blocked")
        raise PDTViolation(
            f"PDT limit reached ({status['day_trades_used']}/3 in 5 days). "
            f"Cannot day-trade {ticker}. Hold overnight."
        )

    if status["warning"]:
        log_event("pdt_guard", "warning", {
            "ticker": ticker,
            "warning": status["warning"],
        })
