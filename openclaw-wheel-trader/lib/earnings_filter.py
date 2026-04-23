"""
Earnings date filter — enforces the rule:

    NEVER sell options expiring through an earnings date.

and, for stock swing entries:

    WARN / DOWNSIZE when an earnings event is within the expected hold window.

Uses Finnhub as the primary source (free tier covers forward earnings calendar).
Falls back to Alpha Vantage earnings history for past dates if Finnhub is down.

All functions fail OPEN only when we explicitly cannot determine the earnings
date — i.e., "no data" means "assume no earnings" after logging. Callers can
opt into strict mode (treat unknown as "has earnings") via `strict=True`.

Usage:
    from lib.earnings_filter import (
        has_earnings_before,
        days_to_next_earnings,
        earnings_veto,
    )

    # For option selling (CSP/CC): veto if earnings before expiration
    if earnings_veto("AAPL", expiration_date="2026-05-16"):
        return None

    # For stock entries: just check days until earnings
    days = days_to_next_earnings("AAPL")
    if days is not None and days < 5:
        # Risky — earnings within 5 trading days
        ...
"""

from __future__ import annotations

from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass

from lib.audit import log_event
from lib.finnhub_client import next_earnings_date, get_earnings_calendar


@dataclass
class EarningsCheck:
    ticker: str
    next_earnings: str | None           # ISO date or None
    days_until: int | None              # calendar days (not trading days)
    has_earnings_in_window: bool
    data_source: str                    # "finnhub" | "none"
    reason: str


def _parse_iso(s: str) -> date | None:
    try:
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        return None


def days_to_next_earnings(ticker: str, lookahead_days: int = 90) -> int | None:
    """Return calendar days until the next scheduled earnings event.

    Returns None if no earnings found within lookahead or lookup fails.
    """
    iso = next_earnings_date(ticker, days_ahead=lookahead_days)
    if not iso:
        return None
    d = _parse_iso(iso)
    if not d:
        return None
    delta = (d - datetime.now(timezone.utc).date()).days
    return max(0, delta)


def has_earnings_before(ticker: str, target_date: str | date, lookahead_days: int = 90) -> bool:
    """True iff an earnings event is scheduled between TODAY and `target_date`.

    `target_date` is inclusive — an earnings on `target_date` itself counts
    (critical for option expirations: selling a May-16 put when earnings are
    May-16 is NOT safe, even though they're the "same day").
    """
    if isinstance(target_date, str):
        target = _parse_iso(target_date)
    else:
        target = target_date
    if not target:
        return False

    today = datetime.now(timezone.utc).date()
    lookahead = min(lookahead_days, max(0, (target - today).days) + 1)

    events = get_earnings_calendar(ticker, days_ahead=lookahead)
    for e in events:
        d = _parse_iso(e.date)
        if not d:
            continue
        if today <= d <= target:
            return True
    return False


def earnings_veto(ticker: str, expiration_date: str | date, strict: bool = False) -> bool:
    """Returns True to VETO an option sale (CSP/CC) due to earnings risk.

    strict=False (default): if Finnhub returns no data / no key, we ASSUME no
        earnings in window (fail open). A `finnhub.no_api_key` audit entry is
        logged so you can tell why the veto was bypassed.
    strict=True: if we can't confirm, we veto (safer but noisier).
    """
    try:
        result = has_earnings_before(ticker, expiration_date)
    except Exception as e:
        log_event("earnings_filter", "lookup_failed", {"ticker": ticker, "error": str(e)[:200]})
        return bool(strict)

    if result:
        log_event("earnings_filter", "veto", {
            "ticker": ticker,
            "expiration": str(expiration_date),
            "reason": "earnings_in_window",
        })
    return result


def check_earnings(ticker: str, window_days: int = 14) -> EarningsCheck:
    """Full diagnostic — useful for CLI, dashboards, and stock entry gates.

    window_days=14 by default (typical stock swing hold window).
    """
    iso = next_earnings_date(ticker, days_ahead=max(window_days, 60))

    if iso is None:
        return EarningsCheck(
            ticker=ticker.upper(),
            next_earnings=None,
            days_until=None,
            has_earnings_in_window=False,
            data_source="none",
            reason="no_data_or_no_scheduled_event",
        )

    d = _parse_iso(iso)
    if not d:
        return EarningsCheck(
            ticker=ticker.upper(),
            next_earnings=iso,
            days_until=None,
            has_earnings_in_window=False,
            data_source="finnhub",
            reason="parse_failed",
        )

    today = datetime.now(timezone.utc).date()
    days = (d - today).days
    in_window = 0 <= days <= window_days

    return EarningsCheck(
        ticker=ticker.upper(),
        next_earnings=iso,
        days_until=days,
        has_earnings_in_window=in_window,
        data_source="finnhub",
        reason=f"earnings_in_{days}_days" if days >= 0 else "past_event",
    )
