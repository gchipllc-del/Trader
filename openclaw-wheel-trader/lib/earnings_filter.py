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
from lib.finnhub_client import (
    next_earnings_date,
    get_earnings_calendar,
    get_earnings_calendar_with_status,
)

# Fail-safe window: earnings happen ~quarterly (~90d). If we can't reach
# Finnhub AND the option expires within this window from now, the chance
# of an earnings event in that span is high enough that we veto rather than
# trade blind. Outside the window, fall back to caller's strict flag.
FAIL_SAFE_VETO_DAYS = 45


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

    Backward-compatible: returns False on lookup failure. Callers that need
    to distinguish "no earnings" from "lookup failed" should use
    `has_earnings_before_with_status` instead.
    """
    has, _ok = has_earnings_before_with_status(ticker, target_date, lookahead_days)
    return has


def has_earnings_before_with_status(
    ticker: str, target_date: str | date, lookahead_days: int = 90
) -> tuple[bool, bool]:
    """Returns (has_earnings_in_window, lookup_ok).

    lookup_ok=False means Finnhub did not respond — caller should treat
    a False has_earnings_in_window result as unconfirmed, not negative.
    """
    if isinstance(target_date, str):
        target = _parse_iso(target_date)
    else:
        target = target_date
    if not target:
        return False, True  # parse error is on the caller, not Finnhub

    today = datetime.now(timezone.utc).date()
    lookahead = min(lookahead_days, max(0, (target - today).days) + 1)

    events, lookup_ok = get_earnings_calendar_with_status(ticker, days_ahead=lookahead)
    for e in events:
        d = _parse_iso(e.date)
        if not d:
            continue
        if today <= d <= target:
            return True, lookup_ok
    return False, lookup_ok


def earnings_veto(ticker: str, expiration_date: str | date, strict: bool = False) -> bool:
    """Returns True to VETO an option sale (CSP/CC) due to earnings risk.

    Three outcomes:
      - Confirmed earnings in window → veto.
      - Confirmed no earnings in window → don't veto.
      - Lookup failed:
          * If expiration is within FAIL_SAFE_VETO_DAYS (~one earnings cycle),
            veto regardless of `strict` — the prior of an earnings event in
            45d is too high to trade blind.
          * If expiration is beyond that window, defer to `strict`
            (fail-open by default for far-dated trades where retry is cheap).

    Prior to 2026-04-29 this fail-open'd unconditionally; an outage during
    earnings season could route a CSP through an earnings gap. Tightened
    after the Wave 1 audit.
    """
    try:
        has_earn, lookup_ok = has_earnings_before_with_status(ticker, expiration_date)
    except Exception as e:
        log_event("earnings_filter", "lookup_failed",
                  {"ticker": ticker, "error": str(e)[:200]})
        return bool(strict)

    if has_earn:
        log_event("earnings_filter", "veto", {
            "ticker": ticker,
            "expiration": str(expiration_date),
            "reason": "earnings_in_window",
        })
        return True

    if not lookup_ok:
        # Fail-safe: how soon does the option expire?
        if isinstance(expiration_date, str):
            exp = _parse_iso(expiration_date)
        else:
            exp = expiration_date
        if exp:
            days_to_exp = (exp - datetime.now(timezone.utc).date()).days
            if days_to_exp <= FAIL_SAFE_VETO_DAYS:
                log_event("earnings_filter", "veto", {
                    "ticker": ticker,
                    "expiration": str(expiration_date),
                    "reason": "lookup_failed_within_fail_safe_window",
                    "days_to_exp": days_to_exp,
                    "fail_safe_days": FAIL_SAFE_VETO_DAYS,
                })
                return True
        # Far-dated: defer to strict flag
        log_event("earnings_filter", "lookup_unavailable", {
            "ticker": ticker,
            "expiration": str(expiration_date),
            "strict": strict,
        })
        return bool(strict)

    return False


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
