"""
Tests for lib/earnings_filter.py.

Per CLAUDE.md, the rule "NEVER sell options expiring through an earnings
date" is ABSOLUTE. The filter is the only enforcement, so we test:

  * earnings_veto returns True iff an event falls between today and the
    expiration (inclusive on both ends — same-day earnings are NOT safe)
  * earnings_veto fails OPEN (returns False) when the upstream errors,
    unless `strict=True`
  * has_earnings_before respects both the iso-string and date forms
  * days_to_next_earnings handles missing data gracefully and never returns
    a negative number
  * check_earnings produces the right reason strings for each branch
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from lib.finnhub_client import EarningsEvent
from lib.earnings_filter import (
    EarningsCheck,
    check_earnings,
    days_to_next_earnings,
    earnings_veto,
    has_earnings_before,
)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _iso(d: date) -> str:
    return d.isoformat()


def _event(ticker: str, on: date) -> EarningsEvent:
    return EarningsEvent(
        ticker=ticker.upper(),
        date=_iso(on),
        hour="bmo",
        eps_estimate=None,
        eps_actual=None,
        revenue_estimate=None,
        revenue_actual=None,
    )


# --------------------------------------------------------------------------
# days_to_next_earnings
# --------------------------------------------------------------------------

class TestDaysToNextEarnings:
    def test_returns_none_when_no_data(self):
        with patch("lib.earnings_filter.next_earnings_date", return_value=None):
            assert days_to_next_earnings("AAPL") is None

    def test_returns_calendar_days_until(self):
        future = _today() + timedelta(days=12)
        with patch("lib.earnings_filter.next_earnings_date",
                   return_value=_iso(future)):
            assert days_to_next_earnings("AAPL") == 12

    def test_clamps_to_zero_for_past_dates(self):
        past = _today() - timedelta(days=3)
        with patch("lib.earnings_filter.next_earnings_date",
                   return_value=_iso(past)):
            # max(0, -3) = 0 — never negative
            assert days_to_next_earnings("AAPL") == 0

    def test_garbage_iso_returns_none(self):
        with patch("lib.earnings_filter.next_earnings_date",
                   return_value="not-a-date"):
            assert days_to_next_earnings("AAPL") is None


# --------------------------------------------------------------------------
# has_earnings_before — the inclusive-window check
# --------------------------------------------------------------------------

class TestHasEarningsBefore:
    def test_no_events_returns_false(self):
        with patch("lib.earnings_filter.get_earnings_calendar", return_value=[]):
            assert has_earnings_before("AAPL", _today() + timedelta(days=30)) is False

    def test_event_before_target_returns_true(self):
        target = _today() + timedelta(days=30)
        event_day = _today() + timedelta(days=10)
        with patch("lib.earnings_filter.get_earnings_calendar",
                   return_value=[_event("AAPL", event_day)]):
            assert has_earnings_before("AAPL", target) is True

    def test_event_after_target_returns_false(self):
        target = _today() + timedelta(days=10)
        event_day = _today() + timedelta(days=20)
        with patch("lib.earnings_filter.get_earnings_calendar",
                   return_value=[_event("AAPL", event_day)]):
            assert has_earnings_before("AAPL", target) is False

    def test_event_exactly_on_expiration_returns_true(self):
        # CRITICAL: the docstring states "earnings on target_date itself
        # counts" — selling a May-16 put when earnings are May-16 is NOT safe.
        target = _today() + timedelta(days=14)
        with patch("lib.earnings_filter.get_earnings_calendar",
                   return_value=[_event("AAPL", target)]):
            assert has_earnings_before("AAPL", target) is True

    def test_event_today_returns_true(self):
        today = _today()
        target = today + timedelta(days=10)
        with patch("lib.earnings_filter.get_earnings_calendar",
                   return_value=[_event("AAPL", today)]):
            assert has_earnings_before("AAPL", target) is True

    def test_accepts_iso_string_target(self):
        target = _today() + timedelta(days=20)
        event_day = _today() + timedelta(days=5)
        with patch("lib.earnings_filter.get_earnings_calendar",
                   return_value=[_event("AAPL", event_day)]):
            assert has_earnings_before("AAPL", _iso(target)) is True

    def test_unparseable_target_returns_false(self):
        # Parse failure → cannot check → fail open at this layer.
        # earnings_veto is the gate that decides what to do with the answer.
        with patch("lib.earnings_filter.get_earnings_calendar", return_value=[]):
            assert has_earnings_before("AAPL", "garbage") is False

    def test_event_with_garbage_date_skipped(self):
        target = _today() + timedelta(days=30)
        bad = EarningsEvent(
            ticker="AAPL", date="not-a-date",
            hour="", eps_estimate=None, eps_actual=None,
            revenue_estimate=None, revenue_actual=None,
        )
        with patch("lib.earnings_filter.get_earnings_calendar",
                   return_value=[bad]):
            assert has_earnings_before("AAPL", target) is False


# --------------------------------------------------------------------------
# earnings_veto — the function the engines call
# --------------------------------------------------------------------------

class TestEarningsVeto:
    def test_no_earnings_no_veto(self, isolated_audit):
        with patch("lib.earnings_filter.get_earnings_calendar", return_value=[]):
            assert earnings_veto("AAPL", _today() + timedelta(days=30)) is False

    def test_earnings_in_window_vetos(self, isolated_audit):
        target = _today() + timedelta(days=30)
        event_day = _today() + timedelta(days=10)
        with patch("lib.earnings_filter.get_earnings_calendar",
                   return_value=[_event("AAPL", event_day)]):
            assert earnings_veto("AAPL", target) is True

    def test_lookup_exception_fails_open_by_default(self, isolated_audit):
        # Default strict=False → unknown means "assume no earnings"
        with patch("lib.earnings_filter.get_earnings_calendar",
                   side_effect=RuntimeError("finnhub down")):
            assert earnings_veto("AAPL", _today() + timedelta(days=30)) is False

    def test_lookup_exception_strict_mode_vetos(self, isolated_audit):
        # strict=True → unknown means "play it safe, veto"
        with patch("lib.earnings_filter.get_earnings_calendar",
                   side_effect=RuntimeError("finnhub down")):
            assert earnings_veto(
                "AAPL", _today() + timedelta(days=30), strict=True,
            ) is True

    def test_logs_audit_event_on_veto(self, isolated_audit):
        target = _today() + timedelta(days=30)
        event_day = _today() + timedelta(days=10)
        with patch("lib.earnings_filter.get_earnings_calendar",
                   return_value=[_event("AAPL", event_day)]):
            earnings_veto("AAPL", target)
        contents = isolated_audit.read_text()
        assert "earnings_filter" in contents
        assert "veto" in contents


# --------------------------------------------------------------------------
# check_earnings — diagnostic for CLI / dashboard
# --------------------------------------------------------------------------

class TestCheckEarnings:
    def test_no_data_returns_none_branch(self):
        with patch("lib.earnings_filter.next_earnings_date", return_value=None):
            result = check_earnings("AAPL", window_days=14)
        assert isinstance(result, EarningsCheck)
        assert result.next_earnings is None
        assert result.days_until is None
        assert result.has_earnings_in_window is False
        assert result.data_source == "none"
        assert result.reason == "no_data_or_no_scheduled_event"

    def test_event_within_window_flags_in_window(self):
        future = _today() + timedelta(days=7)
        with patch("lib.earnings_filter.next_earnings_date",
                   return_value=_iso(future)):
            result = check_earnings("AAPL", window_days=14)
        assert result.next_earnings == _iso(future)
        assert result.days_until == 7
        assert result.has_earnings_in_window is True
        assert result.data_source == "finnhub"
        assert "earnings_in_7_days" == result.reason

    def test_event_outside_window_not_in_window(self):
        future = _today() + timedelta(days=30)
        with patch("lib.earnings_filter.next_earnings_date",
                   return_value=_iso(future)):
            result = check_earnings("AAPL", window_days=14)
        assert result.has_earnings_in_window is False
        assert result.days_until == 30

    def test_garbage_date_returns_parse_failed(self):
        with patch("lib.earnings_filter.next_earnings_date",
                   return_value="not-a-date"):
            result = check_earnings("AAPL")
        assert result.next_earnings == "not-a-date"
        assert result.days_until is None
        assert result.has_earnings_in_window is False
        assert result.reason == "parse_failed"

    def test_uppercases_ticker(self):
        with patch("lib.earnings_filter.next_earnings_date", return_value=None):
            assert check_earnings("aapl").ticker == "AAPL"
