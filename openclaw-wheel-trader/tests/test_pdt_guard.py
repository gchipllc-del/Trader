"""
Tests for lib/pdt_guard.py — FINRA Pattern Day Trader rule enforcement.

Rules under test:
  * Accounts >= $25k are exempt (return "999 remaining", no veto)
  * `pdt.enabled: false` short-circuits to exempt
  * count_day_trades counts only positions that opened-and-closed on the
    same calendar day, within the lookback window
  * check_pdt produces the right warning at 2/3 used and at-limit
  * guard_day_trade raises PDTViolation when at the limit
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import pytest


def _seed_positions(path, day_trade_count=0, ticker="AAPL"):
    """Write `day_trade_count` same-day round trips into a positions file.

    We anchor the timestamps to noon UTC so subtracting/adding a few hours
    keeps both open_at and close_at on the same calendar day — regardless
    of when the test happens to run (a test running just after midnight
    would otherwise see open at 22:00 yesterday, close at 01:00 today).
    """
    today_noon = datetime.now(timezone.utc).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    positions = []
    for _ in range(day_trade_count):
        opened = (today_noon - timedelta(hours=2)).isoformat()
        closed = (today_noon + timedelta(hours=2)).isoformat()
        positions.append({
            "ticker": ticker,
            "status": "closed",
            "opened_at": opened,
            "closed_at": closed,
        })
    path.write_text(json.dumps(positions))


# --------------------------------------------------------------------------
# count_day_trades
# --------------------------------------------------------------------------

class TestCountDayTrades:
    def test_zero_when_no_positions(self, isolated_positions, stub_load_settings):
        from lib.pdt_guard import count_day_trades
        assert count_day_trades() == 0

    def test_counts_same_day_round_trips(self, isolated_positions, stub_load_settings):
        from lib.pdt_guard import count_day_trades
        _seed_positions(isolated_positions, day_trade_count=2)
        assert count_day_trades() == 2

    def test_ignores_overnight_holds(self, isolated_positions, stub_load_settings):
        from lib.pdt_guard import count_day_trades

        today = datetime.now(timezone.utc)
        positions = [{
            "ticker": "X",
            "status": "closed",
            # Opened yesterday, closed today — NOT a day trade.
            "opened_at": (today - timedelta(days=1)).isoformat(),
            "closed_at": today.isoformat(),
        }]
        isolated_positions.write_text(json.dumps(positions))
        assert count_day_trades() == 0

    def test_ignores_open_positions(self, isolated_positions, stub_load_settings):
        from lib.pdt_guard import count_day_trades

        positions = [{
            "ticker": "X",
            "status": "open",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }]
        isolated_positions.write_text(json.dumps(positions))
        assert count_day_trades() == 0

    def test_drops_records_outside_lookback(self, isolated_positions, stub_load_settings):
        from lib.pdt_guard import count_day_trades

        old = datetime.now(timezone.utc) - timedelta(days=30)
        positions = [{
            "ticker": "X",
            "status": "closed",
            "opened_at": old.isoformat(),
            "closed_at": old.isoformat(),
        }]
        isolated_positions.write_text(json.dumps(positions))
        assert count_day_trades() == 0

    def test_unparseable_dates_skipped(self, isolated_positions, stub_load_settings):
        from lib.pdt_guard import count_day_trades

        positions = [{
            "ticker": "X",
            "status": "closed",
            "opened_at": "garbage",
            "closed_at": "also garbage",
        }]
        isolated_positions.write_text(json.dumps(positions))
        assert count_day_trades() == 0


# --------------------------------------------------------------------------
# check_pdt
# --------------------------------------------------------------------------

class TestCheckPdt:
    def test_exempt_when_disabled(self, isolated_positions, paper_settings, monkeypatch):
        from lib import pdt_guard

        paper_settings["pdt"]["enabled"] = False
        monkeypatch.setattr(pdt_guard, "_load_settings", lambda: paper_settings)

        # Even with 5 same-day round trips, disabled means no veto
        _seed_positions(isolated_positions, day_trade_count=5)
        result = pdt_guard.check_pdt(portfolio_value=1_000)
        assert result["can_day_trade"] is True
        assert result["day_trades_remaining"] == 999
        assert result["warning"] is None

    def test_exempt_when_portfolio_above_25k(
        self, isolated_positions, stub_load_settings
    ):
        from lib.pdt_guard import check_pdt

        _seed_positions(isolated_positions, day_trade_count=5)
        result = check_pdt(portfolio_value=25_000)
        assert result["can_day_trade"] is True
        assert result["day_trades_remaining"] == 999

    def test_exempt_boundary_exactly_25k(self, isolated_positions, stub_load_settings):
        # The rule is `>= 25000` — exactly $25k must be exempt.
        from lib.pdt_guard import check_pdt
        _seed_positions(isolated_positions, day_trade_count=5)
        assert check_pdt(portfolio_value=25_000.00)["can_day_trade"] is True

    def test_no_warning_below_threshold(self, isolated_positions, stub_load_settings):
        from lib.pdt_guard import check_pdt

        _seed_positions(isolated_positions, day_trade_count=1)
        result = check_pdt(portfolio_value=10_000)
        assert result["day_trades_used"] == 1
        assert result["day_trades_remaining"] == 2
        assert result["can_day_trade"] is True
        assert result["warning"] is None

    def test_warning_at_threshold(self, isolated_positions, stub_load_settings):
        from lib.pdt_guard import check_pdt

        _seed_positions(isolated_positions, day_trade_count=2)
        result = check_pdt(portfolio_value=10_000)
        assert result["day_trades_used"] == 2
        assert result["day_trades_remaining"] == 1
        assert result["can_day_trade"] is True
        assert result["warning"] is not None
        assert "2/3" in result["warning"]

    def test_blocked_at_limit(self, isolated_positions, stub_load_settings):
        from lib.pdt_guard import check_pdt

        _seed_positions(isolated_positions, day_trade_count=3)
        result = check_pdt(portfolio_value=10_000)
        assert result["day_trades_used"] == 3
        assert result["day_trades_remaining"] == 0
        assert result["can_day_trade"] is False
        assert "LIMIT REACHED" in result["warning"]


# --------------------------------------------------------------------------
# guard_day_trade
# --------------------------------------------------------------------------

class TestGuardDayTrade:
    def test_passes_when_under_limit(
        self, isolated_positions, isolated_audit, stub_load_settings
    ):
        from lib.pdt_guard import guard_day_trade
        _seed_positions(isolated_positions, day_trade_count=1)
        # Should not raise
        guard_day_trade("AAPL", portfolio_value=10_000)

    def test_raises_at_limit(
        self, isolated_positions, isolated_audit, stub_load_settings
    ):
        from lib.pdt_guard import guard_day_trade, PDTViolation
        _seed_positions(isolated_positions, day_trade_count=3)
        with pytest.raises(PDTViolation, match="PDT limit"):
            guard_day_trade("TSLA", portfolio_value=10_000)

    def test_passes_when_exempt_by_balance(
        self, isolated_positions, isolated_audit, stub_load_settings
    ):
        from lib.pdt_guard import guard_day_trade
        _seed_positions(isolated_positions, day_trade_count=10)
        # Above $25k → no PDT regardless of count
        guard_day_trade("NVDA", portfolio_value=100_000)
