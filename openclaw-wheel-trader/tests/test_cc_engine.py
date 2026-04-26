"""
Tests for lib/cc_engine.py — Covered Call execution + wheel-cycle completion.

We exercise four behaviours:

  1. find_assigned_positions filters correctly
  2. check_dividend_conflict respects the ex-div window
  3. execute_cc passes the order gate, persists CC state, reduces cost basis
     by the premium (the source of "wheel cost-basis grind"), and short-
     circuits cleanly on every failure mode (earnings, dividend, consensus,
     gate, broker)
  4. handle_call_assignment completes the wheel — math (capital_gain +
     premiums), invalidates KG facts, and writes a trade-history record
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def patched_cc(monkeypatch, isolated_audit, isolated_positions, isolated_trade_history):
    earnings_veto = MagicMock(return_value=False)
    seek_consensus = MagicMock(return_value={
        "approved": True,
        "decision": "EXECUTE",
        "blocking_agent": None,
        "reason": "",
    })
    remember_trade_decision = MagicMock()
    diary_write = MagicMock()
    kg_invalidate = MagicMock()
    kg_add = MagicMock()
    send_alert = MagicMock()
    next_ex_div = MagicMock(return_value=None)

    monkeypatch.setattr("lib.cc_engine.earnings_veto", earnings_veto)
    monkeypatch.setattr("agents.consensus.seek_consensus", seek_consensus)
    monkeypatch.setattr("lib.cc_engine.remember_trade_decision", remember_trade_decision)
    monkeypatch.setattr("lib.cc_engine.diary_write", diary_write)
    monkeypatch.setattr("lib.cc_engine.kg_invalidate", kg_invalidate)
    monkeypatch.setattr("lib.cc_engine.kg_add", kg_add)
    monkeypatch.setattr("lib.finnhub_client.next_ex_dividend_date", next_ex_div)
    # send_alert lives in lib.monitor — cc_engine imports it lazily inside
    # handle_call_assignment, so patch the source.
    monkeypatch.setattr("lib.monitor.send_alert", send_alert)

    return type("CCPatches", (), {
        "earnings_veto": earnings_veto,
        "seek_consensus": seek_consensus,
        "remember_trade_decision": remember_trade_decision,
        "diary_write": diary_write,
        "kg_invalidate": kg_invalidate,
        "kg_add": kg_add,
        "send_alert": send_alert,
        "next_ex_div": next_ex_div,
    })()


# --------------------------------------------------------------------------
# find_assigned_positions
# --------------------------------------------------------------------------

class TestFindAssignedPositions:
    def test_returns_only_assigned_uncovered(self, isolated_positions):
        from lib import cc_engine

        positions = [
            {"ticker": "A", "status": "assigned", "assigned_shares": 100, "cc_active": False},
            {"ticker": "B", "status": "assigned", "assigned_shares": 100, "cc_active": True},   # already covered
            {"ticker": "C", "status": "open"},                                                    # CSP, not assigned
            {"ticker": "D", "status": "assigned", "assigned_shares": 50},                         # < 100 shares
        ]
        isolated_positions.write_text(json.dumps(positions))

        result = cc_engine.find_assigned_positions()
        assert [p["ticker"] for p in result] == ["A"]


# --------------------------------------------------------------------------
# check_dividend_conflict
# --------------------------------------------------------------------------

class TestDividendConflict:
    def test_no_ex_div_no_conflict(self, patched_cc):
        from lib import cc_engine

        patched_cc.next_ex_div.return_value = None
        assert cc_engine.check_dividend_conflict("AAPL", "2099-12-31") is False

    def test_ex_div_within_window_blocks(self, patched_cc):
        from lib import cc_engine
        from datetime import datetime, timezone, timedelta

        # ex-div in 5 days, expiration in 30 days → conflict
        ex_div = (datetime.now(timezone.utc) + timedelta(days=5)).date().isoformat()
        exp = (datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat()
        patched_cc.next_ex_div.return_value = ex_div
        assert cc_engine.check_dividend_conflict("AAPL", exp) is True

    def test_ex_div_after_expiration_no_conflict(self, patched_cc):
        from lib import cc_engine
        from datetime import datetime, timezone, timedelta

        exp = (datetime.now(timezone.utc) + timedelta(days=10)).date().isoformat()
        ex_div_late = (datetime.now(timezone.utc) + timedelta(days=20)).date().isoformat()
        patched_cc.next_ex_div.return_value = ex_div_late
        assert cc_engine.check_dividend_conflict("AAPL", exp) is False

    def test_unparseable_expiration_fails_open(self, patched_cc):
        from lib import cc_engine

        # Garbage date string — cannot enforce, so we must NOT block.
        assert cc_engine.check_dividend_conflict("AAPL", "not-a-date") is False

    def test_finnhub_exception_fails_open(self, patched_cc):
        from lib import cc_engine

        patched_cc.next_ex_div.side_effect = RuntimeError("finnhub down")
        # Fail open — we don't want a flaky upstream to halt all CCs
        assert cc_engine.check_dividend_conflict("AAPL", "2099-12-31") is False


# --------------------------------------------------------------------------
# execute_cc
# --------------------------------------------------------------------------

def _seed_assigned_position(positions_path, ticker="AAPL", cost_basis=170.0):
    pos = {
        "ticker": ticker,
        "type": "csp",
        "status": "assigned",
        "assigned_shares": 100,
        "cost_basis": cost_basis,
        "cc_active": False,
    }
    positions_path.write_text(json.dumps([pos]))
    return pos


class TestExecuteCcHappyPath:
    def test_successful_cc_persists_state_and_reduces_cost_basis(
        self, patched_cc, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import cc_engine

        position = _seed_assigned_position(isolated_positions, cost_basis=170.0)
        candidate = make_candidate(
            trade_type="cc",
            strike=180.0,        # above cost basis
            premium=2.50,
            expiration="2099-11-15",
        )

        result = cc_engine.execute_cc(
            candidate=candidate,
            position=position,
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )
        assert result == {"id": "order-abc-123", "status": "accepted"}

        submitted = fake_alpaca_client.submit_order.call_args.args[0]
        assert submitted.option_type == "call"
        assert submitted.side == "sell_to_open"
        assert submitted.strike == 180.0

        with open(isolated_positions) as f:
            stored = json.load(f)[0]
        assert stored["cc_active"] is True
        assert stored["cc_strike"] == 180.0
        assert stored["cc_premium"] == 2.50
        # Cost basis reduced by exactly the premium collected
        assert stored["cost_basis"] == pytest.approx(170.0 - 2.50)


class TestExecuteCcShortCircuits:
    def test_earnings_blocks_before_consensus(
        self, patched_cc, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import cc_engine

        position = _seed_assigned_position(isolated_positions)
        patched_cc.earnings_veto.return_value = True
        result = cc_engine.execute_cc(
            candidate=make_candidate(trade_type="cc"),
            position=position,
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )
        assert result is None
        patched_cc.seek_consensus.assert_not_called()
        fake_alpaca_client.submit_order.assert_not_called()

    def test_dividend_conflict_blocks(
        self, patched_cc, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import cc_engine
        from datetime import datetime, timezone, timedelta

        position = _seed_assigned_position(isolated_positions)
        ex = (datetime.now(timezone.utc) + timedelta(days=3)).date().isoformat()
        patched_cc.next_ex_div.return_value = ex
        exp = (datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat()

        result = cc_engine.execute_cc(
            candidate=make_candidate(trade_type="cc", expiration=exp),
            position=position,
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )
        assert result is None
        fake_alpaca_client.submit_order.assert_not_called()

    def test_consensus_veto_blocks(
        self, patched_cc, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import cc_engine

        position = _seed_assigned_position(isolated_positions)
        patched_cc.seek_consensus.return_value = {
            "approved": False,
            "decision": "VETOED",
            "blocking_agent": "compliance_agent",
            "reason": "wash sale",
        }
        result = cc_engine.execute_cc(
            candidate=make_candidate(trade_type="cc"),
            position=position,
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )
        assert result is None
        fake_alpaca_client.submit_order.assert_not_called()

    def test_broker_error_does_not_persist_cc_state(
        self, patched_cc, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import cc_engine

        position = _seed_assigned_position(isolated_positions, cost_basis=170.0)
        fake_alpaca_client.submit_order.side_effect = RuntimeError("boom")

        result = cc_engine.execute_cc(
            candidate=make_candidate(trade_type="cc", strike=180.0),
            position=position,
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )
        assert result is None
        # Cost basis MUST NOT be mutated when the broker call failed.
        with open(isolated_positions) as f:
            stored = json.load(f)[0]
        assert stored["cost_basis"] == 170.0
        assert stored.get("cc_active") in (False, None)


# --------------------------------------------------------------------------
# handle_call_assignment — wheel-cycle completion math
# --------------------------------------------------------------------------

class TestHandleCallAssignment:
    def test_completes_wheel_with_correct_pnl(
        self, patched_cc, isolated_positions, isolated_trade_history
    ):
        from lib import cc_engine

        # Wheel: sold $170 put for $2.00, got assigned, sold $180 call for $2.50.
        # Capital gain = (180 - 170) * 100 = $1,000
        # Premiums = (2.00 + 2.50) * 100 = $450
        # Total P/L = $1,450
        position = {
            "ticker": "AAPL",
            "status": "assigned",
            "assigned_shares": 100,
            "strike": 170.0,
            "premium_collected": 2.00,
            "cc_strike": 180.0,
            "cc_premium": 2.50,
            "cost_basis": 167.5,
        }
        isolated_positions.write_text(json.dumps([position]))

        cc_engine.handle_call_assignment("AAPL", position)

        with open(isolated_positions) as f:
            updated = json.load(f)[0]
        assert updated["status"] == "completed"
        assert updated["total_pnl"] == pytest.approx(1450.0)
        assert "completed_at" in updated

        with open(isolated_trade_history) as f:
            history = json.load(f)
        assert len(history) == 1
        record = history[0]
        assert record["type"] == "wheel_cycle"
        assert record["capital_gain"] == pytest.approx(1000.0)
        assert record["total_pnl"] == pytest.approx(1450.0)
        assert record["put_strike"] == 170.0
        assert record["call_strike"] == 180.0

        # Knowledge graph: old "assigned" fact invalidated, "wheel_completed" added
        patched_cc.kg_invalidate.assert_called_once()
        patched_cc.kg_add.assert_called_once()
        # Telegram alert fires
        patched_cc.send_alert.assert_called_once()
