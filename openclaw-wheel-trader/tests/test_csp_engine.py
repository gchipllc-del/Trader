"""
Tests for lib/csp_engine.py — Cash-Secured Put execution pipeline.

The engine threads multiple subsystems together: agent consensus → order_gate
(propose / validate / execute) → broker → memory palace. We do NOT exercise the
real subsystems here — those have their own tests. Instead we mock at module
boundaries and verify that csp_engine:

  * vetoes earnings-window trades at execute time (defense-in-depth)
  * aborts cleanly when consensus rejects
  * surfaces step1/step2/step3 failures as None (no exception leaks)
  * persists positions only after a successful step3 execute
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import json
import pytest


@pytest.fixture
def patched_engine(monkeypatch, isolated_audit, isolated_positions):
    """csp_engine with all heavy collaborators stubbed.

    Yields a SimpleNamespace of the patches so individual tests can tweak
    return values.
    """
    from lib import csp_engine

    earnings_veto = MagicMock(return_value=False)
    seek_consensus = MagicMock(return_value={
        "approved": True,
        "decision": "EXECUTE",
        "blocking_agent": None,
        "reason": "",
    })
    remember_trade_decision = MagicMock()
    diary_write = MagicMock()

    monkeypatch.setattr("lib.csp_engine.earnings_veto", earnings_veto)
    monkeypatch.setattr("agents.consensus.seek_consensus", seek_consensus)
    monkeypatch.setattr("lib.csp_engine.remember_trade_decision", remember_trade_decision)
    monkeypatch.setattr("lib.csp_engine.diary_write", diary_write)

    return type("Patches", (), {
        "earnings_veto": earnings_veto,
        "seek_consensus": seek_consensus,
        "remember_trade_decision": remember_trade_decision,
        "diary_write": diary_write,
    })()


# --------------------------------------------------------------------------
# execute_csp — happy path
# --------------------------------------------------------------------------

class TestExecuteCspHappyPath:
    def test_passes_all_gates_and_persists_position(
        self, patched_engine, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import csp_engine

        # $170 strike × 100 = $17,000 collateral. Portfolio $200k → 8.5%, well
        # under both the 10% strict and 30% prod position-size caps.
        candidate = make_candidate()
        result = csp_engine.execute_csp(
            candidate=candidate,
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )

        assert result == {"id": "order-abc-123", "status": "accepted"}
        fake_alpaca_client.submit_order.assert_called_once()
        patched_engine.remember_trade_decision.assert_called_once()

        with open(isolated_positions) as f:
            stored = json.load(f)
        assert len(stored) == 1
        assert stored[0]["ticker"] == "AAPL"
        assert stored[0]["type"] == "csp"
        assert stored[0]["status"] == "open"
        assert stored[0]["strike"] == 170.0
        assert stored[0]["order_id"] == "order-abc-123"


# --------------------------------------------------------------------------
# execute_csp — earnings veto (CLAUDE.md ABSOLUTE rule)
# --------------------------------------------------------------------------

class TestEarningsVeto:
    def test_earnings_in_window_blocks_execute(
        self, patched_engine, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import csp_engine

        patched_engine.earnings_veto.return_value = True
        result = csp_engine.execute_csp(
            candidate=make_candidate(),
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )

        assert result is None
        fake_alpaca_client.submit_order.assert_not_called()
        patched_engine.seek_consensus.assert_not_called()
        # Position file untouched
        with open(isolated_positions) as f:
            assert json.load(f) == []


# --------------------------------------------------------------------------
# execute_csp — consensus rejection
# --------------------------------------------------------------------------

class TestConsensusVeto:
    def test_risk_agent_veto_aborts(
        self, patched_engine, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import csp_engine

        patched_engine.seek_consensus.return_value = {
            "approved": False,
            "decision": "VETOED",
            "blocking_agent": "risk_agent",
            "reason": "sector concentration",
        }
        result = csp_engine.execute_csp(
            candidate=make_candidate(),
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )
        assert result is None
        fake_alpaca_client.submit_order.assert_not_called()
        with open(isolated_positions) as f:
            assert json.load(f) == []


# --------------------------------------------------------------------------
# execute_csp — order_gate failures (each step short-circuits cleanly)
# --------------------------------------------------------------------------

class TestOrderGateFailures:
    def test_step1_duplicate_returns_none(
        self, patched_engine, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import csp_engine

        # First call succeeds — second call within 60s window must be blocked
        # by step1_propose's internal deduplication.
        candidate = make_candidate()
        first = csp_engine.execute_csp(
            candidate=candidate,
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )
        assert first is not None

        second = csp_engine.execute_csp(
            candidate=candidate,
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )
        assert second is None
        # Only the first call should have hit the broker
        assert fake_alpaca_client.submit_order.call_count == 1

    def test_step2_low_score_returns_none(
        self, patched_engine, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import csp_engine

        bad = make_candidate(composite_score=4)  # below 7/9 threshold
        result = csp_engine.execute_csp(
            candidate=bad,
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )
        assert result is None
        fake_alpaca_client.submit_order.assert_not_called()

    def test_step2_circuit_breaker_returns_none(
        self, patched_engine, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import csp_engine
        from lib.circuit_breaker import CircuitBreakerTripped

        with patch("lib.order_gate.run_all_checks",
                   side_effect=CircuitBreakerTripped("daily loss")):
            result = csp_engine.execute_csp(
                candidate=make_candidate(),
                client=fake_alpaca_client,
                portfolio_value=200_000,
                current_daily_pnl=-9_999,
                current_open_orders=0,
            )
        assert result is None
        fake_alpaca_client.submit_order.assert_not_called()

    def test_step3_broker_error_returns_none(
        self, patched_engine, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import csp_engine

        fake_alpaca_client.submit_order.side_effect = RuntimeError("alpaca 500")
        result = csp_engine.execute_csp(
            candidate=make_candidate(),
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )
        assert result is None
        # Position must NOT be persisted on broker failure
        with open(isolated_positions) as f:
            assert json.load(f) == []
        # remember_trade_decision must NOT fire on broker failure
        patched_engine.remember_trade_decision.assert_not_called()


# --------------------------------------------------------------------------
# OrderIntent shape — verify the engine builds a put with correct fields
# --------------------------------------------------------------------------

class TestOrderIntentShape:
    def test_intent_is_put_sell_to_open_one_contract(
        self, patched_engine, fake_alpaca_client, make_candidate, isolated_positions
    ):
        from lib import csp_engine

        candidate = make_candidate(strike=145.0, premium=1.85, expiration="2099-11-15")
        csp_engine.execute_csp(
            candidate=candidate,
            client=fake_alpaca_client,
            portfolio_value=200_000,
            current_daily_pnl=0,
            current_open_orders=0,
        )

        submitted = fake_alpaca_client.submit_order.call_args.args[0]
        assert submitted.option_type == "put"
        assert submitted.side == "sell_to_open"
        assert submitted.asset_type == "option"
        assert submitted.quantity == 1
        assert submitted.strike == 145.0
        assert submitted.limit_price == 1.85
        assert submitted.expiration == "2099-11-15"
        # step2_validate flips this to True; step3 enforces it
        assert submitted._validated is True
