"""
Reject-path and audit-trail tests for lib/order_gate.py.

The existing happy-path tests cover hash creation, duplicate detection at the
60-second window, low score rejection, and the "must be validated" guard on
step3. This file fills in:

  * step1 only logs ("step1_proposed") and does NOT mutate broker state
  * step2 audit entries on each rejection path
  * step3 audit entry on broker exception bubbles up + trail is preserved
  * order_value math for both option (collateral) and equity legs
  * full propose → validate → execute path with a fake client
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lib.audit import get_recent_events
from lib.circuit_breaker import CircuitBreakerTripped
from lib.order_gate import (
    OrderIntent,
    step1_propose,
    step2_validate,
    step3_execute,
)


def _intent(**overrides) -> OrderIntent:
    base = dict(
        ticker="AAPL",
        side="sell_to_open",
        order_type="limit",
        asset_type="option",
        quantity=1,
        limit_price=2.50,
        option_type="put",
        strike=170.0,
        expiration="2099-12-31",
        reason="test",
        composite_score=8,
    )
    base.update(overrides)
    return OrderIntent(**base)


# --------------------------------------------------------------------------
# OrderIntent — hash determinism + uniqueness
# --------------------------------------------------------------------------

class TestIntentHash:
    def test_same_inputs_same_hash(self):
        a = _intent()
        b = _intent()
        assert a.intent_hash == b.intent_hash

    def test_different_strike_different_hash(self):
        assert _intent(strike=170).intent_hash != _intent(strike=175).intent_hash

    def test_different_expiration_different_hash(self):
        assert (
            _intent(expiration="2099-01-15").intent_hash
            != _intent(expiration="2099-02-15").intent_hash
        )

    def test_validated_flag_defaults_to_false(self):
        # Critical invariant — step3 trusts this.
        assert _intent()._validated is False


# --------------------------------------------------------------------------
# step1_propose — audit trail
# --------------------------------------------------------------------------

class TestStep1Propose:
    def test_logs_step1_proposed(self, isolated_audit):
        step1_propose(_intent())
        events = [json.loads(line) for line in isolated_audit.read_text().splitlines()]
        assert any(
            e["event_type"] == "order_gate" and e["action"] == "step1_proposed"
            for e in events
        )

    def test_duplicate_within_window_logs_blocked_event(self, isolated_audit):
        intent = _intent(ticker="DUPE-TEST")
        step1_propose(intent)

        with pytest.raises(ValueError, match="Duplicate"):
            step1_propose(intent)

        events = [json.loads(line) for line in isolated_audit.read_text().splitlines()]
        blocked = [
            e for e in events
            if e["event_type"] == "order_gate" and e["action"] == "duplicate_blocked"
        ]
        assert len(blocked) == 1
        assert blocked[0]["result"] == "blocked"


# --------------------------------------------------------------------------
# step2_validate — order_value math + reject paths
# --------------------------------------------------------------------------

class TestStep2OrderValueMath:
    def test_option_uses_strike_x_100_x_qty(self, monkeypatch, isolated_audit):
        captured = {}

        def fake_run(**kw):
            captured.update(kw)
            return True

        monkeypatch.setattr("lib.order_gate.run_all_checks", fake_run)

        intent = _intent(strike=150.0, quantity=2)
        intent._validated = False
        intent.intent_hash = "uniq-opt-math-1"
        step2_validate(intent, portfolio_value=200_000, current_daily_pnl=0,
                       current_open_orders=0)
        # CSP collateral = 150 * 100 * 2 = $30,000
        assert captured["order_value"] == 30_000

    def test_equity_uses_limit_x_qty(self, monkeypatch, isolated_audit):
        captured = {}

        def fake_run(**kw):
            captured.update(kw)
            return True

        monkeypatch.setattr("lib.order_gate.run_all_checks", fake_run)

        intent = _intent(
            asset_type="equity", side="buy", order_type="limit",
            limit_price=42.50, quantity=100, option_type=None, strike=None,
            expiration=None,
        )
        intent.intent_hash = "uniq-eq-math-1"
        step2_validate(intent, portfolio_value=200_000, current_daily_pnl=0,
                       current_open_orders=0)
        assert captured["order_value"] == 4_250.00


class TestStep2RejectPaths:
    def test_circuit_breaker_logs_step2_breaker(
        self, monkeypatch, isolated_audit
    ):
        monkeypatch.setattr(
            "lib.order_gate.run_all_checks",
            MagicMock(side_effect=CircuitBreakerTripped("daily loss")),
        )
        intent = _intent()
        intent.intent_hash = "uniq-breaker-1"

        with pytest.raises(CircuitBreakerTripped):
            step2_validate(intent, 100_000, 0, 0)

        events = [json.loads(l) for l in isolated_audit.read_text().splitlines()]
        breaker_events = [
            e for e in events if e["action"] == "step2_breaker_tripped"
        ]
        assert len(breaker_events) == 1
        assert breaker_events[0]["result"] == "blocked"
        # Validated flag must still be False after a rejection
        assert intent._validated is False

    def test_low_score_logs_step2_low_score(
        self, monkeypatch, isolated_audit
    ):
        monkeypatch.setattr("lib.order_gate.run_all_checks", lambda **k: True)
        intent = _intent(composite_score=3)
        intent.intent_hash = "uniq-lowscore-1"

        with pytest.raises(ValueError, match="Composite score"):
            step2_validate(intent, 100_000, 0, 0)

        events = [json.loads(l) for l in isolated_audit.read_text().splitlines()]
        score_events = [e for e in events if e["action"] == "step2_low_score"]
        assert len(score_events) == 1
        assert score_events[0]["details"]["score"] == 3
        assert score_events[0]["details"]["required"] == 7

    def test_validated_flag_set_only_on_success(
        self, monkeypatch, isolated_audit
    ):
        monkeypatch.setattr("lib.order_gate.run_all_checks", lambda **k: True)
        intent = _intent()
        intent.intent_hash = "uniq-ok-1"

        step2_validate(intent, 100_000, 0, 0)
        assert intent._validated is True


# --------------------------------------------------------------------------
# step3_execute — broker error preserves audit trail
# --------------------------------------------------------------------------

class TestStep3Execute:
    def test_broker_failure_logs_step3_failed_then_reraises(
        self, isolated_audit
    ):
        intent = _intent()
        intent._validated = True
        broker = MagicMock()
        broker.submit_order.side_effect = RuntimeError("rate limited")

        with pytest.raises(RuntimeError, match="rate limited"):
            step3_execute(intent, broker)

        events = [json.loads(l) for l in isolated_audit.read_text().splitlines()]
        failed = [e for e in events if e["action"] == "step3_failed"]
        assert len(failed) == 1
        assert failed[0]["result"] == "failed"
        # The "executing" pending event was logged BEFORE the failure
        executing = [e for e in events if e["action"] == "step3_executing"]
        assert len(executing) == 1

    def test_unvalidated_intent_logs_blocked_and_does_not_call_broker(
        self, isolated_audit
    ):
        intent = _intent()
        # _validated is False by default
        broker = MagicMock()

        with pytest.raises(RuntimeError, match="not validated"):
            step3_execute(intent, broker)

        broker.submit_order.assert_not_called()
        events = [json.loads(l) for l in isolated_audit.read_text().splitlines()]
        assert any(e["action"] == "step3_not_validated" for e in events)

    def test_success_records_step3_executed_with_order_id(
        self, isolated_audit
    ):
        intent = _intent()
        intent._validated = True
        broker = MagicMock()
        broker.submit_order.return_value = {"id": "abc-789", "status": "accepted"}

        result = step3_execute(intent, broker)
        assert result == {"id": "abc-789", "status": "accepted"}

        events = [json.loads(l) for l in isolated_audit.read_text().splitlines()]
        executed = [e for e in events if e["action"] == "step3_executed"]
        assert len(executed) == 1
        assert executed[0]["details"]["order_id"] == "abc-789"
        assert executed[0]["result"] == "success"


# --------------------------------------------------------------------------
# Full pipeline — propose → validate → execute
# --------------------------------------------------------------------------

class TestFullPipeline:
    def test_end_to_end_with_fake_broker(
        self, monkeypatch, fake_alpaca_client, isolated_audit
    ):
        monkeypatch.setattr("lib.order_gate.run_all_checks", lambda **k: True)
        intent = _intent(ticker="E2E", strike=100, expiration="2099-10-10")

        proposed = step1_propose(intent)
        step2_validate(proposed, 200_000, 0, 0)
        result = step3_execute(proposed, fake_alpaca_client)

        assert result == {"id": "order-abc-123", "status": "accepted"}
        fake_alpaca_client.submit_order.assert_called_once()
