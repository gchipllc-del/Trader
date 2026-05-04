"""
Tests for lib/kill_switch.py — emergency liquidation.

The kill switch is the last line of defense. It must:
  * always return a result dict, even on partial failure
  * never let one broker call's exception prevent the other from running
  * record an "activated" audit event before doing any work
  * record a "completed" audit event with status "success" only when no errors,
    "partial" otherwise
  * be invocable as a CLI script (smoke check)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_alpaca_class(monkeypatch):
    """Replace AlpacaClient inside kill_switch with a controllable factory."""
    instance = MagicMock()
    instance.cancel_all_orders.return_value = 7
    instance.close_all_positions.return_value = 4

    factory = MagicMock(return_value=instance)
    monkeypatch.setattr("lib.kill_switch.AlpacaClient", factory)
    return type("Box", (), {"factory": factory, "instance": instance})()


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------

class TestKillSwitchHappyPath:
    def test_full_success_reports_counts(self, fake_alpaca_class, isolated_audit):
        from lib.kill_switch import activate_kill_switch

        result = activate_kill_switch(reason="manual_test")
        assert result["reason"] == "manual_test"
        assert result["orders_cancelled"] == 7
        assert result["positions_closed"] == 4
        assert result["errors"] == []

        fake_alpaca_class.instance.cancel_all_orders.assert_called_once()
        fake_alpaca_class.instance.close_all_positions.assert_called_once()


# --------------------------------------------------------------------------
# Partial-failure modes — the kill switch MUST keep going.
# --------------------------------------------------------------------------

class TestKillSwitchResilience:
    def test_cancel_orders_failure_still_closes_positions(
        self, fake_alpaca_class, isolated_audit
    ):
        from lib.kill_switch import activate_kill_switch

        fake_alpaca_class.instance.cancel_all_orders.side_effect = RuntimeError("api 500")
        result = activate_kill_switch()

        # cancel failed but close still ran — that's the whole point
        fake_alpaca_class.instance.close_all_positions.assert_called_once()
        assert result["positions_closed"] == 4
        assert result["orders_cancelled"] == 0
        assert any("cancel_orders" in e for e in result["errors"])

    def test_close_positions_failure_still_returns_result(
        self, fake_alpaca_class, isolated_audit
    ):
        from lib.kill_switch import activate_kill_switch

        fake_alpaca_class.instance.close_all_positions.side_effect = RuntimeError("oof")
        result = activate_kill_switch()

        assert result["orders_cancelled"] == 7
        assert result["positions_closed"] == 0
        assert any("close_positions" in e for e in result["errors"])

    def test_both_failures_collected_independently(
        self, fake_alpaca_class, isolated_audit
    ):
        from lib.kill_switch import activate_kill_switch

        fake_alpaca_class.instance.cancel_all_orders.side_effect = RuntimeError("a")
        fake_alpaca_class.instance.close_all_positions.side_effect = RuntimeError("b")
        result = activate_kill_switch()

        assert result["orders_cancelled"] == 0
        assert result["positions_closed"] == 0
        assert len(result["errors"]) == 2

    def test_client_init_failure_does_not_raise(self, monkeypatch, isolated_audit):
        from lib import kill_switch

        # AlpacaClient() itself blows up — typical "API keys missing" scenario.
        # We must still return cleanly with the failure logged.
        monkeypatch.setattr(
            "lib.kill_switch.AlpacaClient",
            MagicMock(side_effect=RuntimeError("no keys")),
        )
        result = kill_switch.activate_kill_switch(reason="missing_keys")
        assert result["orders_cancelled"] == 0
        assert result["positions_closed"] == 0
        assert any("client_init" in e for e in result["errors"])


# --------------------------------------------------------------------------
# Audit trail — both 'activated' and 'completed' events must be written.
# --------------------------------------------------------------------------

class TestKillSwitchAuditTrail:
    def _read_events(self, log_path):
        return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]

    def test_writes_activated_event_first_then_completed(
        self, fake_alpaca_class, isolated_audit
    ):
        from lib.kill_switch import activate_kill_switch

        activate_kill_switch(reason="cron_failsafe")
        events = self._read_events(isolated_audit)
        actions = [e["action"] for e in events]
        assert actions[0] == "activated"
        assert "completed" in actions

    def test_completed_status_is_success_when_clean(
        self, fake_alpaca_class, isolated_audit
    ):
        from lib.kill_switch import activate_kill_switch

        activate_kill_switch()
        completed = [
            e for e in self._read_events(isolated_audit) if e["action"] == "completed"
        ][0]
        assert completed["result"] == "success"

    def test_completed_status_is_partial_when_errors_present(
        self, fake_alpaca_class, isolated_audit
    ):
        from lib.kill_switch import activate_kill_switch

        fake_alpaca_class.instance.cancel_all_orders.side_effect = RuntimeError("boom")
        activate_kill_switch()
        completed = [
            e for e in self._read_events(isolated_audit) if e["action"] == "completed"
        ][0]
        assert completed["result"] == "partial"
