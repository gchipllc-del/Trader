"""
Boundary and integration tests for lib/circuit_breaker.py.

The original tests in test_sprint01.py cover the basic pass/fail behaviour of
each individual check. This file fills two gaps:

  1. Off-by-one boundaries — the "ABSOLUTE" rules in CLAUDE.md are stated
     as inequalities, and the failure mode of "exactly at the limit" is
     where bugs hide.
  2. check_paper_mode + run_all_checks — neither was exercised at all.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from lib.circuit_breaker import (
    CircuitBreakerTripped,
    check_contracts_per_order,
    check_cooldown,
    check_daily_loss,
    check_open_orders,
    check_paper_mode,
    check_position_size,
    run_all_checks,
)


STRICT = {
    "circuit_breakers": {
        "max_daily_loss": -500,
        "max_position_pct": 0.10,
        "max_open_orders": 5,
        "max_contracts_per_order": 3,
        "cooldown_after_loss_minutes": 30,
    }
}


# --------------------------------------------------------------------------
# Daily loss — `<=` boundary: exactly at limit MUST trip
# --------------------------------------------------------------------------

class TestDailyLossBoundary:
    def test_one_cent_above_limit_passes(self):
        # -$499.99 > -$500 → safe
        assert check_daily_loss(-499.99, settings=STRICT) is True

    def test_exactly_at_limit_trips(self):
        # The check is `pnl <= max_loss` so -$500 trips.
        with pytest.raises(CircuitBreakerTripped):
            check_daily_loss(-500.00, settings=STRICT)

    def test_one_cent_below_limit_trips(self):
        with pytest.raises(CircuitBreakerTripped):
            check_daily_loss(-500.01, settings=STRICT)

    def test_positive_pnl_passes(self):
        assert check_daily_loss(250.00, settings=STRICT) is True


# --------------------------------------------------------------------------
# Position size — `>` boundary: exactly at limit must NOT trip
# --------------------------------------------------------------------------

class TestPositionSizeBoundary:
    def test_just_under_limit_passes(self):
        # 9.99% of portfolio, max 10%
        assert check_position_size(999, 10_000, settings=STRICT) is True

    def test_exactly_at_limit_passes(self):
        # The check is `pct > max_pct` (strict) — exactly 10% is OK.
        assert check_position_size(1_000, 10_000, settings=STRICT) is True

    def test_just_over_limit_trips(self):
        with pytest.raises(CircuitBreakerTripped):
            check_position_size(1_001, 10_000, settings=STRICT)

    def test_zero_portfolio_value_blocks(self):
        # Division-by-zero guard: defaults to 100% of portfolio → trips
        with pytest.raises(CircuitBreakerTripped):
            check_position_size(1, 0, settings=STRICT)


# --------------------------------------------------------------------------
# Open orders — `>=` boundary: exactly at limit MUST trip
# --------------------------------------------------------------------------

class TestOpenOrdersBoundary:
    def test_one_below_limit_passes(self):
        assert check_open_orders(4, settings=STRICT) is True

    def test_exactly_at_limit_trips(self):
        # `count >= max_orders` → 5 of 5 trips.
        with pytest.raises(CircuitBreakerTripped):
            check_open_orders(5, settings=STRICT)

    def test_above_limit_trips(self):
        with pytest.raises(CircuitBreakerTripped):
            check_open_orders(99, settings=STRICT)


# --------------------------------------------------------------------------
# Contracts per order
# --------------------------------------------------------------------------

class TestContractsBoundary:
    def test_exactly_at_limit_passes(self):
        # `contracts > max` strict — exactly 3 is fine.
        assert check_contracts_per_order(3, settings=STRICT) is True

    def test_one_above_trips(self):
        with pytest.raises(CircuitBreakerTripped):
            check_contracts_per_order(4, settings=STRICT)


# --------------------------------------------------------------------------
# Cooldown
# --------------------------------------------------------------------------

class TestCooldownBoundary:
    def test_no_loss_passes(self):
        assert check_cooldown(None, settings=STRICT) is True

    def test_just_inside_cooldown_trips(self):
        # 29 minutes ago, cooldown is 30 → trips
        recent = datetime.now(timezone.utc) - timedelta(minutes=29)
        with pytest.raises(CircuitBreakerTripped):
            check_cooldown(recent, settings=STRICT)

    def test_just_outside_cooldown_passes(self):
        old = datetime.now(timezone.utc) - timedelta(minutes=31)
        assert check_cooldown(old, settings=STRICT) is True


# --------------------------------------------------------------------------
# Paper-mode gate — the *only* thing standing between "live: false" in
# settings.yaml and a real-money order. We ABSOLUTELY need tests on this.
# --------------------------------------------------------------------------

class TestPaperModeGate:
    def test_paper_mode_passes(self):
        check_paper_mode(settings={"mode": "paper", "live_migration_approved": False})

    def test_live_without_approval_blocks(self):
        with pytest.raises(CircuitBreakerTripped, match="Live trading not approved"):
            check_paper_mode(settings={"mode": "live", "live_migration_approved": False})

    def test_live_with_approval_passes(self):
        # The combination required for actual live trading.
        check_paper_mode(settings={"mode": "live", "live_migration_approved": True})

    def test_missing_mode_blocks(self):
        with pytest.raises(CircuitBreakerTripped):
            check_paper_mode(settings={})


# --------------------------------------------------------------------------
# run_all_checks — integration: any single failing check trips the whole
# pipeline; pre-trade order placement uses this exclusively.
# --------------------------------------------------------------------------

class TestRunAllChecks:
    OK_KW = dict(
        order_value=500,
        portfolio_value=100_000,
        current_daily_pnl=-50,
        current_open_orders=1,
        contracts=1,
        last_loss_time=None,
    )

    def _patch_settings(self, monkeypatch, **overrides):
        s = dict(mode="paper", live_migration_approved=False, **STRICT)
        s["circuit_breakers"] = {**STRICT["circuit_breakers"], **overrides}
        monkeypatch.setattr("lib.circuit_breaker._load_settings", lambda: s)

    def test_all_checks_pass(self, monkeypatch, isolated_audit):
        self._patch_settings(monkeypatch)
        assert run_all_checks(**self.OK_KW) is True

    def test_paper_mode_violation_short_circuits(self, monkeypatch, isolated_audit):
        bad = {"mode": "live", "live_migration_approved": False, **STRICT}
        monkeypatch.setattr("lib.circuit_breaker._load_settings", lambda: bad)
        with pytest.raises(CircuitBreakerTripped, match="Live trading"):
            run_all_checks(**self.OK_KW)

    def test_daily_loss_blocks_pipeline(self, monkeypatch, isolated_audit):
        self._patch_settings(monkeypatch)
        kw = {**self.OK_KW, "current_daily_pnl": -1_000}
        with pytest.raises(CircuitBreakerTripped):
            run_all_checks(**kw)

    def test_contracts_blocks_pipeline(self, monkeypatch, isolated_audit):
        self._patch_settings(monkeypatch)
        kw = {**self.OK_KW, "contracts": 99}
        with pytest.raises(CircuitBreakerTripped):
            run_all_checks(**kw)
