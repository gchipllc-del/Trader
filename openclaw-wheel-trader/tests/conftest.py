"""
Shared pytest fixtures for the OpenClaw Wheel Trader test suite.

Goal: Each individual test file no longer needs to know how to:
  - put the project root on sys.path
  - redirect audit logs into a tmp directory
  - redirect data/positions.json into a tmp directory
  - synthesise OHLCV frames or WheelCandidate objects
  - build a fake Alpaca client

Anything we expect more than one test file to need lives here.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------
# Filesystem isolation
# --------------------------------------------------------------------------

@pytest.fixture
def isolated_audit(tmp_path, monkeypatch):
    """Redirect all audit-log writes to a tmp file for the duration of a test."""
    log_file = tmp_path / "audit_log.jsonl"
    monkeypatch.setattr("lib.audit.AUDIT_FILE", log_file)
    monkeypatch.setattr("lib.audit.LOG_DIR", tmp_path)
    return log_file


@pytest.fixture
def isolated_positions(tmp_path, monkeypatch):
    """Redirect positions.json reads/writes to a tmp file.

    Returns a callable that yields the tmp path so tests can assert on it.
    Every module that owns its own POSITIONS_PATH constant must be patched.
    """
    pos_file = tmp_path / "positions.json"
    pos_file.write_text("[]")

    for module_path in (
        "lib.csp_engine.POSITIONS_PATH",
        "lib.cc_engine.POSITIONS_PATH",
        "lib.pdt_guard.POSITIONS_PATH",
    ):
        monkeypatch.setattr(module_path, pos_file, raising=False)

    return pos_file


@pytest.fixture
def isolated_trade_history(tmp_path, monkeypatch):
    history = tmp_path / "trade_history.json"
    history.write_text("[]")
    monkeypatch.setattr("lib.cc_engine.TRADE_HISTORY_PATH", history, raising=False)
    return history


# --------------------------------------------------------------------------
# Settings fixtures
# --------------------------------------------------------------------------

PAPER_SETTINGS = {
    "mode": "paper",
    "live_migration_approved": False,
    "circuit_breakers": {
        "max_daily_loss": -500,
        "max_position_pct": 0.10,
        "max_open_orders": 5,
        "max_contracts_per_order": 3,
        "cooldown_after_loss_minutes": 30,
    },
    "pdt": {
        "enabled": True,
        "max_day_trades_5d": 3,
        "warning_at": 2,
    },
}


@pytest.fixture
def paper_settings():
    """Deep copy of canonical paper-mode settings — safe to mutate per-test."""
    import copy
    return copy.deepcopy(PAPER_SETTINGS)


@pytest.fixture
def stub_load_settings(monkeypatch, paper_settings):
    """Make every module that reads settings.yaml see the canonical paper config."""
    for target in (
        "lib.circuit_breaker._load_settings",
        "lib.pdt_guard._load_settings",
    ):
        monkeypatch.setattr(target, lambda s=paper_settings: s, raising=False)
    return paper_settings


# --------------------------------------------------------------------------
# Synthetic market data
# --------------------------------------------------------------------------

def _make_ohlcv(n=100, trend="up", seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    if trend == "up":
        base = 100 + np.cumsum(rng.randn(n) * 0.5 + 0.1)
    elif trend == "down":
        base = 200 + np.cumsum(rng.randn(n) * 0.5 - 0.1)
    else:
        base = 150 + rng.randn(n) * 2

    df = pd.DataFrame({
        "open": base + rng.randn(n) * 0.5,
        "high": base + abs(rng.randn(n)) * 1.5,
        "low": base - abs(rng.randn(n)) * 1.5,
        "close": base + rng.randn(n) * 0.5,
        "volume": rng.randint(100_000, 1_000_000, n),
    }, index=dates)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


@pytest.fixture
def make_ohlcv():
    return _make_ohlcv


# --------------------------------------------------------------------------
# Domain-object factories
# --------------------------------------------------------------------------

@pytest.fixture
def make_candidate():
    """Factory for valid WheelCandidate objects with sensible defaults."""
    from lib.screener import WheelCandidate

    def _factory(**overrides):
        defaults = dict(
            ticker="AAPL",
            trade_type="csp",
            strike=170.0,
            expiration="2099-12-31",  # far future to avoid earnings calendar edge cases
            premium=2.50,
            delta=-0.25,
            dte=35,
            annualized_return=0.18,
            trend_score=3,
            level_score=3,
            signal_score=2,
            composite_score=8,
            zone_level=168.0,
            zone_touches=4,
            iv_rank=0.55,
            candlestick_pattern="hammer",
            tradeable=True,
        )
        defaults.update(overrides)
        return WheelCandidate(**defaults)

    return _factory


# --------------------------------------------------------------------------
# Fake broker
# --------------------------------------------------------------------------

@pytest.fixture
def fake_alpaca_client():
    """A MagicMock with realistic return shapes for AlpacaClient methods."""
    client = MagicMock()
    client.get_account.return_value = {
        "portfolio_value": 50_000.0,
        "buying_power": 50_000.0,
        "cash": 50_000.0,
    }
    client.get_open_orders.return_value = []
    client.submit_order.return_value = {
        "id": "order-abc-123",
        "status": "accepted",
    }
    client.cancel_all_orders.return_value = 0
    client.close_all_positions.return_value = 0
    return client


# --------------------------------------------------------------------------
# Order-gate state hygiene
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_order_gate_dedupe():
    """Clear the in-process duplicate-intent cache between every test.

    The order_gate module holds a process-wide dict that blocks the same
    intent_hash within a 60s window. Without this autouse reset, a test
    proposing AAPL 170P would poison every later test that uses the same
    intent.
    """
    from lib import order_gate
    order_gate._recent_intents.clear()
    yield
    order_gate._recent_intents.clear()
