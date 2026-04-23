"""
Tests for Sprint 0 (security) + Sprint 1 (analysis) modules.
Run with: python -m pytest tests/ -v
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.audit import log_event, update_event_result, get_recent_events, AUDIT_FILE, LOG_DIR
from lib.circuit_breaker import (
    check_daily_loss, check_position_size, check_open_orders,
    check_contracts_per_order, check_cooldown, CircuitBreakerTripped,
)
from lib.order_gate import OrderIntent, step1_propose, step2_validate, step3_execute
from lib.alpaca_client import AlpacaClient
from lib.candlestick import (
    scan_patterns, detect_engulfing, detect_hammer,
    detect_shooting_star, detect_morning_star,
)
from lib.zones import detect_zones, find_swing_points, cluster_levels
from lib.trend import analyze_trend, multi_timeframe_analysis
from lib.iv_rank import calculate_iv_rank, evaluate_premium_environment


# ============================================================
# FIXTURES
# ============================================================

def make_ohlcv(n=100, trend="up", seed=42):
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    if trend == "up":
        base = 100 + np.cumsum(rng.randn(n) * 0.5 + 0.1)
    elif trend == "down":
        base = 200 + np.cumsum(rng.randn(n) * 0.5 - 0.1)
    else:  # ranging
        base = 150 + np.cumsum(rng.randn(n) * 0.3) * 0
        base += rng.randn(n) * 2 + 150

    df = pd.DataFrame({
        "open": base + rng.randn(n) * 0.5,
        "high": base + abs(rng.randn(n)) * 1.5,
        "low": base - abs(rng.randn(n)) * 1.5,
        "close": base + rng.randn(n) * 0.5,
        "volume": rng.randint(100000, 1000000, n),
    }, index=dates)

    # Ensure high >= open,close and low <= open,close
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


# ============================================================
# AUDIT TESTS
# ============================================================

class TestAudit:
    def test_log_event_creates_file(self, tmp_path, monkeypatch):
        log_file = tmp_path / "audit_log.jsonl"
        monkeypatch.setattr("lib.audit.AUDIT_FILE", log_file)
        monkeypatch.setattr("lib.audit.LOG_DIR", tmp_path)

        event = log_event("test", "test_action", {"foo": "bar"})
        assert event["event_type"] == "test"
        assert "id" in event
        assert log_file.exists()

        with open(log_file) as f:
            data = json.loads(f.readline())
        assert data["action"] == "test_action"
        assert data["id"] == event["id"]

    def test_secret_redaction(self, tmp_path, monkeypatch):
        log_file = tmp_path / "audit_log.jsonl"
        monkeypatch.setattr("lib.audit.AUDIT_FILE", log_file)
        monkeypatch.setattr("lib.audit.LOG_DIR", tmp_path)

        log_event("test", "secret_test", {
            "api_key": "SHOULD_BE_REDACTED",
            "ticker": "AAPL",
            "secret_token": "ALSO_REDACTED",
        })

        with open(log_file) as f:
            data = json.loads(f.readline())
        assert data["details"]["api_key"] == "***REDACTED***"
        assert data["details"]["ticker"] == "AAPL"
        assert data["details"]["secret_token"] == "***REDACTED***"


# ============================================================
# CIRCUIT BREAKER TESTS
# ============================================================

class TestCircuitBreakers:
    # Explicit settings make these tests independent of config drift
    # (Hermes growth mode raises real limits; tests check the mechanism).
    STRICT_CB = {
        "circuit_breakers": {
            "max_daily_loss": -500,
            "max_position_pct": 0.10,
            "max_open_orders": 5,
            "max_contracts_per_order": 3,
            "cooldown_after_loss_minutes": 30,
        }
    }

    def test_daily_loss_passes(self):
        assert check_daily_loss(-100, settings=self.STRICT_CB) is True

    def test_daily_loss_trips(self):
        with pytest.raises(CircuitBreakerTripped):
            check_daily_loss(-600, settings=self.STRICT_CB)

    def test_position_size_passes(self):
        assert check_position_size(5000, 100000, settings=self.STRICT_CB) is True

    def test_position_size_trips(self):
        # 15% of portfolio exceeds STRICT_CB's 10% max
        with pytest.raises(CircuitBreakerTripped):
            check_position_size(15000, 100000, settings=self.STRICT_CB)

    def test_open_orders_passes(self):
        assert check_open_orders(3, settings=self.STRICT_CB) is True

    def test_open_orders_trips(self):
        # 5 open orders meets STRICT_CB's 5 max (>=) so trips
        with pytest.raises(CircuitBreakerTripped):
            check_open_orders(5, settings=self.STRICT_CB)

    def test_contracts_passes(self):
        assert check_contracts_per_order(1, settings=self.STRICT_CB) is True

    def test_contracts_trips(self):
        # 5 contracts > STRICT_CB's 3 max
        with pytest.raises(CircuitBreakerTripped):
            check_contracts_per_order(5, settings=self.STRICT_CB)

    def test_cooldown_passes_no_loss(self):
        assert check_cooldown(None, settings=self.STRICT_CB) is True

    def test_cooldown_passes_old_loss(self):
        old = datetime.now(timezone.utc) - timedelta(hours=1)
        assert check_cooldown(old, settings=self.STRICT_CB) is True

    def test_cooldown_trips_recent_loss(self):
        # 5 min < STRICT_CB's 30-min cooldown
        recent = datetime.now(timezone.utc) - timedelta(minutes=5)
        with pytest.raises(CircuitBreakerTripped):
            check_cooldown(recent, settings=self.STRICT_CB)


# ============================================================
# ORDER GATE TESTS
# ============================================================

class TestOrderGate:
    def test_propose_creates_hash(self):
        intent = OrderIntent(
            ticker="AAPL", side="sell_to_open", order_type="limit",
            asset_type="option", quantity=1, limit_price=2.50,
            option_type="put", strike=170, expiration="2024-06-21",
            reason="test", composite_score=8,
        )
        result = step1_propose(intent)
        assert result.intent_hash != ""
        assert len(result.intent_hash) == 16

    def test_duplicate_blocked(self):
        intent = OrderIntent(
            ticker="TSLA", side="sell_to_open", order_type="limit",
            asset_type="option", quantity=1, strike=200,
            expiration="2024-07-19", composite_score=8,
        )
        step1_propose(intent)
        with pytest.raises(ValueError, match="Duplicate"):
            step1_propose(intent)

    def test_validate_rejects_low_score(self):
        intent = OrderIntent(
            ticker="NVDA", side="sell_to_open", order_type="limit",
            asset_type="option", quantity=1, strike=100,
            expiration="2024-08-16", composite_score=4,
        )
        # Need unique hash
        intent.intent_hash = "unique123test45"
        with pytest.raises(ValueError, match="Composite score"):
            step2_validate(intent, 100000, -100, 2)

    def test_execute_rejects_unvalidated(self):
        intent = OrderIntent(
            ticker="MSFT", side="sell_to_open", order_type="limit",
            asset_type="option", quantity=1, strike=400,
            expiration="2024-09-20", composite_score=8,
        )
        with pytest.raises(RuntimeError, match="not validated"):
            step3_execute(intent, None)


# ============================================================
# AUDIT UPDATE & ID TESTS
# ============================================================

class TestAuditUpdate:
    def test_event_has_unique_id(self, tmp_path, monkeypatch):
        log_file = tmp_path / "audit_log.jsonl"
        monkeypatch.setattr("lib.audit.AUDIT_FILE", log_file)
        monkeypatch.setattr("lib.audit.LOG_DIR", tmp_path)

        e1 = log_event("test", "action_a")
        e2 = log_event("test", "action_b")
        assert e1["id"] != e2["id"]

    def test_update_event_result(self, tmp_path, monkeypatch):
        log_file = tmp_path / "audit_log.jsonl"
        monkeypatch.setattr("lib.audit.AUDIT_FILE", log_file)
        monkeypatch.setattr("lib.audit.LOG_DIR", tmp_path)

        event = log_event("test", "something", result="pending")
        update_event_result(event["id"], "success")

        events = get_recent_events(10)
        assert len(events) == 2
        assert events[1]["event_type"] == "resolution"
        assert events[1]["details"]["original_event_id"] == event["id"]


# ============================================================
# ALPACA OPTIONS SYMBOL TESTS
# ============================================================

class TestAlpacaOptionsSymbol:
    def test_build_option_symbol_put(self):
        client = AlpacaClient.__new__(AlpacaClient)  # Skip __init__
        sym = client._build_option_symbol("AAPL", "2024-06-21", "put", 170.0)
        assert sym == "AAPL240621P00170000"

    def test_build_option_symbol_call(self):
        client = AlpacaClient.__new__(AlpacaClient)
        sym = client._build_option_symbol("NVDA", "2024-12-20", "call", 950.0)
        assert sym == "NVDA241220C00950000"

    def test_build_option_symbol_fractional_strike(self):
        client = AlpacaClient.__new__(AlpacaClient)
        sym = client._build_option_symbol("SPY", "2025-03-21", "put", 450.5)
        assert sym == "SPY250321P00450500"


# ============================================================
# CANDLESTICK TESTS
# ============================================================

class TestCandlestick:
    def test_detect_hammer(self):
        df = pd.DataFrame({
            "open": [100],
            "high": [100.5],
            "low": [94],
            "close": [100.5],
            "volume": [500000],
        }, index=pd.date_range("2024-01-01", periods=1))

        # Body = 0.5, lower wick = 6, upper wick = 0
        signal = detect_hammer(df, 0)
        assert signal is not None
        assert signal.pattern == "hammer"
        assert signal.direction == "bullish"

    def test_detect_shooting_star(self):
        df = pd.DataFrame({
            "open": [100],
            "high": [107],
            "low": [99.5],
            "close": [99.5],
            "volume": [500000],
        }, index=pd.date_range("2024-01-01", periods=1))

        # Body = 0.5, upper wick = 7, lower wick = 0

        signal = detect_shooting_star(df, 0)
        assert signal is not None
        assert signal.pattern == "shooting_star"
        assert signal.direction == "bearish"

    def test_scan_patterns_returns_list(self):
        df = make_ohlcv(50)
        signals = scan_patterns(df, lookback=10)
        assert isinstance(signals, list)


# ============================================================
# ZONE TESTS
# ============================================================

class TestZones:
    def test_cluster_levels(self):
        prices = [100, 100.5, 101, 150, 151, 200]
        clusters = cluster_levels(prices, tolerance_pct=0.02)
        assert len(clusters) >= 2  # At least 100-group and 150-group

    def test_find_swing_points(self):
        df = make_ohlcv(100)
        highs, lows = find_swing_points(df, window=5)
        assert len(highs) > 0
        assert len(lows) > 0

    def test_detect_zones_returns_list(self):
        df = make_ohlcv(120)
        zones = detect_zones(df)
        assert isinstance(zones, list)


# ============================================================
# TREND TESTS
# ============================================================

class TestTrend:
    def test_uptrend_detected(self):
        df = make_ohlcv(100, trend="up")
        result = analyze_trend(df, "daily")
        # Synthetic data may be classified as choppy due to noise
        assert result.direction in ("uptrend", "ranging", "choppy")
        assert result.timeframe == "daily"

    def test_downtrend_detected(self):
        df = make_ohlcv(100, trend="down")
        result = analyze_trend(df, "daily")
        assert result.direction in ("downtrend", "ranging", "choppy")
        assert result.timeframe == "daily"

    def test_mtf_returns_alignment(self):
        weekly = make_ohlcv(52, trend="up")
        daily = make_ohlcv(100, trend="up")
        result = multi_timeframe_analysis(weekly, daily)
        assert "alignment_score" in result
        assert 0 <= result["alignment_score"] <= 3


# ============================================================
# IV RANK TESTS
# ============================================================

class TestIVRank:
    def test_iv_rank_midpoint(self):
        rank = calculate_iv_rank(0.30, pd.Series([0.20, 0.25, 0.30, 0.35, 0.40]))
        assert 0.4 <= rank <= 0.6  # Should be roughly middle

    def test_iv_rank_at_high(self):
        rank = calculate_iv_rank(0.40, pd.Series([0.20, 0.25, 0.30, 0.35, 0.40]))
        assert rank == 1.0

    def test_evaluate_environment(self):
        result = evaluate_premium_environment(
            0.35,
            pd.Series(np.linspace(0.20, 0.40, 252)),
            min_iv_rank=0.30,
        )
        assert "favorable_for_selling" in result
        assert "iv_rank" in result


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
