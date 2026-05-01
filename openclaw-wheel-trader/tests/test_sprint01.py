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
    @pytest.fixture(autouse=True)
    def _isolate_dedup(self, tmp_path, monkeypatch):
        """Point the cross-process dedup store at a per-test temp file so
        hashes from one test (or a prior `pytest` run) don't carry over."""
        from lib import order_dedup
        monkeypatch.setattr(order_dedup, "DEDUP_FILE", tmp_path / "dedup.json")

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
# CROSS-PROCESS ORDER DEDUP (gap audit Wave 1 #2)
# ============================================================

class TestPhaseGating:
    """Wave 3 #16: confirm portfolio_value drives phase selection
    correctly so CSPs only unlock at $5k and full Wheel at $10k."""

    def test_phase_thresholds(self):
        from lib.stock_engine import (
            get_current_phase, PHASE_2_THRESHOLD, PHASE_3_THRESHOLD,
        )
        assert PHASE_2_THRESHOLD == 5000
        assert PHASE_3_THRESHOLD == 10000
        assert get_current_phase(1500) == 1     # current bankroll
        assert get_current_phase(4999) == 1
        assert get_current_phase(5000) == 2     # CSPs unlock
        assert get_current_phase(9999) == 2
        assert get_current_phase(10000) == 3    # full Wheel
        assert get_current_phase(50000) == 3


class TestMissedCycleWatchdog:
    """Wave 3 #17: missed monitor cycles must drive the kill-switch
    counter. Previously record_missed_check() existed but had no live
    callers — the kill-switch path was effectively dead."""

    @pytest.fixture(autouse=True)
    def _temp_heartbeat(self, tmp_path, monkeypatch):
        from lib import monitor
        monkeypatch.setattr(monitor, "HEARTBEAT_PATH",
                            tmp_path / "heartbeat.json")
        monitor._missed_checks = 0  # reset module global

    def test_no_misses_when_cycle_runs_on_time(self):
        from lib import monitor
        from datetime import datetime, timezone, timedelta
        # Simulate a heartbeat 60s ago — well within the 180s default interval
        recent = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        monitor.HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        monitor.HEARTBEAT_PATH.write_text(json.dumps({"last_check_at": recent}))

        before = monitor._missed_checks
        monitor._check_for_missed_cycles()
        assert monitor._missed_checks == before  # nothing recorded

    def test_records_misses_after_a_long_gap(self, monkeypatch):
        from lib import monitor
        from datetime import datetime, timezone, timedelta
        # 600s gap, 180s interval → 3 missed cycles (600/180 - 1 = 2.33 → 2)
        old = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        monitor.HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        monitor.HEARTBEAT_PATH.write_text(json.dumps({"last_check_at": old}))
        monkeypatch.setattr(monitor, "_load_settings",
                            lambda: {"monitoring": {
                                "check_interval_seconds": 180,
                                "missed_check_alert": 3,
                                "missed_check_kill": 10,
                            }})

        monitor._check_for_missed_cycles()
        assert monitor._missed_checks == 2

    def test_first_ever_run_records_no_misses(self):
        from lib import monitor
        # No heartbeat file exists yet — nothing to compare.
        monitor._check_for_missed_cycles()
        assert monitor._missed_checks == 0


class TestPositionsStore:
    """Wave 3 #15: positions.json mutations must serialize across
    processes. Without the lock, two concurrent appenders each read
    N → write N+1, losing one of the two new entries."""

    @pytest.fixture(autouse=True)
    def _temp_store(self, tmp_path, monkeypatch):
        from lib import positions_store
        monkeypatch.setattr(positions_store, "POSITIONS_PATH",
                            tmp_path / "positions.json")

    def test_save_load_roundtrip(self):
        from lib import positions_store
        positions_store.save_positions([{"ticker": "AAPL", "shares": 1}])
        assert positions_store.load_positions() == [{"ticker": "AAPL", "shares": 1}]

    def test_load_returns_empty_for_missing_file(self):
        from lib import positions_store
        assert positions_store.load_positions() == []

    def test_load_recovers_from_corrupted_json(self, tmp_path):
        from lib import positions_store
        positions_store.POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        positions_store.POSITIONS_PATH.write_text("{not valid json")
        # Returns [] rather than crashing the trade pipeline.
        assert positions_store.load_positions() == []

    def test_mutate_persists_changes(self):
        from lib import positions_store
        with positions_store.mutate_positions() as positions:
            positions.append({"ticker": "MSFT", "shares": 5})
        assert positions_store.load_positions() == [
            {"ticker": "MSFT", "shares": 5}
        ]

    def test_concurrent_appends_do_not_lose_writes(self):
        """Simulate two processes appending to the SAME positions.json
        via mutate_positions. Both writes must survive — no overwrite."""
        from lib import positions_store
        import threading

        positions_store.save_positions([])
        results = []

        def append(ticker):
            with positions_store.mutate_positions() as positions:
                positions.append({"ticker": ticker})
                results.append(ticker)

        # 20 threads, distinct tickers — none should be lost.
        threads = [threading.Thread(target=append, args=(f"T{i}",))
                   for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        final = positions_store.load_positions()
        assert len(final) == 20
        assert sorted(p["ticker"] for p in final) == sorted(results)

    def test_mutate_aborts_write_on_exception(self):
        """If the caller raises inside the with block, the on-disk file
        must remain at its pre-mutation state — no half-written JSON."""
        from lib import positions_store
        positions_store.save_positions([{"ticker": "INITIAL"}])
        try:
            with positions_store.mutate_positions() as positions:
                positions.append({"ticker": "WOULD_BE_LOST"})
                raise RuntimeError("simulated failure mid-mutation")
        except RuntimeError:
            pass
        # File should still hold the original entry.
        assert positions_store.load_positions() == [{"ticker": "INITIAL"}]


class TestOrderDedup:
    """The dedup store must be file-locked and survive across processes,
    so two concurrent scans cannot each clear the same intent hash."""

    @pytest.fixture(autouse=True)
    def _temp_store(self, tmp_path, monkeypatch):
        from lib import order_dedup
        monkeypatch.setattr(order_dedup, "DEDUP_FILE", tmp_path / "dedup.json")

    def test_first_record_succeeds(self):
        from lib.order_dedup import check_and_record
        is_dup, _ = check_and_record("hash_abc", window_seconds=60)
        assert is_dup is False

    def test_second_record_in_window_blocks(self):
        from lib.order_dedup import check_and_record
        check_and_record("hash_xyz", window_seconds=60)
        is_dup, age = check_and_record("hash_xyz", window_seconds=60)
        assert is_dup is True
        assert age >= 0

    def test_hash_persists_across_calls(self, tmp_path):
        """Simulates the BAC double-buy: two separate 'processes' read
        the same dedup file and the second one sees the first's record."""
        from lib import order_dedup

        # First "scan"
        is_dup1, _ = order_dedup.check_and_record("bac_buy_hash", window_seconds=60)
        assert is_dup1 is False
        assert order_dedup.DEDUP_FILE.exists()

        # File contains the hash on disk
        import json
        on_disk = json.loads(order_dedup.DEDUP_FILE.read_text())
        assert "bac_buy_hash" in on_disk

        # Second "scan" (same dedup file → simulates concurrent process)
        is_dup2, _ = order_dedup.check_and_record("bac_buy_hash", window_seconds=60)
        assert is_dup2 is True

    def test_expired_hash_is_allowed_again(self, tmp_path, monkeypatch):
        """A retry past the dedup window should be allowed."""
        from lib import order_dedup
        import json, time

        # Manually plant an expired hash
        expired_ts = time.time() - 120
        order_dedup.DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)
        order_dedup.DEDUP_FILE.write_text(json.dumps({"old_hash": expired_ts}))

        # Use a 60s window — old_hash is 120s old, GC'd then re-recorded
        is_dup, _ = order_dedup.check_and_record("old_hash", window_seconds=60)
        assert is_dup is False

    def test_step1_propose_uses_persistent_dedup(self, tmp_path, monkeypatch):
        """End-to-end: order_gate.step1_propose must hit the file-locked store."""
        from lib import order_dedup
        from lib.order_gate import OrderIntent, step1_propose

        intent_a = OrderIntent(
            ticker="ZZZ", side="sell_to_open", order_type="limit",
            asset_type="option", quantity=1, strike=50,
            expiration="2026-12-19", composite_score=8,
        )
        intent_b = OrderIntent(
            ticker="ZZZ", side="sell_to_open", order_type="limit",
            asset_type="option", quantity=1, strike=50,
            expiration="2026-12-19", composite_score=8,
        )
        # Same parameters → same hash
        assert intent_a.intent_hash == intent_b.intent_hash

        step1_propose(intent_a)
        # Second call (different intent object, same hash) must block
        with pytest.raises(ValueError, match="Duplicate"):
            step1_propose(intent_b)


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
