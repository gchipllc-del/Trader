"""
Tests for Sprint 2 (CSP Engine), Sprint 3 (Monitor), Sprint 4 (CC Engine),
Sprint 6 (Backtest), Sprint 7 (Agents/Consensus), Sprint 8 (Enhancements),
Sprint 9 (main.py), and MemPalace.

Run with: python -m pytest tests/test_sprint02_09.py -v
"""

import json
import os
import sys
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.order_gate import OrderIntent
from lib.screener import WheelCandidate


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
    else:
        base = 150 + rng.randn(n) * 2 + 150

    df = pd.DataFrame({
        "open": base + rng.randn(n) * 0.5,
        "high": base + abs(rng.randn(n)) * 1.5,
        "low": base - abs(rng.randn(n)) * 1.5,
        "close": base + rng.randn(n) * 0.5,
        "volume": rng.randint(100000, 1000000, n),
    }, index=dates)

    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


def make_candidate(ticker="AAPL", trade_type="csp", strike=170, score=8):
    """Create a test WheelCandidate."""
    return WheelCandidate(
        ticker=ticker,
        trade_type=trade_type,
        strike=strike,
        expiration="2024-06-21",
        premium=3.50,
        delta=-0.25,
        dte=35,
        annualized_return=0.15,
        trend_score=3,
        level_score=2,
        signal_score=score - 5,
        composite_score=score,
        zone_level=168.0,
        zone_touches=3,
        iv_rank=0.45,
        candlestick_pattern="hammer",
        tradeable=True,
    )


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with required files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    palace_dir = data_dir / "palace"
    palace_dir.mkdir()
    diaries_dir = palace_dir / "diaries"
    diaries_dir.mkdir()

    # Empty positions
    (data_dir / "positions.json").write_text("[]")
    (data_dir / "trade_history.json").write_text("[]")

    return tmp_path


# ============================================================
# SPRINT 2: CSP ENGINE TESTS
# ============================================================

class TestCSPEngine:
    def test_scan_skips_existing_csp(self, tmp_path, monkeypatch):
        """Should skip tickers with existing open CSPs."""
        from lib import csp_engine

        pos_file = tmp_path / "positions.json"
        pos_file.write_text(json.dumps([{
            "ticker": "AAPL", "type": "csp", "status": "open",
            "strike": 170, "expiration": "2024-06-21",
        }]))
        monkeypatch.setattr(csp_engine, "POSITIONS_PATH", pos_file)

        # Mock dependencies
        monkeypatch.setattr("lib.csp_engine.recall_ticker_history", lambda t: {})
        monkeypatch.setattr("lib.csp_engine.get_current_regime", lambda: "bull")

        client = MagicMock()
        daily = {"AAPL": make_ohlcv()}
        weekly = {"AAPL": make_ohlcv(52)}
        chains = {"AAPL": [{"strike": 165, "delta": -0.25, "dte": 35,
                            "bid": 3.0, "ask": 3.10, "open_interest": 500,
                            "expiration": "2024-06-21"}]}
        iv = {"AAPL": {"iv_rank": 0.5, "favorable_for_selling": True}}

        result = csp_engine.scan_for_csps(client, daily, weekly, chains, iv)
        # AAPL should be skipped — already has open CSP
        aapl_candidates = [c for c in result if c.ticker == "AAPL"]
        assert len(aapl_candidates) == 0

    def test_execute_csp_uses_order_gate(self, tmp_path, monkeypatch):
        """Verify execute_csp goes through consensus → propose → validate → execute."""
        from lib import csp_engine

        pos_file = tmp_path / "positions.json"
        pos_file.write_text("[]")
        monkeypatch.setattr(csp_engine, "POSITIONS_PATH", pos_file)

        # Mock consensus gate — approve everything
        import agents.consensus
        monkeypatch.setattr(agents.consensus, "seek_consensus",
                            lambda c, pv, **kw: {"approved": True, "decision": "APPROVED"})

        # Track order gate calls
        calls = []
        monkeypatch.setattr("lib.csp_engine.step1_propose",
                            lambda i: (calls.append("propose"), setattr(i, '_validated', False), i)[-1])
        monkeypatch.setattr("lib.csp_engine.step2_validate",
                            lambda **kw: (calls.append("validate"), setattr(kw['intent'], '_validated', True), True)[-1])
        monkeypatch.setattr("lib.csp_engine.step3_execute",
                            lambda i, c: (calls.append("execute"), {"id": "test123", "status": "accepted"})[-1])
        monkeypatch.setattr("lib.csp_engine.remember_trade_decision", lambda **kw: None)
        monkeypatch.setattr("lib.csp_engine.diary_write", lambda a, e: None)

        candidate = make_candidate()
        client = MagicMock()

        result = csp_engine.execute_csp(
            candidate, client, portfolio_value=100000,
            current_daily_pnl=0, current_open_orders=0,
        )

        assert result is not None
        assert calls == ["propose", "validate", "execute"]

    def test_intent_uses_underlying_ticker(self, tmp_path, monkeypatch):
        """OrderIntent.ticker should be the underlying, not OCC symbol."""
        from lib import csp_engine

        pos_file = tmp_path / "positions.json"
        pos_file.write_text("[]")
        monkeypatch.setattr(csp_engine, "POSITIONS_PATH", pos_file)

        # Mock consensus gate — approve everything
        import agents.consensus
        monkeypatch.setattr(agents.consensus, "seek_consensus",
                            lambda c, pv, **kw: {"approved": True, "decision": "APPROVED"})

        captured_intent = []
        def mock_propose(intent):
            captured_intent.append(intent)
            return intent

        monkeypatch.setattr("lib.csp_engine.step1_propose", mock_propose)
        monkeypatch.setattr("lib.csp_engine.step2_validate",
                            lambda **kw: (_ for _ in ()).throw(ValueError("test stop")))
        monkeypatch.setattr("lib.csp_engine.diary_write", lambda a, e: None)

        candidate = make_candidate(ticker="NVDA", strike=950)
        csp_engine.execute_csp(candidate, MagicMock(), 100000, 0, 0)

        assert len(captured_intent) == 1
        assert captured_intent[0].ticker == "NVDA"  # Not OCC symbol


# ============================================================
# SPRINT 3: MONITOR TESTS
# ============================================================

class TestMonitor:
    def test_early_close_triggers(self):
        """80% profit with >14 DTE should trigger early close."""
        from lib.monitor import check_early_close

        position = {
            "premium_collected": 3.50,
            "expiration": (datetime.now(timezone.utc) + timedelta(days=20)).strftime("%Y-%m-%d"),
        }
        # Option now worth $0.50 → profit = $300 out of $350 max = 85.7%
        result = check_early_close(position, current_price=0.50)
        assert result is not None
        assert result["action"] == "early_close"

    def test_early_close_skips_low_profit(self):
        """Under 80% profit should not trigger."""
        from lib.monitor import check_early_close

        position = {
            "premium_collected": 3.50,
            "expiration": (datetime.now(timezone.utc) + timedelta(days=20)).strftime("%Y-%m-%d"),
        }
        # Option still worth $2.00 → profit only ~43%
        result = check_early_close(position, current_price=2.00)
        assert result is None

    def test_roll_candidate_itm_low_dte(self):
        """ITM with <7 DTE should trigger roll suggestion."""
        from lib.monitor import check_roll_candidate

        position = {
            "type": "csp",
            "strike": 170,
            "expiration": (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%d"),
        }
        # Stock at 165 → below strike → ITM for put
        result = check_roll_candidate(position, current_price=165)
        assert result is not None
        assert result["action"] == "roll"

    def test_roll_not_triggered_otm(self):
        """OTM should not trigger roll."""
        from lib.monitor import check_roll_candidate

        position = {
            "type": "csp",
            "strike": 170,
            "expiration": (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%d"),
        }
        # Stock at 180 → above strike → OTM for put
        result = check_roll_candidate(position, current_price=180)
        assert result is None

    def test_assignment_detection(self):
        """Should detect when a CSP gets assigned."""
        from lib.monitor import check_assignment

        position = {"ticker": "AAPL", "type": "csp", "status": "open"}
        broker_positions = [
            {"symbol": "AAPL", "qty": "100", "avg_entry_price": "170.00"},
        ]
        result = check_assignment(position, broker_positions)
        assert result is not None
        assert result["action"] == "assigned"
        assert result["shares"] == 100

    def test_heartbeat_kills_after_threshold(self, monkeypatch):
        """10 missed checks should trigger kill switch."""
        from lib import monitor

        monkeypatch.setattr(monitor, "_missed_checks", 0)
        kill_called = []
        monkeypatch.setattr("lib.kill_switch.activate_kill_switch",
                            lambda reason: kill_called.append(reason))
        monkeypatch.setattr(monitor, "send_alert", lambda msg: None)

        for _ in range(10):
            monitor.record_missed_check()

        assert len(kill_called) == 1
        assert "missed_10" in kill_called[0]


# ============================================================
# SPRINT 4: COVERED CALL ENGINE TESTS
# ============================================================

class TestCCEngine:
    def test_find_assigned_positions(self, tmp_path, monkeypatch):
        """Should find positions that need covered calls."""
        from lib import cc_engine

        pos_file = tmp_path / "positions.json"
        pos_file.write_text(json.dumps([
            {"ticker": "AAPL", "status": "assigned", "assigned_shares": 100, "cost_basis": 167},
            {"ticker": "MSFT", "status": "open", "type": "csp"},  # Not assigned
            {"ticker": "NVDA", "status": "assigned", "assigned_shares": 100, "cc_active": True},  # Already has CC
        ]))
        monkeypatch.setattr(cc_engine, "POSITIONS_PATH", pos_file)

        result = cc_engine.find_assigned_positions()
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_handle_call_assignment_pnl(self, tmp_path, monkeypatch):
        """Verify P/L calculation when shares are called away."""
        from lib import cc_engine

        positions = [{
            "ticker": "AAPL", "status": "assigned", "strike": 170,
            "premium_collected": 3.50, "cc_strike": 180, "cc_premium": 2.00,
            "assigned_shares": 100,
        }]
        pos_file = tmp_path / "positions.json"
        pos_file.write_text(json.dumps(positions))
        monkeypatch.setattr(cc_engine, "POSITIONS_PATH", pos_file)

        hist_file = tmp_path / "trade_history.json"
        hist_file.write_text("[]")
        monkeypatch.setattr(cc_engine, "TRADE_HISTORY_PATH", hist_file)

        monkeypatch.setattr("lib.cc_engine.kg_invalidate", lambda *a, **kw: None)
        monkeypatch.setattr("lib.cc_engine.kg_add", lambda *a, **kw: None)
        monkeypatch.setattr("lib.cc_engine.diary_write", lambda a, e: None)
        monkeypatch.setattr("lib.monitor.send_alert", lambda m: None)

        cc_engine.handle_call_assignment("AAPL", positions[0])

        # Check trade history
        history = json.loads(hist_file.read_text())
        assert len(history) == 1
        # Capital gain: (180-170)*100 = $1000
        # Premiums: (3.50+2.00)*100 = $550
        # Total: $1550
        assert history[0]["capital_gain"] == 1000
        assert history[0]["total_pnl"] == 1550

    def test_intent_uses_underlying_ticker_cc(self, tmp_path, monkeypatch):
        """CC OrderIntent.ticker should be the underlying."""
        from lib import cc_engine

        pos_file = tmp_path / "positions.json"
        pos_file.write_text("[]")
        monkeypatch.setattr(cc_engine, "POSITIONS_PATH", pos_file)

        # Mock consensus gate — approve everything
        import agents.consensus
        monkeypatch.setattr(agents.consensus, "seek_consensus",
                            lambda c, pv, **kw: {"approved": True, "decision": "APPROVED"})

        captured = []
        def mock_propose(intent):
            captured.append(intent)
            raise ValueError("test stop")

        monkeypatch.setattr("lib.cc_engine.step1_propose", mock_propose)
        monkeypatch.setattr("lib.cc_engine.diary_write", lambda a, e: None)
        monkeypatch.setattr("lib.cc_engine.check_dividend_conflict", lambda t, e: False)

        candidate = make_candidate(ticker="AAPL", trade_type="cc", strike=180)
        position = {"cost_basis": 167}

        cc_engine.execute_cc(candidate, position, MagicMock(), 100000, 0, 0)

        assert len(captured) == 1
        assert captured[0].ticker == "AAPL"


# ============================================================
# SPRINT 6: BACKTEST TESTS
# ============================================================

class TestBacktest:
    def test_backtest_returns_result(self):
        """Basic backtest should complete and return metrics."""
        from lib.backtest import run_wheel_backtest

        df = make_ohlcv(252, trend="up")
        result = run_wheel_backtest(df, initial_capital=100000)

        assert result.total_trades > 0
        assert result.win_rate >= 0
        assert result.max_drawdown >= 0
        assert result.sharpe_ratio != 0

    def test_monte_carlo_runs(self):
        """Monte Carlo should run N simulations."""
        from lib.backtest import run_monte_carlo

        df = make_ohlcv(252, trend="up")
        result = run_monte_carlo(df, n_simulations=50, initial_capital=100000)

        assert result.n_simulations == 50
        assert result.probability_of_loss >= 0
        assert result.percentile_5 <= result.percentile_95

    def test_benchmark_comparison(self):
        """Should compare wheel returns to buy-and-hold."""
        from lib.backtest import run_wheel_backtest, compare_to_benchmark

        df = make_ohlcv(252, trend="up")
        wheel = run_wheel_backtest(df)
        comparison = compare_to_benchmark(df, wheel)

        assert "wheel" in comparison
        assert "buy_and_hold" in comparison
        assert "outperformance" in comparison

    def test_csp_simulation(self):
        """Single CSP simulation should return assignment status."""
        from lib.backtest import simulate_csp_outcome

        df = make_ohlcv(100)
        returns = df["close"].pct_change().dropna().values

        result = simulate_csp_outcome(
            stock_price=100, strike=95, premium=2.0,
            days_to_exp=35, daily_returns=returns,
        )

        assert "assigned" in result
        assert "premium_pnl" in result
        assert result["premium_pnl"] == 200.0  # 2.0 * 100


# ============================================================
# SPRINT 7: AGENT & CONSENSUS TESTS
# ============================================================

class TestStrategyAgent:
    def test_propose_csp(self, monkeypatch):
        from agents.strategy_agent import StrategyAgent

        monkeypatch.setattr("agents.strategy_agent.recall_ticker_history",
                            lambda t: {"kg_facts": []})
        monkeypatch.setattr("agents.strategy_agent.get_current_regime", lambda: "bull")
        monkeypatch.setattr("agents.strategy_agent.diary_read", lambda a, last_n: [])
        monkeypatch.setattr("agents.strategy_agent.diary_write", lambda a, e: None)

        agent = StrategyAgent()
        proposal = agent.propose_csp(make_candidate())

        assert proposal["action"] == "sell_csp"
        assert proposal["ticker"] == "AAPL"
        assert proposal["composite_score"] == 8
        assert proposal["regime"] == "bull"

    def test_propose_cc(self, monkeypatch):
        from agents.strategy_agent import StrategyAgent

        monkeypatch.setattr("agents.strategy_agent.diary_write", lambda a, e: None)

        agent = StrategyAgent()
        candidate = make_candidate(trade_type="cc", strike=180)
        proposal = agent.propose_cc(candidate, cost_basis=167)

        assert proposal["action"] == "sell_cc"
        assert proposal["above_basis_by"] == 13.0


class TestRiskAgent:
    def test_approve_within_limits(self, tmp_path, monkeypatch):
        from agents.risk_agent import RiskAgent

        pos_file = tmp_path / "positions.json"
        pos_file.write_text("[]")
        monkeypatch.setattr("agents.risk_agent.POSITIONS_PATH", pos_file)
        monkeypatch.setattr("agents.risk_agent.get_current_regime", lambda: "bull")
        monkeypatch.setattr("agents.risk_agent.diary_write", lambda a, e: None)

        agent = RiskAgent()
        proposal = {"ticker": "AAPL", "action": "sell_csp", "strike": 170,
                     "composite_score": 8}

        result = agent.review(proposal, portfolio_value=200000)
        assert result["approved"] is True

    def test_veto_position_too_large(self, tmp_path, monkeypatch):
        from agents.risk_agent import RiskAgent

        pos_file = tmp_path / "positions.json"
        pos_file.write_text("[]")
        monkeypatch.setattr("agents.risk_agent.POSITIONS_PATH", pos_file)
        monkeypatch.setattr("agents.risk_agent.get_current_regime", lambda: "bull")
        monkeypatch.setattr("agents.risk_agent.diary_write", lambda a, e: None)

        agent = RiskAgent()
        # Strike 500 * 100 = $50,000 collateral on $100,000 portfolio = 50%
        proposal = {"ticker": "AAPL", "action": "sell_csp", "strike": 500,
                     "composite_score": 8}

        result = agent.review(proposal, portfolio_value=100000)
        assert result["approved"] is False
        assert "10%" in result["reason"]

    def test_veto_sector_concentration(self, tmp_path, monkeypatch):
        from agents.risk_agent import RiskAgent

        # Already have a bunch of tech positions
        positions = [
            {"ticker": "MSFT", "status": "open", "strike": 400},
            {"ticker": "NVDA", "status": "open", "strike": 900},
        ]
        pos_file = tmp_path / "positions.json"
        pos_file.write_text(json.dumps(positions))
        monkeypatch.setattr("agents.risk_agent.POSITIONS_PATH", pos_file)
        monkeypatch.setattr("agents.risk_agent.get_current_regime", lambda: "bull")
        monkeypatch.setattr("agents.risk_agent.diary_write", lambda a, e: None)

        agent = RiskAgent()
        # Adding AAPL (tech) would push sector past 30%
        proposal = {"ticker": "AAPL", "action": "sell_csp", "strike": 170,
                     "composite_score": 8}

        result = agent.review(proposal, portfolio_value=200000)
        # MSFT: 40k + NVDA: 90k + AAPL: 17k = 147k / 200k = 73.5% tech
        assert result["approved"] is False
        assert "sector" in result["reason"].lower() or "Sector" in result["reason"]


class TestComplianceAgent:
    def test_clear_no_issues(self, tmp_path, monkeypatch):
        from agents.compliance_agent import ComplianceAgent

        hist_file = tmp_path / "trade_history.json"
        hist_file.write_text("[]")
        monkeypatch.setattr("agents.compliance_agent.TRADE_HISTORY_PATH", hist_file)
        monkeypatch.setattr("agents.compliance_agent.kg_query", lambda t, current_only: [])
        monkeypatch.setattr("agents.compliance_agent.diary_write", lambda a, e: None)

        agent = ComplianceAgent()
        proposal = {"ticker": "AAPL", "expiration": "2024-06-21"}

        # Patch market hours to pass
        monkeypatch.setattr(agent, "_check_market_hours",
                            lambda: {"pass": True, "reason": "ok"})

        result = agent.review(proposal)
        assert result["approved"] is True

    def test_wash_sale_blocks(self, tmp_path, monkeypatch):
        from agents.compliance_agent import ComplianceAgent

        # Recent loss on AAPL
        history = [{
            "ticker": "AAPL",
            "total_pnl": -250.0,
            "completed_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        }]
        hist_file = tmp_path / "trade_history.json"
        hist_file.write_text(json.dumps(history))
        monkeypatch.setattr("agents.compliance_agent.TRADE_HISTORY_PATH", hist_file)
        monkeypatch.setattr("agents.compliance_agent.kg_query", lambda t, current_only: [])
        monkeypatch.setattr("agents.compliance_agent.diary_write", lambda a, e: None)

        agent = ComplianceAgent()
        monkeypatch.setattr(agent, "_check_market_hours",
                            lambda: {"pass": True, "reason": "ok"})

        proposal = {"ticker": "AAPL", "expiration": "2024-06-21"}
        result = agent.review(proposal)
        assert result["approved"] is False
        assert "wash sale" in result["reason"].lower()


class TestConsensus:
    def test_unanimous_approval(self, tmp_path, monkeypatch):
        from agents.consensus import seek_consensus

        monkeypatch.setattr("agents.strategy_agent.recall_ticker_history",
                            lambda t: {"kg_facts": []})
        monkeypatch.setattr("agents.strategy_agent.get_current_regime", lambda: "bull")
        monkeypatch.setattr("agents.strategy_agent.diary_read", lambda a, last_n: [])
        monkeypatch.setattr("agents.strategy_agent.diary_write", lambda a, e: None)
        monkeypatch.setattr("agents.risk_agent.get_current_regime", lambda: "bull")
        monkeypatch.setattr("agents.risk_agent.diary_write", lambda a, e: None)
        monkeypatch.setattr("agents.risk_agent.POSITIONS_PATH", tmp_path / "positions.json")
        (tmp_path / "positions.json").write_text("[]")
        monkeypatch.setattr("agents.compliance_agent.TRADE_HISTORY_PATH", tmp_path / "history.json")
        (tmp_path / "history.json").write_text("[]")
        monkeypatch.setattr("agents.compliance_agent.kg_query", lambda t, current_only: [])
        monkeypatch.setattr("agents.compliance_agent.diary_write", lambda a, e: None)

        # Patch market hours
        from agents.compliance_agent import ComplianceAgent
        monkeypatch.setattr(ComplianceAgent, "_check_market_hours",
                            lambda self: {"pass": True, "reason": "ok"})

        monkeypatch.setattr("agents.consensus.diary_write", lambda a, e: None)

        candidate = make_candidate(strike=50)  # Small position relative to portfolio
        result = seek_consensus(candidate, portfolio_value=200000)

        assert result["approved"] is True
        assert result["decision"] == "EXECUTE"

    def test_risk_veto_stops_consensus(self, tmp_path, monkeypatch):
        from agents.consensus import seek_consensus

        monkeypatch.setattr("agents.strategy_agent.recall_ticker_history",
                            lambda t: {"kg_facts": []})
        monkeypatch.setattr("agents.strategy_agent.get_current_regime", lambda: "bull")
        monkeypatch.setattr("agents.strategy_agent.diary_read", lambda a, last_n: [])
        monkeypatch.setattr("agents.strategy_agent.diary_write", lambda a, e: None)
        monkeypatch.setattr("agents.risk_agent.get_current_regime", lambda: "bull")
        monkeypatch.setattr("agents.risk_agent.diary_write", lambda a, e: None)
        monkeypatch.setattr("agents.risk_agent.POSITIONS_PATH", tmp_path / "positions.json")
        (tmp_path / "positions.json").write_text("[]")

        monkeypatch.setattr("agents.consensus.diary_write", lambda a, e: None)

        # Huge position that'll fail risk check
        candidate = make_candidate(strike=500)  # 50k collateral on 100k portfolio
        result = seek_consensus(candidate, portfolio_value=100000)

        assert result["approved"] is False
        assert result["decision"] == "VETOED"
        assert result["blocking_agent"] == "risk_agent"


# ============================================================
# SPRINT 8: ENHANCEMENTS TESTS
# ============================================================

class TestEnhancements:
    def test_earnings_conflict_detection(self):
        from lib.enhancements import set_earnings_date, is_earnings_conflict

        set_earnings_date.__wrapped__ = None  # Reset if needed
        # Manually set cache
        from lib import enhancements
        enhancements._earnings_cache["AAPL"] = "2024-06-15"

        assert is_earnings_conflict("AAPL", "2024-06-21") is True
        assert is_earnings_conflict("AAPL", "2024-06-10") is False
        assert is_earnings_conflict("MSFT", "2024-06-21") is False

    def test_bayesian_assignment_with_history(self):
        from lib.enhancements import bayesian_assignment_probability

        # With historical data showing higher assignment rate
        prob = bayesian_assignment_probability(
            delta=-0.25, dte=35,
            historical_assignments=15, historical_trades=50,
        )
        assert 0 < prob < 1.0

    def test_bayesian_assignment_no_history(self):
        from lib.enhancements import bayesian_assignment_probability

        # No history — should use delta as prior
        prob = bayesian_assignment_probability(
            delta=-0.25, dte=35,
            historical_assignments=0, historical_trades=0,
        )
        assert abs(prob - 0.25) < 0.15  # Should be near delta

    def test_capitol_trades_placeholder(self):
        from lib.enhancements import check_capitol_trades, score_capitol_signal

        assert check_capitol_trades("AAPL") is None
        assert score_capitol_signal([]) == 0.0
        assert score_capitol_signal([
            {"type": "buy", "member": "A"},
            {"type": "buy", "member": "B"},
            {"type": "buy", "member": "C"},
        ]) == 1.0

    def test_anomaly_detection_no_crash(self, tmp_path, monkeypatch):
        from lib.enhancements import detect_anomalies
        from lib import audit

        monkeypatch.setattr(audit, "AUDIT_FILE", tmp_path / "audit.jsonl")
        monkeypatch.setattr(audit, "LOG_DIR", tmp_path)
        monkeypatch.setattr("lib.enhancements.diary_write", lambda a, e: None)

        anomalies = detect_anomalies()
        assert isinstance(anomalies, list)


# ============================================================
# MEMPALACE TESTS
# ============================================================

class TestMemPalace:
    def test_init_creates_dirs(self, tmp_path, monkeypatch):
        from lib import memory_palace as mp

        monkeypatch.setattr(mp, "PALACE_DIR", tmp_path / "palace")
        monkeypatch.setattr(mp, "DIARY_DIR", tmp_path / "palace" / "diaries")
        monkeypatch.setattr(mp, "KG_DB", tmp_path / "palace" / "kg.db")
        monkeypatch.setattr(mp, "HAS_CHROMA", False)

        mp.init_palace()

        assert (tmp_path / "palace").exists()
        assert (tmp_path / "palace" / "diaries").exists()
        assert (tmp_path / "palace" / "kg.db").exists()

    def test_kg_add_and_query(self, tmp_path, monkeypatch):
        from lib import memory_palace as mp

        monkeypatch.setattr(mp, "KG_DB", tmp_path / "kg.db")

        mp.kg_add("AAPL", "entered_csp", "170P_2024-06-21")
        mp.kg_add("AAPL", "zone_support", "168.50")

        facts = mp.kg_query("AAPL", current_only=True)
        assert len(facts) == 2

    def test_kg_invalidate(self, tmp_path, monkeypatch):
        from lib import memory_palace as mp

        monkeypatch.setattr(mp, "KG_DB", tmp_path / "kg.db")

        mp.kg_add("AAPL", "entered_csp", "170P")
        mp.kg_invalidate("AAPL", "entered_csp", "170P")

        current = mp.kg_query("AAPL", current_only=True)
        assert len(current) == 0

        all_facts = mp.kg_query("AAPL", current_only=False)
        assert len(all_facts) == 1  # Still in history

    def test_kg_timeline(self, tmp_path, monkeypatch):
        from lib import memory_palace as mp

        monkeypatch.setattr(mp, "KG_DB", tmp_path / "kg.db")

        mp.kg_add("AAPL", "entered_csp", "170P")
        mp.kg_add("AAPL", "assigned", "100_shares")

        timeline = mp.kg_timeline("AAPL")
        assert len(timeline) == 2

    def test_diary_write_and_read(self, tmp_path, monkeypatch):
        from lib import memory_palace as mp

        monkeypatch.setattr(mp, "DIARY_DIR", tmp_path / "diaries")

        mp.diary_write("strategy_agent", "AAPL|CSP_170P|score_8")
        mp.diary_write("strategy_agent", "NVDA|CSP_950P|score_7")

        entries = mp.diary_read("strategy_agent", last_n=10)
        assert len(entries) == 2
        assert "AAPL" in entries[0]["entry"]
        assert "NVDA" in entries[1]["entry"]

    def test_diary_read_empty(self, tmp_path, monkeypatch):
        from lib import memory_palace as mp

        monkeypatch.setattr(mp, "DIARY_DIR", tmp_path / "diaries")

        entries = mp.diary_read("nonexistent_agent")
        assert entries == []

    def test_remember_trade_decision(self, tmp_path, monkeypatch):
        from lib import memory_palace as mp

        monkeypatch.setattr(mp, "PALACE_DIR", tmp_path / "palace")
        monkeypatch.setattr(mp, "KG_DB", tmp_path / "palace" / "kg.db")
        monkeypatch.setattr(mp, "HAS_CHROMA", False)

        mp.init_palace()

        mp.remember_trade_decision(
            ticker="AAPL", trade_type="csp",
            details={"strike": 170, "expiration": "2024-06-21", "premium": 3.50},
            reasoning="Sold put at support zone with hammer confirmation",
        )

        facts = mp.kg_query("AAPL")
        assert len(facts) == 1
        assert "entered_csp" in facts[0]["predicate"]

    def test_regime_change(self, tmp_path, monkeypatch):
        from lib import memory_palace as mp

        monkeypatch.setattr(mp, "PALACE_DIR", tmp_path / "palace")
        monkeypatch.setattr(mp, "KG_DB", tmp_path / "palace" / "kg.db")
        monkeypatch.setattr(mp, "DIARY_DIR", tmp_path / "palace" / "diaries")
        monkeypatch.setattr(mp, "HAS_CHROMA", False)

        mp.init_palace()

        mp.remember_regime_change("bull", "Higher highs, higher lows on SPY weekly")
        assert mp.get_current_regime() == "bull"

        mp.remember_regime_change("bear", "Lower highs, lower lows, VIX spike")
        assert mp.get_current_regime() == "bear"

        # Old regime should be invalidated
        facts = mp.kg_query("market", current_only=True)
        regime_facts = [f for f in facts if f["predicate"] == "regime"]
        assert len(regime_facts) == 1
        assert regime_facts[0]["object"] == "bear"

    def test_recall_ticker_history(self, tmp_path, monkeypatch):
        from lib import memory_palace as mp

        monkeypatch.setattr(mp, "PALACE_DIR", tmp_path / "palace")
        monkeypatch.setattr(mp, "KG_DB", tmp_path / "palace" / "kg.db")
        monkeypatch.setattr(mp, "DIARY_DIR", tmp_path / "palace" / "diaries")
        monkeypatch.setattr(mp, "HAS_CHROMA", False)

        mp.init_palace()

        mp.kg_add("AAPL", "entered_csp", "170P")
        mp.diary_write("strategy_agent", "AAPL|CSP_170P|score_8")

        history = mp.recall_ticker_history("AAPL")
        assert history["ticker"] == "AAPL"
        assert len(history["kg_facts"]) >= 1
        assert "strategy_agent" in history["agent_mentions"]


# ============================================================
# SPRINT 9: MAIN.PY / CHAOS TESTS
# ============================================================

class TestMain:
    def test_chaos_command_runs(self, monkeypatch, capsys):
        """Chaos tests should all pass."""
        from lib import audit
        # Ensure audit writes to temp
        monkeypatch.setattr(audit, "AUDIT_FILE", Path(tempfile.mkdtemp()) / "audit.jsonl")
        monkeypatch.setattr(audit, "LOG_DIR", Path(tempfile.mkdtemp()))
        monkeypatch.setattr("lib.enhancements.diary_write", lambda a, e: None)

        from main import cmd_chaos
        cmd_chaos()

        captured = capsys.readouterr()
        assert "5/5 passed" in captured.out

    def test_migrate_checklist(self, capsys):
        from main import cmd_migrate
        cmd_migrate()

        captured = capsys.readouterr()
        assert "MIGRATION CHECKLIST" in captured.out
        assert "live_migration_approved" in captured.out


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
