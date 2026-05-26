"""
Tests for the bear agent (adversarial stress-test layer).

Inspired by TauricResearch/TradingAgents bull/bear debate, adapted to
deterministic signal scoring. Validates:
  - score thresholds (PASS / DOWNSIZE / VETO)
  - individual signal contributions
  - tolerant input (None values, missing fields, dataclass-like objects)
  - integration paths (consensus + stock_engine)
"""

import pytest
from unittest.mock import MagicMock, patch


class TestBearAgentScoring:
    def test_clean_setup_passes(self):
        """High-conviction trade with no bearish signals → PASS, multiplier 1.0."""
        from agents.bear_agent import BearAgent
        candidate = {
            "ticker": "NVDA",
            "composite_score": 8,
            "kronos_direction": "bullish",
            "kronos_expected_return": 0.04,
            "news_sentiment": 0.3,
            "bayesian_win_prob": 0.65,
            "earnings_days": 60,
            "correlation_penalty": 1.0,
            "candlestick_pattern": "bullish_engulfing",
        }
        result = BearAgent().review(candidate, regime="bull")
        assert result["action"] == "PASS"
        assert result["size_multiplier"] == 1.0
        assert result["score"] == 0

    def test_overwhelming_bear_vetoes(self):
        """Score ≥ 6 → VETO."""
        from agents.bear_agent import BearAgent
        candidate = {
            "ticker": "WBD",
            "composite_score": 3,           # +1 (low)
            "kronos_direction": "bearish",  # +1
            "news_sentiment": -0.6,         # +1
            "bayesian_win_prob": 0.42,      # +2
            "earnings_days": 3,             # +1
            "correlation_penalty": 0.5,     # +1
            "candlestick_pattern": "bearish_engulfing",  # +1
        }
        result = BearAgent().review(candidate, regime="bear")  # +2
        assert result["score"] >= BearAgent.VETO_THRESHOLD
        assert result["action"] == "VETO"
        assert result["size_multiplier"] == 0.0

    def test_moderate_bear_downsizes(self):
        """Score 3-5 → DOWNSIZE, multiplier 0.5."""
        from agents.bear_agent import BearAgent
        candidate = {
            "ticker": "AAPL",
            "composite_score": 6,
            "kronos_direction": "bearish",  # +1
            "news_sentiment": -0.4,         # +1
            "bayesian_win_prob": 0.55,
            "earnings_days": 5,             # +1
            "correlation_penalty": 1.0,
            "candlestick_pattern": "doji",
        }
        result = BearAgent().review(candidate, regime="sideways")
        assert 3 <= result["score"] < 6
        assert result["action"] == "DOWNSIZE"
        assert result["size_multiplier"] == 0.5

    def test_handles_missing_fields_gracefully(self):
        """A candidate dict with only a ticker shouldn't crash; bear scores 0."""
        from agents.bear_agent import BearAgent
        result = BearAgent().review({"ticker": "ZZZ"}, regime="unknown")
        assert result["score"] == 0
        assert result["action"] == "PASS"

    def test_handles_none_values(self):
        """None values in candidate fields shouldn't crash float coercion."""
        from agents.bear_agent import BearAgent
        candidate = {
            "ticker": "TST",
            "composite_score": None,
            "kronos_direction": None,
            "news_sentiment": None,
            "bayesian_win_prob": None,
            "earnings_days": None,
            "correlation_penalty": None,
            "candlestick_pattern": None,
        }
        result = BearAgent().review(candidate, regime="unknown")
        assert result["score"] == 0
        assert result["action"] == "PASS"

    def test_bear_regime_alone_doesnt_veto(self):
        """Even in bear regime, a clean trade should only earn 2 points → still PASS."""
        from agents.bear_agent import BearAgent
        candidate = {
            "ticker": "KO",
            "composite_score": 8,
            "bayesian_win_prob": 0.7,
            "correlation_penalty": 1.0,
        }
        result = BearAgent().review(candidate, regime="bear")
        assert result["score"] == 2  # only bear_regime hit
        assert result["action"] == "PASS"

    def test_low_bayesian_alone_downsizes(self):
        """Low Bayesian (worth 2 points) + nothing else → 2pts → still PASS."""
        from agents.bear_agent import BearAgent
        candidate = {
            "ticker": "F",
            "composite_score": 7,
            "bayesian_win_prob": 0.45,
            "correlation_penalty": 1.0,
        }
        result = BearAgent().review(candidate, regime="bull")
        assert result["score"] == 2
        assert result["action"] == "PASS"

    def test_signals_carry_evidence(self):
        """Each signal in the output should carry human-readable evidence."""
        from agents.bear_agent import BearAgent
        candidate = {
            "ticker": "AMD",
            "composite_score": 3,
            "candlestick_pattern": "shooting_star",
        }
        result = BearAgent().review(candidate, regime="unknown")
        assert any("composite" in s["name"] for s in result["signals"])
        assert any("candle" in s["name"] for s in result["signals"])
        for s in result["signals"]:
            assert s["evidence"]


class TestBearAgentIntegration:
    def test_stock_engine_vetoes_overwhelmingly_bearish(self, tmp_path, monkeypatch):
        """execute_stock_buy must short-circuit when bear vetoes."""
        from lib import stock_engine

        pos_file = tmp_path / "positions.json"
        pos_file.write_text("[]")
        monkeypatch.setattr(stock_engine, "POSITIONS_PATH", pos_file)
        monkeypatch.setattr(stock_engine, "diary_write", lambda a, e: None)

        # The strategy is already mocked away in other tests; here we only
        # care that bear's VETO short-circuits before step1_propose runs.
        called = []
        monkeypatch.setattr("lib.stock_engine.step1_propose",
                            lambda i: (called.append(i), i)[-1])

        client = MagicMock()
        candidate = {
            "ticker": "WBD",
            "shares": 10,
            "current_price": 27.0,
            "target_price": 30.0,
            "stop_loss": 25.0,
            "composite_score": 3,
            "trend_score": 1, "level_score": 1, "signal_score": 0,
            "zone_level": 26.5, "zone_touches": 2,
            "pattern": "bearish_engulfing",
            "kronos_direction": "bearish",
            "news_sentiment": -0.6,
            "bayesian_win_prob": 0.4,
            "earnings_days": 3,
            "correlation_penalty": 0.4,
        }
        # Force regime=bear so total score ≥ 6
        monkeypatch.setattr("lib.memory_palace.get_current_regime",
                            lambda: "bear")

        result = stock_engine.execute_stock_buy(
            candidate, client, portfolio_value=10000,
            current_daily_pnl=0, current_open_orders=0,
        )
        assert result is None
        assert called == []  # step1_propose never reached

    def test_stock_engine_downsizes_on_moderate_bear(self, tmp_path, monkeypatch):
        """DOWNSIZE multiplier should reduce intended shares before order_gate."""
        from lib import stock_engine

        pos_file = tmp_path / "positions.json"
        pos_file.write_text("[]")
        monkeypatch.setattr(stock_engine, "POSITIONS_PATH", pos_file)
        monkeypatch.setattr(stock_engine, "diary_write", lambda a, e: None)
        monkeypatch.setattr(stock_engine, "remember_trade_decision", lambda **kw: None)
        # Bypass the SPY-gap stock-buys gate — it's a separate live-state
        # circuit breaker, not what this test is asserting.
        monkeypatch.setattr("lib.circuit_breaker.check_stock_buys_enabled",
                            lambda: True)

        captured_intent = []
        monkeypatch.setattr("lib.stock_engine.step1_propose",
                            lambda i: (captured_intent.append(i), i)[-1])
        monkeypatch.setattr("lib.stock_engine.step2_validate",
                            lambda intent, **kw: setattr(intent, "_validated", True) or True)
        monkeypatch.setattr("lib.stock_engine.step3_execute",
                            lambda i, c: {"id": "ord_x", "status": "filled",
                                          "symbol": i.ticker, "qty": str(i.quantity),
                                          "side": "buy", "filled_qty": str(i.quantity)})
        monkeypatch.setattr("lib.memory_palace.get_current_regime", lambda: "sideways")

        client = MagicMock()
        client.get_positions.return_value = []
        client._get_trading_client.return_value.get_orders.return_value = []
        client.limiter.wait_if_needed = lambda: None
        client.wait_for_fill.return_value = {
            "id": "ord_x", "status": "filled",
            "filled_qty": "5", "filled_avg_price": "100.00",
        }

        candidate = {
            "ticker": "AAPL",
            "shares": 10,           # original
            "current_price": 100.0,
            "target_price": 110.0,
            "stop_loss": 95.0,
            "composite_score": 6,
            "trend_score": 2, "level_score": 1, "signal_score": 0,
            "zone_level": 100.0, "zone_touches": 3, "pattern": None,
            # Force exactly DOWNSIZE-tier score:
            "kronos_direction": "bearish",       # +1
            "news_sentiment": -0.4,              # +1
            "earnings_days": 5,                  # +1
        }
        stock_engine.execute_stock_buy(
            candidate, client, portfolio_value=100000,
            current_daily_pnl=0, current_open_orders=0,
        )
        # Intended 10 shares → bear DOWNSIZE to 5
        assert captured_intent
        assert captured_intent[0].quantity == 5
