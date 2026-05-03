"""
Tests for BullAgent and the bull/bear combiner.

Mirrors test_bear_agent.py for the symmetric agent. The combiner is
where the asymmetric risk discipline lives — these tests lock that in.
"""

import pytest
from unittest.mock import MagicMock


class TestBullAgentScoring:
    def test_clean_setup_passes_with_no_signals(self):
        """A neutral candidate with no extreme signals → score=0 → WEAK."""
        from agents.bull_agent import BullAgent
        candidate = {
            "ticker": "F",
            "composite_score": 5,
            "kronos_direction": "neutral",
            "news_sentiment": 0.0,
            "bayesian_win_prob": 0.55,
            "earnings_days": 20,
            "correlation_penalty": 0.95,  # not perfect
            "candlestick_pattern": None,
        }
        result = BullAgent().review(candidate, regime="sideways")
        assert result["score"] == 0
        assert result["action"] == "WEAK"
        assert result["size_multiplier"] == 1.0

    def test_strong_bull_boosts(self):
        """Multiple bullish signals → ≥7 → BOOST with 1.25x."""
        from agents.bull_agent import BullAgent
        candidate = {
            "ticker": "NVDA",
            "composite_score": 9,           # +1
            "kronos_direction": "bullish",  # +1
            "news_sentiment": 0.6,          # +1
            "bayesian_win_prob": 0.72,      # +2
            "earnings_days": 45,            # +1
            "correlation_penalty": 1.0,     # +1
            "candlestick_pattern": "morning_star",  # +1
        }
        result = BullAgent().review(candidate, regime="bull")  # +2
        assert result["score"] >= BullAgent.BOOST_THRESHOLD
        assert result["action"] == "BOOST"
        assert result["size_multiplier"] == BullAgent.BOOST_MULTIPLIER

    def test_handles_missing_fields(self):
        from agents.bull_agent import BullAgent
        result = BullAgent().review({"ticker": "ZZZ"}, regime="unknown")
        assert result["score"] == 0
        assert result["action"] == "WEAK"

    def test_handles_none_values(self):
        from agents.bull_agent import BullAgent
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
        result = BullAgent().review(candidate, regime="unknown")
        assert result["score"] == 0


class TestBullBearCombiner:
    def _bear_pass(self):
        return {"action": "PASS", "size_multiplier": 1.0, "score": 1, "signals": []}

    def _bear_downsize(self):
        return {"action": "DOWNSIZE", "size_multiplier": 0.5, "score": 4, "signals": []}

    def _bear_veto(self):
        return {"action": "VETO", "size_multiplier": 0.0, "score": 7, "signals": []}

    def test_bear_veto_always_wins_even_with_strong_bull(self):
        from agents.bull_agent import combine_bull_bear
        bull = {"action": "BOOST", "size_multiplier": 1.25, "score": 9, "signals": []}
        result = combine_bull_bear(bull, self._bear_veto())
        assert result["decision"] == "VETO"
        assert result["size_multiplier"] == 0.0

    def test_bear_downsize_suppresses_bull_boost(self):
        """Even a strong bull cannot override bear DOWNSIZE."""
        from agents.bull_agent import combine_bull_bear
        bull = {"action": "BOOST", "size_multiplier": 1.25, "score": 8, "signals": []}
        result = combine_bull_bear(bull, self._bear_downsize())
        assert result["decision"] == "DOWNSIZE"
        assert result["size_multiplier"] == 0.5

    def test_bull_boost_fires_when_bear_is_silent_and_delta_is_large(self):
        from agents.bull_agent import combine_bull_bear
        bull = {"action": "BOOST", "size_multiplier": 1.25, "score": 7, "signals": []}
        bear = {"action": "PASS", "size_multiplier": 1.0, "score": 2, "signals": []}
        result = combine_bull_bear(bull, bear)
        assert result["decision"] == "BOOST"
        assert result["size_multiplier"] == 1.25
        assert result["delta"] == 5

    def test_bull_boost_suppressed_when_delta_too_small(self):
        """Bull score barely beats bear score → don't boost (need delta ≥ 4)."""
        from agents.bull_agent import combine_bull_bear
        bull = {"action": "BOOST", "size_multiplier": 1.25, "score": 7, "signals": []}
        bear = {"action": "PASS", "size_multiplier": 1.0, "score": 5, "signals": []}
        result = combine_bull_bear(bull, bear)
        assert result["decision"] == "PASS"
        assert result["size_multiplier"] == 1.0

    def test_neutral_bull_doesnt_boost(self):
        """Bull NEUTRAL → no boost regardless of bear score."""
        from agents.bull_agent import combine_bull_bear
        bull = {"action": "NEUTRAL", "size_multiplier": 1.0, "score": 5, "signals": []}
        bear = {"action": "PASS", "size_multiplier": 1.0, "score": 0, "signals": []}
        result = combine_bull_bear(bull, bear)
        assert result["decision"] == "PASS"
        assert result["size_multiplier"] == 1.0


class TestBullBearStockEngineIntegration:
    def test_stock_engine_upsizes_on_strong_bull_silent_bear(self, tmp_path, monkeypatch):
        """Strong bull + silent bear → 1.25x upsize within max_position."""
        from lib import stock_engine

        pos_file = tmp_path / "positions.json"
        pos_file.write_text("[]")
        monkeypatch.setattr(stock_engine, "POSITIONS_PATH", pos_file)
        monkeypatch.setattr(stock_engine, "diary_write", lambda a, e: None)
        monkeypatch.setattr(stock_engine, "remember_trade_decision", lambda **kw: None)

        captured_intent = []
        monkeypatch.setattr("lib.stock_engine.step1_propose",
                            lambda i: (captured_intent.append(i), i)[-1])
        monkeypatch.setattr("lib.stock_engine.step2_validate",
                            lambda intent, **kw: setattr(intent, "_validated", True) or True)
        monkeypatch.setattr("lib.stock_engine.step3_execute",
                            lambda i, c: {"id": "ord_x", "status": "filled",
                                          "symbol": i.ticker, "qty": str(i.quantity),
                                          "side": "buy", "filled_qty": str(i.quantity)})
        monkeypatch.setattr("lib.memory_palace.get_current_regime", lambda: "bull")

        client = MagicMock()
        client.get_positions.return_value = []
        client._get_trading_client.return_value.get_orders.return_value = []
        client.limiter.wait_if_needed = lambda: None
        client.wait_for_fill.return_value = {
            "id": "ord_x", "status": "filled",
            "filled_qty": "10", "filled_avg_price": "100.00",
        }

        # Strong bull, no bear signals.
        candidate = {
            "ticker": "NVDA",
            "shares": 8,
            "current_price": 100.0,
            "target_price": 110.0,
            "stop_loss": 95.0,
            "composite_score": 9,
            "trend_score": 3, "level_score": 3, "signal_score": 3,
            "zone_level": 100.0, "zone_touches": 5, "pattern": "morning_star",
            "kronos_direction": "bullish",
            "kronos_expected_return": 0.05,
            "news_sentiment": 0.6,
            "bayesian_win_prob": 0.75,
            "earnings_days": 45,
            "correlation_penalty": 1.0,
        }
        stock_engine.execute_stock_buy(
            candidate, client, portfolio_value=100000,
            current_daily_pnl=0, current_open_orders=0,
        )
        # 8 shares × 1.25 = 10 → captured intent should hold 10
        assert captured_intent
        assert captured_intent[0].quantity == 10
