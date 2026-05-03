"""
Tests for agents/fund_manager — portfolio-level review layer.

Adapted from TauricResearch/TradingAgents Fund Manager stage. Validates
each portfolio-level rule fires when expected and stays silent when not.
"""

import pytest


class TestFundManagerSectorConcentration:
    def test_no_action_when_balanced(self):
        from agents.fund_manager import FundManager
        positions = [
            {"ticker": "NVDA", "status": "open", "sector": "tech",
             "shares": 1, "entry_price": 200, "market_value": 200},
            {"ticker": "KO", "status": "open", "sector": "consumer_staples",
             "shares": 5, "entry_price": 80, "market_value": 400},
            {"ticker": "JPM", "status": "open", "sector": "financials",
             "shares": 2, "entry_price": 150, "market_value": 300},
        ]
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=9_100,
        )
        assert review["actions"] == []
        assert "balanced" in review["summary"].lower()

    def test_sector_overconcentration_triggers_trim(self):
        from agents.fund_manager import FundManager
        # 60% of bankroll in tech — over 50% sector cap
        positions = [
            {"ticker": "NVDA", "status": "open", "sector": "tech",
             "shares": 10, "entry_price": 300, "market_value": 3_000},
            {"ticker": "AMD", "status": "open", "sector": "tech",
             "shares": 20, "entry_price": 150, "market_value": 3_000},
            {"ticker": "KO", "status": "open", "sector": "consumer_staples",
             "shares": 5, "entry_price": 80, "market_value": 400},
        ]
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=3_600,
            max_sector_pct=0.50,
        )
        sector_actions = [a for a in review["actions"] if a["type"] == "SECTOR_TRIM"]
        assert len(sector_actions) == 1
        assert "tech" in sector_actions[0]["summary"]
        assert sorted(sector_actions[0]["affected_tickers"]) == ["AMD", "NVDA"]

    def test_unknown_sector_doesnt_trigger(self):
        """Crypto / un-tagged positions shouldn't fire sector alarms."""
        from agents.fund_manager import FundManager
        positions = [
            {"ticker": "BTC/USD", "status": "open",  # no sector tag
             "shares": 0.1, "entry_price": 100_000, "market_value": 10_000},
            {"ticker": "KO", "status": "open", "sector": "consumer_staples",
             "shares": 1, "entry_price": 100, "market_value": 100},
        ]
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_100, cash=0,
        )
        sector_actions = [a for a in review["actions"] if a["type"] == "SECTOR_TRIM"]
        assert len(sector_actions) == 0


class TestFundManagerCorrelation:
    def test_correlation_cluster_triggers_when_over_cap(self):
        from agents.fund_manager import FundManager
        positions = [
            {"ticker": "F", "status": "open", "correlation_group": "ev",
             "shares": 10, "entry_price": 12, "market_value": 120},
            {"ticker": "NIO", "status": "open", "correlation_group": "ev",
             "shares": 20, "entry_price": 8, "market_value": 160},
            {"ticker": "RIVN", "status": "open", "correlation_group": "ev",
             "shares": 5, "entry_price": 16, "market_value": 80},
            {"ticker": "LCID", "status": "open", "correlation_group": "ev",
             "shares": 30, "entry_price": 3, "market_value": 90},
        ]
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=9_550,
            max_correlation_group_count=3,
        )
        corr_actions = [a for a in review["actions"] if a["type"] == "CORRELATION_TRIM"]
        assert len(corr_actions) == 1
        assert "ev" in corr_actions[0]["summary"]
        assert len(corr_actions[0]["affected_tickers"]) == 4


class TestFundManagerCashBuffer:
    def test_low_cash_buffer_triggers(self):
        from agents.fund_manager import FundManager
        positions = [
            {"ticker": "X", "status": "open", "sector": "industrials",
             "shares": 100, "entry_price": 100, "market_value": 10_000},
        ]
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_200, cash=200,  # 1.96% cash
            min_cash_buffer_pct=0.05,
        )
        cash_actions = [a for a in review["actions"] if a["type"] == "CASH_BUFFER_LOW"]
        assert len(cash_actions) == 1


class TestFundManagerEdgeCases:
    def test_empty_book_passes(self):
        from agents.fund_manager import FundManager
        review = FundManager().review_portfolio(
            positions=[], bankroll=10_000, cash=10_000,
        )
        assert review["actions"] == []
        assert review["stats"]["n_positions"] == 0

    def test_zero_bankroll_skipped(self):
        from agents.fund_manager import FundManager
        review = FundManager().review_portfolio(
            positions=[{"ticker": "X", "status": "open"}],
            bankroll=0, cash=0,
        )
        assert review["actions"] == []
        assert "skipped" in review["summary"].lower()

    def test_non_open_positions_excluded(self):
        from agents.fund_manager import FundManager
        positions = [
            {"ticker": "X", "status": "closed", "sector": "tech",
             "shares": 100, "entry_price": 100, "market_value": 10_000},
        ]
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=10_000,
        )
        # No open positions → no concentration concern
        assert review["actions"] == []
        assert review["stats"]["n_positions"] == 0
