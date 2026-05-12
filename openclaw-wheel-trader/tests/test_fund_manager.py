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


class TestFundManagerHRP:
    """HRP rebalance check — added 2026-05-12 (PyPortfolioOpt integration)."""

    def _stocks(self, *tickers_and_values):
        """Helper to build stock position dicts.

        Each arg is ``(ticker, market_value)``. Used by every HRP test.
        """
        return [
            {"ticker": t, "type": "stock", "status": "open",
             "sector": "tech", "shares": int(mv / 100), "entry_price": 100,
             "market_value": float(mv)}
            for t, mv in tickers_and_values
        ]

    def _returns_df(self, tickers, days=60, *, seed=0, drift=0.0001, scale=0.015):
        """Build a synthetic returns DataFrame for the given tickers."""
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(seed)
        data = {
            t: rng.normal(drift, scale, days) for t in tickers
        }
        return pd.DataFrame(data)

    def test_hrp_skipped_when_no_returns_passed(self):
        """Default review (no historical_returns kwarg) skips HRP cleanly."""
        from agents.fund_manager import FundManager
        positions = self._stocks(("NVDA", 3_000), ("AMD", 3_000), ("KO", 3_000))
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=1_000,
        )
        assert all(a["type"] != "HRP_REBALANCE" for a in review["actions"])
        assert review["stats"]["hrp_weights"] == {}

    def test_hrp_skipped_when_under_3_stock_positions(self):
        from agents.fund_manager import FundManager
        positions = self._stocks(("NVDA", 3_000), ("AMD", 3_000))
        rets = self._returns_df(["NVDA", "AMD"])
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=4_000,
            historical_returns=rets,
        )
        assert all(a["type"] != "HRP_REBALANCE" for a in review["actions"])
        assert review["stats"]["hrp_weights"] == {}

    def test_hrp_skipped_when_under_30_days_of_returns(self):
        from agents.fund_manager import FundManager
        positions = self._stocks(("NVDA", 3_000), ("AMD", 3_000), ("KO", 3_000))
        rets = self._returns_df(["NVDA", "AMD", "KO"], days=20)
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=1_000,
            historical_returns=rets,
        )
        assert all(a["type"] != "HRP_REBALANCE" for a in review["actions"])

    def test_hrp_skipped_when_returns_not_dataframe(self):
        from agents.fund_manager import FundManager
        positions = self._stocks(("NVDA", 3_000), ("AMD", 3_000), ("KO", 3_000))
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=1_000,
            historical_returns={"NVDA": [0.01, 0.02]},  # not a DataFrame
        )
        assert all(a["type"] != "HRP_REBALANCE" for a in review["actions"])

    def test_hrp_emits_action_on_drift(self):
        """One position is 60% of invested capital but HRP would weight it
        far lower — expect an HRP_REBALANCE action targeting that ticker.
        """
        from agents.fund_manager import FundManager
        # NVDA dominates the book: $7k of $10k invested = 70%
        # KO and JPM split the rest at 15% each
        positions = self._stocks(
            ("NVDA", 7_000), ("KO", 1_500), ("JPM", 1_500),
        )
        rets = self._returns_df(["NVDA", "KO", "JPM"], days=60, scale=0.015)
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=0,
            historical_returns=rets,
        )
        hrp_actions = [a for a in review["actions"] if a["type"] == "HRP_REBALANCE"]
        assert hrp_actions, f"expected HRP_REBALANCE action; got {review['actions']}"
        # HRP w/ ~uncorrelated synthetic returns produces roughly equal
        # weights, so NVDA at 70% will be flagged for trim.
        nvda_actions = [a for a in hrp_actions if "NVDA" in a["affected_tickers"]]
        assert nvda_actions, "expected NVDA to be flagged for trim"
        assert "trim" in nvda_actions[0]["summary"].lower()

    def test_hrp_stats_present_when_ran(self):
        from agents.fund_manager import FundManager
        positions = self._stocks(("NVDA", 3_000), ("AMD", 3_000), ("KO", 3_000))
        rets = self._returns_df(["NVDA", "AMD", "KO"], days=60)
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=1_000,
            historical_returns=rets,
        )
        weights = review["stats"]["hrp_weights"]
        assert weights, "expected hrp_weights to be populated"
        assert set(weights.keys()) == {"NVDA", "AMD", "KO"}
        # Weights sum to ~1.0 by HRP construction
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_hrp_excludes_options_from_optimization(self):
        """Options positions in the book are ignored by HRP — only stocks
        get optimized."""
        from agents.fund_manager import FundManager
        positions = [
            {"ticker": "NVDA", "type": "stock", "status": "open", "sector": "tech",
             "shares": 30, "entry_price": 100, "market_value": 3_000},
            {"ticker": "AMD", "type": "stock", "status": "open", "sector": "tech",
             "shares": 20, "entry_price": 100, "market_value": 2_000},
            {"ticker": "KO", "type": "stock", "status": "open", "sector": "consumer",
             "shares": 25, "entry_price": 100, "market_value": 2_500},
            # CSP — should NOT be in HRP optimization
            {"ticker": "MSFT", "type": "csp", "status": "open", "sector": "tech",
             "strike": 400, "market_value": -200},
        ]
        rets = self._returns_df(["NVDA", "AMD", "KO"], days=60)
        review = FundManager().review_portfolio(
            positions=positions, bankroll=10_000, cash=2_500,
            historical_returns=rets,
        )
        weights = review["stats"]["hrp_weights"]
        assert "MSFT" not in weights, "options should not appear in HRP weights"


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
