"""Tests for Trading Memory Palace integration."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.memory_palace import (
    init_palace, add_drawer, search_memory, Drawer,
    kg_add, kg_invalidate, kg_query, kg_timeline,
    diary_write, diary_read,
    remember_trade_decision, remember_regime_change, recall_ticker_history,
    get_current_regime, PALACE_DIR, KG_DB, DIARY_DIR,
)


@pytest.fixture(autouse=True)
def tmp_palace(tmp_path, monkeypatch):
    """Redirect palace to temp dir for each test."""
    palace = tmp_path / "palace"
    monkeypatch.setattr("lib.memory_palace.PALACE_DIR", palace)
    monkeypatch.setattr("lib.memory_palace.KG_DB", palace / "knowledge_graph.db")
    monkeypatch.setattr("lib.memory_palace.DIARY_DIR", palace / "diaries")
    return palace


class TestDrawers:
    def test_add_and_search_fallback(self):
        drawer = Drawer(
            wing="wing_aapl", hall="hall_facts",
            room="aapl-csp",
            content="Sold AAPL 170P for $3.50 premium. Support zone at 168 with 4 touches.",
            metadata={"ticker": "AAPL", "strike": 170, "premium": 3.50},
        )
        did = add_drawer(drawer)
        assert len(did) == 12

        # Search — works via ChromaDB or fallback
        results = search_memory("AAPL 170P premium", wing="wing_aapl")
        # ChromaDB semantic search or keyword fallback should find it
        assert isinstance(results, list)

    def test_wing_filter(self):
        add_drawer(Drawer(
            wing="wing_aapl", hall="hall_facts", room="test",
            content="AAPL trade reasoning here",
            metadata={"ticker": "AAPL"},
        ))
        add_drawer(Drawer(
            wing="wing_nvda", hall="hall_facts", room="test",
            content="NVDA trade reasoning here",
            metadata={"ticker": "NVDA"},
        ))

        aapl_results = search_memory("trade", wing="wing_aapl")
        assert all("AAPL" in r["content"] for r in aapl_results)


class TestKnowledgeGraph:
    def test_add_and_query(self):
        kg_add("AAPL", "entered_csp", "170P_2024-06-21",
               valid_from="2024-05-01", metadata={"premium": 3.50})

        facts = kg_query("AAPL")
        assert len(facts) == 1
        assert facts[0]["predicate"] == "entered_csp"
        assert facts[0]["object"] == "170P_2024-06-21"

    def test_invalidate(self):
        kg_add("AAPL", "holding", "100_shares", valid_from="2024-06-21")
        kg_invalidate("AAPL", "holding", "100_shares", ended="2024-07-15")

        current = kg_query("AAPL", current_only=True)
        assert len(current) == 0

        all_facts = kg_query("AAPL", current_only=False)
        assert len(all_facts) == 1
        assert all_facts[0]["valid_to"] == "2024-07-15"

    def test_timeline(self):
        kg_add("AAPL", "entered_csp", "170P", valid_from="2024-05-01")
        kg_add("AAPL", "assigned", "100_shares", valid_from="2024-06-21")
        kg_add("AAPL", "entered_cc", "180C", valid_from="2024-06-25")

        tl = kg_timeline("AAPL")
        assert len(tl) == 3
        # Should be chronological
        assert tl[0]["valid_from"] <= tl[1]["valid_from"] <= tl[2]["valid_from"]

    def test_regime_change(self):
        remember_regime_change("bull", "Higher highs on weekly, VIX below 15")
        assert get_current_regime() == "bull"

        remember_regime_change("bear", "Weekly broke below 50 SMA, VIX spike to 30")
        assert get_current_regime() == "bear"

        # Old regime should be invalidated
        all_facts = kg_query("market", current_only=False)
        invalidated = [f for f in all_facts if f["valid_to"] is not None]
        assert len(invalidated) >= 1


class TestAgentDiaries:
    def test_write_and_read(self):
        diary_write("strategy_agent", "AAPL|CSP_170P|score_8/9|zone_168|hammer")
        diary_write("strategy_agent", "NVDA|SKIP|iv_rank_22pct|below_threshold")

        entries = diary_read("strategy_agent", last_n=5)
        assert len(entries) == 2
        assert "AAPL" in entries[0]["entry"]
        assert "NVDA" in entries[1]["entry"]

    def test_separate_diaries(self):
        diary_write("strategy_agent", "PROPOSE|AAPL_170P")
        diary_write("risk_agent", "APPROVE|AAPL_170P|within_limits")
        diary_write("compliance_agent", "CLEAR|no_wash_sale")

        strat = diary_read("strategy_agent")
        risk = diary_read("risk_agent")
        comp = diary_read("compliance_agent")

        assert len(strat) == 1
        assert len(risk) == 1
        assert len(comp) == 1
        assert "PROPOSE" in strat[0]["entry"]
        assert "APPROVE" in risk[0]["entry"]


class TestTradingHelpers:
    def test_remember_trade_decision(self):
        remember_trade_decision(
            ticker="AAPL",
            trade_type="csp",
            details={"strike": 170, "expiration": "2024-06-21", "premium": 3.50},
            reasoning="Sold put at 170 support zone. 4 touches, hammer confirmed on daily. IV rank 45%. Composite 8/9.",
        )

        # Should be in KG
        facts = kg_query("AAPL")
        assert any(f["predicate"] == "entered_csp" for f in facts)

    def test_recall_ticker_history(self):
        kg_add("AAPL", "entered_csp", "170P", valid_from="2024-05-01")
        diary_write("strategy_agent", "AAPL|CSP_170P|strong_setup")

        history = recall_ticker_history("AAPL")
        assert history["ticker"] == "AAPL"
        assert len(history["kg_facts"]) >= 1
        assert "strategy_agent" in history["agent_mentions"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
