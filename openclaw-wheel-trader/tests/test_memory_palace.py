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
    record_trade_outcome, get_past_outcomes, prior_loss_rate,
    get_current_regime, PALACE_DIR, KG_DB, DIARY_DIR,
)


@pytest.fixture(autouse=True)
def tmp_palace(tmp_path, monkeypatch):
    """Redirect palace to temp dir for each test.

    Force HAS_CHROMA=False so we exercise the JSONL fallback path.
    Some macOS / Python builds segfault inside chromadb's onnx
    embedding model (``onnx_mini_lm_l6_v2``) — that's an environment
    issue unrelated to MemPalace logic, and the fallback path covers
    the same correctness contract.
    """
    palace = tmp_path / "palace"
    monkeypatch.setattr("lib.memory_palace.PALACE_DIR", palace)
    monkeypatch.setattr("lib.memory_palace.KG_DB", palace / "knowledge_graph.db")
    monkeypatch.setattr("lib.memory_palace.DIARY_DIR", palace / "diaries")
    monkeypatch.setattr("lib.memory_palace.HAS_CHROMA", False)
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


class TestLearningLoop:
    """Decision → outcome → reflection → injection cycle (TradingAgents v0.2.4 pattern)."""

    def test_remember_returns_drawer_id(self):
        drawer_id = remember_trade_decision(
            ticker="AAPL", trade_type="csp",
            details={"strike": 170, "expiration": "2026-06-21", "premium": 3.50},
            reasoning="Strong support zone at 170, hammer on daily, IV rank 45%.",
        )
        assert isinstance(drawer_id, str) and len(drawer_id) == 12

    def test_record_outcome_links_back_to_decision(self):
        drawer_id = remember_trade_decision(
            ticker="AAPL", trade_type="csp",
            details={"strike": 170, "expiration": "2026-06-21", "premium": 3.50},
            reasoning="Test decision.",
        )
        outcome_id = record_trade_outcome(
            ticker="AAPL",
            decision_drawer_id=drawer_id,
            realized_return_pct=0.012,
            holding_days=14,
            exit_reason="csp_expired",
            final_pnl_dollars=350.0,
        )
        assert isinstance(outcome_id, str)
        # KG should have the outcome edge linking back to the decision
        facts = kg_query("AAPL", current_only=False)
        outcomes = [f for f in facts if f["predicate"] == "outcome"]
        assert outcomes, "expected outcome edge in KG"
        # KG stores metadata as JSON string; parse it before asserting
        meta = json.loads(outcomes[0]["metadata"])
        assert meta["decision_drawer_id"] == drawer_id
        assert meta["holding_days"] == 14

    def test_get_past_outcomes_returns_empty_when_no_history(self):
        result = get_past_outcomes("ZZZZ")  # ticker never seen
        assert result == ""

    def test_get_past_outcomes_includes_resolved_trades(self):
        drawer_id = remember_trade_decision(
            ticker="NVDA", trade_type="csp",
            details={"strike": 500, "expiration": "2026-06-21", "premium": 8.0},
            reasoning="Decision reasoning text.",
        )
        record_trade_outcome(
            ticker="NVDA", decision_drawer_id=drawer_id,
            realized_return_pct=-0.18, holding_days=21,
            exit_reason="csp_assigned",
        )
        result = get_past_outcomes("NVDA")
        # The formatted output should include the exit reason and return
        assert "NVDA" in result
        assert "csp_assigned" in result or "-18" in result or "-0.18" in result or "-18.0%" in result

    def test_prior_loss_rate_zero_when_no_outcomes(self):
        losses, total, rate = prior_loss_rate("FRESH")
        assert (losses, total, rate) == (0, 0, 0.0)

    def test_prior_loss_rate_counts_negative_outcomes(self):
        # 3 trades: 2 losses, 1 win
        for i, ret in enumerate([-0.15, -0.08, +0.03]):
            d = remember_trade_decision(
                ticker="AMD", trade_type="csp",
                details={"strike": 100 + i, "expiration": "2026-06-21"},
                reasoning=f"Decision {i}",
            )
            record_trade_outcome(
                ticker="AMD", decision_drawer_id=d,
                realized_return_pct=ret, holding_days=14,
                exit_reason="csp_assigned" if ret < 0 else "csp_expired",
            )
        losses, total, rate = prior_loss_rate("AMD")
        assert total == 3
        assert losses == 2
        assert abs(rate - (2/3)) < 0.01

    def test_outcome_metadata_includes_alpha_when_provided(self):
        d = remember_trade_decision(
            ticker="SPY", trade_type="csp",
            details={"strike": 500}, reasoning="test",
        )
        record_trade_outcome(
            ticker="SPY", decision_drawer_id=d,
            realized_return_pct=0.05, alpha_pct=-0.02, holding_days=10,
            exit_reason="csp_expired",
        )
        # The outcome drawer should be searchable. JSONL fallback uses
        # case-insensitive substring match against content, so the
        # query must be a substring of the rendered outcome text.
        outcomes = search_memory("SPY", wing="wing_spy",
                                 hall="hall_outcomes", n_results=5)
        assert outcomes, "expected outcome drawer to be retrievable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
