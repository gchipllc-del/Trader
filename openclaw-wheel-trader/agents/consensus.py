"""
Sprint 7: Agent Consensus Protocol

Trade proceeds ONLY if:
  1. Strategy Agent PROPOSES
  2. Risk Agent APPROVES (does not VETO)
  3. Compliance Agent APPROVES (no regulatory issues)
  4. LLM Analyst ADVISES (advisory by default; configurable veto)

Unanimous consent required from the 3 governance agents. The LLM analyst
is an advisory 4th voice — logged on every decision; it only blocks when
`llm.consensus_veto_enabled: true` AND it returns "skip" with high confidence.

The execution layer is behind all gates — no agent can directly call the broker.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import yaml

from agents.strategy_agent import StrategyAgent
from agents.risk_agent import RiskAgent
from agents.compliance_agent import ComplianceAgent
from agents.bear_agent import BearAgent
from agents.bull_agent import BullAgent, combine_bull_bear
from lib.audit import log_event
from lib.memory_palace import diary_write, get_current_regime
from lib.screener import WheelCandidate


strategy = StrategyAgent()
risk = RiskAgent()
compliance = ComplianceAgent()
bear = BearAgent()
bull = BullAgent()

STRATEGY_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"


def _load_llm_config() -> dict:
    try:
        with open(STRATEGY_PATH) as f:
            return (yaml.safe_load(f) or {}).get("llm", {}) or {}
    except Exception:
        return {}


def _llm_advisory_vote(
    candidate: WheelCandidate,
    cost_basis: float | None,
) -> dict:
    """Call LLM analyst for a 4th-voice vote. Never raises — always returns a dict.

    Returns:
        {
            "ran": bool,              # did we actually query the LLM
            "action": str,            # "sell"|"wait"|"skip"|"skipped"
            "win_probability": float | None,
            "confidence": float | None,
            "reasoning": str,
            "veto": bool,             # should we block? (only true if config enables + skip + high conf)
            "provider": str | None,
            "model": str | None,
        }
    """
    cfg = _load_llm_config()
    if not cfg.get("enabled", False) or not cfg.get("consensus_enabled", False):
        return {"ran": False, "action": "skipped", "veto": False,
                "win_probability": None, "confidence": None, "reasoning": "llm_consensus_disabled",
                "provider": None, "model": None}

    try:
        from lib.llm_analyst import analyze_option_setup
        analysis = analyze_option_setup(
            ticker=candidate.ticker,
            trade_type=candidate.trade_type,
            strike=candidate.strike,
            premium=candidate.premium,
            delta=candidate.delta,
            dte=candidate.dte,
            composite_score=candidate.composite_score,
            zone_level=candidate.zone_level,
            zone_touches=candidate.zone_touches,
            iv_rank=candidate.iv_rank,
            annualized_return=candidate.annualized_return,
            candlestick_pattern=candidate.candlestick_pattern,
            cost_basis=cost_basis,
        )
    except Exception as e:
        log_event("consensus", "llm_error", {"ticker": candidate.ticker, "error": str(e)[:200]})
        return {"ran": False, "action": "error", "veto": False,
                "win_probability": None, "confidence": None, "reasoning": str(e)[:200],
                "provider": None, "model": None}

    if analysis is None:
        return {"ran": False, "action": "unavailable", "veto": False,
                "win_probability": None, "confidence": None, "reasoning": "llm_unavailable_or_parse_fail",
                "provider": None, "model": None}

    # Veto logic: only block if explicitly enabled AND skip suggested AND confident
    veto_threshold = float(cfg.get("consensus_veto_confidence", 0.7))
    veto_enabled = bool(cfg.get("consensus_veto_enabled", False))
    veto = (
        veto_enabled
        and analysis.suggested_action == "skip"
        and analysis.confidence >= veto_threshold
    )

    return {
        "ran": True,
        "action": analysis.suggested_action,
        "win_probability": analysis.win_probability,
        "confidence": analysis.confidence,
        "reasoning": analysis.reasoning,
        "bullish_factors": analysis.bullish_factors,
        "bearish_factors": analysis.bearish_factors,
        "veto": veto,
        "provider": analysis.provider,
        "model": analysis.model,
    }


def seek_consensus(
    candidate: WheelCandidate,
    portfolio_value: float,
    cost_basis: float | None = None,
    *,
    broker_positions: list[dict] | None = None,
) -> dict:
    """
    Run the 3-agent consensus process.

    Args:
        candidate: The screener's trade candidate
        portfolio_value: Current portfolio value for risk checks
        cost_basis: For covered calls, the share cost basis
        broker_positions: Optional pre-fetched broker position list.
            When provided, the risk agent runs the global
            capital-at-risk gate (sum of short-put collateral +
            long-share value) in addition to per-position checks.

    Returns:
        {
            "approved": bool,
            "proposal": dict,
            "risk_review": dict,
            "compliance_review": dict,
            "decision": "EXECUTE" | "VETOED" | "BLOCKED",
            "blocking_agent": str | None,
        }
    """
    # Step 1: Strategy proposes
    if candidate.trade_type == "csp":
        proposal = strategy.propose_csp(candidate)
    elif candidate.trade_type == "cc" and cost_basis is not None:
        proposal = strategy.propose_cc(candidate, cost_basis)
    else:
        return {
            "approved": False,
            "decision": "BLOCKED",
            "blocking_agent": "strategy_agent",
            "reason": f"Unknown trade type: {candidate.trade_type}",
        }

    # Step 2: Risk reviews
    risk_review = risk.review(
        proposal, portfolio_value, broker_positions=broker_positions,
    )

    if not risk_review["approved"]:
        log_event("consensus", "vetoed_by_risk", {
            "ticker": candidate.ticker,
            "reason": risk_review["reason"],
        })
        return {
            "approved": False,
            "proposal": proposal,
            "risk_review": risk_review,
            "compliance_review": None,
            "decision": "VETOED",
            "blocking_agent": "risk_agent",
            "reason": risk_review["reason"],
        }

    # Step 3: Compliance reviews
    compliance_review = compliance.review(proposal)

    if not compliance_review["approved"]:
        log_event("consensus", "blocked_by_compliance", {
            "ticker": candidate.ticker,
            "reason": compliance_review["reason"],
        })
        return {
            "approved": False,
            "proposal": proposal,
            "risk_review": risk_review,
            "compliance_review": compliance_review,
            "decision": "BLOCKED",
            "blocking_agent": "compliance_agent",
            "reason": compliance_review["reason"],
        }

    # Step 4: Bull/Bear adversarial scoring (deterministic, no LLM cost).
    # Inspired by TauricResearch/TradingAgents bullish/bearish researcher
    # debate. Both agents read the same candidate fields; bear can VETO
    # or DOWNSIZE, bull can suggest BOOST when its case materially
    # exceeds bear's. combine_bull_bear is asymmetric: bear always wins
    # ties — never let bull's enthusiasm override a bear veto.
    regime_now = get_current_regime() or "unknown"
    bear_review = bear.review(candidate, regime=regime_now)
    bull_review = bull.review(candidate, regime=regime_now)
    combined = combine_bull_bear(bull_review, bear_review)

    if combined["decision"] == "VETO":
        log_event("consensus", "vetoed_by_bear", {
            "ticker": candidate.ticker,
            "bear_score": bear_review["score"],
            "bull_score": bull_review["score"],
            "signals": [s["name"] for s in bear_review["signals"]],
        })
        return {
            "approved": False,
            "proposal": proposal,
            "risk_review": risk_review,
            "compliance_review": compliance_review,
            "bear_review": bear_review,
            "bull_review": bull_review,
            "combined": combined,
            "decision": "VETOED",
            "blocking_agent": "bear_agent",
            "reason": combined["reasoning"],
        }

    # Step 5: LLM advisory vote (non-blocking unless config explicitly enables veto)
    llm_review = _llm_advisory_vote(candidate, cost_basis)
    if llm_review.get("ran"):
        log_event("consensus", "llm_advisory", {
            "ticker": candidate.ticker,
            "action": llm_review["action"],
            "win_prob": llm_review["win_probability"],
            "confidence": llm_review["confidence"],
            "veto": llm_review["veto"],
        })
        diary_write("llm_analyst",
            f"{candidate.ticker}|{candidate.trade_type.upper()}_{llm_review['action'].upper()}|"
            f"win_{llm_review['win_probability']:.0%}|conf_{llm_review['confidence']:.2f}|"
            f"{(llm_review.get('reasoning') or '')[:80]}")

    if llm_review.get("veto"):
        log_event("consensus", "vetoed_by_llm", {
            "ticker": candidate.ticker,
            "reason": llm_review.get("reasoning", "")[:200],
        })
        return {
            "approved": False,
            "proposal": proposal,
            "risk_review": risk_review,
            "compliance_review": compliance_review,
            "llm_review": llm_review,
            "decision": "VETOED",
            "blocking_agent": "llm_analyst",
            "reason": f"LLM skip w/ confidence {llm_review['confidence']:.2f}: "
                      f"{llm_review.get('reasoning', '')[:160]}",
        }

    # UNANIMOUS CONSENT — proceed
    log_event("consensus", "approved", {
        "ticker": candidate.ticker,
        "trade_type": candidate.trade_type,
        "strike": candidate.strike,
        "score": candidate.composite_score,
    }, result="success")

    diary_write("strategy_agent",
        f"CONSENSUS_APPROVED|{candidate.ticker}|{candidate.trade_type}|{candidate.strike}")

    return {
        "approved": True,
        "proposal": proposal,
        "risk_review": risk_review,
        "compliance_review": compliance_review,
        "bear_review": bear_review,
        "bull_review": bull_review,
        "combined": combined,
        "llm_review": llm_review,
        "decision": "EXECUTE",
        "blocking_agent": None,
        # Combined sizing: bear can downsize (0.5) or veto (0.0); bull can
        # boost (1.25) only when bear is silent. Caller multiplies Kelly.
        "size_multiplier": combined["size_multiplier"],
    }
