"""
Sprint 7: Agent Consensus Protocol

Trade proceeds ONLY if:
  1. Strategy Agent PROPOSES
  2. Risk Agent APPROVES (does not VETO)
  3. Compliance Agent APPROVES (no regulatory issues)

Unanimous consent required. Any single agent can block a trade.
The execution layer is behind all three gates — no agent can
directly call the broker.
"""

from datetime import datetime, timezone

from agents.strategy_agent import StrategyAgent
from agents.risk_agent import RiskAgent
from agents.compliance_agent import ComplianceAgent
from lib.audit import log_event
from lib.memory_palace import diary_write
from lib.screener import WheelCandidate


strategy = StrategyAgent()
risk = RiskAgent()
compliance = ComplianceAgent()


def seek_consensus(
    candidate: WheelCandidate,
    portfolio_value: float,
    cost_basis: float | None = None,
) -> dict:
    """
    Run the 3-agent consensus process.

    Args:
        candidate: The screener's trade candidate
        portfolio_value: Current portfolio value for risk checks
        cost_basis: For covered calls, the share cost basis

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
    risk_review = risk.review(proposal, portfolio_value)

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
        "decision": "EXECUTE",
        "blocking_agent": None,
    }
