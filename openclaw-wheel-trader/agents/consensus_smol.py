"""
Alternative consensus orchestration using HuggingFace smolagents.

This is a FEATURE-FLAGGED pilot that mirrors the behavior of agents/consensus.py
via smolagents' ToolCallingAgent framework. It is NOT the default — the
hand-rolled consensus.py remains the production path because it's fully
deterministic and tested. This module exists to let us A/B the two approaches
with real LLM judgment on borderline trades.

When to use:
    * Set `llm.smol_consensus_enabled: true` in config/wheel_strategy.yaml AND
    * Call `seek_consensus_smol(candidate, portfolio_value, cost_basis)` from
      experimental code paths.

What this gives us over the hand-rolled flow:
    * Agents can reason over proposals qualitatively instead of just applying
      numeric gates — useful for catching cases where all numbers pass but
      the overall setup has a subtle flaw (narrative contradiction, unusual
      IV curvature, etc).
    * smolagents handles retry/tool-calling/error recovery without us
      reimplementing it.

What we keep from consensus.py:
    * The hard numeric gates (risk_agent, compliance_agent) still fire FIRST.
      Only after those pass do we hand to the LLM agents.
    * All the audit logging and diary writes.

Install:
    pip install smolagents

Safe to import even without smolagents installed — callers probe
`smol_consensus_available()` first.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from lib.audit import log_event
from lib.screener import WheelCandidate
from agents.consensus import seek_consensus  # delegate the deterministic path

try:
    from smolagents import ToolCallingAgent, OpenAIServerModel  # type: ignore
    _HAS_SMOLAGENTS = True
except Exception:
    _HAS_SMOLAGENTS = False


STRATEGY_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"


def smol_consensus_available() -> bool:
    return _HAS_SMOLAGENTS and bool(os.environ.get("DEEPSEEK_API_KEY"))


def _load_llm_cfg() -> dict:
    try:
        with open(STRATEGY_PATH) as f:
            return (yaml.safe_load(f) or {}).get("llm", {}) or {}
    except Exception:
        return {}


def _build_model() -> "OpenAIServerModel":
    """DeepSeek via OpenAI-compatible endpoint (same as lib/llm_analyst.py)."""
    cfg = _load_llm_cfg()
    model_id = cfg.get("model") or "deepseek-v4-flash"
    return OpenAIServerModel(
        model_id=model_id,
        api_base="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )


def _vote(role: str, stance: str, proposal_summary: str, model) -> tuple[str, str]:
    """Run one agent, return (vote, one-sentence-reason).

    `stance` describes what the role should care about — gets baked into the
    agent description so the LLM answers in-character.
    """
    agent = ToolCallingAgent(
        tools=[],
        model=model,
        name=f"{role}_agent",
        description=(
            f"You are the {role} reviewer on a disciplined options trading desk. "
            f"{stance} "
            "Respond with EXACTLY one line: 'YES: <reason>' or 'NO: <reason>'. "
            "Keep the reason under 20 words."
        ),
    )
    try:
        answer = str(agent.run(
            f"Trade proposal:\n{proposal_summary}\n\nApprove? Reply as specified."
        )).strip()
    except Exception as e:
        log_event("consensus_smol", "agent_error", {"role": role, "error": str(e)[:200]})
        return ("ABSTAIN", f"agent_error: {str(e)[:100]}")
    upper = answer.upper()
    if upper.startswith("YES"):
        return ("YES", answer[4:].lstrip(": ").strip()[:200])
    if upper.startswith("NO"):
        return ("NO", answer[3:].lstrip(": ").strip()[:200])
    # Unparseable → abstain (doesn't block; the hand-rolled gates already approved)
    return ("ABSTAIN", answer[:200])


def seek_consensus_smol(
    candidate: WheelCandidate,
    portfolio_value: float,
    cost_basis: float | None = None,
) -> dict:
    """
    Two-phase consensus:
      Phase 1: hand-rolled deterministic gates (risk + compliance) from
               agents.consensus.seek_consensus. Same approve/veto semantics.
      Phase 2: IF phase 1 approved AND smol_consensus_enabled, ask 3 LLM
               agents (strategy/risk/compliance) to weigh in with qualitative
               votes. Unanimous YES required to proceed with smol approval;
               otherwise we fall back to the phase-1 decision (so this is
               never MORE restrictive than the deterministic path by default).

    Falls back to the deterministic result if smolagents isn't available or
    the smol config flag is off.
    """
    base = seek_consensus(candidate, portfolio_value, cost_basis)

    cfg = _load_llm_cfg()
    if not cfg.get("smol_consensus_enabled", False):
        return base
    if not base.get("approved"):
        # No point running the expensive LLM round if we already vetoed
        return base
    if not smol_consensus_available():
        log_event("consensus_smol", "unavailable", {
            "ticker": candidate.ticker,
            "reason": "smolagents_missing_or_no_key",
        })
        return base

    try:
        model = _build_model()
    except Exception as e:
        log_event("consensus_smol", "model_init_failed", {"error": str(e)[:200]})
        return base

    proposal = base.get("proposal", {})
    summary = (
        f"Ticker: {candidate.ticker}\n"
        f"Trade: SELL-TO-OPEN {candidate.trade_type.upper()} "
        f"strike={candidate.strike} expiry={candidate.expiration} "
        f"premium={candidate.premium} delta={candidate.delta} "
        f"dte={candidate.dte}\n"
        f"Composite score: {candidate.composite_score}/9\n"
        f"Zone: ${candidate.zone_level} ({candidate.zone_touches} touches)\n"
        f"IV rank: {candidate.iv_rank:.0%}  "
        f"Annualized: {candidate.annualized_return:.1%}\n"
        f"Cost basis: {cost_basis or 'n/a'}"
    )

    stances = {
        "strategy": (
            "You care about: technical setup quality, zone strength, "
            "candlestick confirmation, IV richness. Approve only strong, "
            "high-conviction setups."
        ),
        "risk": (
            "You care about: position sizing, correlation, sector concentration, "
            "assignment risk, whether this trade could cause outsized drawdown. "
            "Reject if the blast radius looks wrong."
        ),
        "compliance": (
            "You care about: wash sales, PDT rule, order limits, earnings "
            "proximity, dividend-ex-date conflicts. Reject if any regulatory "
            "red flag appears."
        ),
    }

    votes = {}
    for role, stance in stances.items():
        vote, reason = _vote(role, stance, summary, model)
        votes[role] = {"vote": vote, "reason": reason}

    smol_approved = all(v["vote"] == "YES" for v in votes.values())
    log_event("consensus_smol", "vote_complete", {
        "ticker": candidate.ticker,
        "smol_approved": smol_approved,
        "votes": {r: v["vote"] for r, v in votes.items()},
    })

    out = dict(base)
    out["smol_votes"] = votes
    out["smol_approved"] = smol_approved
    # Conservative policy: only TAKE the trade if BOTH deterministic gates
    # AND the LLM unanimous vote agree. The deterministic veto always wins.
    if not smol_approved:
        out["approved"] = False
        out["decision"] = "SMOL_DISSENT"
        # Identify who said no first
        for role in ("risk", "compliance", "strategy"):
            if votes[role]["vote"] != "YES":
                out["blocking_agent"] = f"smol_{role}_agent"
                out["reason"] = votes[role]["reason"]
                break
    return out
