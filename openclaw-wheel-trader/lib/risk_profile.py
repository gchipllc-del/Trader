"""
Risk personalities — risky / neutral / safe parameter profiles.

Inspired by TauricResearch/TradingAgents Risk Management Team
(Risky/Neutral/Safe debaters + Fund Manager). Where their paper uses
LLM debate, we use deterministic profile selection based on observable
state — no LLM cost, fully predictable, easy to audit.

Selection rules (first match wins):

  SAFE      — bear regime, OR daily loss > 5% of bankroll, OR cooldown active
  RISKY     — bull regime AND bankroll ≥ $5k AND no daily loss
  NEUTRAL   — everything else (default)

Each profile produces an override dict applied ON TOP of the base
strategy/settings YAML — same fields, different values. Callers pull
e.g. `effective_kelly_fraction()` instead of reading
strategy.yaml directly.

The profiles are tighter / looser than base in three dimensions:
  • kelly_fraction       — how much of full-Kelly to use
  • max_position_pct     — per-position cap (within breaker ceiling)
  • min_composite_score  — entry quality bar

Why deterministic instead of LLM debate: at $1.5k bankroll the paper's
"11 LLM calls per decision" pricing eats 3-5% of every trade. Profile
selection from observable state captures the same calibration without
the bleed.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.audit import log_event


@dataclass(frozen=True)
class RiskProfile:
    """A named parameter override applied on top of base strategy config."""
    name: str
    kelly_fraction: float
    max_position_pct: float
    min_composite_score: int
    rationale: str


# Profiles ordered conservative → aggressive. Tunable but bounded — never
# exceed circuit_breakers.max_position_pct (0.30) or the breaker blocks.
SAFE = RiskProfile(
    name="safe",
    kelly_fraction=0.10,            # 1/10 Kelly — defensive
    max_position_pct=0.15,
    min_composite_score=7,
    rationale=(
        "bear regime / drawdown / post-loss cooldown — preserve capital, "
        "wait for high-conviction setups only"
    ),
)

NEUTRAL = RiskProfile(
    name="neutral",
    kelly_fraction=0.25,            # quarter-Kelly — current default
    max_position_pct=0.30,
    min_composite_score=5,
    rationale="default operating regime — quarter-Kelly with full position cap",
)

RISKY = RiskProfile(
    name="risky",
    kelly_fraction=0.50,            # half-Kelly — aggressive
    max_position_pct=0.30,           # circuit-breaker ceiling stays sacred
    min_composite_score=4,
    rationale=(
        "bull regime + bankroll ≥ $5k + clean P/L — let high-edge candidates "
        "size up to half-Kelly within the breaker cap"
    ),
)


def select_profile(
    *,
    regime: str = "unknown",
    bankroll: float = 0.0,
    daily_loss_pct: float = 0.0,
    cooldown_active: bool = False,
    risky_bankroll_threshold: float = 5000.0,
    safe_drawdown_threshold: float = -0.05,
) -> RiskProfile:
    """
    Pick the active risk profile from observable state.

    Args:
        regime: "bull" | "bear" | "sideways" | "unknown" (from memory_palace)
        bankroll: current portfolio value in USD
        daily_loss_pct: today's P/L as fraction of bankroll (e.g. -0.04 = -4%)
        cooldown_active: True if we're inside the post-loss cooldown window
        risky_bankroll_threshold: minimum bankroll to qualify for RISKY
        safe_drawdown_threshold: daily P/L pct that triggers SAFE
                                 (negative; -0.05 = -5%)

    Returns:
        The selected RiskProfile.
    """
    regime_l = str(regime).lower()

    # SAFE first — any defensive trigger wins
    if regime_l == "bear":
        return SAFE
    if daily_loss_pct <= safe_drawdown_threshold:
        return SAFE
    if cooldown_active:
        return SAFE

    # RISKY only when ALL conditions clear
    if (regime_l == "bull"
            and bankroll >= risky_bankroll_threshold
            and daily_loss_pct >= 0.0):
        return RISKY

    return NEUTRAL


def apply_profile_to(base_params: dict, profile: RiskProfile) -> dict:
    """
    Return a copy of base_params with the profile's overrides applied.

    Only overrides the three knobs the profile owns; leaves everything
    else untouched. Caller should treat the result as read-only for
    that decision cycle.
    """
    out = dict(base_params)
    out["kelly_fraction"] = profile.kelly_fraction
    out["max_position_pct"] = profile.max_position_pct
    out["min_composite_score"] = profile.min_composite_score
    out["_active_risk_profile"] = profile.name
    return out


def log_active_profile(profile: RiskProfile, context: dict | None = None) -> None:
    """Audit which profile fired and why. Call once per scan/cycle."""
    log_event("risk_profile", "active", {
        "profile": profile.name,
        "kelly_fraction": profile.kelly_fraction,
        "max_position_pct": profile.max_position_pct,
        "min_composite_score": profile.min_composite_score,
        "rationale": profile.rationale,
        "context": context or {},
    })
