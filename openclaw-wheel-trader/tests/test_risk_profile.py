"""
Tests for lib/risk_profile — Risky/Neutral/Safe deterministic selection.

Locks in the rule that SAFE always wins on any defensive trigger
(bear regime / drawdown / cooldown), so a future tweak can't accidentally
let RISKY fire while we're bleeding.
"""

import pytest


class TestSelectProfile:
    def test_default_state_returns_neutral(self):
        from lib.risk_profile import select_profile, NEUTRAL
        # Sideways regime, no drawdown, no cooldown, modest bankroll
        p = select_profile(
            regime="sideways", bankroll=1500.0,
            daily_loss_pct=0.0, cooldown_active=False,
        )
        assert p is NEUTRAL

    def test_bear_regime_forces_safe(self):
        from lib.risk_profile import select_profile, SAFE
        # Even with high bankroll, bear regime → SAFE
        p = select_profile(
            regime="bear", bankroll=50_000.0,
            daily_loss_pct=0.02, cooldown_active=False,
        )
        assert p is SAFE

    def test_deep_drawdown_forces_safe(self):
        from lib.risk_profile import select_profile, SAFE
        # Bull regime but daily loss exceeds threshold
        p = select_profile(
            regime="bull", bankroll=50_000.0,
            daily_loss_pct=-0.06,  # -6% > -5% threshold
        )
        assert p is SAFE

    def test_cooldown_forces_safe(self):
        from lib.risk_profile import select_profile, SAFE
        p = select_profile(
            regime="bull", bankroll=50_000.0,
            daily_loss_pct=0.0, cooldown_active=True,
        )
        assert p is SAFE

    def test_risky_requires_all_three_conditions(self):
        """RISKY only when bull regime AND bankroll ≥ $5k AND no daily loss."""
        from lib.risk_profile import select_profile, RISKY, NEUTRAL

        # All conditions met → RISKY
        p = select_profile(regime="bull", bankroll=10_000.0, daily_loss_pct=0.01)
        assert p is RISKY

        # Bankroll too low → NEUTRAL (not RISKY)
        p = select_profile(regime="bull", bankroll=2_000.0, daily_loss_pct=0.01)
        assert p is NEUTRAL

        # Slight loss → NEUTRAL
        p = select_profile(regime="bull", bankroll=10_000.0, daily_loss_pct=-0.01)
        assert p is NEUTRAL

        # Sideways regime → NEUTRAL
        p = select_profile(regime="sideways", bankroll=10_000.0, daily_loss_pct=0.0)
        assert p is NEUTRAL

    def test_unknown_regime_defaults_to_neutral(self):
        from lib.risk_profile import select_profile, NEUTRAL
        p = select_profile(regime="unknown", bankroll=10_000.0, daily_loss_pct=0.0)
        assert p is NEUTRAL


class TestApplyProfile:
    def test_overrides_three_knobs(self):
        from lib.risk_profile import apply_profile_to, RISKY, NEUTRAL, SAFE
        base = {
            "kelly_fraction": 0.25,
            "max_position_pct": 0.30,
            "min_composite_score": 5,
            "other_param": "preserved",
        }
        for profile in (RISKY, NEUTRAL, SAFE):
            out = apply_profile_to(base, profile)
            assert out["kelly_fraction"] == profile.kelly_fraction
            assert out["max_position_pct"] == profile.max_position_pct
            assert out["min_composite_score"] == profile.min_composite_score
            assert out["other_param"] == "preserved"
            assert out["_active_risk_profile"] == profile.name
            # Ensure base wasn't mutated
            assert base["kelly_fraction"] == 0.25

    def test_safe_is_more_conservative_than_risky(self):
        from lib.risk_profile import SAFE, RISKY
        assert SAFE.kelly_fraction < RISKY.kelly_fraction
        assert SAFE.max_position_pct < RISKY.max_position_pct
        assert SAFE.min_composite_score > RISKY.min_composite_score


class TestProfileBoundsRespectBreakers:
    """Profile values must never exceed circuit_breaker ceilings — that
    would let a profile silently smash through the absolute cap."""

    def test_max_position_pct_within_breaker_ceiling(self):
        """settings.yaml circuit_breakers.max_position_pct = 0.30."""
        from lib.risk_profile import RISKY, NEUTRAL, SAFE
        BREAKER_CEILING = 0.30
        for p in (RISKY, NEUTRAL, SAFE):
            assert p.max_position_pct <= BREAKER_CEILING, (
                f"{p.name} max_position_pct {p.max_position_pct} exceeds "
                f"breaker ceiling {BREAKER_CEILING}"
            )

    def test_kelly_fraction_capped_at_half_kelly(self):
        """Full Kelly = ruin risk. Half-Kelly is the cap."""
        from lib.risk_profile import RISKY, NEUTRAL, SAFE
        for p in (RISKY, NEUTRAL, SAFE):
            assert p.kelly_fraction <= 0.50
            assert p.kelly_fraction > 0
