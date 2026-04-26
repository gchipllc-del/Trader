"""
Tests for lib/kelly.py — Kelly Criterion position sizing.

This is pure math. The risk if it's wrong is direct dollar loss: an
overconfident position size on a low-edge trade is exactly how Kelly
"goes to ruin" warnings are written. So we test:

  * boundary inputs (win_prob 0/1, risk 0, reward <= 0) safely return 0
  * monotonicity: more edge → bigger size, all else equal
  * fractional Kelly is < full Kelly
  * the position-size pipeline never exceeds the max_position_pct cap
  * Kronos adjustment is bounded and bidirectional
"""

from __future__ import annotations

import pytest

from lib.kelly import (
    composite_to_win_prob,
    expected_value_stock,
    fractional_kelly_stock,
    kelly_fraction,
    kelly_fraction_stock,
    kelly_position_size,
)


# --------------------------------------------------------------------------
# composite_to_win_prob
# --------------------------------------------------------------------------

class TestCompositeToWinProb:
    def test_zero_score_returns_floor(self):
        # Linear: 0.35 + (0/13)*0.40 = 0.35
        assert composite_to_win_prob(0) == pytest.approx(0.35)

    def test_max_score_returns_ceiling(self):
        # 0.35 + (13/13)*0.40 = 0.75
        assert composite_to_win_prob(13) == pytest.approx(0.75)

    def test_midpoint(self):
        # 0.35 + (6.5/13)*0.40 = 0.55
        assert composite_to_win_prob(6, max_score=13) == pytest.approx(0.5346, abs=1e-3)

    def test_zero_max_score_returns_neutral(self):
        # Defensive guard against div-by-zero — should not raise
        assert composite_to_win_prob(5, max_score=0) == 0.5

    def test_monotonic_in_score(self):
        prev = -1.0
        for s in range(0, 14):
            curr = composite_to_win_prob(s)
            assert curr > prev
            prev = curr


# --------------------------------------------------------------------------
# kelly_fraction_stock — pure formula
# --------------------------------------------------------------------------

class TestKellyFractionStock:
    def test_known_value(self):
        # p=0.6, reward=10%, risk=5%, b=2 → f = (0.6*2 - 0.4)/2 = 0.4
        assert kelly_fraction_stock(0.6, 0.10, 0.05) == pytest.approx(0.4)

    def test_zero_edge_returns_zero(self):
        # 50/50 with symmetric reward/risk → 0
        assert kelly_fraction_stock(0.5, 0.05, 0.05) == pytest.approx(0.0)

    def test_negative_edge_returns_negative_kelly(self):
        # p=0.4, b=1 → (0.4 - 0.6)/1 = -0.2 (caller treats this as "don't trade")
        assert kelly_fraction_stock(0.4, 0.05, 0.05) == pytest.approx(-0.2)

    def test_zero_win_prob_returns_zero(self):
        assert kelly_fraction_stock(0.0, 0.10, 0.05) == 0.0

    def test_certain_win_returns_zero(self):
        # The formula breaks at p=1 (q=0). Per code, must guard and return 0.
        assert kelly_fraction_stock(1.0, 0.10, 0.05) == 0.0

    def test_zero_risk_returns_zero(self):
        # Division-by-zero guard
        assert kelly_fraction_stock(0.6, 0.10, 0.0) == 0.0

    def test_negative_risk_returns_zero(self):
        assert kelly_fraction_stock(0.6, 0.10, -0.05) == 0.0

    def test_monotonic_in_win_prob(self):
        prev = -10
        for p in [0.4, 0.5, 0.6, 0.7, 0.8]:
            curr = kelly_fraction_stock(p, 0.10, 0.05)
            assert curr > prev
            prev = curr


# --------------------------------------------------------------------------
# fractional_kelly_stock — safety multiplier
# --------------------------------------------------------------------------

class TestFractionalKelly:
    def test_quarter_kelly_is_quarter_of_full(self):
        full = kelly_fraction_stock(0.6, 0.10, 0.05)
        quarter = fractional_kelly_stock(0.6, 0.10, 0.05, fraction=0.25)
        assert quarter == pytest.approx(full * 0.25)

    def test_half_kelly_is_half_of_full(self):
        full = kelly_fraction_stock(0.6, 0.10, 0.05)
        half = fractional_kelly_stock(0.6, 0.10, 0.05, fraction=0.5)
        assert half == pytest.approx(full * 0.5)

    def test_negative_full_kelly_returns_zero(self):
        # Bad-edge trade → don't bet, regardless of multiplier
        assert fractional_kelly_stock(0.4, 0.05, 0.05, fraction=0.5) == 0.0

    def test_capped_at_one(self):
        # Even very strong edge × half should never exceed 100%
        out = fractional_kelly_stock(0.95, 0.50, 0.01, fraction=0.5)
        assert out <= 1.0

    def test_default_fraction_loaded_from_config(self):
        # Smoke: the function must work without an explicit fraction
        # (loads from wheel_strategy.yaml). Ratio doesn't matter here —
        # just that no exception is raised.
        out = fractional_kelly_stock(0.6, 0.10, 0.05)
        assert 0.0 <= out <= 1.0


# --------------------------------------------------------------------------
# expected_value_stock
# --------------------------------------------------------------------------

class TestExpectedValueStock:
    def test_positive_ev_at_breakeven_plus(self):
        # 60% × 10% - 40% × 5% = 0.06 - 0.02 = 0.04
        assert expected_value_stock(0.6, 0.10, 0.05) == pytest.approx(0.04)

    def test_zero_ev_at_breakeven(self):
        # p such that p*r = (1-p)*risk; symmetric here → p=0.5
        assert expected_value_stock(0.5, 0.05, 0.05) == pytest.approx(0.0)

    def test_negative_ev_below_breakeven(self):
        assert expected_value_stock(0.3, 0.10, 0.05) < 0


# --------------------------------------------------------------------------
# kelly_position_size — full pipeline
# --------------------------------------------------------------------------

class TestKellyPositionSize:
    GOOD_TRADE = dict(
        portfolio_value=100_000,
        current_price=50.0,
        target_price=55.0,    # +10%
        stop_loss=47.5,        # -5%
        composite_score=10,
    )

    def test_happy_path_returns_positive_shares(self):
        result = kelly_position_size(**self.GOOD_TRADE)
        assert result["shares"] > 0
        assert result["reason"] == "kelly_sized"
        assert 0 < result["pct_of_portfolio"] <= 1.0
        assert result["reward_pct"] == pytest.approx(0.10)
        assert result["risk_pct"] == pytest.approx(0.05)

    def test_invalid_portfolio_returns_zero(self):
        result = kelly_position_size(
            portfolio_value=0,
            current_price=50.0,
            target_price=55.0,
            stop_loss=47.5,
            composite_score=10,
        )
        assert result["shares"] == 0
        assert result["reason"] == "invalid_inputs"

    def test_negative_current_price_returns_zero(self):
        result = kelly_position_size(
            portfolio_value=100_000,
            current_price=-1.0,
            target_price=55.0,
            stop_loss=47.5,
            composite_score=10,
        )
        assert result["shares"] == 0

    def test_target_below_entry_returns_zero(self):
        # If target < entry, reward_pct < 0 → don't trade
        result = kelly_position_size(
            portfolio_value=100_000,
            current_price=50.0,
            target_price=49.0,    # below entry — invalid
            stop_loss=47.5,
            composite_score=10,
        )
        assert result["shares"] == 0
        assert result["reason"] == "invalid_target_or_stop"

    def test_stop_above_entry_returns_zero(self):
        result = kelly_position_size(
            portfolio_value=100_000,
            current_price=50.0,
            target_price=55.0,
            stop_loss=51.0,       # stop above entry — invalid
            composite_score=10,
        )
        assert result["shares"] == 0
        assert result["reason"] == "invalid_target_or_stop"

    def test_low_score_negative_edge_returns_zero(self):
        # Score 0 → win_prob 0.35; reward 10%, risk 5% (b=2)
        # Kelly = (0.35*2 - 0.65)/2 = 0.025 — barely positive, marginal.
        # Drop reward to 4% so b=0.8: (0.35*0.8 - 0.65)/0.8 = -0.4625 → 0 shares.
        result = kelly_position_size(
            portfolio_value=100_000,
            current_price=50.0,
            target_price=52.0,    # +4%
            stop_loss=47.5,        # -5%
            composite_score=0,
        )
        assert result["shares"] == 0
        assert result["reason"] == "negative_edge"

    def test_max_position_pct_caps_size(self):
        # Force a strong edge that would otherwise size huge:
        # p=0.75 (score 13), b = 0.20/0.02 = 10 → Kelly = (0.75*10 - 0.25)/10 = 0.725
        # × 0.25 quarter Kelly = 0.181. Cap at 0.10 to verify.
        result = kelly_position_size(
            portfolio_value=100_000,
            current_price=10.0,
            target_price=12.0,    # +20%
            stop_loss=9.8,        # -2%
            composite_score=13,
            max_position_pct=0.10,
            fraction=0.5,         # half Kelly to push past 10%
        )
        assert result["pct_of_portfolio"] == pytest.approx(0.10)
        assert "capped_at" in result["reason"]
        # shares should equal portfolio * cap / price
        assert result["shares"] == int(100_000 * 0.10 / 10.0)

    def test_kronos_bullish_bumps_win_prob(self):
        baseline = kelly_position_size(
            kronos_expected_return=None,
            **self.GOOD_TRADE,
        )
        bullish = kelly_position_size(
            kronos_expected_return=0.05,   # +5% forecast
            **self.GOOD_TRADE,
        )
        assert bullish["win_prob"] > baseline["win_prob"]
        assert bullish["kronos_adjustment"] > 0

    def test_kronos_bearish_lowers_win_prob(self):
        baseline = kelly_position_size(
            kronos_expected_return=None,
            **self.GOOD_TRADE,
        )
        bearish = kelly_position_size(
            kronos_expected_return=-0.05,
            **self.GOOD_TRADE,
        )
        assert bearish["win_prob"] < baseline["win_prob"]
        assert bearish["kronos_adjustment"] < 0

    def test_kronos_adjustment_clamped(self):
        # Extreme Kronos forecast must be clamped to ±0.15.
        result = kelly_position_size(
            kronos_expected_return=0.99,
            **self.GOOD_TRADE,
        )
        assert result["kronos_adjustment"] == pytest.approx(0.15)

    def test_win_prob_clamped_after_kronos(self):
        # Baseline win_prob (score 10) ≈ 0.658; +0.15 = 0.808; cap at 0.85.
        # Build an extreme case to verify.
        result = kelly_position_size(
            portfolio_value=100_000,
            current_price=50.0,
            target_price=55.0,
            stop_loss=47.5,
            composite_score=13,            # 0.75
            kronos_expected_return=0.50,   # clamped to +0.15 → 0.90 → cap 0.85
        )
        assert result["win_prob"] <= 0.85


# --------------------------------------------------------------------------
# Legacy prediction-market kelly_fraction
# --------------------------------------------------------------------------

class TestLegacyKellyFraction:
    def test_zero_market_prob_returns_zero(self):
        assert kelly_fraction(0.6, 0.0) == 0.0

    def test_one_market_prob_returns_zero(self):
        assert kelly_fraction(0.6, 1.0) == 0.0

    def test_zero_our_prob_returns_zero(self):
        assert kelly_fraction(0.0, 0.5) == 0.0

    def test_one_our_prob_returns_zero(self):
        assert kelly_fraction(1.0, 0.5) == 0.0

    def test_known_value(self):
        # our=0.6, market=0.5 → b = 0.5/0.5 = 1, kelly = (0.6 - 0.4)/1 = 0.2
        assert kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
