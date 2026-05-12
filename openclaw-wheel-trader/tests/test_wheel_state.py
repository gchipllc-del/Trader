"""Tests for lib.wheel_state — global wheel state classification + capital-at-risk."""

from __future__ import annotations

import pytest

from lib.wheel_state import (
    IllegalWheelState,
    WheelLeg,
    WheelState,
    classify_book,
    parse_occ_symbol,
    total_capital_at_risk,
)


# ── OCC parsing ──────────────────────────────────────────────────────

class TestOccParsing:
    def test_stock_symbol_returns_none(self):
        assert parse_occ_symbol("SPY") is None
        assert parse_occ_symbol("AAPL") is None
        assert parse_occ_symbol("BRK.B") is None

    def test_crypto_symbol_returns_none(self):
        assert parse_occ_symbol("BTC/USD") is None
        assert parse_occ_symbol("ETH/USD") is None

    def test_short_string_returns_none(self):
        # Less than the 15-char OCC suffix → can't be an option
        assert parse_occ_symbol("ABC123") is None
        assert parse_occ_symbol("") is None

    def test_valid_call_symbol(self):
        result = parse_occ_symbol("SPY250620C00500000")
        assert result == ("SPY", "2025-06-20", "call", 500.0)

    def test_valid_put_symbol(self):
        result = parse_occ_symbol("AAPL260117P00170500")
        assert result == ("AAPL", "2026-01-17", "put", 170.5)

    def test_fractional_strike(self):
        # Strike of 170.50
        _, _, _, strike = parse_occ_symbol("AAPL260117P00170500")
        assert strike == 170.5

    def test_high_strike(self):
        # NVDA at $1234.567
        result = parse_occ_symbol("NVDA260117C01234567")
        assert result == ("NVDA", "2026-01-17", "call", 1234.567)

    def test_garbage_suffix(self):
        # Right length but wrong format
        assert parse_occ_symbol("SPYXXXXXXXXXXXXXX") is None
        assert parse_occ_symbol("SPY250620X00500000") is None  # neither C nor P


# ── Legal-state classification ───────────────────────────────────────

def _leg(symbol, qty, market_value=0.0, side="long"):
    """Build a minimal broker position dict matching AlpacaClient.get_positions shape."""
    return {
        "symbol": symbol, "qty": qty, "market_value": market_value, "side": side,
    }


class TestLegalStates:
    def test_empty_book(self):
        assert classify_book([]) == {}

    def test_flat_when_no_positions_for_ticker(self):
        # The book has SPY but no AAPL — AAPL won't appear in the result
        book = classify_book([_leg("SPY", 100, 50_000)])
        assert "AAPL" not in book
        assert "SPY" in book

    def test_short_put_only(self):
        # Short 1 contract of AAPL170P
        book = classify_book([
            _leg("AAPL260117P00170000", -1, -50, side="short"),
        ])
        assert book["AAPL"].stage == "short_put"
        assert book["AAPL"].short_put_collateral == 170 * 100 * 1
        assert book["AAPL"].long_share_count == 0

    def test_long_shares_only(self):
        # Assigned but no CC yet
        book = classify_book([_leg("AAPL", 100, 17_000)])
        assert book["AAPL"].stage == "long_shares"
        assert book["AAPL"].long_share_count == 100
        assert book["AAPL"].long_share_value == 17_000

    def test_long_shares_with_cc(self):
        book = classify_book([
            _leg("AAPL", 100, 17_500),
            _leg("AAPL260117C00180000", -1, -25, side="short"),
        ])
        assert book["AAPL"].stage == "long_shares_with_cc"
        assert book["AAPL"].short_call_count == 1
        assert book["AAPL"].long_share_count == 100

    def test_long_shares_with_csp(self):
        # Shares held AND a new CSP at a lower strike — legal pyramid
        book = classify_book([
            _leg("AAPL", 100, 17_000),
            _leg("AAPL260117P00160000", -1, -40, side="short"),
        ])
        assert book["AAPL"].stage == "long_shares_with_csp"

    def test_multiple_short_puts_same_underlying(self):
        # Two CSPs at different strikes — legal, both contribute to collateral
        book = classify_book([
            _leg("AAPL260117P00170000", -1, -50, side="short"),
            _leg("AAPL260117P00165000", -1, -30, side="short"),
        ])
        assert book["AAPL"].stage == "short_put"
        assert book["AAPL"].short_put_collateral == (170 + 165) * 100

    def test_book_with_mixed_underlyings(self):
        # AAPL has shares, NVDA has CSP, SPY is flat shares
        book = classify_book([
            _leg("AAPL", 100, 17_000),
            _leg("NVDA260117P00500000", -1, -200, side="short"),
            _leg("SPY", 50, 25_000),
        ])
        assert book["AAPL"].stage == "long_shares"
        assert book["NVDA"].stage == "short_put"
        assert book["SPY"].stage == "long_shares"
        assert book["NVDA"].short_put_collateral == 500 * 100


# ── Illegal-state detection ──────────────────────────────────────────

class TestIllegalStates:
    def test_short_equity_raises(self):
        with pytest.raises(IllegalWheelState) as exc:
            classify_book([_leg("AAPL", -100, -17_000, side="short")])
        assert "short equity" in str(exc.value).lower()
        assert exc.value.per_underlying["AAPL"].stage == "illegal"

    def test_long_put_raises(self):
        with pytest.raises(IllegalWheelState) as exc:
            classify_book([
                _leg("AAPL260117P00170000", 1, 50, side="long"),
            ])
        assert "long option" in str(exc.value).lower()

    def test_long_call_raises(self):
        with pytest.raises(IllegalWheelState) as exc:
            classify_book([
                _leg("AAPL260117C00180000", 1, 50, side="long"),
            ])
        assert "long option" in str(exc.value).lower()

    def test_uncovered_short_call_raises(self):
        # 2 short calls but only 100 shares — covers 1, leaves 1 naked
        with pytest.raises(IllegalWheelState) as exc:
            classify_book([
                _leg("AAPL", 100, 17_000),
                _leg("AAPL260117C00180000", -2, -50, side="short"),
            ])
        assert "uncovered" in str(exc.value).lower()

    def test_short_call_with_no_shares_raises(self):
        with pytest.raises(IllegalWheelState) as exc:
            classify_book([
                _leg("AAPL260117C00180000", -1, -25, side="short"),
            ])
        # Detected by the uncovered-shares check (0 shares < 100 needed)
        assert "uncovered" in str(exc.value).lower()

    def test_raise_on_illegal_false_returns_book(self):
        # Diagnostic mode — return result without raising
        book = classify_book(
            [_leg("AAPL", -100, -17_000, side="short")],
            raise_on_illegal=False,
        )
        assert book["AAPL"].stage == "illegal"
        assert "short equity" in book["AAPL"].illegal_reason.lower()


# ── Capital-at-risk aggregation ──────────────────────────────────────

class TestCapitalAtRisk:
    def test_empty_book(self):
        result = total_capital_at_risk({})
        assert result == {"short_put_collateral": 0.0,
                          "long_share_value": 0.0, "total": 0.0}

    def test_sum_across_book(self):
        book = classify_book([
            # AAPL: 100 shares @ $17k + short put @ 165 strike → +$16,500 collateral
            _leg("AAPL", 100, 17_000),
            _leg("AAPL260117P00165000", -1, -30, side="short"),
            # NVDA: short put only @ 500 strike → +$50,000 collateral
            _leg("NVDA260117P00500000", -1, -200, side="short"),
            # SPY: 50 shares @ $25k
            _leg("SPY", 50, 25_000),
        ])
        car = total_capital_at_risk(book)
        assert car["short_put_collateral"] == 16_500 + 50_000
        assert car["long_share_value"] == 17_000 + 25_000
        assert car["total"] == 16_500 + 50_000 + 17_000 + 25_000
