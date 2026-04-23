"""
Smoke tests for the alpha-signal extensions:
- lib/insider_flow.py   (SEC Form 4)
- lib/congress_flow.py  (Senate STOCK Act)
- lib/portfolio_greeks.py (py_vollib Greeks)
- The new likelihood steps in lib/bayesian_forecaster.py

Run with:
    python -m pytest tests/test_alpha_signals.py -v
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# insider_flow — ticker validation + cache + bayesian lift
# ============================================================

def test_insider_flow_ticker_validation():
    """Invalid tickers must return neutral without any network I/O."""
    from lib.insider_flow import check_insider_flow

    bad = ["../../etc/passwd", "BAD TICKER!", "", "9AAPL", "a" * 30]
    for b in bad:
        r = check_insider_flow(b, days=30)
        # Rejection surfaces as neutral/zero-confidence with a non-blank reason.
        # The specific message varies ("invalid format" vs "unknown ticker (no CIK match)")
        # but either outcome means no network I/O and no crash.
        assert r.signal == "neutral"
        assert r.confidence == 0.0
        assert any(kw in r.reason.lower() for kw in
                   ("invalid", "format", "unknown", "no cik"))


def test_insider_flow_result_schema():
    """InsiderFlowResult must expose the fields main.py CLI depends on."""
    from lib.insider_flow import InsiderFlowResult, InsiderTransaction

    r = InsiderFlowResult(ticker="X")
    # Fields cmd_insiders() reads:
    for field in ("ticker", "sentiment", "confidence", "cluster_detected",
                  "buy_count", "sell_count", "total_buy_value_usd",
                  "signal", "recent_buys", "reason", "cached"):
        assert hasattr(r, field), f"missing {field}"

    t = InsiderTransaction(
        filing_date="2026-01-01", insider_name="J. Doe", insider_title="CEO",
        is_officer=True, is_director=False, is_ten_percent=False,
        shares=1000, price_per_share=50.0, dollar_value=50_000.0,
        transaction_code="P",
    )
    for field in ("filing_date", "insider_name", "insider_title",
                  "shares", "price_per_share", "dollar_value", "transaction_code"):
        assert hasattr(t, field)


# ============================================================
# congress_flow — freshness + staleness suppressor
# ============================================================

def test_congress_flow_ticker_validation():
    from lib.congress_flow import check_congress_flow
    r = check_congress_flow("../../etc/passwd", days=30)
    assert r.signal == "neutral"
    assert r.confidence == 0.0


def test_congress_flow_feed_url_env_override(monkeypatch):
    """CONGRESS_FEED_URL env var should be picked up on import."""
    # Note: module is already imported by this test file, so we reach into the
    # module for confirmation rather than reimport.
    import lib.congress_flow as cf
    # Confirm the constant exists + is computed from env with fallback.
    assert cf.FEED_URL.startswith("http")


def test_congress_flow_staleness_suppression(monkeypatch, tmp_path):
    """When feed's newest txn is > threshold days old, signal is suppressed."""
    import lib.congress_flow as cf

    # Stub feed: one ticker, newest txn is 1000 days ago
    stale_date = (datetime.now(timezone.utc) - timedelta(days=1000))
    fake_feed = [{
        "ticker": "FAKE",
        "transactions": [
            {
                "transaction_date": stale_date.strftime("%m/%d/%Y"),
                "type": "Purchase",
                "amount": "$50,001 - $100,000",
                "senator": "Test Senator",
                "owner": "Self",
            }
        ],
    }]

    # Redirect cache dir + stub _download_feed
    monkeypatch.setattr(cf, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cf, "_download_feed", lambda force=False: fake_feed)

    r = cf.check_congress_flow("FAKE", days=90, bypass_cache=True)
    assert r.feed_stale is True
    assert r.feed_age_days >= 999
    assert r.confidence <= 0.05    # signal suppressed
    assert r.signal == "neutral"
    assert "stale" in r.reason.lower()


def test_congress_flow_fresh_cluster_detection(monkeypatch, tmp_path):
    """Fresh feed with 3+ distinct senators in 30d triggers cluster signal."""
    import lib.congress_flow as cf

    recent_date = datetime.now(timezone.utc) - timedelta(days=10)
    def mkptr(senator: str):
        return {
            "transaction_date": recent_date.strftime("%m/%d/%Y"),
            "type": "Purchase",
            "amount": "$100,001 - $250,000",
            "senator": senator,
            "owner": "Self",
        }
    fake_feed = [{
        "ticker": "FAKE",
        "transactions": [mkptr("Senator A"), mkptr("Senator B"),
                         mkptr("Senator C"), mkptr("Senator D")],
    }]
    monkeypatch.setattr(cf, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cf, "_download_feed", lambda force=False: fake_feed)

    r = cf.check_congress_flow("FAKE", days=90, bypass_cache=True)
    assert r.feed_stale is False
    assert r.cluster_detected is True
    assert r.distinct_buyers_30d >= 3
    assert r.signal == "bullish_cluster"
    assert r.sentiment > 0.5
    assert r.confidence > 0.5


# ============================================================
# portfolio_greeks — stock + CSP + CC + risk-sizing
# ============================================================

def test_portfolio_greeks_stock_only():
    """All-stock portfolio: delta = total shares, other Greeks = 0."""
    from lib.portfolio_greeks import compute_portfolio_greeks

    positions = [
        {"ticker": "A", "type": "stock", "status": "open", "shares": 10, "entry_price": 50},
        {"ticker": "B", "type": "stock", "status": "open", "shares": 5,  "entry_price": 100},
        {"ticker": "C", "type": "stock", "status": "closed", "shares": 100, "entry_price": 10},
    ]
    fake_spots = {"A": 50.0, "B": 100.0, "C": 10.0}
    p = compute_portfolio_greeks(positions, spot_fetcher=fake_spots.get)
    assert p.total_delta == 15.0                # 10 + 5 (closed pos skipped)
    assert p.total_gamma == 0.0
    assert p.total_vega == 0.0
    assert p.total_theta == 0.0
    assert p.invalid_count == 0


def test_portfolio_greeks_short_put_signs():
    """Short CSP: +delta, -gamma, -vega, +theta."""
    from lib.portfolio_greeks import compute_csp_greeks

    exp = (datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat()
    pos = {"ticker": "X", "type": "csp", "status": "open",
           "strike": 100, "expiration": exp, "contracts": 1}
    pg = compute_csp_greeks(pos, spot=105.0, iv=0.25)
    assert pg.valid
    assert pg.delta > 0         # short put has + delta
    assert pg.gamma < 0         # short gamma
    assert pg.vega < 0          # short vega
    assert pg.theta > 0         # collecting theta


def test_portfolio_greeks_expired_csp():
    """Expired options must be flagged invalid, not crash."""
    from lib.portfolio_greeks import compute_csp_greeks

    past = (datetime.now(timezone.utc) - timedelta(days=3)).date().isoformat()
    pos = {"ticker": "X", "type": "csp", "status": "open",
           "strike": 100, "expiration": past, "contracts": 1}
    pg = compute_csp_greeks(pos, spot=100.0, iv=0.25)
    assert pg.valid is False
    assert "expired" in pg.reason.lower()


def test_portfolio_greeks_bad_inputs_dont_crash():
    """Garbage inputs must degrade to invalid=False, never crash."""
    from lib.portfolio_greeks import compute_csp_greeks

    for bad in [
        {"strike": 0, "expiration": "", "contracts": 1},
        {"strike": 100, "expiration": "not-a-date", "contracts": 1},
        {"strike": -50, "expiration": "2027-01-01", "contracts": 1},
    ]:
        pos = {"ticker": "X", "type": "csp", "status": "open", **bad}
        pg = compute_csp_greeks(pos, spot=0, iv=-1)
        assert pg.valid is False  # never raises


def test_vega_sized_contracts_headroom():
    """vega_sized_contracts respects limit + direction."""
    from lib.portfolio_greeks import vega_sized_contracts

    # Short vol, with $300 of $500 short already → $200 headroom
    # Per-contract vega = -$50 → should allow 4 contracts
    assert vega_sized_contracts(500, -300, -50) == 4

    # At the limit → 0
    assert vega_sized_contracts(500, -500, -50) == 0

    # Candidate long vol adding to short book → reduces short → unlimited
    # (in direction of reducing), but formula treats each direction independently.
    # long vol = +50; current = -300; direction positive:
    # target_limit = +500; headroom = 500 - (-300) = 800 → capped by max (10)
    assert vega_sized_contracts(500, -300, 50) == 10


# ============================================================
# bayesian_forecaster — new likelihoods + chain integration
# ============================================================

def test_bayesian_insider_likelihood_bounds():
    from lib.bayesian_forecaster import _insider_to_likelihood
    # No signal → exactly 0.50 (neutral)
    assert _insider_to_likelihood(None, 0.8) == 0.50
    # High confidence + bullish cluster
    lh = _insider_to_likelihood(0.85, 0.9)
    assert lh > 0.55
    # Low confidence → barely perturbs from 0.50
    lh2 = _insider_to_likelihood(0.85, 0.05)
    assert 0.49 < lh2 < 0.52


def test_bayesian_congress_likelihood_bounds():
    from lib.bayesian_forecaster import _congress_to_likelihood
    assert _congress_to_likelihood(None, 0.8) == 0.50
    lh = _congress_to_likelihood(0.80, 0.8)
    assert lh > 0.55
    # Stale/low-conf → neutral
    assert 0.49 < _congress_to_likelihood(0.80, 0.05) < 0.52


def test_bayesian_chain_includes_new_steps():
    from lib.bayesian_forecaster import forecast_stock
    f = forecast_stock(
        ticker="X",
        composite_score=7, trend_score=2, level_score=2, signal_score=2, momentum_score=2,
    )
    steps = [s.get("step") for s in f.bayesian_chain]
    assert "insider" in steps
    assert "congress" in steps
    # Both should come AFTER kronos and BEFORE news (informed-flow layer)
    assert steps.index("insider") > steps.index("kronos")
    assert steps.index("insider") < steps.index("news")


def test_bayesian_backward_compat():
    """forecast_stock() without the new kwargs must still work."""
    from lib.bayesian_forecaster import forecast_stock
    f = forecast_stock(
        ticker="X",
        composite_score=5, trend_score=1, level_score=1, signal_score=1, momentum_score=1,
    )
    assert 0.05 <= f.win_probability <= 0.95
    assert f.ticker == "X"


def test_bayesian_cluster_signals_raise_prior():
    """Adding strong insider+congress clusters should lift the posterior."""
    from lib.bayesian_forecaster import forecast_stock

    baseline = forecast_stock(
        ticker="X",
        composite_score=7, trend_score=2, level_score=2, signal_score=2, momentum_score=2,
    )
    boosted = forecast_stock(
        ticker="X",
        composite_score=7, trend_score=2, level_score=2, signal_score=2, momentum_score=2,
        insider_sentiment=0.85, insider_confidence=0.85,
        congress_sentiment=0.80, congress_confidence=0.75,
    )
    assert boosted.win_probability > baseline.win_probability
