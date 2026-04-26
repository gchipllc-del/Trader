"""
Tests for Kronos batch inference path.

Covers:
- predict_prices_batch happy path (3 tickers in, 3 KronosForecast out, in order)
- partial cache hit: only the misses go through predict_batch
- empty input: returns [] without invoking the predictor
- mixed lookback lengths: aligned to a common min and an audit event is logged
- stock_engine._apply_kronos_gate falls back to per-ticker when batch raises

The model itself is too heavy to load in CI — we mock at the predictor level
(`lib.kronos_forecaster._load_predictor` returns a MagicMock whose
`.predict_batch` returns a list of pandas DataFrames in input order).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200, start_close: float = 100.0) -> pd.DataFrame:
    """Build a synthetic OHLCV frame Kronos's wrapper accepts."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = [start_close + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
        },
        index=idx,
    )


def _make_pred_df(pred_bars: int, base: float) -> pd.DataFrame:
    """Build a DataFrame with the columns predict_batch returns."""
    idx = pd.date_range("2024-12-01", periods=pred_bars, freq="B")
    return pd.DataFrame(
        {
            "open":   [base + i * 0.1 for i in range(pred_bars)],
            "high":   [base + 1.0 + i * 0.1 for i in range(pred_bars)],
            "low":    [base - 1.0 + i * 0.1 for i in range(pred_bars)],
            "close":  [base + 0.5 + i * 0.1 for i in range(pred_bars)],
            "volume": [1_000_000] * pred_bars,
            "amount": [base * 1_000_000] * pred_bars,
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the file cache at a tmp dir so tests don't read/write real cache."""
    from lib import kronos_forecaster as kf

    monkeypatch.setattr(kf, "CACHE_DIR", tmp_path / "kronos_cache")
    yield


# ─────────────────────────────────────────────────────────────────
# Happy path: 3 tickers, no cache
# ─────────────────────────────────────────────────────────────────

def test_predict_prices_batch_happy_path():
    """3 tickers in, 3 forecasts out, in input order, single predict_batch call."""
    from lib import kronos_forecaster as kf

    tickers = ["AAA", "BBB", "CCC"]
    pred_bars = 10

    fake_predictor = MagicMock()
    fake_predictor.predict_batch.return_value = [
        _make_pred_df(pred_bars, base=110.0),  # AAA
        _make_pred_df(pred_bars, base=210.0),  # BBB
        _make_pred_df(pred_bars, base=310.0),  # CCC
    ]

    fetch_returns = {
        "AAA": _make_ohlcv(200, start_close=100.0),
        "BBB": _make_ohlcv(200, start_close=200.0),
        "CCC": _make_ohlcv(200, start_close=300.0),
    }

    with patch.object(kf, "_load_predictor", return_value=fake_predictor) as load_mock, \
         patch.object(kf, "_fetch_ohlcv", side_effect=lambda t, **kw: fetch_returns[t]) as fetch_mock:
        out = kf.predict_prices_batch(tickers=tickers, pred_bars=pred_bars, lookback=200)

    assert load_mock.call_count == 1, "predictor must load exactly once"
    assert fake_predictor.predict_batch.call_count == 1, "exactly one batch call"
    assert fetch_mock.call_count == 3, "one fetch per cache miss"

    assert len(out) == 3
    assert [f.ticker for f in out] == tickers, "order must match input"
    for f in out:
        assert isinstance(f, kf.KronosForecast)
        assert f.pred_bars == pred_bars
        assert f.direction in {"bullish", "bearish", "neutral"}
        # New shape parity with predict_price
        assert isinstance(f.predicted_close, list) and len(f.predicted_close) == pred_bars


# ─────────────────────────────────────────────────────────────────
# One cache hit → only misses go through predict_batch
# ─────────────────────────────────────────────────────────────────

def test_predict_prices_batch_partial_cache():
    """One ticker is already cached → only the other 2 hit predict_batch."""
    from lib import kronos_forecaster as kf

    tickers = ["AAA", "BBB", "CCC"]
    pred_bars = 10

    # Pre-populate cache for BBB
    cached_forecast = {
        "ticker": "BBB",
        "interval": "1d",
        "lookback_bars": 200,
        "pred_bars": pred_bars,
        "predicted_close": [205.0] * pred_bars,
        "predicted_high": [206.0] * pred_bars,
        "predicted_low": [204.0] * pred_bars,
        "current_price": 200.0,
        "pred_final_close": 205.0,
        "pred_high_watermark": 206.0,
        "pred_low_watermark": 204.0,
        "direction": "bullish",
        "expected_return": 0.025,
        "confidence": 0.7,
    }
    bbb_key = kf._cache_key("BBB", "1d", pred_bars, model_name="NeoQuasar/Kronos-base")
    kf._cache_put(bbb_key, cached_forecast)

    fake_predictor = MagicMock()
    fake_predictor.predict_batch.return_value = [
        _make_pred_df(pred_bars, base=110.0),  # AAA
        _make_pred_df(pred_bars, base=310.0),  # CCC
    ]

    fetch_returns = {
        "AAA": _make_ohlcv(200, start_close=100.0),
        "CCC": _make_ohlcv(200, start_close=300.0),
    }

    with patch.object(kf, "_load_predictor", return_value=fake_predictor), \
         patch.object(kf, "_fetch_ohlcv", side_effect=lambda t, **kw: fetch_returns[t]) as fetch_mock:
        out = kf.predict_prices_batch(tickers=tickers, pred_bars=pred_bars, lookback=200)

    # predict_batch called with exactly 2 series, in the AAA, CCC order
    fake_predictor.predict_batch.assert_called_once()
    call_kwargs = fake_predictor.predict_batch.call_args.kwargs
    assert len(call_kwargs["df_list"]) == 2

    # Only fetched for misses
    assert fetch_mock.call_count == 2
    fetched_tickers = [c.args[0] for c in fetch_mock.call_args_list]
    assert set(fetched_tickers) == {"AAA", "CCC"}

    # Output preserves AAA, BBB, CCC order
    assert [f.ticker for f in out] == tickers
    # BBB's data came from cache verbatim
    bbb = next(f for f in out if f.ticker == "BBB")
    assert bbb.expected_return == 0.025
    assert bbb.direction == "bullish"


# ─────────────────────────────────────────────────────────────────
# Empty input
# ─────────────────────────────────────────────────────────────────

def test_predict_prices_batch_empty_input_no_predictor_load():
    """Empty list returns [] and never touches the predictor or fetcher."""
    from lib import kronos_forecaster as kf

    with patch.object(kf, "_load_predictor") as load_mock, \
         patch.object(kf, "_fetch_ohlcv") as fetch_mock:
        out = kf.predict_prices_batch(tickers=[], pred_bars=10)

    assert out == []
    load_mock.assert_not_called()
    fetch_mock.assert_not_called()


# ─────────────────────────────────────────────────────────────────
# Mixed lookback lengths get aligned + logged
# ─────────────────────────────────────────────────────────────────

def test_predict_prices_batch_aligns_unequal_lookback():
    """When fetched data lengths differ, all series truncate to common min."""
    from lib import kronos_forecaster as kf

    tickers = ["AAA", "BBB"]
    pred_bars = 5

    fake_predictor = MagicMock()
    fake_predictor.predict_batch.return_value = [
        _make_pred_df(pred_bars, base=110.0),
        _make_pred_df(pred_bars, base=210.0),
    ]

    # AAA has 250 bars, BBB has only 120. Common min = 120.
    fetch_returns = {
        "AAA": _make_ohlcv(250, start_close=100.0),
        "BBB": _make_ohlcv(120, start_close=200.0),
    }

    captured_events: list[tuple[str, dict]] = []

    def fake_log(event_type, action, details=None, result="pending"):
        captured_events.append((action, details or {}))
        return {}

    with patch.object(kf, "_load_predictor", return_value=fake_predictor), \
         patch.object(kf, "_fetch_ohlcv", side_effect=lambda t, **kw: fetch_returns[t]), \
         patch.object(kf, "log_event", side_effect=fake_log):
        out = kf.predict_prices_batch(
            tickers=tickers, pred_bars=pred_bars, lookback=200
        )

    # The batch call's df_list series must all be the same length (= 120, the min)
    call_kwargs = fake_predictor.predict_batch.call_args.kwargs
    lengths = [len(df) for df in call_kwargs["df_list"]]
    assert len(set(lengths)) == 1, f"expected equal lengths, got {lengths}"
    assert lengths[0] == 120

    # Audit event must record the alignment
    actions = [a for a, _ in captured_events]
    assert "batch_lookback_aligned" in actions, f"missing alignment audit, got {actions}"

    # Results still in order
    assert [f.ticker for f in out] == tickers
    for f in out:
        assert f.lookback_bars == 120


# ─────────────────────────────────────────────────────────────────
# Stock engine fallback when predict_prices_batch raises
# ─────────────────────────────────────────────────────────────────

def test_apply_kronos_gate_batch_failure_falls_back_to_per_ticker():
    """If predict_prices_batch raises, the gate falls back to predict_price."""
    from lib import stock_engine as se
    from lib import kronos_forecaster as kf

    candidates = [
        {"ticker": "AAA"},
        {"ticker": "BBB"},
    ]

    # Build distinct per-ticker forecasts the per-ticker fallback should produce
    def make_forecast(t: str, exp_ret: float, direction: str) -> kf.KronosForecast:
        return kf.KronosForecast(
            ticker=t, interval="1d", lookback_bars=200, pred_bars=10,
            predicted_close=[100.0] * 10, predicted_high=[101.0] * 10, predicted_low=[99.0] * 10,
            current_price=100.0, pred_final_close=100.0 + exp_ret * 100.0,
            pred_high_watermark=101.0, pred_low_watermark=99.0,
            direction=direction, expected_return=exp_ret, confidence=0.6,
        )

    aaa_forecast = make_forecast("AAA", 0.05, "bullish")
    bbb_forecast = make_forecast("BBB", -0.05, "bearish")

    def fake_predict_price(*, ticker, **kwargs):
        return {"AAA": aaa_forecast, "BBB": bbb_forecast}[ticker]

    def boom(*args, **kwargs):
        raise RuntimeError("simulated batch failure")

    with patch.object(kf, "predict_prices_batch", side_effect=boom) as batch_mock, \
         patch.object(kf, "predict_price", side_effect=fake_predict_price) as single_mock:
        out = se._apply_kronos_gate(candidates, kronos_cfg={"veto_return_threshold": -0.02, "pred_bars": 10})

    # Batch was attempted exactly once
    assert batch_mock.call_count == 1
    # Per-ticker fallback was used for both
    assert single_mock.call_count == 2

    # AAA passed through; BBB was vetoed (expected_return -0.05 < -0.02)
    out_tickers = [c["ticker"] for c in out]
    assert "AAA" in out_tickers
    assert "BBB" not in out_tickers

    aaa = next(c for c in out if c["ticker"] == "AAA")
    assert aaa["kronos_direction"] == "bullish"
    assert aaa["kronos_expected_return"] == 0.05


# ─────────────────────────────────────────────────────────────────
# Screening preset wiring
# ─────────────────────────────────────────────────────────────────

def test_screening_gate_preset_exists_and_uses_n3():
    """PAPER_PRESETS must expose screening_gate with sample_count=3."""
    from lib.kronos_forecaster import PAPER_PRESETS

    assert "screening_gate" in PAPER_PRESETS
    sg = PAPER_PRESETS["screening_gate"]
    assert sg["sample_count"] == 3
    assert sg["T"] == 0.6
    assert sg["top_p"] == 0.90


def test_predict_prices_batch_all_cache_hits_skips_predictor():
    """If every ticker is already cached, predict_batch must not be called."""
    from lib import kronos_forecaster as kf

    tickers = ["AAA", "BBB"]
    pred_bars = 10

    for t, base in [("AAA", 100.0), ("BBB", 200.0)]:
        kf._cache_put(
            kf._cache_key(t, "1d", pred_bars, model_name="NeoQuasar/Kronos-base"),
            {
                "ticker": t, "interval": "1d", "lookback_bars": 200, "pred_bars": pred_bars,
                "predicted_close": [base] * pred_bars,
                "predicted_high": [base + 1] * pred_bars,
                "predicted_low": [base - 1] * pred_bars,
                "current_price": base, "pred_final_close": base,
                "pred_high_watermark": base + 1, "pred_low_watermark": base - 1,
                "direction": "neutral", "expected_return": 0.0, "confidence": 0.5,
            },
        )

    with patch.object(kf, "_load_predictor") as load_mock, \
         patch.object(kf, "_fetch_ohlcv") as fetch_mock:
        out = kf.predict_prices_batch(tickers=tickers, pred_bars=pred_bars)

    load_mock.assert_not_called()
    fetch_mock.assert_not_called()
    assert [f.ticker for f in out] == tickers


def test_predict_prices_batch_all_fetches_fail_returns_only_cached():
    """When every fetch raises and nothing is cached, predictor isn't called."""
    from lib import kronos_forecaster as kf

    def boom_fetch(*args, **kwargs):
        raise RuntimeError("network down")

    with patch.object(kf, "_load_predictor") as load_mock, \
         patch.object(kf, "_fetch_ohlcv", side_effect=boom_fetch):
        out = kf.predict_prices_batch(tickers=["AAA", "BBB"], pred_bars=10)

    load_mock.assert_not_called()
    assert out == []


@pytest.mark.parametrize("interval", ["1h", "5m"])
def test_predict_prices_batch_handles_intraday_intervals(interval):
    """1h and 5m intervals pick the right pandas frequency."""
    from lib import kronos_forecaster as kf

    pred_bars = 4
    fake_predictor = MagicMock()
    fake_predictor.predict_batch.return_value = [_make_pred_df(pred_bars, base=110.0)]

    df = _make_ohlcv(120, start_close=100.0)
    if interval == "1h":
        df.index = pd.date_range("2024-06-03 09:30", periods=120, freq="h")
    else:  # 5m
        df.index = pd.date_range("2024-06-03 09:30", periods=120, freq="5min")

    with patch.object(kf, "_load_predictor", return_value=fake_predictor), \
         patch.object(kf, "_fetch_ohlcv", return_value=df):
        out = kf.predict_prices_batch(
            tickers=["AAA"], pred_bars=pred_bars, interval=interval, lookback=100
        )

    assert len(out) == 1
    assert out[0].interval == interval


def test_predict_prices_batch_size_mismatch_raises():
    """If predict_batch returns the wrong length, we raise (defensive guard)."""
    from lib import kronos_forecaster as kf

    fake_predictor = MagicMock()
    # Predictor mistakenly returns 1 frame for 2 inputs
    fake_predictor.predict_batch.return_value = [_make_pred_df(10, base=110.0)]

    fetch_returns = {
        "AAA": _make_ohlcv(200, start_close=100.0),
        "BBB": _make_ohlcv(200, start_close=200.0),
    }

    with patch.object(kf, "_load_predictor", return_value=fake_predictor), \
         patch.object(kf, "_fetch_ohlcv", side_effect=lambda t, **kw: fetch_returns[t]):
        with pytest.raises(RuntimeError, match="predict_batch returned"):
            kf.predict_prices_batch(tickers=["AAA", "BBB"], pred_bars=10)


def test_predict_prices_batch_cache_put_failure_does_not_break_inference(monkeypatch):
    """A cache-write error must be logged but never break the call."""
    from lib import kronos_forecaster as kf

    fake_predictor = MagicMock()
    fake_predictor.predict_batch.return_value = [_make_pred_df(10, base=110.0)]

    def boom_cache(key, data):
        raise OSError("disk full")

    with patch.object(kf, "_load_predictor", return_value=fake_predictor), \
         patch.object(kf, "_fetch_ohlcv", return_value=_make_ohlcv(200, 100.0)), \
         patch.object(kf, "_cache_put", side_effect=boom_cache):
        out = kf.predict_prices_batch(tickers=["AAA"], pred_bars=10)

    assert len(out) == 1 and out[0].ticker == "AAA"


def test_apply_kronos_gate_passes_screening_preset_to_batch():
    """Gate 4 calls predict_prices_batch with sample_count=3 (screening_gate)."""
    from lib import stock_engine as se
    from lib import kronos_forecaster as kf

    candidates = [{"ticker": "AAA"}, {"ticker": "BBB"}]

    def make_forecast(t: str) -> kf.KronosForecast:
        return kf.KronosForecast(
            ticker=t, interval="1d", lookback_bars=200, pred_bars=10,
            predicted_close=[100.0] * 10, predicted_high=[101.0] * 10, predicted_low=[99.0] * 10,
            current_price=100.0, pred_final_close=101.0,
            pred_high_watermark=101.0, pred_low_watermark=99.0,
            direction="bullish", expected_return=0.01, confidence=0.5,
        )

    captured_kwargs: dict = {}

    def fake_batch(**kwargs):
        captured_kwargs.update(kwargs)
        return [make_forecast(t) for t in kwargs["tickers"]]

    with patch.object(kf, "predict_prices_batch", side_effect=fake_batch):
        se._apply_kronos_gate(candidates, kronos_cfg={"veto_return_threshold": -0.02, "pred_bars": 10})

    assert captured_kwargs["sample_count"] == 3
    assert captured_kwargs["temperature"] == 0.6
    assert captured_kwargs["top_p"] == 0.90
