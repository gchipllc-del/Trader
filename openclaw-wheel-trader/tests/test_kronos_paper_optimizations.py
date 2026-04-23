"""
Smoke tests for the Kronos paper-optimization additions.

Validates that the public API matches what the Kronos paper
(arXiv:2508.02739) recommends:
- Table 1:  model sizes (small/base/large)
- Table 6:  inference hyperparameters (T, top-p, N) per task
- Table 8:  lookback × forecast-horizon windows per interval
- Section 2: 6-dim OHLCVA input

These tests do NOT hit the network or load Kronos weights — they only
verify the public configuration constants, helpers, and function
signatures so we catch accidental regressions on every commit.

Run with:
    python -m pytest tests/test_kronos_paper_optimizations.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Table 1 — Model Sizes
# ============================================================

def test_kronos_models_table1():
    """KRONOS_MODELS must expose paper Table 1's three sizes."""
    from lib.kronos_forecaster import KRONOS_MODELS

    for size in ("small", "base", "large"):
        assert size in KRONOS_MODELS, f"missing size '{size}'"
        cfg = KRONOS_MODELS[size]
        assert cfg["model"].startswith("NeoQuasar/Kronos-")
        assert cfg["tokenizer"].startswith("NeoQuasar/Kronos-Tokenizer-")
        assert isinstance(cfg["params_m"], (int, float))
        assert cfg["params_m"] > 0

    # Paper exact parameter counts
    assert KRONOS_MODELS["small"]["params_m"] == 24.7
    assert KRONOS_MODELS["base"]["params_m"] == 102.3
    assert KRONOS_MODELS["large"]["params_m"] == 499.2


def test_resolve_model_size_validation():
    """_resolve_model rejects unknown sizes but accepts explicit model_name."""
    from lib.kronos_forecaster import _resolve_model

    # Valid sizes
    m, t = _resolve_model(model_size="small")
    assert m == "NeoQuasar/Kronos-small"
    m, t = _resolve_model(model_size="base")
    assert m == "NeoQuasar/Kronos-base"
    m, t = _resolve_model(model_size="large")
    assert m == "NeoQuasar/Kronos-large"

    # Default → base
    m, t = _resolve_model()
    assert m == "NeoQuasar/Kronos-base"

    # Explicit model_name bypasses size
    m, t = _resolve_model(model_name="NeoQuasar/Kronos-base")
    assert m == "NeoQuasar/Kronos-base"

    # Unknown size raises
    with pytest.raises(ValueError, match="Unknown model_size"):
        _resolve_model(model_size="xlarge")


# ============================================================
# Table 6 — Inference Hyperparameters
# ============================================================

def test_paper_presets_table6():
    """PAPER_PRESETS must match paper Table 6 verbatim."""
    from lib.kronos_forecaster import PAPER_PRESETS

    # Paper Table 6 exact values
    assert PAPER_PRESETS["forecast"] == {"T": 0.6, "top_p": 0.90, "sample_count": 10}
    assert PAPER_PRESETS["return"]   == {"T": 0.6, "top_p": 0.90, "sample_count": 10}
    assert PAPER_PRESETS["volatility"] == {"T": 0.9, "top_p": 0.90, "sample_count": 1}
    assert PAPER_PRESETS["generate"] == {"T": 1.0, "top_p": 0.95, "sample_count": 1}
    assert PAPER_PRESETS["simulate"] == {"T": 0.6, "top_p": 0.90, "sample_count": 10}


def test_predict_price_defaults_match_paper():
    """predict_price's default T / top_p / N must match paper Table 6."""
    import inspect
    from lib.kronos_forecaster import predict_price

    sig = inspect.signature(predict_price)
    defaults = {k: v.default for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty}

    assert defaults["temperature"] == 0.6, "Paper Table 6: T=0.6 for price forecasting"
    assert defaults["top_p"] == 0.90, "Paper Table 6: top_p=0.90"
    assert defaults["sample_count"] == 10, "Paper Table 6: N=10"


def test_price_to_probability_defaults_match_paper():
    """price_to_probability's default T / top_p / N must match paper Table 6."""
    import inspect
    from lib.kronos_forecaster import price_to_probability

    sig = inspect.signature(price_to_probability)
    defaults = {k: v.default for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty}

    assert defaults["temperature"] == 0.6
    assert defaults["top_p"] == 0.90
    assert defaults["sample_count"] == 10


def test_predict_volatility_defaults_match_paper():
    """predict_volatility's defaults must match paper Table 6 for volatility."""
    import inspect
    from lib.kronos_forecaster import predict_volatility

    sig = inspect.signature(predict_volatility)
    defaults = {k: v.default for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty}

    assert defaults["temperature"] == 0.9, "Paper Table 6: T=0.9 for volatility"
    assert defaults["top_p"] == 0.90
    assert defaults["sample_count"] == 1, "Paper Table 6: N=1 for volatility"


# ============================================================
# Table 8 — Look-back × Forecast Horizon
# ============================================================

def test_paper_windows_table8():
    """PAPER_WINDOWS must match paper Table 8."""
    from lib.kronos_forecaster import PAPER_WINDOWS

    # Paper Table 8 exact values
    assert PAPER_WINDOWS["5m"]  == (480, 96)
    assert PAPER_WINDOWS["10m"] == (240, 48)
    assert PAPER_WINDOWS["15m"] == (160, 32)
    assert PAPER_WINDOWS["20m"] == (120, 24)
    assert PAPER_WINDOWS["40m"] == (90,  24)
    assert PAPER_WINDOWS["1h"]  == (80,  12)
    assert PAPER_WINDOWS["2h"]  == (60,  12)
    assert PAPER_WINDOWS["4h"]  == (90,  18)
    assert PAPER_WINDOWS["1d"]  == (40,  12)


def test_paper_window_fallback():
    """paper_window falls back to 1d for unknown intervals."""
    from lib.kronos_forecaster import paper_window

    # Known intervals
    assert paper_window("1d") == (40, 12)
    assert paper_window("5m") == (480, 96)

    # Unknown falls back to 1d
    assert paper_window("weekly") == (40, 12)
    assert paper_window("") == (40, 12)


# ============================================================
# Preset-driven API
# ============================================================

def test_predict_with_preset_rejects_unknown_task():
    from lib.kronos_forecaster import predict_with_preset

    with pytest.raises(ValueError, match="Unknown task"):
        predict_with_preset(ticker="AAPL", task="astrology")


# ============================================================
# Max Context (Paper Section "Training Procedure": 512)
# ============================================================

def test_max_context_is_paper_limit():
    """MAX_CONTEXT_LEN must be 512 per the paper."""
    from lib.kronos_forecaster import MAX_CONTEXT_LEN
    assert MAX_CONTEXT_LEN == 512


# ============================================================
# VolatilityForecast dataclass schema
# ============================================================

def test_volatility_forecast_schema():
    """VolatilityForecast must expose all fields the CLI/agents depend on."""
    from lib.kronos_forecaster import VolatilityForecast

    v = VolatilityForecast(
        ticker="X", interval="1d", horizon_bars=30,
        current_price=100.0,
        realized_vol_annualized=0.25,
        realized_vol_period=0.05,
        historical_vol_annualized=0.22,
        vol_regime="normal",
        confidence=0.8,
    )
    for field in ("ticker", "interval", "horizon_bars", "current_price",
                  "realized_vol_annualized", "realized_vol_period",
                  "historical_vol_annualized", "vol_regime", "confidence"):
        assert hasattr(v, field), f"missing {field}"


# ============================================================
# Public-API surface check
# ============================================================

def test_public_api_exports():
    """All optimization additions must be importable from the module root."""
    from lib import kronos_forecaster

    for name in (
        # Constants
        "PAPER_PRESETS", "KRONOS_MODELS", "PAPER_WINDOWS", "MAX_CONTEXT_LEN",
        "DEFAULT_MODEL_SIZE",
        # Helpers
        "paper_window", "_resolve_model",
        # Prediction functions
        "predict_price", "price_to_probability", "predict_volatility",
        "predict_with_preset",
        # Dataclasses
        "KronosForecast", "PriceProbability", "VolatilityForecast",
        # Printers
        "print_forecast_report", "print_probability_report",
        "print_volatility_report",
    ):
        assert hasattr(kronos_forecaster, name), f"missing export: {name}"
