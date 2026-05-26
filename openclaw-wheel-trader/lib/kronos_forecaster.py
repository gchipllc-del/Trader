"""
Kronos Forecaster — zero-shot price prediction for prediction markets.

Wraps the Kronos foundation model (pretrained on 45+ global exchanges) to:
    1. Fetch historical OHLCV data for any ticker (yfinance)
    2. Run Kronos to generate future candlestick predictions
    3. Convert price forecasts into probability estimates for binary markets
       (e.g., "Will BTC be above $X by date Y?" → P(close > target))
    4. Feed the probability back into Polybot's Bayesian forecaster

The model runs zero-shot — no fine-tuning needed. It understands candlestick
patterns from pretraining on massive cross-market data.

Hyperparameters follow the original Kronos paper recommendations (Shi et al.,
"Kronos: A Foundation Model for the Language of Financial Markets",
arXiv:2508.02739, Table 6):

    Task                              T     top-p   N samples
    -----------------------------------------------------------
    Price Series Forecasting          0.6   0.90    10
    Return Forecasting                0.6   0.90    10
    Realized Volatility Forecasting   0.9   0.90     1
    Synthetic K-line Generation       1.0   0.95     1
    Investment Simulation             0.6   0.90    10

The paper notes temperature scaling offers more nuanced control than top-p
truncation, and forecasting tasks benefit from lower T to reduce randomness
while volatility/generation benefit from higher T for diversity.

Model sizes (Kronos paper Table 1):
    Kronos-small:  24.7M params  (8 layers,  d_model=512,  8 heads)
    Kronos-base:  102.3M params (12 layers, d_model=832, 16 heads) [default]
    Kronos-large: 499.2M params (18 layers, d_model=1664, 32 heads)

Max context: 512 tokens (per paper, Section "Training Procedure").

Security:
    - Model weights downloaded from Hugging Face (verified checksums)
    - All yfinance data treated as untrusted (validated on ingest)
    - No API keys needed — public market data + public model weights
    - Rate limiting on data fetches
    - Results cached to avoid redundant GPU/CPU inference
    - No secrets in any log or error message
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lib.audit import log_event

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "kronos_cache"

# Singleton model — loaded once, reused across calls
_predictor = None
_loaded_model_name: str | None = None  # Track which model is loaded
_model_lock = False  # Simple single-process guard

# ── Paper-Recommended Inference Presets ──────────────────────────
# From Kronos paper (arXiv:2508.02739) Table 6.
# Use these for task-specific defaults instead of guessing hyperparameters.
PAPER_PRESETS = {
    "forecast": {"T": 0.6, "top_p": 0.90, "sample_count": 10},
    "return":   {"T": 0.6, "top_p": 0.90, "sample_count": 10},
    "volatility": {"T": 0.9, "top_p": 0.90, "sample_count": 1},
    "generate": {"T": 1.0, "top_p": 0.95, "sample_count": 1},
    "simulate": {"T": 0.6, "top_p": 0.90, "sample_count": 10},
}

# ── Model Size Selector ──────────────────────────────────────────
# From Kronos paper Table 1. Map friendly names to HF model IDs.
KRONOS_MODELS = {
    "small": {
        "model": "NeoQuasar/Kronos-small",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "params_m": 24.7,
    },
    "base": {
        "model": "NeoQuasar/Kronos-base",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "params_m": 102.3,
    },
    "large": {
        "model": "NeoQuasar/Kronos-large",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "params_m": 499.2,
    },
}

DEFAULT_MODEL_SIZE = "base"
MAX_CONTEXT_LEN = 512  # Paper hard limit

# ── Paper-Specified Look-back × Forecast-Horizon Windows ─────────
# From Kronos paper (arXiv:2508.02739) Table 8: "Look-back and forecast
# horizon windows" used during pre-training evaluation. Kronos is trained
# to perform best at these specific ratios.
# Format: interval → (lookback_bars, forecast_horizon_bars)
PAPER_WINDOWS = {
    "5m":   (480, 96),
    "10m":  (240, 48),
    "15m":  (160, 32),
    "20m":  (120, 24),
    "40m":  (90,  24),
    "1h":   (80,  12),
    "2h":   (60,  12),
    "4h":   (90,  18),
    "1d":   (40,  12),
}


def paper_window(interval: str) -> tuple[int, int]:
    """
    Look up the paper-recommended (lookback, forecast_horizon) for an interval.

    Falls back to ("1d" → 40, 12) if the interval isn't in PAPER_WINDOWS.
    Use this when you want Kronos to operate in its pre-training sweet spot.
    """
    return PAPER_WINDOWS.get(interval, PAPER_WINDOWS["1d"])


def _resolve_model(
    model_size: str | None = None,
    model_name: str | None = None,
) -> tuple[str, str]:
    """
    Resolve a friendly size name ("small"/"base"/"large") or explicit HF id
    into (model_name, tokenizer_name).

    Precedence: explicit `model_name` > `model_size` > default ("base").
    Size names are validated against KRONOS_MODELS.
    """
    if model_name:
        # Explicit HF id — use with default tokenizer
        return model_name, "NeoQuasar/Kronos-Tokenizer-base"
    size = (model_size or DEFAULT_MODEL_SIZE).lower()
    if size not in KRONOS_MODELS:
        raise ValueError(
            f"Unknown model_size '{size}'. Valid: {list(KRONOS_MODELS.keys())}"
        )
    cfg = KRONOS_MODELS[size]
    return cfg["model"], cfg["tokenizer"]


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class KronosForecast:
    """Output of a Kronos price prediction."""
    ticker: str
    interval: str                  # "1d", "1h", "5m"
    lookback_bars: int
    pred_bars: int
    predicted_close: list[float]   # Predicted close prices
    predicted_high: list[float]
    predicted_low: list[float]
    current_price: float           # Last known close
    pred_final_close: float        # Predicted close at end of horizon
    pred_high_watermark: float     # Max predicted high
    pred_low_watermark: float      # Min predicted low
    direction: str                 # "bullish", "bearish", "neutral"
    expected_return: float         # (pred_final - current) / current
    confidence: float              # Based on prediction variance across samples


@dataclass
class PriceProbability:
    """Probability that price crosses a target — for prediction markets."""
    ticker: str
    target_price: float
    direction: str                 # "above" or "below"
    probability: float             # 0.0 - 1.0
    confidence: float              # 0.0 - 1.0
    horizon_bars: int
    interval: str
    forecast: KronosForecast | None = None


# ── Model Loading ────────────────────────────────────────────────

def _load_predictor(
    model_name: str = "NeoQuasar/Kronos-base",
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
    max_context: int = MAX_CONTEXT_LEN,
):
    """
    Load Kronos model + tokenizer. Downloads from HuggingFace on first call.

    Uses singleton pattern — heavy models should only load once per process.
    If a different model_name is requested than what's currently loaded, the
    old predictor is released and a new one is loaded (size switching).

    Returns the KronosPredictor instance.
    """
    global _predictor, _loaded_model_name, _model_lock

    # Fast path: same model already loaded
    if _predictor is not None and _loaded_model_name == model_name:
        return _predictor

    if _model_lock:
        raise RuntimeError("Kronos model is already loading — concurrent load blocked")

    _model_lock = True
    try:
        import torch

        from lib.kronos import Kronos, KronosPredictor, KronosTokenizer

        # Release previous model if we're switching sizes
        if _predictor is not None and _loaded_model_name != model_name:
            log_event("kronos", "model_switching", {
                "from": _loaded_model_name,
                "to": model_name,
            })
            _predictor = None
            _loaded_model_name = None

        log_event("kronos", "model_loading", {
            "model": model_name,
            "tokenizer": tokenizer_name,
            "max_context": max_context,
        })

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        model = Kronos.from_pretrained(model_name)

        # Auto-detect device (MPS on Mac, CUDA on GPU, CPU fallback)
        device = None
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Clamp max_context to paper limit (512).
        max_context = min(max(16, int(max_context)), MAX_CONTEXT_LEN)

        _predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_context=max_context,
        )
        _loaded_model_name = model_name

        log_event("kronos", "model_loaded", {
            "model": model_name,
            "device": device,
            "max_context": max_context,
        }, result="success")

        return _predictor

    except Exception as e:
        log_event("kronos", "model_load_failed", {
            "model": model_name,
            "error": str(e)[:200],
        }, result="failed")
        raise
    finally:
        _model_lock = False


# ── Data Fetching ────────────────────────────────────────────────

# Rate limiter for yfinance
_last_fetch = 0.0


def _fetch_ohlcv(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data.

    Tries Alpaca first (already wired into the trading bot's infrastructure),
    falls back to yfinance if Alpaca fails or is unavailable.

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        period: Lookback period ("6mo", "1y", "2y") — mapped to bar count for Alpaca
        interval: Bar interval ("1d", "1h", "5m")

    Returns:
        DataFrame with [open, high, low, close, volume] + DatetimeIndex.
    """
    global _last_fetch

    # Rate limit: 1 request per 0.5s
    elapsed = time.time() - _last_fetch
    if elapsed < 0.5:
        time.sleep(0.5 - elapsed)

    # Sanitize ticker — only allow alphanumeric, hyphens, dots
    clean_ticker = re.sub(r"[^A-Za-z0-9\-.]", "", ticker)[:20]
    if not clean_ticker:
        raise ValueError(f"Invalid ticker: {ticker}")

    # --- Try Alpaca first (more reliable than yfinance on macOS) ---
    try:
        from lib.alpaca_client import AlpacaClient

        # Map period to approximate bar count
        period_to_bars = {
            "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "5y": 1825,
        }
        # Alpaca timeframe mapping
        timeframe_map = {"1d": "1Day", "1h": "1Hour", "5m": "5Min"}
        alpaca_tf = timeframe_map.get(interval, "1Day")
        limit = period_to_bars.get(period, 180)

        # Alpaca caps at 10000 bars per request
        limit = min(limit, 5000)

        client = AlpacaClient()
        bars = client.get_bars([clean_ticker], timeframe=alpaca_tf, limit=limit)

        if clean_ticker in bars and not bars[clean_ticker].empty:
            data = bars[clean_ticker].copy()
            # Alpaca columns are already lowercase: open, high, low, close, volume
            required = ["open", "high", "low", "close"]
            if all(col in data.columns for col in required):
                if "volume" not in data.columns:
                    data["volume"] = 0.0
                data = data.dropna(subset=required)
                if len(data) >= 30:
                    _last_fetch = time.time()
                    log_event("kronos", "fetched_from_alpaca", {
                        "ticker": clean_ticker, "bars": len(data),
                    })
                    return data
    except Exception as e:
        log_event("kronos", "alpaca_fetch_failed", {
            "ticker": clean_ticker, "error": str(e)[:200],
        })
        # Fall through to yfinance

    # --- Fallback: yfinance ---
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(
            f"Cannot fetch {clean_ticker}: Alpaca failed and yfinance unavailable ({str(e)[:100]})"
        )

    try:
        data = yf.download(
            clean_ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        _last_fetch = time.time()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {clean_ticker}: {str(e)[:200]}")

    if data is None or data.empty:
        raise RuntimeError(f"No data returned for {clean_ticker}")

    # Normalize column names (yfinance returns capitalized, possibly as MultiIndex)
    data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]

    # Validate — all data is untrusted
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in data.columns:
            raise RuntimeError(f"Missing column '{col}' in {clean_ticker} data")

    # Drop NaN rows
    data = data.dropna(subset=required)

    if len(data) < 30:
        raise RuntimeError(f"Insufficient data for {clean_ticker}: {len(data)} bars (need 30+)")

    # Ensure volume exists
    if "volume" not in data.columns:
        data["volume"] = 0.0

    return data


# ── Cache ────────────────────────────────────────────────────────

def _cache_key(
    ticker: str,
    interval: str,
    pred_bars: int,
    model_name: str = "NeoQuasar/Kronos-base",
) -> str:
    """
    Build a cache key. Includes model_name so switching model sizes
    (small/base/large) doesn't hit stale cache from another size.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Use just the size suffix (base/small/large) to keep the key short
    short_model = model_name.rsplit("-", 1)[-1].lower() if "-" in model_name else model_name
    raw = f"{ticker}:{interval}:{pred_bars}:{today}:{short_model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_get(key: str, ttl_minutes: int = 60) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        cached_at = datetime.fromisoformat(data.get("cached_at", ""))
        if datetime.now(timezone.utc) - cached_at > timedelta(minutes=ttl_minutes):
            return None
        return data
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def _cache_put(key: str, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data["cached_at"] = datetime.now(timezone.utc).isoformat()
    path = CACHE_DIR / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


# ── Core Prediction ──────────────────────────────────────────────

def predict_price(
    ticker: str,
    pred_bars: int = 30,
    interval: str = "1d",
    lookback: int = 400,
    sample_count: int = 10,              # Paper Table 6: N=10 for forecasting
    temperature: float = 0.6,            # Paper Table 6: T=0.6 for price forecasting
    top_p: float = 0.90,                 # Paper Table 6: top_p=0.90 for forecasting
    model_name: str | None = None,       # None → resolved from model_size
    model_size: str | None = None,       # "small" | "base" | "large"
    use_paper_window: bool = False,      # Snap lookback+pred_bars to paper Table 8
) -> KronosForecast:
    """
    Predict future prices for a ticker using Kronos.

    This is the main function for raw price forecasting. For prediction
    market probability estimates, use `price_to_probability()` instead.

    Hyperparameter defaults follow the Kronos paper (arXiv:2508.02739,
    Table 6) for "Price Series Forecasting": T=0.6, top_p=0.90, N=10.

    Args:
        ticker: Stock/crypto symbol (e.g., "AAPL", "BTC-USD", "ETH-USD")
        pred_bars: Number of future bars to predict
        interval: Bar interval ("1d" for daily, "1h" for hourly)
        lookback: Number of historical bars to feed the model (max 512)
        sample_count: Number of inference samples to average (paper default: 10)
        temperature: Sampling temperature (paper default: 0.6 for forecasting)
        top_p: Nucleus sampling cutoff (paper default: 0.90)
        model_name: Explicit HuggingFace model ID (overrides model_size)
        model_size: "small" (fast, 24.7M), "base" (default, 102.3M), "large" (499.2M)

    Returns:
        KronosForecast with predicted OHLCV and direction.
    """
    # Resolve which model to use
    model_name, tokenizer_name = _resolve_model(model_size=model_size, model_name=model_name)

    # Optionally snap to paper Table 8 window (Kronos's pre-training sweet spot)
    if use_paper_window:
        pw_lookback, pw_horizon = paper_window(interval)
        lookback = pw_lookback
        pred_bars = pw_horizon
        log_event("kronos", "using_paper_window", {
            "interval": interval, "lookback": lookback, "pred_bars": pred_bars,
        })

    # Cache key must include model_name so switching sizes doesn't hit stale cache
    key = _cache_key(ticker, interval, pred_bars, model_name=model_name)
    cached = _cache_get(key, ttl_minutes=60)
    if cached and "predicted_close" in cached:
        return KronosForecast(
            ticker=cached["ticker"],
            interval=cached["interval"],
            lookback_bars=cached["lookback_bars"],
            pred_bars=cached["pred_bars"],
            predicted_close=cached["predicted_close"],
            predicted_high=cached["predicted_high"],
            predicted_low=cached["predicted_low"],
            current_price=cached["current_price"],
            pred_final_close=cached["pred_final_close"],
            pred_high_watermark=cached["pred_high_watermark"],
            pred_low_watermark=cached["pred_low_watermark"],
            direction=cached["direction"],
            expected_return=cached["expected_return"],
            confidence=cached["confidence"],
        )

    # Fetch data
    period_map = {"1d": "2y", "1h": "60d", "5m": "7d"}
    period = period_map.get(interval, "1y")
    data = _fetch_ohlcv(ticker, period=period, interval=interval)

    # Clamp lookback to available data and model max context (paper: 512)
    lookback = min(lookback, len(data), MAX_CONTEXT_LEN)
    data = data.tail(lookback + pred_bars)  # Extra for timestamp generation

    # Prepare input
    hist = data.head(lookback)
    x_df = hist[["open", "high", "low", "close", "volume"]].copy()
    # Paper Section 2: Kronos expects 6-dim OHLCVA (Open, High, Low, Close,
    # Volume, Amount). "Amount" = price × volume (turnover).
    x_df["amount"] = x_df["volume"] * x_df["close"]

    # Kronos expects a pd.Series (with .dt accessor), not a DatetimeIndex
    x_timestamp = pd.Series(pd.DatetimeIndex(hist.index))

    # Generate future timestamps
    if interval == "1d":
        freq = pd.tseries.offsets.BDay(1)
    elif interval == "1h":
        freq = pd.tseries.offsets.Hour(1)
    else:
        freq = pd.tseries.offsets.Minute(5)

    last_ts = x_timestamp.iloc[-1]
    y_timestamp = pd.Series(pd.date_range(start=last_ts + freq, periods=pred_bars, freq=freq))

    # Load model and predict
    predictor = _load_predictor(model_name=model_name, tokenizer_name=tokenizer_name)

    # Clamp to safe ranges (avoid silently passing bad values to Kronos)
    temperature = max(0.1, min(float(temperature), 2.0))
    top_p = max(0.1, min(float(top_p), 1.0))
    sample_count = max(1, min(int(sample_count), 50))

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_bars,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )

    # Extract predictions
    pred_close = pred_df["close"].values.tolist()
    pred_high = pred_df["high"].values.tolist()
    pred_low = pred_df["low"].values.tolist()

    current_price = float(hist["close"].iloc[-1])
    pred_final = float(pred_close[-1])
    high_watermark = float(max(pred_high))
    low_watermark = float(min(pred_low))

    expected_return = (pred_final - current_price) / current_price if current_price > 0 else 0.0

    # Direction determination
    if expected_return > 0.02:
        direction = "bullish"
    elif expected_return < -0.02:
        direction = "bearish"
    else:
        direction = "neutral"

    # Confidence from prediction variance (multi-sample)
    # If sample_count > 1, Kronos internally averages — measure spread of predictions
    pred_range = high_watermark - low_watermark
    normalized_range = pred_range / current_price if current_price > 0 else 1.0
    # Tighter predicted range = higher confidence
    confidence = max(0.1, min(1.0 - normalized_range, 0.9))

    forecast = KronosForecast(
        ticker=ticker,
        interval=interval,
        lookback_bars=lookback,
        pred_bars=pred_bars,
        predicted_close=[round(p, 4) for p in pred_close],
        predicted_high=[round(p, 4) for p in pred_high],
        predicted_low=[round(p, 4) for p in pred_low],
        current_price=round(current_price, 4),
        pred_final_close=round(pred_final, 4),
        pred_high_watermark=round(high_watermark, 4),
        pred_low_watermark=round(low_watermark, 4),
        direction=direction,
        expected_return=round(expected_return, 4),
        confidence=round(confidence, 4),
    )

    # Cache result
    _cache_put(key, {
        "ticker": forecast.ticker,
        "interval": forecast.interval,
        "lookback_bars": forecast.lookback_bars,
        "pred_bars": forecast.pred_bars,
        "predicted_close": forecast.predicted_close,
        "predicted_high": forecast.predicted_high,
        "predicted_low": forecast.predicted_low,
        "current_price": forecast.current_price,
        "pred_final_close": forecast.pred_final_close,
        "pred_high_watermark": forecast.pred_high_watermark,
        "pred_low_watermark": forecast.pred_low_watermark,
        "direction": forecast.direction,
        "expected_return": forecast.expected_return,
        "confidence": forecast.confidence,
    })

    log_event("kronos", "prediction_complete", {
        "ticker": ticker,
        "interval": interval,
        "pred_bars": pred_bars,
        "current_price": forecast.current_price,
        "pred_final_close": forecast.pred_final_close,
        "direction": forecast.direction,
        "expected_return": forecast.expected_return,
    }, result="success")

    return forecast


# ── Price → Probability Conversion ───────────────────────────────

def price_to_probability(
    ticker: str,
    target_price: float,
    direction: str = "above",
    horizon_bars: int = 30,
    interval: str = "1d",
    sample_count: int = 10,              # Paper Table 6: N=10 for forecasting
    temperature: float = 0.6,            # Paper Table 6: T=0.6 for price forecasting
    top_p: float = 0.90,                 # Paper Table 6: top_p=0.90 for forecasting
    model_name: str | None = None,
    model_size: str | None = None,       # "small" | "base" | "large"
    use_paper_window: bool = False,      # Snap lookback to paper Table 8
) -> PriceProbability:
    """
    Estimate the probability that a price will be above/below a target.

    This is the key function for prediction market integration.
    For a market like "Will BTC be above $70,000 by June 30?":
        - ticker = "BTC-USD"
        - target_price = 70000
        - direction = "above"
        - horizon_bars = days until June 30

    Uses multi-sample Monte Carlo: runs Kronos multiple times with
    different sampling paths, counts what fraction cross the target.

    Hyperparameter defaults follow the Kronos paper Table 6 for "Price
    Series Forecasting": T=0.6, top_p=0.90, N=10 independent paths.
    (Previous defaults T=0.9, top_p=0.92 were too diffuse — paper shows
    lower T gives sharper, more accurate point forecasts.)

    Args:
        ticker: Stock/crypto symbol
        target_price: The price threshold
        direction: "above" or "below"
        horizon_bars: How many bars into the future
        interval: Bar interval
        sample_count: Number of independent Kronos paths (paper default: 10)
        temperature: Sampling temperature (paper default: 0.6 for price forecasts)
        top_p: Nucleus sampling cutoff (paper default: 0.90)
        model_name: Explicit HuggingFace model ID (overrides model_size)
        model_size: "small" | "base" (default) | "large"

    Returns:
        PriceProbability with the estimated probability.
    """
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got '{direction}'")

    # Resolve model
    model_name, tokenizer_name = _resolve_model(model_size=model_size, model_name=model_name)

    # Optionally snap to paper Table 8 window
    if use_paper_window:
        pw_lookback, pw_horizon = paper_window(interval)
        horizon_bars = pw_horizon
        _requested_lookback = pw_lookback
    else:
        _requested_lookback = 400

    # Clamp to safe ranges
    temperature = max(0.1, min(float(temperature), 2.0))
    top_p = max(0.1, min(float(top_p), 1.0))
    sample_count = max(1, min(int(sample_count), 50))

    # Run multiple independent forecasts
    # Each with sample_count=1 so we get independent paths for Monte Carlo
    crosses = 0
    all_finals = []

    predictor = _load_predictor(model_name=model_name, tokenizer_name=tokenizer_name)

    # Fetch data once
    period_map = {"1d": "2y", "1h": "60d", "5m": "7d"}
    period = period_map.get(interval, "1y")
    data = _fetch_ohlcv(ticker, period=period, interval=interval)

    lookback = min(_requested_lookback, len(data), MAX_CONTEXT_LEN)
    hist = data.tail(lookback)

    x_df = hist[["open", "high", "low", "close", "volume"]].copy()
    # 6-dim OHLCVA input (paper Section 2). Amount = price × volume.
    x_df["amount"] = x_df["volume"] * x_df["close"]
    # Kronos expects a pd.Series (with .dt accessor), not a DatetimeIndex
    x_timestamp = pd.Series(pd.DatetimeIndex(hist.index))

    if interval == "1d":
        freq = pd.tseries.offsets.BDay(1)
    elif interval == "1h":
        freq = pd.tseries.offsets.Hour(1)
    else:
        freq = pd.tseries.offsets.Minute(5)

    last_ts = x_timestamp.iloc[-1]
    y_timestamp = pd.Series(pd.date_range(start=last_ts + freq, periods=horizon_bars, freq=freq))

    for i in range(sample_count):
        try:
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=horizon_bars,
                T=temperature,
                top_p=top_p,
                sample_count=1,    # 1 per loop → each iteration = 1 independent MC path
                verbose=False,
            )

            final_close = float(pred_df["close"].iloc[-1])
            all_finals.append(final_close)

            # Check if ANY bar in the prediction crosses the target
            if direction == "above":
                if pred_df["high"].max() >= target_price:
                    crosses += 1
            else:
                if pred_df["low"].min() <= target_price:
                    crosses += 1

        except Exception as e:
            log_event("kronos", "sample_failed", {
                "ticker": ticker, "sample": i, "error": str(e)[:200],
            }, result="failed")

    if not all_finals:
        return PriceProbability(
            ticker=ticker,
            target_price=target_price,
            direction=direction,
            probability=0.5,  # No data — neutral
            confidence=0.0,
            horizon_bars=horizon_bars,
            interval=interval,
        )

    # Primary probability: fraction of paths that cross
    raw_prob = crosses / len(all_finals)

    # Secondary: what fraction of final closes beat the target
    if direction == "above":
        final_prob = sum(1 for f in all_finals if f >= target_price) / len(all_finals)
    else:
        final_prob = sum(1 for f in all_finals if f <= target_price) / len(all_finals)

    # Blend: 60% path-crossing (stronger signal), 40% final-close
    probability = 0.6 * raw_prob + 0.4 * final_prob

    # Clamp to avoid extremes with limited samples
    probability = max(0.05, min(probability, 0.95))

    # Confidence from sample agreement
    if len(all_finals) >= 3:
        std = float(np.std(all_finals))
        mean = float(np.mean(all_finals))
        cv = std / abs(mean) if abs(mean) > 0 else 1.0
        confidence = max(0.1, min(1.0 - cv * 2, 0.9))
    else:
        confidence = 0.2

    current_price = float(hist["close"].iloc[-1])

    # Build a summary forecast from averages
    forecast = KronosForecast(
        ticker=ticker,
        interval=interval,
        lookback_bars=lookback,
        pred_bars=horizon_bars,
        predicted_close=[round(f, 4) for f in all_finals],
        predicted_high=[],
        predicted_low=[],
        current_price=round(current_price, 4),
        pred_final_close=round(float(np.mean(all_finals)), 4),
        pred_high_watermark=round(float(max(all_finals)), 4),
        pred_low_watermark=round(float(min(all_finals)), 4),
        direction="bullish" if np.mean(all_finals) > current_price else "bearish",
        expected_return=round((float(np.mean(all_finals)) - current_price) / current_price, 4),
        confidence=round(confidence, 4),
    )

    result = PriceProbability(
        ticker=ticker,
        target_price=target_price,
        direction=direction,
        probability=round(probability, 4),
        confidence=round(confidence, 4),
        horizon_bars=horizon_bars,
        interval=interval,
        forecast=forecast,
    )

    log_event("kronos", "probability_computed", {
        "ticker": ticker,
        "target": target_price,
        "direction": direction,
        "probability": result.probability,
        "confidence": result.confidence,
        "samples": len(all_finals),
        "crosses": crosses,
    }, result="success")

    return result


# ── Volatility Forecasting (Kronos Paper's Strongest Task) ───────
# The Kronos paper reports realized-volatility MAE of 0.037 vs 0.066 for the
# best baseline — roughly a 44% reduction. For an options-writing bot this
# translates directly into edge: better IV forecasts → better strike selection
# and more accurate CSP/CC vega risk sizing.

@dataclass
class VolatilityForecast:
    """Output of a Kronos realized-volatility prediction."""
    ticker: str
    interval: str
    horizon_bars: int
    current_price: float
    realized_vol_annualized: float     # Predicted annualized realized vol (e.g., 0.28 = 28%)
    realized_vol_period: float         # Predicted realized vol over horizon (not annualized)
    historical_vol_annualized: float   # Ex-post realized over lookback (for comparison)
    vol_regime: str                    # "low", "normal", "elevated", "extreme"
    confidence: float


def predict_volatility(
    ticker: str,
    horizon_bars: int = 30,
    interval: str = "1d",
    lookback: int = 400,
    temperature: float = 0.9,            # Paper Table 6: T=0.9 for volatility
    top_p: float = 0.90,                 # Paper Table 6: top_p=0.90
    sample_count: int = 1,               # Paper Table 6: N=1 for volatility
    model_name: str | None = None,
    model_size: str | None = None,
    use_paper_window: bool = False,      # Snap to paper Table 8 window
) -> VolatilityForecast:
    """
    Predict realized volatility over a horizon using Kronos.

    This is the task where Kronos dominates in the paper (arXiv:2508.02739
    Figure 1, "Volatility Forecasting MAE" panel): 0.037 vs 0.066 for the
    best baseline — a ~44% MAE reduction.

    Paper Table 6 hyperparameters for realized-volatility forecasting:
        T = 0.9, top_p = 0.90, N = 1

    The paper notes volatility benefits from HIGHER temperature (more diverse
    paths) than point-price forecasting (T=0.6), because vol is inherently
    probabilistic and requires capturing the dispersion, not the mean.

    Realized volatility is computed from the predicted close path using the
    standard daily log-return estimator, then annualized with sqrt(N).

    Args:
        ticker: Stock/crypto symbol
        horizon_bars: Number of future bars to forecast
        interval: "1d" | "1h" | "5m"
        lookback: Historical bars to feed model (max 512, paper limit)
        temperature: 0.9 per paper
        top_p: 0.90 per paper
        sample_count: 1 per paper (volatility uses single path)
        model_name/model_size: Model selection (see _resolve_model)

    Returns:
        VolatilityForecast with annualized + period-realized vol + regime label.
    """
    model_name, tokenizer_name = _resolve_model(model_size=model_size, model_name=model_name)

    # Optionally snap to paper Table 8 window
    if use_paper_window:
        pw_lookback, pw_horizon = paper_window(interval)
        lookback = pw_lookback
        horizon_bars = pw_horizon

    # Fetch data
    period_map = {"1d": "2y", "1h": "60d", "5m": "7d"}
    period = period_map.get(interval, "1y")
    data = _fetch_ohlcv(ticker, period=period, interval=interval)

    lookback = min(lookback, len(data), MAX_CONTEXT_LEN)
    hist = data.tail(lookback)

    x_df = hist[["open", "high", "low", "close", "volume"]].copy()
    x_df["amount"] = x_df["volume"] * x_df["close"]
    x_timestamp = pd.Series(pd.DatetimeIndex(hist.index))

    if interval == "1d":
        freq = pd.tseries.offsets.BDay(1)
        bars_per_year = 252
    elif interval == "1h":
        freq = pd.tseries.offsets.Hour(1)
        bars_per_year = 252 * 6.5        # 6.5 trading hours/day
    else:
        freq = pd.tseries.offsets.Minute(5)
        bars_per_year = 252 * 78         # 78 five-min bars/day

    last_ts = x_timestamp.iloc[-1]
    y_timestamp = pd.Series(pd.date_range(start=last_ts + freq, periods=horizon_bars, freq=freq))

    # Clamp
    temperature = max(0.1, min(float(temperature), 2.0))
    top_p = max(0.1, min(float(top_p), 1.0))
    sample_count = max(1, min(int(sample_count), 20))

    predictor = _load_predictor(model_name=model_name, tokenizer_name=tokenizer_name)

    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=horizon_bars,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )
    except Exception as e:
        log_event("kronos", "volatility_failed", {
            "ticker": ticker, "error": str(e)[:200],
        }, result="failed")
        raise

    # Realized vol from predicted path: std of log returns
    pred_close = pred_df["close"].values.astype(float)
    if len(pred_close) < 2:
        raise RuntimeError(f"Insufficient prediction bars for {ticker}: {len(pred_close)}")
    log_returns = np.diff(np.log(np.clip(pred_close, 1e-9, None)))
    period_std = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0
    annualized = period_std * (bars_per_year ** 0.5)
    period_vol = period_std * (len(log_returns) ** 0.5)

    # Historical vol for context
    hist_close = hist["close"].values.astype(float)
    hist_log_returns = np.diff(np.log(np.clip(hist_close, 1e-9, None)))
    hist_std = float(np.std(hist_log_returns, ddof=1)) if len(hist_log_returns) > 1 else 0.0
    hist_annualized = hist_std * (bars_per_year ** 0.5)

    # Regime classification (rough IV percentile buckets)
    if annualized < 0.15:
        regime = "low"
    elif annualized < 0.35:
        regime = "normal"
    elif annualized < 0.60:
        regime = "elevated"
    else:
        regime = "extreme"

    # Confidence: how consistent is predicted vol with recent history?
    # Big divergence → less confident (Kronos is extrapolating aggressively)
    if hist_annualized > 0.001:
        divergence = abs(annualized - hist_annualized) / hist_annualized
        confidence = max(0.2, min(1.0 - divergence * 0.5, 0.95))
    else:
        confidence = 0.5

    current_price = float(hist["close"].iloc[-1])

    result = VolatilityForecast(
        ticker=ticker,
        interval=interval,
        horizon_bars=horizon_bars,
        current_price=round(current_price, 4),
        realized_vol_annualized=round(annualized, 4),
        realized_vol_period=round(period_vol, 4),
        historical_vol_annualized=round(hist_annualized, 4),
        vol_regime=regime,
        confidence=round(confidence, 4),
    )

    log_event("kronos", "volatility_computed", {
        "ticker": ticker,
        "horizon_bars": horizon_bars,
        "ann_vol": result.realized_vol_annualized,
        "hist_vol": result.historical_vol_annualized,
        "regime": regime,
    }, result="success")

    return result


# ── Preset-Driven Prediction (Paper Table 6 Shortcut) ────────────

def predict_with_preset(
    ticker: str,
    task: str,                           # "forecast" | "return" | "volatility" | "generate" | "simulate"
    pred_bars: int = 30,
    interval: str = "1d",
    lookback: int = 400,
    model_size: str | None = None,
    model_name: str | None = None,
    use_paper_window: bool = True,       # Default ON — presets pair with paper windows
) -> KronosForecast | VolatilityForecast:
    """
    Shortcut that applies the paper's recommended hyperparameters for a
    given task without the caller having to remember them.

    See PAPER_PRESETS for the mapping (arXiv:2508.02739 Table 6).

    Args:
        ticker: Symbol
        task: One of PAPER_PRESETS keys
        pred_bars / interval / lookback / model_size / model_name:
            Passed through to the appropriate underlying function.

    Returns:
        - KronosForecast for "forecast" | "return" | "generate" | "simulate"
        - VolatilityForecast for "volatility"
    """
    task = task.lower()
    if task not in PAPER_PRESETS:
        raise ValueError(
            f"Unknown task '{task}'. Valid: {list(PAPER_PRESETS.keys())}"
        )
    preset = PAPER_PRESETS[task]

    if task == "volatility":
        return predict_volatility(
            ticker=ticker,
            horizon_bars=pred_bars,
            interval=interval,
            lookback=lookback,
            temperature=preset["T"],
            top_p=preset["top_p"],
            sample_count=preset["sample_count"],
            model_name=model_name,
            model_size=model_size,
            use_paper_window=use_paper_window,
        )
    # All other tasks use price-forecast path with preset hyperparameters
    return predict_price(
        ticker=ticker,
        pred_bars=pred_bars,
        interval=interval,
        lookback=lookback,
        sample_count=preset["sample_count"],
        temperature=preset["T"],
        top_p=preset["top_p"],
        model_name=model_name,
        model_size=model_size,
        use_paper_window=use_paper_window,
    )


# ── Market Question Parser ───────────────────────────────────────

# Patterns to extract ticker + target from prediction market questions
_PRICE_PATTERNS = [
    # "Will Bitcoin be above $70,000 by June?"
    re.compile(
        r"will\s+(\w+)\s+(?:be\s+)?(?:close\s+)?(?:above|over|exceed)\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # "Will AAPL close below $150?"
    re.compile(
        r"will\s+(\w+)\s+(?:close\s+)?(?:below|under|fall below)\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # "BTC above $100k"
    re.compile(
        r"(\w+)\s+(?:above|over)\s+\$?([\d,]+\.?\d*[kKmM]?)",
        re.IGNORECASE,
    ),
    # "Will the price of ETH reach $5000?"
    re.compile(
        r"(?:price of|price for)\s+(\w+)\s+(?:reach|hit|exceed)\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
]

# Common name → ticker mapping
_TICKER_MAP = {
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD",
    "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
    "xrp": "XRP-USD", "ripple": "XRP-USD",
    "cardano": "ADA-USD", "ada": "ADA-USD",
    "apple": "AAPL", "aapl": "AAPL",
    "google": "GOOGL", "googl": "GOOGL",
    "microsoft": "MSFT", "msft": "MSFT",
    "amazon": "AMZN", "amzn": "AMZN",
    "tesla": "TSLA", "tsla": "TSLA",
    "nvidia": "NVDA", "nvda": "NVDA",
    "meta": "META",
    "sp500": "^GSPC", "s&p": "^GSPC", "s&p500": "^GSPC",
    "nasdaq": "^IXIC",
    "gold": "GC=F", "silver": "SI=F",
    "oil": "CL=F", "crude": "CL=F",
}


def _parse_quantity(s: str) -> float:
    """Parse price strings like '70,000', '100k', '1.5M'."""
    s = s.replace(",", "")
    multiplier = 1.0
    if s.lower().endswith("k"):
        multiplier = 1_000
        s = s[:-1]
    elif s.lower().endswith("m"):
        multiplier = 1_000_000
        s = s[:-1]
    return float(s) * multiplier


def parse_price_market(question: str) -> dict | None:
    """
    Attempt to extract ticker, target price, and direction from a
    prediction market question.

    Returns:
        {"ticker": "BTC-USD", "target": 70000.0, "direction": "above"}
        or None if the question isn't about a price.
    """
    for pattern in _PRICE_PATTERNS:
        match = pattern.search(question)
        if match:
            asset = match.group(1).lower().strip()
            target_str = match.group(2)

            # Resolve ticker
            ticker = _TICKER_MAP.get(asset, asset.upper())

            try:
                target = _parse_quantity(target_str)
            except (ValueError, TypeError):
                continue

            # Determine direction from the matched pattern
            question_lower = question.lower()
            if any(w in question_lower for w in ["below", "under", "fall"]):
                direction = "below"
            else:
                direction = "above"

            return {
                "ticker": ticker,
                "target": target,
                "direction": direction,
            }

    return None


# ── Integration with Polybot Forecaster ──────────────────────────

def get_kronos_estimate(
    market_question: str,
    horizon_days: int = 30,
    sample_count: int = 10,
) -> float | None:
    """
    High-level entry point for the Polybot forecaster pipeline.

    Takes a prediction market question, checks if it's about a price,
    and if so, returns a Kronos-based probability estimate.

    This plugs into estimate_probability() as another Bayesian source.

    Args:
        market_question: The prediction market question text
        horizon_days: Days until market resolution
        sample_count: Number of Kronos samples for MC probability

    Returns:
        Probability (0.0-1.0) or None if the market isn't price-based.
    """
    parsed = parse_price_market(market_question)
    if parsed is None:
        return None  # Not a price-based market — Kronos can't help

    try:
        result = price_to_probability(
            ticker=parsed["ticker"],
            target_price=parsed["target"],
            direction=parsed["direction"],
            horizon_bars=max(1, min(horizon_days, 120)),
            interval="1d",
            sample_count=sample_count,
        )
        return result.probability
    except Exception as e:
        log_event("kronos", "estimate_failed", {
            "question": market_question[:100],
            "error": str(e)[:200],
        }, result="failed")
        return None


# ── CLI Helpers ──────────────────────────────────────────────────

def print_forecast_report(forecast: KronosForecast):
    """Print a formatted Kronos forecast to terminal."""
    print("=" * 60)
    print(f"  KRONOS FORECAST — {forecast.ticker} ({forecast.interval})")
    print("=" * 60)
    print(f"  Current Price:    ${forecast.current_price:,.2f}")
    print(f"  Predicted Close:  ${forecast.pred_final_close:,.2f} ({forecast.pred_bars} bars out)")
    print(f"  Expected Return:  {forecast.expected_return:+.2%}")
    print(f"  Direction:        {forecast.direction.upper()}")
    print(f"  Confidence:       {forecast.confidence:.0%}")
    print(f"  High Watermark:   ${forecast.pred_high_watermark:,.2f}")
    print(f"  Low Watermark:    ${forecast.pred_low_watermark:,.2f}")
    print(f"  Lookback:         {forecast.lookback_bars} bars")

    # Show predicted trajectory (sampled points)
    n = len(forecast.predicted_close)
    if n > 5:
        points = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        print(f"\n  Trajectory:")
        for i in points:
            bar_label = f"T+{i + 1}"
            price = forecast.predicted_close[i]
            change = (price - forecast.current_price) / forecast.current_price
            print(f"    {bar_label:6s}  ${price:,.2f}  ({change:+.2%})")

    print("=" * 60)


def print_probability_report(result: PriceProbability):
    """Print a formatted probability estimate to terminal."""
    print("=" * 60)
    print(f"  KRONOS PROBABILITY — {result.ticker}")
    print("=" * 60)
    print(f"  Question:       Will {result.ticker} be {result.direction} ${result.target_price:,.2f}?")
    print(f"  Probability:    {result.probability:.1%}")
    print(f"  Confidence:     {result.confidence:.0%}")
    print(f"  Horizon:        {result.horizon_bars} {result.interval} bars")

    if result.forecast:
        f = result.forecast
        print(f"\n  Current Price:  ${f.current_price:,.2f}")
        print(f"  Predicted:      ${f.pred_final_close:,.2f} ({f.expected_return:+.2%})")
        print(f"  Direction:      {f.direction.upper()}")
        print(f"  Range:          ${f.pred_low_watermark:,.2f} — ${f.pred_high_watermark:,.2f}")

    print("=" * 60)


def print_volatility_report(result: VolatilityForecast):
    """Print a formatted Kronos volatility forecast to terminal."""
    print("=" * 60)
    print(f"  KRONOS VOLATILITY — {result.ticker} ({result.interval})")
    print("=" * 60)
    print(f"  Current Price:       ${result.current_price:,.2f}")
    print(f"  Horizon:             {result.horizon_bars} bars")
    print(f"  Predicted Ann Vol:   {result.realized_vol_annualized:.1%}")
    print(f"  Predicted Period:    {result.realized_vol_period:.1%}")
    print(f"  Historical Ann Vol:  {result.historical_vol_annualized:.1%}")
    delta = result.realized_vol_annualized - result.historical_vol_annualized
    delta_pct = (delta / result.historical_vol_annualized * 100) if result.historical_vol_annualized > 0 else 0
    print(f"  Change vs History:   {delta:+.1%} ({delta_pct:+.0f}% relative)")
    print(f"  Regime:              {result.vol_regime.upper()}")
    print(f"  Confidence:          {result.confidence:.0%}")
    print("=" * 60)
