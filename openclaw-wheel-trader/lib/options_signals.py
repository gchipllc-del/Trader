"""
Options-derived signals for stronger stock-trade entries.

Even when trading equities (Phase 1, no options exposure), the *options
market* often leads price action. Three signals worth gating on:

  1. **IV regime** — SPY's ATM IV vs its 20-day average. Calm regime
     favors trend continuation; stressed regime favors mean-reversion
     and tighter stops. Used as a market-wide context for entries.

  2. **IV vs HV (per ticker)** — implied volatility / 20-day realized
     volatility. Ratio < 0.85 means options are CHEAP relative to
     actual price action — often institutional positioning hint that
     options market expects calm. Boost long entries.

  3. **Volume vs OI imbalance (call side)** — today's call volume
     vs aggregate call OI. Spike = aggressive new positioning. If
     accompanied by price strength, signals institutional accumulation.

All three use the existing Alpaca options snapshot endpoint (free tier
returns bid/ask/volume/OI but NOT greeks — we derive IV via py_vollib's
inverse Black-Scholes solver).

Each signal returns a tuple (score_delta: int, reason: str). The caller
combines them into a single options_score that boosts the composite
trade score by [-2, +2] points.

Wired into score_stock_buy() as a separate component; can be disabled
via `options_signals.enabled: false` in wheel_strategy.yaml.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from lib.audit import log_event

CACHE_DIR = Path(__file__).parent.parent / "data" / "options_signals_cache"
CACHE_TTL_SECONDS = 600  # 10-min cache: options data slow-changing intra-day


# --- Black-Scholes inverse for IV --------------------------------------

def _compute_iv_from_mid(
    spot: float, strike: float, dte_years: float, mid_price: float,
    is_call: bool, risk_free_rate: float = 0.045,
) -> float | None:
    """Solve for IV given mid-market price using py_vollib's solver.

    Returns IV as a decimal (e.g., 0.25 = 25% annualized) or None if
    the solver fails (illiquid contract, bad mid, etc.).
    """
    try:
        from py_vollib.black_scholes.implied_volatility import implied_volatility
        flag = "c" if is_call else "p"
        iv = implied_volatility(
            price=mid_price, S=spot, K=strike, t=dte_years,
            r=risk_free_rate, flag=flag,
        )
        # Sanity-clamp: any IV outside [3%, 500%] is almost certainly junk
        if 0.03 <= iv <= 5.0:
            return float(iv)
        return None
    except Exception:
        return None


# --- Alpaca options snapshot helpers ------------------------------------

def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": os.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
    }


def _fetch_options_snapshots(ticker: str, limit: int = 100) -> dict:
    """Fetch up to `limit` option contracts for `ticker` with bid/ask/IV.

    Returns dict {snapshot_symbol: snapshot_data} or empty on error.
    """
    url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{ticker}"
    try:
        r = requests.get(
            url,
            headers=_alpaca_headers(),
            params={"feed": "indicative", "limit": limit},
            timeout=15,
        )
        if r.status_code == 200:
            return (r.json() or {}).get("snapshots", {}) or {}
    except Exception as e:
        log_event("options_signals", "fetch_failed",
                  {"ticker": ticker, "error": str(e)[:200]}, result="degraded")
    return {}


def _parse_contract_symbol(sym: str) -> tuple[str, str, str, float] | None:
    """Parse OCC option symbol like SPY260428C00505000.

    Returns (ticker, expiry_iso, type, strike) or None on parse failure.
    Format: TICKER + YYMMDD + C/P + STRIKE*1000 zero-padded to 8 digits.
    """
    try:
        # Find the date — first 6 digits after the ticker
        i = 0
        while i < len(sym) and not sym[i].isdigit():
            i += 1
        ticker = sym[:i]
        if i + 7 > len(sym):
            return None
        yy = int(sym[i:i+2])
        mm = int(sym[i+2:i+4])
        dd = int(sym[i+4:i+6])
        cp = sym[i+6]
        strike_raw = int(sym[i+7:i+15])
        strike = strike_raw / 1000.0
        # Assume 20YY for years > 50, else 21YY (handles 2050+ safely)
        year = 2000 + yy
        expiry = f"{year:04d}-{mm:02d}-{dd:02d}"
        return (ticker, expiry, "call" if cp == "C" else "put", strike)
    except Exception:
        return None


# --- Core signal: ATM IV ------------------------------------------------

def compute_atm_iv(
    ticker: str, current_price: float, target_dte_days: int = 30,
) -> float | None:
    """Compute the ATM implied vol for `ticker`.

    Strategy:
      1. Fetch chain snapshots for the ticker
      2. Filter to contracts within ±5% of current price (ATM region)
      3. Pick expiry closest to target_dte_days
      4. For each ATM call+put, derive IV from mid-price using py_vollib
      5. Return the average — typically smoothes minor mispricing

    Returns IV as decimal (e.g., 0.27) or None if no usable contracts.
    """
    if current_price <= 0:
        return None

    snapshots = _fetch_options_snapshots(ticker, limit=200)
    if not snapshots:
        return None

    today = datetime.now(timezone.utc).date()
    target_date = today + timedelta(days=target_dte_days)

    candidates = []
    for sym, snap in snapshots.items():
        parsed = _parse_contract_symbol(sym)
        if parsed is None:
            continue
        _, exp_iso, opt_type, strike = parsed
        # ATM filter: ±5% of current price
        if abs(strike - current_price) / current_price > 0.05:
            continue
        # DTE filter: drop expired
        try:
            exp_date = datetime.fromisoformat(exp_iso).date()
        except Exception:
            continue
        if exp_date <= today:
            continue
        # Mid price from quote
        quote = snap.get("latest_quote") or {}
        bid = float(quote.get("bp", 0) or 0)
        ask = float(quote.get("ap", 0) or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2.0
        # DTE in years
        dte_days = (exp_date - today).days
        dte_years = dte_days / 365.0
        # Distance from target DTE
        dte_distance = abs(dte_days - target_dte_days)
        candidates.append({
            "symbol": sym, "type": opt_type, "strike": strike,
            "mid": mid, "dte_years": dte_years, "dte_distance": dte_distance,
        })

    if not candidates:
        return None

    # Pick the expiry-bucket closest to target DTE
    candidates.sort(key=lambda c: c["dte_distance"])
    nearest_dte = candidates[0]["dte_years"]
    target_bucket = [
        c for c in candidates
        if abs(c["dte_years"] - nearest_dte) < 0.02  # within ~7 days
    ]

    # Compute IVs
    ivs = []
    for c in target_bucket:
        iv = _compute_iv_from_mid(
            spot=current_price, strike=c["strike"],
            dte_years=c["dte_years"], mid_price=c["mid"],
            is_call=(c["type"] == "call"),
        )
        if iv is not None:
            ivs.append(iv)

    if not ivs:
        return None

    return float(np.mean(ivs))


# --- Historical realized volatility -------------------------------------

def realized_vol_20d(daily_df: pd.DataFrame, window: int = 20) -> float | None:
    """20-day annualized realized vol from log returns. None on failure."""
    if daily_df is None or len(daily_df) < window + 1:
        return None
    try:
        log_ret = np.log(daily_df["close"] / daily_df["close"].shift(1))
        sigma = float(log_ret.tail(window).std())
        if np.isnan(sigma) or sigma <= 0:
            return None
        return sigma * np.sqrt(252)  # annualize
    except Exception:
        return None


# --- Signal 1: HV-regime per ticker (vol coiling vs expanding) ----------
#
# 2026-04-28: Alpaca's free `feed=indicative` returns chain *structure*
# but bid/ask are 0, so we can't derive IV from mid-price. Pivoted to
# HV-regime (20d vs 60d realized volatility ratio) which delivers the
# same conceptual signal (coiled spring vs already-moving) using only
# the daily OHLC bars we already have for free. compute_atm_iv() is
# kept as scaffolding for when you upgrade to Alpaca Algo Trader Plus
# ($99/mo, paid OPRA feed enables real bid/ask quotes).

def realized_vol_window(daily_df: pd.DataFrame, window: int) -> float | None:
    """Annualized realized vol over `window` bars."""
    if daily_df is None or len(daily_df) < window + 1:
        return None
    try:
        log_ret = np.log(daily_df["close"] / daily_df["close"].shift(1))
        sigma = float(log_ret.tail(window).std())
        if np.isnan(sigma) or sigma <= 0:
            return None
        return sigma * np.sqrt(252)
    except Exception:
        return None


def hv_regime_signal(
    ticker: str, current_price: float, daily_df: pd.DataFrame,
) -> tuple[int, str]:
    """Per-ticker volatility-regime signal from HV ratio.

    HV20 / HV60 measures how today's realized vol compares to the
    longer-run base rate. Three regimes:

      +1 = HV20/HV60 < 0.80 → vol contracting, "coiled spring" — long
            entries here often resolve into breakouts as compression
            energy releases
       0 = ratio in [0.80, 1.30] → normal, no signal
      −1 = HV20/HV60 > 1.30 → vol expanding, the move is already
            happening — late to enter, higher whipsaw risk
    """
    hv20 = realized_vol_window(daily_df, 20)
    hv60 = realized_vol_window(daily_df, 60)
    if hv20 is None or hv60 is None or hv60 <= 0:
        return 0, "no_hv_data"
    ratio = hv20 / hv60
    if ratio < 0.80:
        return +1, f"vol_coiling_hv20/60={ratio:.2f}"
    if ratio > 1.30:
        return -1, f"vol_expanding_hv20/60={ratio:.2f}"
    return 0, f"vol_normal_hv20/60={ratio:.2f}"


# Backwards-compat alias — old name still works in any external caller
iv_vs_hv_signal = hv_regime_signal


# --- Signal 2: Market HV regime (SPY proxy) -----------------------------

_REGIME_CACHE: dict[str, tuple[float, dict]] = {}
_REGIME_TTL_SECONDS = 30 * 60  # SPY IV slow-changing; 30 min cache


def market_iv_regime(daily_spy_df: pd.DataFrame | None = None) -> tuple[int, str]:
    """Classify the broader market regime via SPY HV ratio.

    Returns (boost: int, reason: str):
      +1 = vol contracting (SPY HV20/HV60 < 0.80) → calm, trend-follow edge
       0 = NORMAL regime
      −1 = vol expanding (SPY HV20/HV60 > 1.30) → suppress momentum entries
              (late, whipsaw risk)

    Cached 30 min so per-candidate calls don't recompute. Caller passes
    SPY daily bars from the scan's data fetch (no extra API call).
    """
    cache_entry = _REGIME_CACHE.get("spy")
    if cache_entry:
        ts, value = cache_entry
        if time.time() - ts < _REGIME_TTL_SECONDS:
            return value["delta"], value["reason"]

    if daily_spy_df is None:
        return 0, "no_spy_data"

    hv20 = realized_vol_window(daily_spy_df, 20)
    hv60 = realized_vol_window(daily_spy_df, 60)
    if hv20 is None or hv60 is None or hv60 <= 0:
        return 0, "no_spy_hv"

    ratio = hv20 / hv60
    if ratio < 0.80:
        delta, reason = +1, f"calm_market_hv20/60={ratio:.2f}"
    elif ratio > 1.30:
        delta, reason = -1, f"stressed_market_hv20/60={ratio:.2f}"
    else:
        delta, reason = 0, f"normal_market_hv20/60={ratio:.2f}"

    _REGIME_CACHE["spy"] = (time.time(), {"delta": delta, "reason": reason})
    return delta, reason


# Alias: this used to be IV-based; now HV-based. Old callers still work.
market_hv_regime = market_iv_regime


# --- Signal 3: Unusual call volume --------------------------------------

def unusual_call_volume(
    ticker: str, current_price: float,
) -> tuple[int, str]:
    """Spot today's call volume vs aggregate call open-interest.

    Heuristic: a call-side aggregate volume that exceeds 50% of total
    call OI in one day signals aggressive new positioning. If
    accompanied by price strength (caller's responsibility to combine
    with momentum signal), this is a high-conviction accumulation hint.

    Returns (boost: int, reason: str).
    """
    snapshots = _fetch_options_snapshots(ticker, limit=200)
    if not snapshots:
        return 0, "no_chain"

    today = datetime.now(timezone.utc).date()
    call_volume_today = 0.0
    call_oi = 0.0

    for sym, snap in snapshots.items():
        parsed = _parse_contract_symbol(sym)
        if parsed is None:
            continue
        _, exp_iso, opt_type, strike = parsed
        if opt_type != "call":
            continue
        # ATM region only — wings are noisy
        if abs(strike - current_price) / current_price > 0.10:
            continue
        try:
            exp_date = datetime.fromisoformat(exp_iso).date()
        except Exception:
            continue
        if exp_date <= today:
            continue
        bar = snap.get("daily_bar") or {}
        call_volume_today += float(bar.get("v", 0) or 0)
        call_oi += float(snap.get("open_interest", 0) or 0)

    if call_oi <= 0:
        return 0, "no_oi_data"
    ratio = call_volume_today / call_oi
    if ratio >= 0.50:
        return +1, f"unusual_call_vol_ratio={ratio:.2f}"
    if ratio >= 0.30:
        return 0, f"elevated_call_vol_ratio={ratio:.2f}"
    return 0, f"normal_call_vol_ratio={ratio:.2f}"


# --- Aggregate: full options-signal score for a candidate ---------------

def options_signal_score(
    ticker: str, current_price: float,
    daily_df: pd.DataFrame,
    market_regime_delta: int = 0,
    enable_iv_hv: bool = True,
    enable_volume: bool = False,  # off by default — slower extra API call
) -> tuple[int, list[str]]:
    """Combine all options-derived signals into a single ±2 score boost.

    Args:
        ticker: stock symbol
        current_price: latest close price
        daily_df: daily OHLC bars for HV calc
        market_regime_delta: pre-computed market regime delta (caller fetches
            once per scan via market_iv_regime()) — defaults to 0 (skip)
        enable_iv_hv: include the IV-vs-HV signal
        enable_volume: include the unusual-call-volume signal (extra API call)

    Returns:
        (score: int in [-2, +2], reasons: list[str])
    """
    score = 0
    reasons: list[str] = []

    # Market regime applies uniformly to all candidates this scan
    if market_regime_delta != 0:
        score += market_regime_delta
        reasons.append(f"regime{market_regime_delta:+d}")

    if enable_iv_hv:
        delta, why = iv_vs_hv_signal(ticker, current_price, daily_df)
        if delta != 0:
            score += delta
            reasons.append(why)

    if enable_volume:
        delta, why = unusual_call_volume(ticker, current_price)
        if delta != 0:
            score += delta
            reasons.append(why)

    # Cap final score at ±2 so options never dominate
    score = max(-2, min(2, score))
    return score, reasons
