"""
Volatility-Aware Position Sizing — vol-targeting modifier on top of Kelly.

Adapted from the Adaptive-Volatility-Regime-Based-Execution-and-Risk-
Framework pattern (github.com/shreejitverma/...): use forward-realized
volatility to scale position size, NOT just the Kelly edge. Same Sharpe,
materially less drawdown.

The intuition: Kelly's optimal-fraction formula uses the EDGE (reward/risk
ratio + win prob), but says nothing about regime. A 5% edge in a calm
market and a 5% edge in a high-vol market deserve different sizes —
the high-vol bet has a wider distribution of outcomes around the edge,
so the same Kelly fraction has more variance. Vol-targeting compensates.

Formula:
    vol_ratio = current_vol / baseline_vol
    vol_multiplier = clip(1.0 / sqrt(vol_ratio), FLOOR, CEILING)

    When current_vol == baseline_vol:  multiplier = 1.0 (no change)
    When current_vol == 2x baseline:   multiplier ≈ 0.71 (downsize)
    When current_vol == 4x baseline:   multiplier = FLOOR (cap)
    When current_vol == 0.5x baseline: multiplier = sqrt(2) ≈ 1.41 → CEILING (cap)

We CAP both directions. We don't upsize aggressively in low-vol regimes
because (a) low vol can flip suddenly (vol clustering), and (b) we have
other signals (composite_score, bull/bear) carrying conviction info.
The asymmetry: downsizing in high vol is the high-value direction; the
ceiling is mostly defensive.

Optional Kronos integration: if `use_kronos_forecast=True`, the current_vol
is replaced with Kronos's predicted realized vol over a forward window.
This is the strongest task in the Kronos paper (44% MAE reduction vs
baselines), so when it's available, we prefer it over backward-looking
realized vol. Kronos calls are expensive (~6-10s on local CPU), so this
is opt-in and reserved for high-conviction candidates.

Public API:
    compute_vol_multiplier(ticker, daily_df, *, cfg) -> (mult, meta)

Falls back to multiplier=1.0 (no adjustment) whenever data is insufficient
or the computation fails — bias is toward NOT modifying sizing in edge
cases. Better to use Kelly's raw output than to apply a noisy modifier.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from lib.audit import log_event


# Sensible defaults. Override via wheel_strategy.yaml.vol_aware_sizing
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "current_window": 20,           # bars used for current realized vol
    "baseline_window": 60,          # bars used for baseline vol comparison
    "use_kronos_forecast": False,   # opt-in: replace current_vol with Kronos's forward forecast
    "kronos_horizon": 20,           # forecast horizon when use_kronos_forecast=True
    "floor": 0.5,                   # min multiplier — never reduce position below 50% Kelly
    "ceiling": 1.0,                 # max multiplier — never upsize past 100% Kelly (defensive)
    "min_baseline_vol": 0.05,       # absolute floor on baseline_vol to prevent divide-near-zero blowup
}


def _resolve(cfg: dict | None, key: str):
    if cfg is None:
        return DEFAULTS[key]
    return cfg.get(key, DEFAULTS[key])


def compute_vol_multiplier(
    ticker: str,
    daily_df: pd.DataFrame | None,
    *,
    cfg: dict | None = None,
) -> tuple[float, dict]:
    """
    Compute the vol-target multiplier for a Kelly-sized position.

    Returns:
        (multiplier, meta) where multiplier is in [FLOOR, CEILING] and
        meta carries the diagnostics (current_vol, baseline_vol, ratio,
        method, fallback_reason).

    Failure mode: returns (1.0, meta_with_reason). The integrator should
    always apply the returned multiplier; on insufficient data we just
    return 1.0 so Kelly's output is used unchanged.
    """
    cfg = cfg or DEFAULTS
    enabled = _resolve(cfg, "enabled")
    if not enabled:
        return 1.0, {"method": "disabled", "multiplier": 1.0}

    current_window = int(_resolve(cfg, "current_window"))
    baseline_window = int(_resolve(cfg, "baseline_window"))
    floor = float(_resolve(cfg, "floor"))
    ceiling = float(_resolve(cfg, "ceiling"))
    min_baseline = float(_resolve(cfg, "min_baseline_vol"))
    use_kronos = bool(_resolve(cfg, "use_kronos_forecast"))

    # Need enough bars for the baseline window.
    if daily_df is None or len(daily_df) < baseline_window + 2:
        return 1.0, {
            "method": "fallback_insufficient_bars",
            "have": 0 if daily_df is None else len(daily_df),
            "need": baseline_window + 2,
            "multiplier": 1.0,
        }

    # Compute backward-looking realized vols. Re-using the existing
    # implementation in options_signals so we don't fork the math.
    try:
        from lib.options_signals import realized_vol_window
        baseline_vol = realized_vol_window(daily_df, baseline_window)
        current_vol_realized = realized_vol_window(daily_df, current_window)
    except Exception as e:
        log_event("vol_sizing", "rv_compute_failed",
                  {"ticker": ticker, "error": str(e)[:200]}, result="degraded")
        return 1.0, {"method": "fallback_rv_error", "multiplier": 1.0,
                     "error": str(e)[:200]}

    if baseline_vol is None or current_vol_realized is None:
        return 1.0, {
            "method": "fallback_none_vol",
            "baseline_vol": baseline_vol,
            "current_vol": current_vol_realized,
            "multiplier": 1.0,
        }

    if baseline_vol < min_baseline:
        # Clamp the denominator. A tiny baseline_vol makes the ratio
        # explode and we'd downsize for noise. Treat as "data weird,
        # don't modify sizing."
        return 1.0, {
            "method": "fallback_baseline_clamp",
            "baseline_vol": baseline_vol,
            "min_baseline_vol": min_baseline,
            "multiplier": 1.0,
        }

    # Forward-looking vol (Kronos). Optional, slow. When available,
    # it's a better numerator than backward-looking current_vol.
    current_vol = current_vol_realized
    method = "realized_rv20_over_rv60"
    if use_kronos:
        kronos_vol = _try_kronos_vol(ticker, cfg)
        if kronos_vol is not None and kronos_vol > 0:
            current_vol = kronos_vol
            method = "kronos_forecast_over_rv60"

    ratio = current_vol / baseline_vol
    raw_mult = 1.0 / math.sqrt(ratio)
    multiplier = max(floor, min(ceiling, raw_mult))

    meta = {
        "method": method,
        "ticker": ticker,
        "current_vol": round(current_vol, 4),
        "baseline_vol": round(baseline_vol, 4),
        "vol_ratio": round(ratio, 4),
        "raw_multiplier": round(raw_mult, 4),
        "multiplier": round(multiplier, 4),
        "clamped": raw_mult != multiplier,
    }
    if multiplier < 1.0:
        meta["interpretation"] = (
            f"vol elevated ({ratio:.2f}x baseline) — downsizing to {multiplier:.0%} of Kelly"
        )
    elif multiplier > 1.0:
        meta["interpretation"] = (
            f"vol compressed ({ratio:.2f}x baseline) — upsizing to {multiplier:.0%}"
        )
    else:
        meta["interpretation"] = "vol at baseline — no sizing adjustment"
    return multiplier, meta


def _try_kronos_vol(ticker: str, cfg: dict | None) -> float | None:
    """Best-effort Kronos forecast volatility. Returns None on any failure
    so the caller falls back to backward-looking RV. This is intentionally
    silent (no exception propagates) — Kronos is a nice-to-have, not a
    requirement.
    """
    try:
        from lib.kronos_forecaster import predict_volatility
        horizon = int(_resolve(cfg, "kronos_horizon"))
        forecast = predict_volatility(
            ticker=ticker,
            horizon_bars=horizon,
            interval="1d",
            sample_count=1,           # paper Table 6 default
            temperature=0.9,          # paper Table 6 default
            top_p=0.90,
        )
        vol = getattr(forecast, "annualized_vol", None) or getattr(
            forecast, "realized_vol", None
        )
        return float(vol) if vol else None
    except Exception:
        return None


def summary_line(meta: dict) -> str:
    """One-line summary for logs / dashboard / per-trade output."""
    if meta.get("method") == "disabled":
        return "vol_sizing: disabled"
    if meta.get("method", "").startswith("fallback"):
        return f"vol_sizing: skipped ({meta.get('method')})"
    return (
        f"vol_sizing: {meta.get('method')} "
        f"current={meta.get('current_vol')} "
        f"baseline={meta.get('baseline_vol')} "
        f"ratio={meta.get('vol_ratio')} "
        f"→ mult={meta.get('multiplier')}"
    )
