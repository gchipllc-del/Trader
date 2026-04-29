"""
Crypto LLM forecaster — Tier A.2 (2026-04-25).

Asks a 3-provider free LLM ensemble (Gemini + Groq + Cerebras) to forecast
the probability of hitting a target_pct gain over a horizon. Inspired by
the AI-Hedge-Fund repo (#8 in the Top-10 list) but kept minimal:

    - Single question, structured JSON response
    - First successful provider wins (no aggregation — keep it simple
      until we have enough data to weight providers empirically)
    - Falls back gracefully through provider list, returns None if all
      fail
    - 60-min in-memory cache keyed by (symbol, recent close, horizon)

Why free providers only:
    Anthropic / DeepSeek / Moonshot accounts had billing issues earlier.
    Gemini / Groq / Cerebras have generous free tiers and are sufficient
    for a single 30-day directional probability question.

Security:
    - API keys ONLY from env vars
    - LLM responses parsed defensively (probability clamped to [0.05, 0.95])
    - No keys logged
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lib.audit import log_event


# ── Provider registry ────────────────────────────────────────────

# Each provider entry: (name, env_var, base_url, model)
#
# 2026-04-28: Added gemini-3.1-pro-preview as the first-tier voice. Provider
# iteration is first-success-wins, so Pro Preview is tried first; on 429
# (Pro free tier ≈5 RPM, 50-100 RPD) the call auto-falls-through to Flash.
_PROVIDERS = [
    ("gemini_pro", "GOOGLE_API_KEY",
     "https://generativelanguage.googleapis.com/v1beta/openai/",
     "gemini-3.1-pro-preview"),  # Tier-1: best reasoning, tight free quota
    ("gemini", "GOOGLE_API_KEY",
     "https://generativelanguage.googleapis.com/v1beta/openai/",
     "gemini-2.5-flash"),  # Flash fallback: 15 RPM, 1500 RPD free
    ("groq", "GROQ_API_KEY",
     "https://api.groq.com/openai/v1",
     "llama-3.3-70b-versatile"),
    ("cerebras", "CEREBRAS_API_KEY",
     "https://api.cerebras.ai/v1",
     "qwen-3-235b-a22b-instruct-2507"),
]


# ── Tiny in-memory cache ─────────────────────────────────────────

_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_SECONDS = 60 * 60  # 1 hour


def _cache_key(symbol: str, last_close: float, target_pct: float, horizon_days: int) -> str:
    raw = f"{symbol}|{round(last_close, 2)}|{target_pct}|{horizon_days}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_get(key: str) -> dict | None:
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_put(key: str, value: dict) -> None:
    _CACHE[key] = (time.time(), value)


# ── Prompt construction ──────────────────────────────────────────

def _build_prompt(symbol: str, df: pd.DataFrame, target_pct: float,
                  horizon_days: int, trend_details: dict) -> str:
    closes = df["close"].tail(30).tolist()
    current = closes[-1] if closes else 0
    high_30d = max(df["high"].tail(30).tolist()) if "high" in df.columns else current
    low_30d = min(df["low"].tail(30).tolist()) if "low" in df.columns else current

    # Compact recent price history for the LLM (last 14 daily closes)
    recent = ", ".join(f"${c:.2f}" for c in closes[-14:])

    return f"""You are a quantitative crypto analyst. Estimate the probability that {symbol} \
gains at least {target_pct*100:.1f}% within the next {horizon_days} days from the \
current price of ${current:,.2f}.

Recent 14-day daily closes: {recent}
30-day high: ${high_30d:,.2f}
30-day low: ${low_30d:,.2f}

Technical signals (computed locally):
- Trend score: {trend_details.get('above_ma20', False) and 'above' or 'below'} 20-day MA
- 7-day ROC: {trend_details.get('roc_7d', 0)*100:+.2f}%
- 14-day ROC: {trend_details.get('roc_14d', 0)*100:+.2f}%
- MACD bullish: {trend_details.get('macd_bullish', False)}
- MACD histogram rising: {trend_details.get('macd_hist_rising', False)}
- Bollinger %B: {trend_details.get('bb_pctb', 0.5):.2f}
- RSI: {trend_details.get('rsi', 50):.1f}
- Volume thrust (5d > 20d avg): {trend_details.get('volume_thrust', False)}
- ATR-strong (close > MA20 by ≥1 ATR): {trend_details.get('atr_strong', False)}

Reason carefully about:
1. Base rate: how often crypto majors gain {target_pct*100:.0f}% in {horizon_days} days historically
2. Current technical picture (signals above)
3. Crypto regime: are we in a bull / bear / chop phase based on the price path?
4. Mean reversion vs continuation: is the asset overextended?

Respond ONLY with valid JSON in this exact form:
{{"probability": 0.55, "reasoning": "Brief 1-2 sentence justification."}}

probability MUST be a float between 0.05 and 0.95."""


# ── Provider call ────────────────────────────────────────────────

def _call_provider(name: str, base_url: str, api_key: str, model: str,
                   prompt: str, timeout: int = 30) -> dict | None:
    """Call one OpenAI-compatible provider. Returns parsed dict or None."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4,
            # Gemini's OpenAI-compat shim doesn't support response_format
            response_format=(
                {"type": "json_object"}
                if not name.startswith("gemini") else None
            ),
        )
        content = resp.choices[0].message.content
        if not content:
            return None

        # Defensive parse — strip code fences, then JSON.loads
        cleaned = content.strip()
        if cleaned.startswith("```"):
            # Strip fenced code block
            lines = cleaned.split("\n")
            cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))
        data = json.loads(cleaned)

        prob = float(data.get("probability", 0.5))
        prob = max(0.05, min(0.95, prob))  # Clamp
        reasoning = str(data.get("reasoning", ""))[:500]

        return {"probability": prob, "reasoning": reasoning, "provider": name, "model": model}

    except Exception as e:
        log_event("crypto_llm", "provider_failed", {
            "provider": name,
            "error": type(e).__name__ + ": " + str(e)[:150],
        })
        return None


# ── Public entry point ───────────────────────────────────────────

def forecast_crypto_target(
    symbol: str,
    df: pd.DataFrame,
    target_pct: float = 0.15,
    horizon_days: int = 30,
    trend_details: dict | None = None,
) -> dict | None:
    """Forecast probability of `symbol` gaining `target_pct` within `horizon_days`.

    Returns dict {"probability": float, "reasoning": str, "provider": str}
    or None if no provider responded.

    Tries providers in order until one succeeds (free-tier providers
    have spotty availability — first-success is fine for a directional
    probability question).
    """
    if df is None or df.empty:
        return None
    if trend_details is None:
        trend_details = {}

    last_close = float(df["close"].iloc[-1])
    cache_key = _cache_key(symbol, last_close, target_pct, horizon_days)

    cached = _cache_get(cache_key)
    if cached:
        log_event("crypto_llm", "cache_hit", {"symbol": symbol})
        return {**cached, "cached": True}

    prompt = _build_prompt(symbol, df, target_pct, horizon_days, trend_details)

    for name, env_var, base_url, model in _PROVIDERS:
        api_key = os.environ.get(env_var, "").strip()
        if not api_key:
            log_event("crypto_llm", "provider_no_key", {"provider": name})
            continue

        result = _call_provider(name, base_url, api_key, model, prompt)
        if result is not None:
            log_event("crypto_llm", "forecast_success", {
                "symbol": symbol,
                "provider": name,
                "probability": result["probability"],
            })
            _cache_put(cache_key, result)
            return result

    log_event("crypto_llm", "all_providers_failed", {"symbol": symbol})
    return None
