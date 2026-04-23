"""
Alpha Vantage API client — economic + fundamental data provider.

Docs: https://www.alphavantage.co/documentation/

Key env var:
    ALPHA_VANTAGE_API_KEY — read from environment (never stored in code/config)

Graceful degradation: every function returns None/[] on missing key or error.

Focus (complements Finnhub, not duplicates it):
    - Economic indicators (GDP, CPI, unemployment, fed funds rate) — regime signals
    - News sentiment (AV has an integrated sentiment model — useful as a 3rd
      source alongside NewsAPI + Yahoo Finance RSS)
    - Company overview (once-per-ticker reference data)
    - Earnings history (complements Finnhub forward calendar)

Rate limiting:
    Free tier = 25 calls/day, 5 calls/minute. We cache aggressively.
    12s between calls = 5/min max.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from lib.audit import log_event

BASE_URL = "https://www.alphavantage.co/query"
CACHE_DIR = Path(__file__).parent.parent / "data" / "alpha_vantage_cache"
DEFAULT_TIMEOUT = 15
RATE_LIMIT_INTERVAL = 12.0  # seconds — 5 calls/min max


# ── Rate limiter ─────────────────────────────────────────────────

_last_call = 0.0


def _rate_limit():
    global _last_call
    now = time.time()
    elapsed = now - _last_call
    if elapsed < RATE_LIMIT_INTERVAL:
        time.sleep(RATE_LIMIT_INTERVAL - elapsed)
    _last_call = time.time()


# ── Cache ────────────────────────────────────────────────────────

def _cache_key(function: str, params: dict) -> str:
    raw = f"{function}:{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _cache_get(key: str, ttl_seconds: int):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > ttl_seconds:
            return None
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _cache_put(key: str, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(path)


# ── Core HTTP ────────────────────────────────────────────────────

def _api_key() -> str | None:
    return os.environ.get("ALPHA_VANTAGE_API_KEY")


def _get(function: str, params: dict | None = None, ttl_seconds: int = 24 * 3600) -> Any:
    """Low-level GET with rate limit, cache, graceful failure."""
    key = _api_key()
    if not key:
        log_event("alpha_vantage", "no_api_key", {"function": function})
        return None

    params = dict(params or {})
    params["function"] = function
    cache_k = _cache_key(function, params)
    cached = _cache_get(cache_k, ttl_seconds)
    if cached is not None:
        return cached

    params["apikey"] = key
    _rate_limit()

    try:
        resp = requests.get(BASE_URL, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            log_event("alpha_vantage", "http_error", {
                "function": function,
                "status": resp.status_code,
                "body": resp.text[:200],
            })
            return None
        data = resp.json()
        # AV returns 200 with an error/info key instead of an HTTP error
        if isinstance(data, dict):
            if "Error Message" in data:
                log_event("alpha_vantage", "api_error", {
                    "function": function,
                    "message": str(data.get("Error Message"))[:200],
                })
                return None
            if "Note" in data:  # Rate limit warning
                log_event("alpha_vantage", "rate_limited", {
                    "function": function,
                    "note": str(data.get("Note"))[:200],
                })
                return None
            if "Information" in data and "API rate" in str(data.get("Information", "")):
                log_event("alpha_vantage", "rate_limited", {
                    "function": function,
                    "info": str(data.get("Information"))[:200],
                })
                return None
        _cache_put(cache_k, data)
        return data
    except requests.exceptions.Timeout:
        log_event("alpha_vantage", "timeout", {"function": function})
        return None
    except Exception as e:
        log_event("alpha_vantage", "request_failed", {"function": function, "error": str(e)[:200]})
        return None


# ── Economic Indicators ──────────────────────────────────────────

@dataclass
class EconomicSnapshot:
    """Latest values of key macro indicators. Used for regime detection."""
    gdp_growth: float | None        # Real GDP YoY (%)
    cpi: float | None               # Latest CPI level
    unemployment: float | None      # Unemployment rate (%)
    fed_funds_rate: float | None    # Fed Funds effective rate (%)
    treasury_10y: float | None      # 10Y Treasury yield (%)
    as_of: str                      # ISO date of freshest data


def get_economic_snapshot() -> EconomicSnapshot | None:
    """Fetch the latest values of core macro indicators.

    Returns None if no key or all lookups failed.
    Partial results are OK — fields default to None.
    """
    if not _api_key():
        return None

    def _first_value(d: dict | None) -> float | None:
        if not d or "data" not in d:
            return None
        data = d.get("data") or []
        if not data:
            return None
        try:
            v = data[0].get("value")
            if v in (None, ".", ""):
                return None
            return float(v)
        except (ValueError, TypeError, KeyError):
            return None

    def _first_date(d: dict | None) -> str:
        if not d or "data" not in d:
            return ""
        data = d.get("data") or []
        if not data:
            return ""
        return str(data[0].get("date", ""))

    # These each cost 1 AV call. We cache them 24h.
    gdp = _get("REAL_GDP", {"interval": "quarterly"}, ttl_seconds=24 * 3600)
    cpi = _get("CPI", {"interval": "monthly"}, ttl_seconds=24 * 3600)
    unemp = _get("UNEMPLOYMENT", {}, ttl_seconds=24 * 3600)
    fed = _get("FEDERAL_FUNDS_RATE", {"interval": "monthly"}, ttl_seconds=24 * 3600)
    t10 = _get("TREASURY_YIELD", {"interval": "monthly", "maturity": "10year"}, ttl_seconds=24 * 3600)

    dates = [d for d in (_first_date(gdp), _first_date(cpi), _first_date(unemp),
                          _first_date(fed), _first_date(t10)) if d]
    as_of = max(dates) if dates else ""

    if all(x is None for x in (_first_value(gdp), _first_value(cpi), _first_value(unemp),
                                _first_value(fed), _first_value(t10))):
        return None

    return EconomicSnapshot(
        gdp_growth=_first_value(gdp),
        cpi=_first_value(cpi),
        unemployment=_first_value(unemp),
        fed_funds_rate=_first_value(fed),
        treasury_10y=_first_value(t10),
        as_of=as_of,
    )


# ── News Sentiment ──────────────────────────────────────────────

@dataclass
class AvNewsItem:
    title: str
    url: str
    time_published: str
    summary: str
    overall_sentiment_score: float  # -1 (bearish) to +1 (bullish)
    overall_sentiment_label: str
    source: str
    tickers: list[str]


def get_news_sentiment(ticker: str, limit: int = 20) -> list[AvNewsItem]:
    """News articles with Alpha Vantage's built-in sentiment scoring.

    Complements our existing news_sentiment module (which uses NewsAPI + Yahoo).
    AV has its own sentiment model, so it's a useful cross-check.
    """
    data = _get("NEWS_SENTIMENT", {"tickers": ticker, "limit": str(limit), "sort": "LATEST"},
                ttl_seconds=30 * 60)
    if not data or not isinstance(data, dict):
        return []

    items = []
    for n in (data.get("feed") or [])[:limit]:
        try:
            items.append(AvNewsItem(
                title=str(n.get("title", ""))[:300],
                url=str(n.get("url", ""))[:300],
                time_published=str(n.get("time_published", "")),
                summary=str(n.get("summary", ""))[:500],
                overall_sentiment_score=float(n.get("overall_sentiment_score", 0.0)),
                overall_sentiment_label=str(n.get("overall_sentiment_label", ""))[:30],
                source=str(n.get("source", ""))[:50],
                tickers=[str(t.get("ticker", ""))[:10] for t in (n.get("ticker_sentiment") or [])[:10]],
            ))
        except Exception:
            continue
    return items


def aggregate_news_sentiment(ticker: str, limit: int = 20) -> dict | None:
    """Compact sentiment summary: mean score, bullish %, bearish %, article count."""
    items = get_news_sentiment(ticker, limit=limit)
    if not items:
        return None
    scores = [i.overall_sentiment_score for i in items if i.overall_sentiment_score is not None]
    if not scores:
        return None
    n = len(scores)
    mean = sum(scores) / n
    bullish = sum(1 for s in scores if s > 0.15)
    bearish = sum(1 for s in scores if s < -0.15)
    return {
        "ticker": ticker.upper(),
        "article_count": n,
        "mean_score": round(mean, 4),
        "bullish_pct": round(bullish / n, 3),
        "bearish_pct": round(bearish / n, 3),
        "neutral_pct": round((n - bullish - bearish) / n, 3),
        "label": "bullish" if mean > 0.15 else ("bearish" if mean < -0.15 else "neutral"),
    }


# ── Company Overview ─────────────────────────────────────────────

def get_company_overview(ticker: str) -> dict | None:
    """One-shot company reference data: name, sector, industry, market cap, PE, etc."""
    data = _get("OVERVIEW", {"symbol": ticker}, ttl_seconds=7 * 24 * 3600)
    if not data or not isinstance(data, dict):
        return None
    if not data.get("Symbol"):
        return None  # AV returns empty object on invalid ticker
    return {
        "ticker": data.get("Symbol"),
        "name": data.get("Name"),
        "sector": data.get("Sector"),
        "industry": data.get("Industry"),
        "market_cap": _safe_float(data.get("MarketCapitalization")),
        "pe_ratio": _safe_float(data.get("PERatio")),
        "peg_ratio": _safe_float(data.get("PEGRatio")),
        "book_value": _safe_float(data.get("BookValue")),
        "dividend_yield": _safe_float(data.get("DividendYield")),
        "eps": _safe_float(data.get("EPS")),
        "profit_margin": _safe_float(data.get("ProfitMargin")),
        "beta": _safe_float(data.get("Beta")),
        "52w_high": _safe_float(data.get("52WeekHigh")),
        "52w_low": _safe_float(data.get("52WeekLow")),
        "analyst_target_price": _safe_float(data.get("AnalystTargetPrice")),
    }


# ── Earnings History ─────────────────────────────────────────────

def get_earnings_history(ticker: str) -> dict | None:
    """Quarterly and annual earnings history. Complements Finnhub forward calendar."""
    data = _get("EARNINGS", {"symbol": ticker}, ttl_seconds=24 * 3600)
    if not data or not isinstance(data, dict):
        return None
    return {
        "ticker": ticker.upper(),
        "quarterly": (data.get("quarterlyEarnings") or [])[:8],
        "annual": (data.get("annualEarnings") or [])[:5],
    }


# ── Helpers ──────────────────────────────────────────────────────

def _safe_float(x) -> float | None:
    if x in (None, "", "None", "-"):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def alpha_vantage_status() -> dict:
    """Diagnostic: key set + test call status."""
    key_set = bool(_api_key())
    reachable = None
    if key_set:
        # Use GLOBAL_QUOTE as a cheap connectivity test (1 call)
        r = _get("GLOBAL_QUOTE", {"symbol": "SPY"}, ttl_seconds=0)
        reachable = r is not None
    return {
        "key_set": key_set,
        "reachable": reachable,
        "cache_dir": str(CACHE_DIR),
    }
