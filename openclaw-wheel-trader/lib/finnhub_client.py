"""
Finnhub API client — financial data provider.

Docs: https://finnhub.io/docs/api

Key env var:
    FINNHUB_API_KEY — read from environment (never stored in code or config)

Graceful degradation: every function returns None on missing key / API error /
parse failure. Callers check for None and proceed without the data.

Supported endpoints:
    - Earnings calendar (critical: the bot's rules forbid selling options
      expiring through an earnings date)
    - Company news
    - Analyst recommendation trends
    - Insider transactions
    - Stock quote (sanity check fallback if Alpaca fails)
    - Basic financials (P/E, market cap, etc.)

Caching:
    - Earnings calendar:  6h TTL (rarely changes)
    - News:               30m TTL (shorter — fresh news matters)
    - Analyst recs:       24h TTL (monthly update frequency)
    - Insider trades:     6h TTL
    - Quote:              60s TTL (near-realtime)
    - Basic financials:   24h TTL

Rate limiting:
    Free tier = 60 calls/minute. We enforce 1.2s between calls as a safety margin.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests

from lib.audit import log_event

BASE_URL = "https://finnhub.io/api/v1"
CACHE_DIR = Path(__file__).parent.parent / "data" / "finnhub_cache"
DEFAULT_TIMEOUT = 10
RATE_LIMIT_INTERVAL = 1.2  # seconds between calls (60/min tier)


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

def _cache_key(endpoint: str, params: dict) -> str:
    raw = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
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
    return os.environ.get("FINNHUB_API_KEY")


def _get(endpoint: str, params: dict | None = None, ttl_seconds: int = 300) -> Any:
    """Low-level GET with rate limit, cache, and graceful failure.

    Returns parsed JSON or None on any failure.
    """
    key = _api_key()
    if not key:
        log_event("finnhub", "no_api_key", {"endpoint": endpoint})
        return None

    params = dict(params or {})
    # Cache key excludes the api token
    cache_k = _cache_key(endpoint, params)
    cached = _cache_get(cache_k, ttl_seconds)
    if cached is not None:
        return cached

    params["token"] = key
    _rate_limit()

    try:
        resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 429:
            log_event("finnhub", "rate_limited", {"endpoint": endpoint})
            return None
        if resp.status_code != 200:
            log_event("finnhub", "http_error", {
                "endpoint": endpoint,
                "status": resp.status_code,
                "body": resp.text[:200],
            })
            return None
        data = resp.json()
        _cache_put(cache_k, data)
        return data
    except requests.exceptions.Timeout:
        log_event("finnhub", "timeout", {"endpoint": endpoint})
        return None
    except Exception as e:
        log_event("finnhub", "request_failed", {"endpoint": endpoint, "error": str(e)[:200]})
        return None


# ── Public API ───────────────────────────────────────────────────

@dataclass
class EarningsEvent:
    ticker: str
    date: str               # ISO date "YYYY-MM-DD"
    hour: str               # "bmo" (before market open) | "amc" (after market close) | ""
    eps_estimate: float | None
    eps_actual: float | None
    revenue_estimate: float | None
    revenue_actual: float | None


def get_earnings_calendar(ticker: str, days_ahead: int = 60) -> list[EarningsEvent]:
    """Return upcoming earnings events for a ticker within `days_ahead` days.

    Returns [] if no key, API error, or no events.
    """
    today = datetime.now(timezone.utc).date()
    end = today + timedelta(days=days_ahead)
    data = _get(
        "/calendar/earnings",
        {"symbol": ticker, "from": today.isoformat(), "to": end.isoformat()},
        ttl_seconds=6 * 3600,
    )
    if not data or not isinstance(data, dict):
        return []

    events = []
    for e in data.get("earningsCalendar", []) or []:
        try:
            events.append(EarningsEvent(
                ticker=str(e.get("symbol", ticker)).upper(),
                date=str(e.get("date", "")),
                hour=str(e.get("hour", "")),
                eps_estimate=_safe_float(e.get("epsEstimate")),
                eps_actual=_safe_float(e.get("epsActual")),
                revenue_estimate=_safe_float(e.get("revenueEstimate")),
                revenue_actual=_safe_float(e.get("revenueActual")),
            ))
        except Exception:
            continue
    events.sort(key=lambda x: x.date)
    return events


def next_earnings_date(ticker: str, days_ahead: int = 60) -> str | None:
    """Return ISO date of next earnings, or None if unknown / too far out."""
    events = get_earnings_calendar(ticker, days_ahead=days_ahead)
    today_iso = datetime.now(timezone.utc).date().isoformat()
    for e in events:
        if e.date >= today_iso:
            return e.date
    return None


@dataclass
class NewsHeadline:
    ticker: str
    datetime: int            # Unix timestamp
    headline: str
    summary: str
    source: str
    url: str
    sentiment: float | None  # Not in free tier, but field kept for future


def get_company_news(ticker: str, days_back: int = 7) -> list[NewsHeadline]:
    """Return company-specific news headlines for the last `days_back` days."""
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)
    data = _get(
        "/company-news",
        {"symbol": ticker, "from": start.isoformat(), "to": today.isoformat()},
        ttl_seconds=30 * 60,
    )
    if not data or not isinstance(data, list):
        return []

    headlines = []
    for n in data[:50]:
        try:
            headlines.append(NewsHeadline(
                ticker=ticker.upper(),
                datetime=int(n.get("datetime", 0)),
                headline=str(n.get("headline", ""))[:300],
                summary=str(n.get("summary", ""))[:500],
                source=str(n.get("source", ""))[:50],
                url=str(n.get("url", ""))[:300],
                sentiment=None,
            ))
        except Exception:
            continue
    return headlines


@dataclass
class AnalystRec:
    period: str              # "YYYY-MM-01"
    buy: int
    hold: int
    sell: int
    strong_buy: int
    strong_sell: int

    @property
    def net_score(self) -> float:
        """Weighted score: strong_buy=+2, buy=+1, hold=0, sell=-1, strong_sell=-2.
        Normalized to [-1, +1] by total analysts."""
        total = self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell
        if total == 0:
            return 0.0
        raw = (2 * self.strong_buy + self.buy - self.sell - 2 * self.strong_sell)
        return max(-1.0, min(1.0, raw / (2 * total)))


def get_analyst_recs(ticker: str) -> list[AnalystRec]:
    """Latest-first list of analyst recommendation trends (monthly aggregates)."""
    data = _get("/stock/recommendation", {"symbol": ticker}, ttl_seconds=24 * 3600)
    if not data or not isinstance(data, list):
        return []

    recs = []
    for r in data:
        try:
            recs.append(AnalystRec(
                period=str(r.get("period", "")),
                buy=int(r.get("buy", 0)),
                hold=int(r.get("hold", 0)),
                sell=int(r.get("sell", 0)),
                strong_buy=int(r.get("strongBuy", 0)),
                strong_sell=int(r.get("strongSell", 0)),
            ))
        except Exception:
            continue
    recs.sort(key=lambda x: x.period, reverse=True)
    return recs


def get_quote(ticker: str) -> dict | None:
    """Real-time-ish stock quote.

    Returns {"price": float, "change": float, "change_pct": float, "high": float,
             "low": float, "open": float, "prev_close": float, "timestamp": int}
    or None on failure.
    """
    data = _get("/quote", {"symbol": ticker}, ttl_seconds=60)
    if not data:
        return None
    try:
        return {
            "ticker": ticker.upper(),
            "price": _safe_float(data.get("c")) or 0.0,
            "change": _safe_float(data.get("d")) or 0.0,
            "change_pct": (_safe_float(data.get("dp")) or 0.0) / 100.0,
            "high": _safe_float(data.get("h")) or 0.0,
            "low": _safe_float(data.get("l")) or 0.0,
            "open": _safe_float(data.get("o")) or 0.0,
            "prev_close": _safe_float(data.get("pc")) or 0.0,
            "timestamp": int(data.get("t", 0)),
        }
    except Exception:
        return None


def get_basic_financials(ticker: str) -> dict | None:
    """Fundamental metrics: P/E, market cap, 52w high/low, etc.

    Returns a dict of selected metrics or None.
    """
    data = _get("/stock/metric", {"symbol": ticker, "metric": "all"}, ttl_seconds=24 * 3600)
    if not data or not isinstance(data, dict):
        return None
    metrics = data.get("metric") or {}
    if not isinstance(metrics, dict):
        return None
    # Pick the most useful subset (Finnhub returns hundreds)
    return {
        "pe_ttm": _safe_float(metrics.get("peTTM")),
        "pb_ttm": _safe_float(metrics.get("pbTTM") or metrics.get("pbAnnual")),
        "ps_ttm": _safe_float(metrics.get("psTTM")),
        "ev_ebitda": _safe_float(metrics.get("currentEv/freeCashFlowTTM")),
        "beta": _safe_float(metrics.get("beta")),
        "market_cap": _safe_float(metrics.get("marketCapitalization")),
        "52w_high": _safe_float(metrics.get("52WeekHigh")),
        "52w_low": _safe_float(metrics.get("52WeekLow")),
        "dividend_yield": _safe_float(metrics.get("dividendYieldIndicatedAnnual")),
        "roe_ttm": _safe_float(metrics.get("roeTTM")),
        "debt_to_equity": _safe_float(metrics.get("totalDebt/totalEquityAnnual")),
        "current_ratio": _safe_float(metrics.get("currentRatioAnnual")),
    }


@dataclass
class DividendEvent:
    ticker: str
    ex_date: str            # ISO date "YYYY-MM-DD" — the critical one for CC risk
    pay_date: str           # ISO date
    amount: float           # Cash dividend per share
    currency: str


def get_dividend_calendar(ticker: str, days_ahead: int = 60) -> list[DividendEvent]:
    """Return upcoming ex-dividend events for a ticker within `days_ahead` days.

    This is the key input for CC risk: ITM calls get early-exercised the day
    before ex-date to capture the dividend, so CC expirations past an ex-date
    need extra care (or a hard skip for conservative operators).
    """
    today = datetime.now(timezone.utc).date()
    end = today + timedelta(days=days_ahead)
    data = _get(
        "/stock/dividend",
        {"symbol": ticker, "from": today.isoformat(), "to": end.isoformat()},
        ttl_seconds=6 * 3600,
    )
    if not data or not isinstance(data, list):
        return []

    events = []
    for d in data:
        try:
            ex_date = str(d.get("date", ""))  # Finnhub uses "date" for ex-date
            if not ex_date:
                continue
            events.append(DividendEvent(
                ticker=str(d.get("symbol", ticker)).upper(),
                ex_date=ex_date,
                pay_date=str(d.get("payDate", "")),
                amount=_safe_float(d.get("amount")) or 0.0,
                currency=str(d.get("currency", "USD")),
            ))
        except Exception:
            continue
    events.sort(key=lambda x: x.ex_date)
    return events


def next_ex_dividend_date(ticker: str, days_ahead: int = 60) -> str | None:
    """ISO date of next ex-dividend event, or None if unknown / none scheduled."""
    events = get_dividend_calendar(ticker, days_ahead=days_ahead)
    today_iso = datetime.now(timezone.utc).date().isoformat()
    for e in events:
        if e.ex_date >= today_iso:
            return e.ex_date
    return None


def get_insider_trades(ticker: str, days_back: int = 90) -> list[dict]:
    """Recent insider buy/sell transactions."""
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)
    data = _get(
        "/stock/insider-transactions",
        {"symbol": ticker, "from": start.isoformat(), "to": today.isoformat()},
        ttl_seconds=6 * 3600,
    )
    if not data or not isinstance(data, dict):
        return []
    trades = []
    for t in (data.get("data") or [])[:30]:
        try:
            trades.append({
                "name": str(t.get("name", ""))[:100],
                "shares": int(t.get("share", 0)),
                "change": int(t.get("change", 0)),  # + = buy, - = sell
                "filing_date": str(t.get("filingDate", "")),
                "transaction_date": str(t.get("transactionDate", "")),
                "price": _safe_float(t.get("transactionPrice")),
            })
        except Exception:
            continue
    return trades


# ── Helpers ──────────────────────────────────────────────────────

def _safe_float(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def finnhub_status() -> dict:
    """Diagnostic: is the key set and does a test call succeed?"""
    key_set = bool(_api_key())
    reachable = None
    if key_set:
        # Use a cheap endpoint to test connectivity
        q = _get("/quote", {"symbol": "SPY"}, ttl_seconds=0)
        reachable = q is not None
    return {
        "key_set": key_set,
        "reachable": reachable,
        "cache_dir": str(CACHE_DIR),
    }
