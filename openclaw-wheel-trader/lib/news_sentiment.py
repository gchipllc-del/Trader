"""
News Sentiment — lightweight stock news checker for trade entry decisions.

Before buying a stock, check if recent news is strongly negative.
Prevents walking into bad earnings, downgrades, lawsuits, etc.

Adapted from polybot's news_feed.py — simplified for stock-specific use.

Sources (in priority order):
    1. NewsAPI — broad headline search (requires NEWSAPI_KEY env var)
    2. RSS feeds — curated financial feeds (no API key needed)

Security:
    - API keys loaded ONLY from environment variables
    - All external responses treated as untrusted input
    - Input sanitized before outbound queries
    - Results cached with TTL (30 min default)
    - No secrets in any log or error message
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

from lib.audit import log_event

CACHE_DIR = Path(__file__).parent.parent / "data" / "news_cache"


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class StockNewsResult:
    """Aggregated news sentiment for a stock ticker."""
    ticker: str
    sentiment: float = 0.5        # 0.0 = very bearish, 1.0 = very bullish
    confidence: float = 0.0       # 0.0 = no data, 1.0 = strong signal
    article_count: int = 0
    headlines: list[str] = field(default_factory=list)
    signal: str = "neutral"       # "bullish", "bearish", "neutral"
    cached: bool = False


# ── Bearish/Bullish Keyword Detection ────────────────────────────

BEARISH_KEYWORDS = {
    "downgrade", "lawsuit", "sec investigation", "fda reject", "recall",
    "bankruptcy", "default", "fraud", "layoff", "layoffs", "cut jobs",
    "miss estimate", "missed estimates", "revenue miss", "earnings miss",
    "profit warning", "guidance cut", "guidance lower", "sell rating",
    "bear case", "short seller", "plunge", "plummet", "crash", "tank",
    "tumble", "dive", "slump", "decline", "drop", "fall", "lose",
    "loss", "weak", "concern", "risk", "warning", "trouble", "problem",
    "crisis", "scandal", "probe", "investigate", "fine", "penalty",
    "suspend", "halt", "delisted", "overvalued",
}

BULLISH_KEYWORDS = {
    "upgrade", "buy rating", "strong buy", "outperform", "overweight",
    "beat estimate", "beat estimates", "earnings beat", "revenue beat",
    "raise guidance", "guidance raise", "guidance higher", "fda approv",
    "partnership", "contract win", "record revenue", "record profit",
    "all-time high", "breakout", "rally", "surge", "soar", "jump",
    "gain", "rise", "climb", "strong", "growth", "expand", "launch",
    "innovation", "breakthrough", "bullish", "momentum", "recovery",
    "rebound", "undervalued", "dividend increase", "buyback",
    "stock split", "analyst positive",
}


def _score_headline(title: str) -> float:
    """
    Score a headline's sentiment. Returns 0.0-1.0.

    Counts bearish vs bullish keyword matches, weighted by match count.
    """
    title_lower = title.lower()

    bearish_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in title_lower)
    bullish_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in title_lower)

    total = bearish_hits + bullish_hits
    if total == 0:
        return 0.5  # Neutral — no signal

    # Score: more bullish hits → higher score
    return bullish_hits / total


# ── Rate Limiter ─────────────────────────────────────────────────

_last_call: dict[str, float] = {}


def _rate_limit(source: str, min_interval: float = 2.0):
    """Simple per-source rate limiter."""
    now = time.time()
    last = _last_call.get(source, 0.0)
    elapsed = now - last
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call[source] = time.time()


# ── Cache ────────────────────────────────────────────────────────

def _cache_key(ticker: str) -> str:
    # 2026-04-28: Removed hour bucket from key. Previously the key included
    # `%Y-%m-%d-%H` so any cache lookup that crossed an hour boundary would
    # miss regardless of TTL — causing redundant NewsAPI fetches for the
    # same 4 tickers within 11 minutes (audit-confirmed). With a stable
    # per-ticker key, the TTL parameter (default 30 min) actually governs
    # freshness as intended.
    raw = f"stock_news:{ticker}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_get(key: str, ttl_minutes: int = 30) -> dict | None:
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


# ── NewsAPI Source ───────────────────────────────────────────────

def _fetch_newsapi(ticker: str, max_articles: int = 8) -> list[dict]:
    """
    Fetch stock news from NewsAPI.org.
    Returns list of {"title": str, "sentiment": float}.
    """
    import requests

    api_key = os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWS_API_KEY")
    if not api_key:
        return []

    _rate_limit("newsapi")

    from_date = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")

    # Search for ticker + company name patterns
    query = f'"{ticker}" stock OR shares OR earnings'

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query[:500],
        "from": from_date,
        "sortBy": "relevancy",
        "pageSize": min(max_articles, 20),
        "language": "en",
    }
    headers = {"X-Api-Key": api_key}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log_event("news_sentiment", "newsapi_failed", {
            "ticker": ticker,
            "error": str(e)[:200],
        }, result="failed")
        return []

    articles = []
    for item in data.get("articles", [])[:max_articles]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", ""))[:500]
        if not title or title == "[Removed]":
            continue
        # Must mention the ticker or be obviously related
        if ticker.lower() not in title.lower():
            continue
        articles.append({
            "title": title,
            "sentiment": _score_headline(title),
        })

    return articles


# ── RSS Source (free, no API key) ────────────────────────────────

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
]


def _fetch_rss(ticker: str, max_articles: int = 5) -> list[dict]:
    """Fetch stock news from RSS feeds. No API key needed."""
    try:
        import xml.etree.ElementTree as ET
        import requests
    except ImportError:
        return []

    articles = []

    for feed_url in RSS_FEEDS:
        url = feed_url.format(ticker=ticker)
        _rate_limit("rss", min_interval=1.0)

        try:
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "OpenClaw-Trader/1.0"
            })
            if resp.status_code != 200:
                continue

            root = ET.fromstring(resp.text[:50000])  # Limit parse size
            items = root.findall(".//item")[:max_articles]

            for item in items:
                title_el = item.find("title")
                if title_el is None or not title_el.text:
                    continue
                title = str(title_el.text)[:500]
                articles.append({
                    "title": title,
                    "sentiment": _score_headline(title),
                })

        except Exception:
            continue

    return articles


# ── Main API ─────────────────────────────────────────────────────

def check_stock_sentiment(ticker: str, cache_ttl: int = 30) -> StockNewsResult:
    """
    Check recent news sentiment for a stock ticker.

    Returns a StockNewsResult with:
        - sentiment: 0.0 (very bearish) to 1.0 (very bullish)
        - confidence: 0.0 (no data) to 1.0 (many articles, strong signal)
        - signal: "bullish", "bearish", or "neutral"

    Used by stock_engine as a pre-trade filter:
        - sentiment < 0.3 → skip trade (strongly bearish news)
        - sentiment > 0.7 → boost confidence (bullish news)
        - sentiment 0.3-0.7 → no effect (neutral/mixed)
    """
    # Check cache
    key = _cache_key(ticker)
    cached = _cache_get(key, ttl_minutes=cache_ttl)
    if cached:
        return StockNewsResult(
            ticker=cached["ticker"],
            sentiment=cached["sentiment"],
            confidence=cached["confidence"],
            article_count=cached["article_count"],
            headlines=cached.get("headlines", []),
            signal=cached["signal"],
            cached=True,
        )

    # Fetch from all sources
    all_articles = []
    all_articles.extend(_fetch_newsapi(ticker))
    all_articles.extend(_fetch_rss(ticker))

    if not all_articles:
        result = StockNewsResult(ticker=ticker)
        log_event("news_sentiment", "no_articles", {"ticker": ticker})
        return result

    # Aggregate sentiment
    sentiments = [a["sentiment"] for a in all_articles]
    avg_sentiment = sum(sentiments) / len(sentiments)

    # Confidence based on article count and agreement
    # More articles + more agreement = higher confidence
    count_factor = min(len(all_articles) / 5.0, 1.0)  # Max out at 5 articles
    agreement = 1.0 - (sum(abs(s - avg_sentiment) for s in sentiments) / len(sentiments))
    confidence = count_factor * agreement

    # Signal classification
    if avg_sentiment < 0.35:
        signal = "bearish"
    elif avg_sentiment > 0.65:
        signal = "bullish"
    else:
        signal = "neutral"

    headlines = [a["title"] for a in all_articles[:5]]

    result = StockNewsResult(
        ticker=ticker,
        sentiment=round(avg_sentiment, 4),
        confidence=round(confidence, 4),
        article_count=len(all_articles),
        headlines=headlines,
        signal=signal,
    )

    # Cache
    _cache_put(key, {
        "ticker": result.ticker,
        "sentiment": result.sentiment,
        "confidence": result.confidence,
        "article_count": result.article_count,
        "headlines": result.headlines,
        "signal": result.signal,
    })

    log_event("news_sentiment", "checked", {
        "ticker": ticker,
        "sentiment": result.sentiment,
        "signal": result.signal,
        "articles": result.article_count,
    })

    return result


def check_batch_sentiment(tickers: list[str]) -> dict[str, StockNewsResult]:
    """
    Check news sentiment for multiple tickers.
    Returns dict of ticker → StockNewsResult.
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = check_stock_sentiment(ticker)
        except Exception as e:
            log_event("news_sentiment", "batch_error", {
                "ticker": ticker,
                "error": str(e)[:200],
            })
            results[ticker] = StockNewsResult(ticker=ticker)
    return results
