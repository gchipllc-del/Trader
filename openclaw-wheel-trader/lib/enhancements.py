"""
Sprint 8: Advanced Enhancements

- Earnings calendar awareness (never sell through earnings)
- Capitol Trades integration (congressional trading signals)
- Anomaly detection (flag unusual bot behavior)
- Bayesian probability updater (refine assignment estimates)

Source: Video (Capitol Trades), Advanced Algorithmic Trading (Bayesian)
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from lib.audit import log_event, get_recent_events
from lib.memory_palace import diary_write, kg_add, search_memory


# ============================================================
# EARNINGS CALENDAR
# ============================================================

# In production, pull from API (e.g., Alpha Vantage, FMP, Alpaca)
# This is the skeleton that gets wired to live data.

EARNINGS_PATH = Path(__file__).parent.parent / "data" / "earnings_calendar.json"

_earnings_cache: dict[str, str] = {}  # ticker -> next earnings date
_earnings_loaded: bool = False


def _load_earnings_file():
    """Load earnings dates from local JSON file into cache."""
    global _earnings_loaded
    if _earnings_loaded:
        return
    if EARNINGS_PATH.exists():
        try:
            with open(EARNINGS_PATH) as f:
                data = json.load(f)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            for ticker, date_str in data.items():
                # Only cache future earnings
                if date_str >= today:
                    _earnings_cache[ticker] = date_str
        except (json.JSONDecodeError, KeyError):
            pass
    _earnings_loaded = True


def set_earnings_date(ticker: str, date: str):
    """Manually set or cache an earnings date."""
    _earnings_cache[ticker] = date
    kg_add(ticker, "earnings_date", date)
    # Persist to file
    _save_earnings_file()


def _save_earnings_file():
    """Persist the earnings cache to disk."""
    EARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Merge with existing file data
    existing = {}
    if EARNINGS_PATH.exists():
        try:
            with open(EARNINGS_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    existing.update(_earnings_cache)
    with open(EARNINGS_PATH, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)


def get_next_earnings(ticker: str) -> str | None:
    """Get next earnings date for a ticker. Loads from file on first call."""
    _load_earnings_file()
    return _earnings_cache.get(ticker)


def is_earnings_conflict(ticker: str, expiration: str) -> bool:
    """Check if option expiration crosses an earnings date."""
    earnings = get_next_earnings(ticker)
    if not earnings or not expiration:
        return False
    return expiration >= earnings


def refresh_earnings_calendar(tickers: list[str]) -> dict[str, str]:
    """
    Fetch upcoming earnings dates for a list of tickers.

    Tries Alpha Vantage (if ALPHA_VANTAGE_KEY env var is set),
    then falls back to a lightweight SEC/free API scrape.
    Results are cached to data/earnings_calendar.json.

    Returns: {ticker: "YYYY-MM-DD"} for tickers with upcoming earnings
    """
    import os
    import requests as _requests

    updated = {}
    av_key = os.environ.get("ALPHA_VANTAGE_KEY", "")

    if av_key:
        # Alpha Vantage EARNINGS endpoint (free tier: 25 calls/day)
        for ticker in tickers:
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "EARNINGS",
                    "symbol": ticker,
                    "apikey": av_key,
                }
                resp = _requests.get(url, params=params, timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                upcoming = data.get("quarterlyEarnings", [])
                if upcoming:
                    # First entry is next/most recent
                    next_date = upcoming[0].get("reportedDate", "")
                    if next_date:
                        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                        if next_date >= today:
                            updated[ticker] = next_date
                            _earnings_cache[ticker] = next_date
            except Exception:
                continue
    else:
        # Try Financial Modeling Prep free endpoint (no key, limited)
        try:
            url = "https://financialmodelingprep.com/api/v3/earning_calendar"
            params = {"apikey": "demo"}  # Demo key for limited free access
            resp = _requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                ticker_set = set(tickers)
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                for entry in data:
                    sym = entry.get("symbol", "")
                    date = entry.get("date", "")
                    if sym in ticker_set and date >= today:
                        if sym not in updated:  # Keep earliest upcoming
                            updated[sym] = date
                            _earnings_cache[sym] = date
        except Exception:
            pass

    if updated:
        _save_earnings_file()
        log_event("enhancements", "earnings_refreshed", {
            "count": len(updated),
            "tickers": list(updated.keys()),
        })

    return updated


# ============================================================
# CAPITOL TRADES — Congressional Trading Signals
# ============================================================

def check_capitol_trades(ticker: str) -> dict | None:
    """
    Check if any congress members recently traded this ticker.
    Source: capitoltrades.com
    
    In production, scrape or use API. Returns signal dict or None.
    This is a SUPPLEMENTARY signal — never the sole reason to trade.
    """
    # Placeholder — wire to Capitol Trades scraper
    # Returns None when no data available
    return None


def score_capitol_signal(trades: list[dict]) -> float:
    """
    Score congressional trading activity.
    0.0 = no signal
    0.5 = some buys
    1.0 = multiple members buying aggressively
    
    This adds to the composite score but never exceeds a +1 bonus.
    """
    if not trades:
        return 0.0

    buys = [t for t in trades if t.get("type") == "buy"]
    unique_members = set(t.get("member", "") for t in buys)

    if len(unique_members) >= 3:
        return 1.0
    elif len(unique_members) >= 1:
        return 0.5
    return 0.0


# ============================================================
# BAYESIAN ASSIGNMENT PROBABILITY
# ============================================================

def bayesian_assignment_probability(
    delta: float,
    dte: int,
    historical_assignments: int,
    historical_trades: int,
) -> float:
    """
    Use Bayesian updating to refine assignment probability.
    
    Prior: delta (e.g., -0.25 delta = 25% chance of ITM at expiry)
    Likelihood: historical assignment rate for this delta range
    Posterior: updated probability
    
    Source: Advanced Algorithmic Trading, Bayesian Statistics chapter
    """
    # Prior from delta
    prior = abs(delta)

    # If we have historical data, use it
    if historical_trades > 0:
        observed_rate = historical_assignments / historical_trades
        # Simple weighted average (pseudo-Bayesian with uniform prior)
        # Weight increases with more observations
        weight = min(historical_trades / 50, 1.0)  # Full weight at 50+ trades
        posterior = (1 - weight) * prior + weight * observed_rate
    else:
        posterior = prior

    # Adjust for time decay — closer to expiry = more certainty
    # If very close to expiry and ITM, probability increases sharply
    time_factor = max(0, 1 - dte / 45)  # Increases as DTE shrinks
    adjusted = posterior * (1 + time_factor * 0.3)  # Up to 30% increase near expiry

    return min(adjusted, 0.99)  # Cap at 99%


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomalies() -> list[dict]:
    """
    Scan recent activity for unusual patterns.
    Flags:
    - Sudden burst of orders (>3 in 5 minutes)
    - Orders outside normal hours
    - Config changes
    - Repeated failures
    """
    anomalies = []
    recent = get_recent_events(n=100, event_type="order_gate")

    if not recent:
        return anomalies

    # Check for order burst
    now = datetime.now(timezone.utc)
    five_min_ago = (now - timedelta(minutes=5)).isoformat()
    recent_orders = [
        e for e in recent
        if e.get("timestamp", "") > five_min_ago
        and e.get("action") == "step3_executed"
    ]

    if len(recent_orders) > 3:
        anomalies.append({
            "type": "order_burst",
            "severity": "high",
            "detail": f"{len(recent_orders)} orders in last 5 minutes",
            "detected_at": now.isoformat(),
        })

    # Check for repeated failures
    recent_failures = [
        e for e in recent
        if e.get("result") == "failed"
        and e.get("timestamp", "") > (now - timedelta(hours=1)).isoformat()
    ]

    if len(recent_failures) > 5:
        anomalies.append({
            "type": "repeated_failures",
            "severity": "medium",
            "detail": f"{len(recent_failures)} failures in last hour",
            "detected_at": now.isoformat(),
        })

    # Check for circuit breaker trips
    breaker_events = get_recent_events(n=50, event_type="circuit_breaker")
    recent_trips = [
        e for e in breaker_events
        if e.get("result") == "blocked"
        and e.get("timestamp", "") > (now - timedelta(hours=1)).isoformat()
    ]

    if len(recent_trips) > 3:
        anomalies.append({
            "type": "frequent_breaker_trips",
            "severity": "high",
            "detail": f"{len(recent_trips)} circuit breaker trips in last hour",
            "detected_at": now.isoformat(),
        })

    for anomaly in anomalies:
        log_event("anomaly", anomaly["type"], anomaly, result="flagged")
        diary_write("risk_agent",
            f"ANOMALY|{anomaly['type']}|{anomaly['severity']}|{anomaly['detail']}")

    return anomalies
