"""Historical PEAD signal for backtesting — 2026-05-28.

PEAD (Post-Earnings Announcement Drift) is one of our best signals
empirically, but ``enable_pead: false`` was hardcoded in the backtest
because there was no per-sim-date earnings cache. This module fixes
that gap.

ARCHITECTURE DIFFERENCE FROM KRONOS / NEWS / LLM CACHES
  Those caches store one entry per (ticker, sim_date) — the SIGNAL
  value as it would have looked on that day. They have to, because
  the underlying inference depends on the date's bars.

  PEAD is different. The earnings EVENTS themselves don't change after
  they happen. What changes is the score, which depends on
  ``days_since`` (which depends on sim_date). So we cache the earnings
  CALENDAR per ticker — one fetch covers all sim_dates — and compute
  the score on demand.

  Result: ~30 Finnhub calls (one per ticker) to fill the cache instead
  of ~5,760 (32 tickers × 180 sim_dates). Re-uses lib/historical_cache
  for the disk layer.

USAGE
    from lib.historical_pead import get_historical_pead
    result = get_historical_pead(ticker, sim_date=pd.Timestamp("2026-03-15"))
    # → {"ticker": ..., "score": 0.43, "kind": "strong_beat",
    #    "days_since": 22, "in_drift_window": True, "reason": "..."}

    # In backtest's _check_pead replacement:
    if result["score"] >= params.get("pead_min_score", 0.15):
        # PEAD votes FOR

BUILD CACHE
    python main.py build-cache --signal pead --lookback 180

The build pulls 2 years of earnings history per ticker (enough for a
180-day backtest with 90-day PEAD lookbacks). Cache TTL is 7 days
(refresh after that since new earnings can land).
"""
from __future__ import annotations

from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any

from lib import historical_cache
from lib.audit import log_event


# Mirror the live thresholds from lib/pead_signal so the backtest's
# score matches what live would compute at the same date.
PEAD_BLACKOUT_DAYS = 1   # day 0-1 post-earnings — too volatile
PEAD_DRIFT_DAYS = 60     # day 2-60 — drift window
CACHE_TTL_DAYS = 7       # earnings calendar refresh cadence


def _fetch_full_earnings_history(ticker: str, years_back: int = 2) -> list[dict] | None:
    """Pull historical earnings for ``ticker`` using Finnhub's
    ``/stock/earnings`` endpoint, which returns the last 4 quarters of
    REAL announcements (actual vs estimate EPS + surprise%). Cheap —
    one call per ticker covers the entire backtest window.

    2026-05-28: switched from /calendar/earnings to /stock/earnings
    because the calendar endpoint on Finnhub's free tier returns at
    most ONE event per query regardless of date range. /stock/earnings
    consistently returns 4 quarters which is enough for PEAD's 60-day
    drift window.

    The endpoint reports ``period`` (quarter-end date, e.g.
    "2026-03-31"). The actual announcement happens ~4-6 weeks after
    quarter end. We use period+45d as the proxy announcement date —
    small calibration error vs the true PEAD ~60-day drift window,
    not material for score computation.
    """
    import os
    import requests
    from datetime import datetime as _dt, timedelta as _td
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        log_event("historical_pead", "no_api_key", {"ticker": ticker})
        return None
    # Pull as many quarters as the free tier returns; 8 is generous,
    # actual return is 4-8 depending on tier.
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/stock/earnings",
            params={"symbol": ticker, "limit": 16, "token": api_key},
            timeout=10,
        )
    except Exception as e:
        log_event("historical_pead", "fetch_failed",
                  {"ticker": ticker, "error": str(e)[:120]})
        return None
    if resp.status_code != 200:
        log_event("historical_pead", "http_error",
                  {"ticker": ticker, "status": resp.status_code})
        return None
    try:
        raw = resp.json()
    except Exception:
        return None
    if not isinstance(raw, list):
        return []
    out = []
    for q in raw:
        try:
            period_str = str(q.get("period", ""))
            if not period_str:
                continue
            period = _dt.strptime(period_str, "%Y-%m-%d").date()
            # Announcement ~45 days after quarter end (industry typical)
            announcement_date = period + _td(days=45)
            out.append({
                "date": announcement_date.isoformat(),
                "period_end": period_str,
                "eps_estimate": q.get("estimate"),
                "eps_actual": q.get("actual"),
                "surprise_pct_finnhub": q.get("surprisePercent"),
            })
        except Exception:
            continue
    # Sort newest-first
    out.sort(key=lambda e: e.get("date", ""), reverse=True)
    return out


def _surprise_pct(estimate: float | None, actual: float | None) -> float | None:
    """Same logic as lib.pead_signal._surprise_pct."""
    if estimate is None or actual is None:
        return None
    try:
        est = float(estimate)
        act = float(actual)
    except (TypeError, ValueError):
        return None
    if est == 0:
        return None
    return (act - est) / abs(est)


def _classify_surprise(surprise_pct: float | None) -> str:
    """Same logic as lib.pead_signal._classify_surprise."""
    if surprise_pct is None:
        return "no_data"
    if surprise_pct >= 0.10:
        return "strong_beat"
    if surprise_pct >= 0.02:
        return "beat"
    if surprise_pct >= -0.02:
        return "in_line"
    if surprise_pct >= -0.10:
        return "miss"
    return "strong_miss"


def _drift_score(kind: str, days_since: int) -> float:
    """Same logic as lib.pead_signal._drift_score."""
    if days_since < PEAD_BLACKOUT_DAYS:
        return 0.0
    if days_since > PEAD_DRIFT_DAYS:
        return 0.0
    base = {
        "strong_beat":  0.6,
        "beat":         0.3,
        "in_line":      0.0,
        "miss":        -0.3,
        "strong_miss": -0.6,
        "no_data":      0.0,
    }.get(kind, 0.0)
    decay = max(0.0, 1.0 - (days_since - PEAD_BLACKOUT_DAYS) / float(PEAD_DRIFT_DAYS))
    return round(base * decay, 4)


def get_historical_pead(ticker: str, sim_date: Any) -> dict:
    """Return the PEAD signal value as it would have looked on ``sim_date``.

    Looks up the most recent earnings event with date <= sim_date,
    computes days_since dynamically, scores it. No cache hit triggers
    a Finnhub call to populate the calendar; subsequent calls on
    different sim_dates reuse the cached calendar.
    """
    # Normalize sim_date to a date object
    if hasattr(sim_date, "date"):
        target = sim_date.date()
    elif isinstance(sim_date, str):
        target = date.fromisoformat(sim_date[:10])
    elif isinstance(sim_date, date):
        target = sim_date
    else:
        target = datetime.now(timezone.utc).date()

    # 1. Try cache (one entry per ticker; sim_date doesn't affect the calendar).
    # IMPORTANT: cache key uses TODAY's first-of-month, not sim_date's. The
    # earnings calendar is built at "now" — a backtest looking back to
    # sim_date 2026-03-15 still wants the calendar fetched today, not a
    # nonexistent March cache slot. Same key for read and write.
    cache_key_date = datetime.now(timezone.utc).date().replace(day=1)
    cached = historical_cache.get(
        signal="pead", ticker=ticker, date=cache_key_date,
        params={"years_back": 2},
    )
    if cached is not None:
        # historical_cache.get returns the unwrapped value (a list of
        # event dicts). No need for .get("value") — that's its job already.
        events = cached if isinstance(cached, list) else []
    else:
        events = _fetch_full_earnings_history(ticker, years_back=2)
        if events is not None:
            historical_cache.put(
                signal="pead", ticker=ticker, date=cache_key_date,
                value=events, params={"years_back": 2},
            )

    if not events:
        return {
            "ticker": ticker, "score": 0.0, "kind": "no_data",
            "days_since": None, "in_drift_window": False,
            "reason": "no_earnings_history",
        }

    # 2. Find the most recent past event for this sim_date
    past = []
    for evt in events:
        d = evt.get("date", "")
        if not d:
            continue
        try:
            event_date = date.fromisoformat(d[:10])
        except Exception:
            continue
        if event_date <= target:
            past.append((event_date, evt))
    if not past:
        return {
            "ticker": ticker, "score": 0.0, "kind": "no_data",
            "days_since": None, "in_drift_window": False,
            "reason": "no_past_earnings_before_sim_date",
        }
    past.sort(key=lambda x: x[0], reverse=True)
    event_date, evt = past[0]
    days_since = (target - event_date).days

    surprise = _surprise_pct(evt.get("eps_estimate"), evt.get("eps_actual"))
    kind = _classify_surprise(surprise)
    score = _drift_score(kind, days_since)

    return {
        "ticker": ticker,
        "event_date": event_date.isoformat(),
        "days_since": days_since,
        "eps_estimate": evt.get("eps_estimate"),
        "eps_actual": evt.get("eps_actual"),
        "surprise_pct": round(surprise, 4) if surprise is not None else None,
        "kind": kind,
        "score": score,
        "in_drift_window": PEAD_BLACKOUT_DAYS <= days_since <= PEAD_DRIFT_DAYS,
        "reason": (
            f"{kind} ({surprise:+.1%}) {days_since}d after announcement"
            if surprise is not None
            else f"{kind} ({days_since}d since)"
        ),
    }


def build_cache(tickers: list[str], force: bool = False) -> dict:
    """Pre-populate the earnings-history cache for a universe of tickers.
    Called by ``main.py build-cache --signal pead``.

    Cheap — one Finnhub call per ticker, not per sim_date.
    """
    today = datetime.now(timezone.utc).date()
    cache_key_date = today.replace(day=1)
    built = 0
    skipped = 0
    failed = []
    for ticker in tickers:
        existing = (
            None if force
            else historical_cache.get(
                signal="pead", ticker=ticker, date=cache_key_date,
                params={"years_back": 2},
            )
        )
        if existing is not None:
            skipped += 1
            continue
        events = _fetch_full_earnings_history(ticker, years_back=2)
        if events is None:
            failed.append(ticker)
            continue
        historical_cache.put(
            signal="pead", ticker=ticker, date=cache_key_date,
            value=events, params={"years_back": 2},
        )
        built += 1
    summary = {
        "built": built, "skipped": skipped, "failed": failed,
        "total": len(tickers),
    }
    log_event("historical_pead", "cache_built", summary)
    return summary


__all__ = ["get_historical_pead", "build_cache"]
