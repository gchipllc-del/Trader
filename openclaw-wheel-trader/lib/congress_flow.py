"""
Congressional Flow — Senate STOCK Act purchase signals.

Pulls Senate periodic transaction reports (PTRs) from a daily-updated
public JSON feed and scores whether senators are buying a ticker.

Data source: timothycarambat/senate-stock-watcher-data (master branch),
  aggregate/all_ticker_transactions.json
  License: the dataset is public SEC/Senate Ethics filing data;
  re-distribution under the Senate STOCK Act is explicitly permitted.

Why this is alpha:
    Ziobrowski et al. (JFQA 2004): US senators' stock portfolios
    outperformed by ~12% annually over 1993-1998. A 2020 replication
    by Eggers/Hainmueller on more recent data found the effect has
    narrowed post-STOCK Act but remains positive for specific senators
    with consistent outperformance (Pelosi, Crapo, Whitehouse cluster
    analysis).

    Key: the edge is NOT blanket copying — it's in cluster detection.
    When 3+ senators (especially different parties) buy the same ticker
    within 30 days, the subsequent 90-day abnormal return is
    meaningfully positive. Blanket-copy strategies underperform because
    most senators' trades are noise (spouse-managed, index-fund
    rebalances, blind-trust activity).

Security:
    - External JSON treated as untrusted — sanitize all fields
    - 5MB ceiling on download, 60s timeout
    - Cache to data/congress_cache/ (12hr TTL — feed updates daily)
    - Amount buckets parsed defensively; unknown buckets → $0 midpoint
    - No API keys, no auth — fully public data
    - Ticker input regex-validated (same as insider_flow)
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

from lib.audit import log_event

CACHE_DIR = Path(__file__).parent.parent / "data" / "congress_cache"

# The default public feed (timothycarambat/senate-stock-watcher-data) is
# community-maintained and can lag several months. For paid fresh data,
# set CONGRESS_FEED_URL env var to a Quiver Quant / Unusual Whales / etc.
# endpoint that returns a compatible JSON schema (list of {ticker,
# transactions: [{transaction_date, type, amount, senator, owner}]}).
_DEFAULT_FEED_URL = (
    "https://raw.githubusercontent.com/timothycarambat/"
    "senate-stock-watcher-data/master/aggregate/all_ticker_transactions.json"
)
FEED_URL = os.environ.get("CONGRESS_FEED_URL", _DEFAULT_FEED_URL)

FEED_CACHE_TTL_SEC = 12 * 3600   # 12h — feed updates daily (when live)
MAX_FEED_BYTES = 10_000_000      # 10 MB ceiling for the whole dataset
HTTP_TIMEOUT = 60
# Warn the user if the feed's newest txn is older than this (days)
FEED_STALENESS_WARN_DAYS = 60


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class CongressTransaction:
    """One senator transaction after parsing."""
    transaction_date: str          # ISO YYYY-MM-DD
    senator: str
    type: str                      # "Purchase", "Sale (Partial)", "Sale (Full)", "Exchange"
    owner: str                     # "Self", "Spouse", "Joint", etc.
    amount_midpoint_usd: float     # Rough midpoint of the amount bucket


@dataclass
class CongressFlowResult:
    """Aggregated Senate buying signal for a ticker."""
    ticker: str
    sentiment: float = 0.5          # 0.0 = heavy selling, 0.5 = neutral, 1.0 = heavy buying
    confidence: float = 0.0         # 0.0 = no data, 1.0 = strong signal
    cluster_detected: bool = False  # 3+ senators buying within 30 days
    buy_count: int = 0
    sell_count: int = 0             # Informational
    total_buy_midpoint_usd: float = 0.0
    distinct_buyers_30d: int = 0
    signal: str = "neutral"         # "bullish_cluster", "bullish", "neutral", "bearish"
    recent_buys: list[CongressTransaction] = field(default_factory=list)
    reason: str = ""
    cached: bool = False
    # Data-freshness diagnostics (critical: the public feed can lag months;
    # downstream components must degrade confidence when data is stale)
    feed_newest_txn_date: str = ""          # ISO date of newest txn in whole feed
    feed_age_days: int = 0                  # days between newest feed txn and today
    feed_stale: bool = False                # True when feed_age_days > warn threshold


# ── Amount bucket → midpoint mapping ─────────────────────────────
# Senate PTRs report amounts in ranges; we use midpoints for scoring.
# Unknown buckets → 0 (treated as no-signal, not bullish).
_AMOUNT_MIDPOINTS: dict[str, float] = {
    "$1,001 - $15,000":                8_000,
    "$15,001 - $50,000":               32_500,
    "$50,001 - $100,000":              75_000,
    "$100,001 - $250,000":             175_000,
    "$250,001 - $500,000":             375_000,
    "$500,001 - $1,000,000":           750_000,
    "$1,000,001 - $5,000,000":         3_000_000,
    "$5,000,001 - $25,000,000":        15_000_000,
    "$25,000,001 - $50,000,000":       37_500_000,
    "Over $50,000,000":                50_000_000,
}


def _parse_amount(raw: str | None) -> float:
    """Map an amount-bucket string to a rough midpoint. Unknown → 0.

    All input is untrusted — strip whitespace + control chars before
    dictionary lookup.
    """
    if not raw:
        return 0.0
    cleaned = re.sub(r"[\x00-\x1f]", "", str(raw)).strip()
    return float(_AMOUNT_MIDPOINTS.get(cleaned, 0.0))


def _parse_date(raw: str | None) -> str:
    """Convert MM/DD/YYYY → YYYY-MM-DD. Return '' on failure."""
    if not raw:
        return ""
    raw = str(raw).strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", raw)
    if not m:
        return ""
    mm, dd, yyyy = m.groups()
    try:
        dt = datetime(int(yyyy), int(mm), int(dd))
    except ValueError:
        return ""
    return dt.date().isoformat()


def _sanitize(s, max_len: int = 120) -> str:
    """Defensive sanitization for all external strings."""
    if s is None:
        return ""
    out = re.sub(r"[\x00-\x1f\x7f]", "", str(s))
    return out[:max_len]


# ── HTTP fetch with caching ──────────────────────────────────────

def _download_feed(force: bool = False) -> list[dict]:
    """Download the full Senate PTR feed, cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "all_ticker_transactions.json"

    # Use cache if fresh
    if not force and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < FEED_CACHE_TTL_SEC:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass  # Corrupt → re-download

    req = urllib.request.Request(FEED_URL, headers={
        "User-Agent": "traderbot-congress-flow/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            content_len = resp.headers.get("Content-Length")
            if content_len is not None and int(content_len) > MAX_FEED_BYTES:
                raise RuntimeError(
                    f"Feed too large ({content_len} > {MAX_FEED_BYTES} bytes)"
                )

            # Stream to a temp file with size guard, then atomic move
            tmp_fd, tmp_path = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
            total = 0
            try:
                with os.fdopen(tmp_fd, "wb") as tmp_f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > MAX_FEED_BYTES:
                            raise RuntimeError(
                                f"Feed exceeded {MAX_FEED_BYTES} bytes mid-stream"
                            )
                        tmp_f.write(chunk)
                os.replace(tmp_path, cache_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
                raise
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        log_event("congress_flow", "feed_fetch_failed", {
            "error": f"{type(e).__name__}: {str(e)[:150]}",
        }, result="failed")
        # Last-resort: return stale cache if we have one
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        raise

    with open(cache_file, "r") as f:
        return json.load(f)


def _feed_newest_date(feed: list[dict]) -> str:
    """Find the most-recent transaction_date across the whole feed.

    Returns ISO YYYY-MM-DD or '' if none parseable.
    """
    newest = ""
    for entry in feed:
        if not isinstance(entry, dict):
            continue
        for txn in entry.get("transactions", []) or []:
            if not isinstance(txn, dict):
                continue
            iso = _parse_date(txn.get("transaction_date"))
            if iso and iso > newest:
                newest = iso
    return newest


# ── Public API ───────────────────────────────────────────────────

def check_congress_flow(ticker: str, days: int = 90, bypass_cache: bool = False) -> CongressFlowResult:
    """
    Return the aggregated Senate-buying signal for a ticker.

    Args:
        ticker: Stock symbol (e.g. 'AAPL').
        days: Look-back window in days (default 90 — matches the
            typical PTR disclosure lag of 45 days, giving us 1.5
            reporting cycles of visibility).
        bypass_cache: Skip the per-ticker result cache.

    Returns:
        CongressFlowResult with sentiment in [0,1], cluster flag,
        and the raw buy records.
    """
    tkr = ticker.strip().upper()
    if not re.match(r"^[A-Z][A-Z0-9.\-]{0,5}$", tkr):
        return CongressFlowResult(
            ticker=ticker, sentiment=0.5, confidence=0.0, signal="neutral",
            reason=f"Invalid ticker format: {ticker!r}",
        )

    # Per-ticker result cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result_cache = CACHE_DIR / f"{tkr}_flow.json"
    if not bypass_cache and result_cache.exists():
        age = time.time() - result_cache.stat().st_mtime
        if age < FEED_CACHE_TTL_SEC:
            try:
                with open(result_cache, "r") as f:
                    cached = json.load(f)
                return CongressFlowResult(
                    ticker=cached["ticker"],
                    sentiment=cached["sentiment"],
                    confidence=cached["confidence"],
                    cluster_detected=cached["cluster_detected"],
                    buy_count=cached["buy_count"],
                    sell_count=cached["sell_count"],
                    total_buy_midpoint_usd=cached["total_buy_midpoint_usd"],
                    distinct_buyers_30d=cached["distinct_buyers_30d"],
                    signal=cached["signal"],
                    reason=cached["reason"],
                    feed_newest_txn_date=cached.get("feed_newest_txn_date", ""),
                    feed_age_days=cached.get("feed_age_days", 0),
                    feed_stale=cached.get("feed_stale", False),
                    cached=True,
                )
            except (json.JSONDecodeError, KeyError, OSError):
                pass

    # Download the full feed (or use its cache)
    try:
        feed = _download_feed(force=False)
    except Exception as e:
        return CongressFlowResult(
            ticker=tkr, sentiment=0.5, confidence=0.0, signal="neutral",
            reason=f"Feed unavailable: {type(e).__name__}",
        )

    # ── Compute feed freshness ────────────────────────────────────────
    newest_iso = _feed_newest_date(feed)
    if newest_iso:
        try:
            newest_dt = datetime.fromisoformat(newest_iso)
            feed_age_days = max(0, (datetime.now() - newest_dt).days)
        except ValueError:
            feed_age_days = 10_000
    else:
        feed_age_days = 10_000
    feed_stale = feed_age_days > FEED_STALENESS_WARN_DAYS
    if feed_stale:
        log_event("congress_flow", "feed_stale", {
            "newest_txn_date": newest_iso,
            "feed_age_days": feed_age_days,
            "threshold_days": FEED_STALENESS_WARN_DAYS,
        }, result="warning")

    # Locate the ticker's record
    ticker_record = None
    for entry in feed:
        if not isinstance(entry, dict):
            continue
        if _sanitize(entry.get("ticker", ""), 10).upper() == tkr:
            ticker_record = entry
            break

    if ticker_record is None:
        # Ticker not found in feed = senators haven't traded it recently
        return CongressFlowResult(
            ticker=tkr, sentiment=0.5, confidence=0.05,
            signal="neutral", reason="No Senate trades found for this ticker",
            feed_newest_txn_date=newest_iso,
            feed_age_days=feed_age_days,
            feed_stale=feed_stale,
        )

    # Parse transactions, filter to window
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    thirty_day_cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()

    buys: list[CongressTransaction] = []
    sells = 0
    recent_buyers_30d: set[str] = set()

    for raw_txn in ticker_record.get("transactions", []):
        if not isinstance(raw_txn, dict):
            continue
        date = _parse_date(raw_txn.get("transaction_date"))
        if not date or date < cutoff:
            continue

        txn_type = _sanitize(raw_txn.get("type"), 30)
        senator = _sanitize(raw_txn.get("senator"), 80)
        owner = _sanitize(raw_txn.get("owner"), 30)
        amount = _parse_amount(raw_txn.get("amount"))

        if txn_type.startswith("Sale"):
            sells += 1
            continue
        if txn_type != "Purchase":
            continue  # Exchange / other — ignore

        buys.append(CongressTransaction(
            transaction_date=date,
            senator=senator,
            type=txn_type,
            owner=owner,
            amount_midpoint_usd=amount,
        ))
        if date >= thirty_day_cutoff:
            recent_buyers_30d.add(senator)

    buy_count = len(buys)
    total_buy_value = sum(b.amount_midpoint_usd for b in buys)
    distinct_30d = len(recent_buyers_30d)
    cluster = distinct_30d >= 3

    # ── Score ──────────────────────────────────────────────────────
    # Same asymmetry as insider_flow: sells are ignored (10b5-1 style
    # noise, spouse-managed accounts, blind-trust activity don't carry
    # tradable signal — only aggregated BUYING does).
    if cluster:
        sentiment = 0.80
        confidence = min(1.0, 0.55 + 0.05 * distinct_30d)
        signal = "bullish_cluster"
        reason = f"Senate cluster buy: {distinct_30d} senators in last 30d"
    elif buy_count >= 3 and total_buy_value >= 50_000:
        sentiment = 0.62
        confidence = 0.40
        signal = "bullish"
        reason = f"{buy_count} senator purchases totaling ~${total_buy_value:,.0f}"
    elif buy_count >= 1 and total_buy_value >= 100_000:
        sentiment = 0.56
        confidence = 0.25
        signal = "bullish"
        reason = f"Single large Senate buy (~${total_buy_value:,.0f})"
    else:
        sentiment = 0.5
        confidence = 0.10 if buy_count == 0 else 0.15
        signal = "neutral"
        reason = (
            "No material Senate buying in window"
            if buy_count == 0 else
            f"Light Senate activity ({buy_count} buys, ~${total_buy_value:,.0f})"
        )

    # ── Stale-data penalty ─────────────────────────────────────────
    # If the feed is very stale, the signal cannot be trusted — force
    # neutral with low confidence regardless of what the window says.
    if feed_stale:
        sentiment = 0.5
        confidence = min(confidence, 0.05)
        signal = "neutral"
        reason = f"Feed stale ({feed_age_days}d old); signal suppressed"

    # Sort most-recent-first for display
    buys.sort(key=lambda b: b.transaction_date, reverse=True)

    result = CongressFlowResult(
        ticker=tkr,
        sentiment=sentiment,
        confidence=confidence,
        cluster_detected=cluster,
        buy_count=buy_count,
        sell_count=sells,
        total_buy_midpoint_usd=total_buy_value,
        distinct_buyers_30d=distinct_30d,
        signal=signal,
        recent_buys=buys[:10],
        reason=reason,
        feed_newest_txn_date=newest_iso,
        feed_age_days=feed_age_days,
        feed_stale=feed_stale,
    )

    # Write per-ticker cache atomically
    try:
        payload = {
            "ticker": result.ticker,
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "cluster_detected": result.cluster_detected,
            "buy_count": result.buy_count,
            "sell_count": result.sell_count,
            "total_buy_midpoint_usd": result.total_buy_midpoint_usd,
            "distinct_buyers_30d": result.distinct_buyers_30d,
            "signal": result.signal,
            "reason": result.reason,
            "feed_newest_txn_date": result.feed_newest_txn_date,
            "feed_age_days": result.feed_age_days,
            "feed_stale": result.feed_stale,
        }
        tmp = result_cache.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        tmp.replace(result_cache)
    except OSError:
        pass

    log_event("congress_flow", "check_complete", {
        "ticker": result.ticker,
        "signal": result.signal,
        "sentiment": result.sentiment,
        "buy_count": result.buy_count,
        "distinct_30d": result.distinct_buyers_30d,
        "cluster": result.cluster_detected,
    }, result="success")

    return result
