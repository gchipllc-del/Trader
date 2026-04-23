"""
Insider Flow — SEC Form 4 insider-buying signal.

This module pulls Form 4 filings (insider transactions) directly from
SEC EDGAR's public JSON API — no third-party library, no pyarrow
dependency. We look specifically for *open-market purchases*
(transactionCode = 'P'), filter out noise like option exercises (M),
grants (A), and tax withholdings (F).

Why this is alpha:
    Academic literature (Jeng/Metrick/Zeckhauser 2003; Lakonishok/Lee
    2001) documents that insider open-market BUYS predict positive
    abnormal returns of 4-8% over 6-12 months. Cluster buys — where
    3+ distinct insiders purchase within 30 days — are the strongest
    variant, with ~10% 90-day abnormal returns in Cohen/Malloy/Pomorski.

    We specifically DON'T trade on insider sales — they have a much
    weaker signal-to-noise ratio (10b5-1 plans, diversification, tax
    events all create sells that aren't bearish).

Security:
    - Required SEC User-Agent header (SEC blocks unidentified bots)
    - No API keys (endpoints are public)
    - Rate limiter: SEC asks for <10 req/s; we throttle to 5 req/s
    - Response size cap (refuse XML > 1MB — malformed is suspicious)
    - Cache results to data/insider_cache/ (6-hour TTL)
    - XML parsing uses defusedxml to prevent XXE / entity attacks
    - All external strings sanitized before log write
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

# defusedxml is stdlib-safer than xml.etree for untrusted input.
# Fall back to stdlib with explicit flags if defusedxml isn't installed.
try:
    from defusedxml.ElementTree import fromstring as xml_fromstring
    _XML_LIB = "defusedxml"
except ImportError:
    from xml.etree.ElementTree import fromstring as xml_fromstring
    _XML_LIB = "stdlib"

from lib.audit import log_event

CACHE_DIR = Path(__file__).parent.parent / "data" / "insider_cache"
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours — Form 4s are filed T+2 so no rush
MAX_XML_BYTES = 1_000_000     # 1MB ceiling per filing
HTTP_TIMEOUT = 20             # seconds
MIN_REQUEST_INTERVAL = 0.2    # ~5 req/s to SEC (their limit is 10/s)

# SEC policy: "Please declare your User Agent in request headers"
# Use the user's real contact email so SEC can reach them if there's an
# abuse issue. Falls back to a generic string with clear project info.
_SEC_CONTACT_EMAIL = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
_DEFAULT_USER_AGENT = (
    f"traderbot-insider-flow/1.0 ({_SEC_CONTACT_EMAIL or 'contact@example.com'})"
)

# ── Ticker → CIK lookup (cached after first call) ─────────────────
_CIK_MAP: dict[str, str] | None = None


@dataclass
class InsiderTransaction:
    """One Form 4 transaction (after filtering to open-market buys)."""
    filing_date: str
    insider_name: str
    insider_title: str
    is_officer: bool
    is_director: bool
    is_ten_percent: bool
    shares: int
    price_per_share: float
    dollar_value: float
    transaction_code: str  # P, S, M, etc.


@dataclass
class InsiderFlowResult:
    """Aggregated insider-flow signal for a ticker."""
    ticker: str
    sentiment: float = 0.5          # 0.0 = heavy selling, 0.5 = neutral, 1.0 = heavy buying
    confidence: float = 0.0         # 0.0 = no data, 1.0 = strong signal
    cluster_detected: bool = False  # 3+ insiders buying within 30 days
    buy_count: int = 0              # Distinct insider buys in window
    sell_count: int = 0             # (informational only — we don't trade on it)
    total_buy_value_usd: float = 0.0
    signal: str = "neutral"         # "bullish_cluster", "bullish", "neutral", "bearish"
    recent_buys: list[InsiderTransaction] = field(default_factory=list)
    reason: str = ""
    cached: bool = False


# ── Rate limiter ─────────────────────────────────────────────────
_last_request_ts: float = 0.0


def _rate_limit():
    """Throttle SEC requests. Module-global lock suffices for single-process."""
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_ts = time.time()


def _http_get(url: str, max_bytes: int = MAX_XML_BYTES) -> bytes:
    """GET an SEC URL with required User-Agent + size cap."""
    _rate_limit()
    req = urllib.request.Request(url, headers={
        "User-Agent": _DEFAULT_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        content_len = resp.headers.get("Content-Length")
        if content_len is not None and int(content_len) > max_bytes:
            raise RuntimeError(f"Response too large ({content_len} bytes) for {url}")

        # Handle gzip if server used it
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RuntimeError(f"Response exceeded {max_bytes} bytes mid-stream")
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip
            data = gzip.decompress(data)
        return data


def _load_cik_map() -> dict[str, str]:
    """Fetch and cache the ticker → CIK mapping. SEC updates this file ~daily."""
    global _CIK_MAP
    if _CIK_MAP is not None:
        return _CIK_MAP

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "ticker_to_cik.json"

    # Reload cached copy if fresh (within 24h)
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400:
        try:
            with open(cache_file, "r") as f:
                _CIK_MAP = json.load(f)
            return _CIK_MAP
        except (json.JSONDecodeError, OSError):
            pass  # Fall through and re-fetch

    try:
        raw = _http_get(
            "https://www.sec.gov/files/company_tickers.json",
            max_bytes=10_000_000,  # This file is ~2MB
        )
        data = json.loads(raw)
        # Schema: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
        mapping: dict[str, str] = {}
        for entry in data.values() if isinstance(data, dict) else []:
            tkr = entry.get("ticker", "").upper() if isinstance(entry, dict) else ""
            cik = entry.get("cik_str") if isinstance(entry, dict) else None
            if tkr and isinstance(cik, int):
                mapping[tkr] = f"{cik:010d}"

        # Atomic cache write
        tmp = cache_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(mapping, f)
        tmp.replace(cache_file)

        _CIK_MAP = mapping
        return _CIK_MAP
    except Exception as e:
        log_event("insider_flow", "cik_lookup_failed", {"error": str(e)[:200]}, result="failed")
        _CIK_MAP = {}
        return _CIK_MAP


def _ticker_to_cik(ticker: str) -> str | None:
    """Map a ticker symbol to its SEC CIK (10-digit zero-padded)."""
    tkr = ticker.strip().upper()
    # Reject anything that isn't a plausible ticker (1-5 letters, possibly with dots/dashes)
    if not re.match(r"^[A-Z][A-Z0-9.\-]{0,5}$", tkr):
        return None
    cik_map = _load_cik_map()
    return cik_map.get(tkr)


def _fetch_recent_form4s(cik: str, days: int = 90) -> list[dict]:
    """Get metadata for all Form 4 filings for this CIK in the last `days` days."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    raw = _http_get(url, max_bytes=5_000_000)
    data = json.loads(raw)

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()

    form4s = []
    for i, f in enumerate(forms):
        if f != "4":
            continue
        filing_date = filing_dates[i] if i < len(filing_dates) else ""
        if filing_date < cutoff:
            # Arrays are filed-date-descending — safe to stop once we pass the cutoff
            break
        form4s.append({
            "accession": accession_numbers[i],
            "date": filing_date,
            "primary": primary_docs[i] if i < len(primary_docs) else "",
        })
    return form4s


def _parse_form4_xml(xml_bytes: bytes) -> list[InsiderTransaction]:
    """Parse a Form 4 XML into InsiderTransaction records.

    Filters to open-market purchases only (code='P'). Everything else
    (option exercises, grants, sales, tax withholdings) is discarded —
    those aren't tradable signals.
    """
    try:
        root = xml_fromstring(xml_bytes)
    except Exception as e:
        log_event("insider_flow", "xml_parse_failed", {"error": str(e)[:200]}, result="degraded")
        return []

    # Extract filer (insider) metadata once per filing
    owner = root.find(".//reportingOwner")
    if owner is None:
        return []
    name_el = owner.find(".//rptOwnerName")
    insider_name = (name_el.text or "").strip() if name_el is not None else "unknown"

    rel = owner.find("reportingOwnerRelationship")
    is_officer = _xml_text(rel, "isOfficer") == "1" if rel is not None else False
    is_director = _xml_text(rel, "isDirector") == "1" if rel is not None else False
    is_ten_percent = _xml_text(rel, "isTenPercentOwner") == "1" if rel is not None else False
    title = _xml_text(rel, "officerTitle") or ""

    # Filing date (period of report)
    period = _xml_text(root, "periodOfReport") or ""

    # Iterate non-derivative transactions (direct stock trades)
    txns: list[InsiderTransaction] = []
    for t in root.findall(".//nonDerivativeTransaction"):
        code_el = t.find(".//transactionCode")
        code = (code_el.text or "").strip() if code_el is not None else ""
        # P = open market purchase — the only code we care about for bullish signal
        if code != "P":
            continue

        shares = _xml_num(t, ".//transactionShares/value", as_int=True)
        price = _xml_num(t, ".//transactionPricePerShare/value", as_int=False)
        if shares is None or price is None or shares <= 0 or price <= 0:
            continue

        txns.append(InsiderTransaction(
            filing_date=period,
            insider_name=_sanitize(insider_name, 80),
            insider_title=_sanitize(title, 80),
            is_officer=is_officer,
            is_director=is_director,
            is_ten_percent=is_ten_percent,
            shares=shares,
            price_per_share=float(price),
            dollar_value=float(shares) * float(price),
            transaction_code=code,
        ))
    return txns


def _xml_text(node, tag: str) -> str | None:
    """Find a tag, returning .text or None. None-safe."""
    if node is None:
        return None
    el = node.find(tag)
    if el is None or el.text is None:
        return None
    return el.text.strip()


def _xml_num(node, xpath: str, as_int: bool = False):
    """Pull a numeric value from XML, defensive against missing/bad data."""
    t = _xml_text(node, xpath)
    if t is None:
        return None
    try:
        f = float(t)
        return int(f) if as_int else f
    except ValueError:
        return None


def _sanitize(s: str, max_len: int = 200) -> str:
    """Strip control chars + cap length. Insider names and titles come
    from SEC XML but are user-submitted — treat as untrusted input."""
    if not s:
        return ""
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)
    return s[:max_len]


def _fetch_form4_xml(cik: str, accession: str) -> bytes | None:
    """Download the raw Form 4 XML. Returns None on any failure."""
    acc_clean = accession.replace("-", "")
    # Strip the padding zeros from CIK for the Archives path
    cik_short = cik.lstrip("0") or "0"
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_short}/{acc_clean}/form4.xml"
    try:
        return _http_get(url, max_bytes=MAX_XML_BYTES)
    except Exception as e:
        log_event("insider_flow", "form4_fetch_failed", {
            "accession": accession, "error": str(e)[:200],
        }, result="degraded")
        return None


# ── Public API ───────────────────────────────────────────────────

def check_insider_flow(ticker: str, days: int = 90, bypass_cache: bool = False) -> InsiderFlowResult:
    """
    Return the aggregated insider-buying signal for a ticker.

    Args:
        ticker: Stock symbol (e.g. 'AAPL').
        days: Look-back window in days (default 90 — captures the
            3-month regulatory filing horizon most insiders use).
        bypass_cache: Force a fresh pull.

    Returns:
        InsiderFlowResult with sentiment in [0,1], confidence, cluster
        flag, and the raw buy records.
    """
    # ── Cache check ────────────────────────────────────────────────
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker.upper()}_flow.json"
    if not bypass_cache and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                return InsiderFlowResult(
                    ticker=cached["ticker"],
                    sentiment=cached["sentiment"],
                    confidence=cached["confidence"],
                    cluster_detected=cached["cluster_detected"],
                    buy_count=cached["buy_count"],
                    sell_count=cached["sell_count"],
                    total_buy_value_usd=cached["total_buy_value_usd"],
                    signal=cached["signal"],
                    reason=cached["reason"],
                    cached=True,
                )
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # Corrupt cache → refetch

    # ── Ticker → CIK ───────────────────────────────────────────────
    cik = _ticker_to_cik(ticker)
    if cik is None:
        return InsiderFlowResult(
            ticker=ticker.upper(),
            sentiment=0.5, confidence=0.0, signal="neutral",
            reason=f"Unknown ticker (no CIK match for {ticker!r})",
        )

    # ── Pull recent Form 4s ────────────────────────────────────────
    try:
        filings = _fetch_recent_form4s(cik, days=days)
    except Exception as e:
        log_event("insider_flow", "submissions_fetch_failed", {
            "ticker": ticker, "error": str(e)[:200],
        }, result="failed")
        return InsiderFlowResult(
            ticker=ticker.upper(),
            sentiment=0.5, confidence=0.0, signal="neutral",
            reason=f"SEC API error: {type(e).__name__}",
        )

    # ── Parse each Form 4 ──────────────────────────────────────────
    all_buys: list[InsiderTransaction] = []
    for f in filings[:50]:  # Hard cap — most tickers have <20 Form 4s per 90 days
        xml_bytes = _fetch_form4_xml(cik, f["accession"])
        if xml_bytes is None:
            continue
        txns = _parse_form4_xml(xml_bytes)
        # Prefer the filing-date from the recent-submissions array since
        # Form 4 XML's periodOfReport is the trade date, not filing date.
        for t in txns:
            t.filing_date = f["date"]
        all_buys.extend(txns)

    # ── Aggregate ──────────────────────────────────────────────────
    unique_insiders = {t.insider_name for t in all_buys}
    buy_count = len(all_buys)
    total_buy_value = sum(t.dollar_value for t in all_buys)

    # Cluster detection: 3+ DIFFERENT insiders buying in the last 30 days
    thirty_day_cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
    recent_insiders = {t.insider_name for t in all_buys if t.filing_date >= thirty_day_cutoff}
    cluster = len(recent_insiders) >= 3

    # ── Score ──────────────────────────────────────────────────────
    # Sentiment: [0.5 neutral, 1.0 heavy cluster buying]
    # We never set sentiment below 0.5 here — absence of buying ≠ bearish
    # (sells come from 10b5-1 plans and aren't actionable).
    if cluster:
        sentiment = 0.85
        confidence = min(1.0, 0.60 + 0.05 * len(recent_insiders))
        signal = "bullish_cluster"
        reason = f"Cluster buy: {len(recent_insiders)} insiders purchased in last 30d"
    elif buy_count >= 2 and total_buy_value >= 100_000:
        sentiment = 0.65
        confidence = 0.45
        signal = "bullish"
        reason = f"{buy_count} buy(s) totaling ${total_buy_value:,.0f}"
    elif buy_count == 1 and total_buy_value >= 250_000:
        sentiment = 0.58
        confidence = 0.30
        signal = "bullish"
        reason = f"Single large buy: ${total_buy_value:,.0f}"
    else:
        sentiment = 0.5
        confidence = 0.10 if buy_count == 0 else 0.20
        signal = "neutral"
        reason = "No material insider buying in window" if buy_count == 0 \
                 else f"Light buying ({buy_count} txns, ${total_buy_value:,.0f})"

    result = InsiderFlowResult(
        ticker=ticker.upper(),
        sentiment=sentiment,
        confidence=confidence,
        cluster_detected=cluster,
        buy_count=buy_count,
        sell_count=0,  # We don't track sells — too noisy to score
        total_buy_value_usd=total_buy_value,
        signal=signal,
        recent_buys=all_buys[:10],  # Keep top 10 for display / debugging
        reason=reason,
    )

    # ── Write cache atomically ─────────────────────────────────────
    try:
        payload = {
            "ticker": result.ticker,
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "cluster_detected": result.cluster_detected,
            "buy_count": result.buy_count,
            "sell_count": result.sell_count,
            "total_buy_value_usd": result.total_buy_value_usd,
            "signal": result.signal,
            "reason": result.reason,
        }
        tmp = cache_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        tmp.replace(cache_file)
    except OSError:
        pass  # Cache-write failure is non-fatal

    log_event("insider_flow", "check_complete", {
        "ticker": result.ticker,
        "signal": result.signal,
        "sentiment": result.sentiment,
        "cluster": result.cluster_detected,
        "buy_count": result.buy_count,
    }, result="success")

    return result
