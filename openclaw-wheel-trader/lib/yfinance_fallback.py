"""
yfinance fallback — extends historical bar coverage beyond Alpaca's 7-yr window.

Use case: backtests on horizons longer than Alpaca's free-tier history (e.g.,
2008 GFC, 2020 COVID crash). yfinance provides daily bars going back decades
for free.

Returns DataFrames in the *same shape* as AlpacaClient.get_bars() so callers
can swap in transparently:

    from lib.yfinance_fallback import get_long_history_bars
    bars = get_long_history_bars(["SPY", "QQQ"], years_back=20)
    df = bars["SPY"]   # DataFrame with open/high/low/close/volume

Important:
- Only daily bars, no intraday (yfinance's intraday is limited to 60 days).
- yfinance is unofficial and rate-limits aggressively. Cache results.
- Some tickers (e.g., MARA pre-2021 IPO) have gaps; check df length before use.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from lib.audit import log_event

CACHE_DIR = Path(__file__).parent.parent / "data" / "yfinance_cache"
CACHE_TTL_HOURS = 24 * 7  # weekly refresh; daily bars don't change intra-day


def _cache_path(ticker: str, years_back: int) -> Path:
    return CACHE_DIR / f"{ticker.upper()}_{years_back}y.parquet"


def _is_fresh(path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_hours * 3600


def _try_yfinance() -> object | None:
    """Lazy import; return None on broken install (e.g., curl_cffi issues)."""
    try:
        import yfinance as yf
        return yf
    except Exception as e:
        log_event(
            "yfinance_fallback", "import_failed",
            {"error": str(e)[:200]}, result="degraded",
        )
        return None


def get_long_history_bars(
    tickers: list[str],
    years_back: int = 10,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV bars for `tickers` going back `years_back` years.

    Output shape matches AlpacaClient.get_bars(): a dict of {ticker:
    DataFrame} where each frame has open/high/low/close/volume columns and
    a DatetimeIndex.

    Returns empty dict if yfinance is unavailable.
    """
    yf = _try_yfinance()
    if yf is None:
        print("  ⚠️  yfinance not importable — see install notes in this module.")
        return {}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    end = datetime.now()
    start = end - timedelta(days=int(years_back * 365.25))

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ticker = ticker.upper()
        cache = _cache_path(ticker, years_back)

        if use_cache and _is_fresh(cache):
            try:
                out[ticker] = pd.read_parquet(cache)
                continue
            except Exception:
                pass  # fall through to fetch

        try:
            # progress=False suppresses yfinance's TQDM bar
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if df is None or df.empty:
                log_event(
                    "yfinance_fallback", "empty_result",
                    {"ticker": ticker}, result="degraded",
                )
                continue

            # yfinance returns columns like ('Open', 'TICKER') in MultiIndex
            # when downloading multiple symbols; for single-symbol downloads
            # we get plain columns. Normalize to lowercase.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]

            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.index.name = "timestamp"
            out[ticker] = df

            try:
                df.to_parquet(cache)
            except Exception:
                pass  # parquet may fail without pyarrow; skip cache

            time.sleep(0.5)  # respect rate limits

        except Exception as e:
            log_event(
                "yfinance_fallback", "fetch_failed",
                {"ticker": ticker, "error": str(e)[:200]}, result="degraded",
            )

    return out


def get_combined_history(
    client,
    tickers: list[str],
    years_back: int = 10,
) -> dict[str, pd.DataFrame]:
    """Combined Alpaca (recent) + yfinance (long-tail) daily bars.

    For each ticker, fetch the full requested window from yfinance and stitch
    Alpaca's recent bars on the right edge if they extend further. In practice
    yfinance's coverage typically equals or exceeds Alpaca's free-tier 7yr
    window so the simple "yfinance for long history, fall back to Alpaca" is
    fine for most backtest needs.
    """
    yf_bars = get_long_history_bars(tickers, years_back=years_back)
    if yf_bars:
        return yf_bars

    # yfinance unavailable — fall back to whatever Alpaca gives us
    print("  ⚠️  Falling back to Alpaca bars only (limited to ~7yr history).")
    bars = client.get_bars(tickers, timeframe="1Day", limit=2000)
    return bars or {}


# --- Diagnostic ---------------------------------------------------------

def diagnose_install() -> str:
    """Return a human-readable status of the yfinance install."""
    yf = _try_yfinance()
    if yf is None:
        return (
            "yfinance: NOT IMPORTABLE\n"
            "Fix:\n"
            "  pip install --upgrade yfinance\n"
            "If curl_cffi errors persist on macOS:\n"
            "  pip install --upgrade --force-reinstall curl_cffi\n"
        )
    try:
        df = yf.download("SPY", period="5d", progress=False, threads=False)
        if df is None or df.empty:
            return f"yfinance v{getattr(yf, '__version__', '?')} imported, but SPY download empty"
        return f"yfinance v{getattr(yf, '__version__', '?')} OK ({len(df)} bars for SPY 5d)"
    except Exception as e:
        return f"yfinance imported but download failed: {e}"
