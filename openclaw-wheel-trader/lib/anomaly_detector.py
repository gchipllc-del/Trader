"""
Anomaly Detector — flag stocks suddenly going parabolic ("skyrocket" events).

The pattern we're catching: a stock that was quiet for 20 days then explodes
on outsized volume + price move + range expansion + news velocity all at once.
That's the signature of a real catalyst (FDA decision, earnings beat, M&A
leak, short squeeze, AI/crypto narrative bid). Single features fire on noise;
the *composite* fires on tail-of-real-catalyst.

Signal architecture (4-feature composite z-score):
    1. Volume z      (35%) — money flowing in
    2. Price-move z  (30%) — direction + magnitude / ATR
    3. Range z       (15%) — vol regime change
    4. News velocity (20%) — confirms there's a catalyst

Triggered when:
    composite_z >= 4.0
    AND momentum confirmed (3 of last 4 bars green)
    AND price/volume gates pass (kill micro-caps + penny stocks)
    AND today's move is positive (we're long-only on this signal)

This is *detection*, not entry. The anomaly log feeds the strategy engine,
which still applies its own gates (Kelly sizing, stop-loss, earnings filter,
etc.) before placing a trade. False positives here are cheap; false
negatives (missing a real squeeze) are expensive.

Usage:
    from lib.alpaca_client import AlpacaClient
    from lib.anomaly_detector import scan_universe, print_anomaly_report

    client = AlpacaClient()
    scores = scan_universe(client, ["NVDA","TSLA","COIN", ...])
    print_anomaly_report(scores)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lib.audit import log_event

# --- Tunable thresholds --------------------------------------------------
DEFAULT_COMPOSITE_THRESHOLD = 4.0   # composite z-score needed to trigger
DEFAULT_MIN_AVG_VOLUME = 1_000_000  # 20d avg daily volume floor (liquidity)
DEFAULT_MIN_PRICE = 5.0             # skip penny stocks (pump-and-dump risk)
DEFAULT_MAX_PRICE = 1500.0          # very-high-price names trade weird
DEFAULT_MOMENTUM_BARS = 4           # last N bars to check for momentum
DEFAULT_MOMENTUM_GREENS = 3         # need this many green bars of N

# Composite weights — sum to 1.0
W_VOLUME = 0.35
W_PRICE = 0.30
W_RANGE = 0.15
W_NEWS = 0.20

ANOMALY_LOG_PATH = Path(__file__).parent.parent / "data" / "anomaly_log.jsonl"


@dataclass
class AnomalyScore:
    symbol: str
    timestamp: str
    last_price: float
    pct_move_today: float
    composite_z: float
    volume_z: float
    price_z: float
    range_z: float
    news_velocity: int
    momentum_confirmed: bool
    triggered: bool
    skip_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --- Feature helpers -----------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range. Returns nan if not enough bars."""
    if df is None or len(df) < period + 1:
        return float("nan")
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    tr = pd.concat(
        [(high - low), (high - close_prev).abs(), (low - close_prev).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.tail(period).mean())


def _safe_zscore(x: float, mean: float, std: float) -> float:
    """Z-score with guard against zero std."""
    if std is None or std <= 0 or np.isnan(std):
        return 0.0
    return float((x - mean) / std)


# --- Core scoring --------------------------------------------------------

def compute_features(
    symbol: str,
    daily_df: pd.DataFrame,
    news_count_60min: int = 0,
    composite_threshold: float = DEFAULT_COMPOSITE_THRESHOLD,
    min_avg_volume: int = DEFAULT_MIN_AVG_VOLUME,
    min_price: float = DEFAULT_MIN_PRICE,
    max_price: float = DEFAULT_MAX_PRICE,
) -> AnomalyScore:
    """Compute the 4-feature anomaly score for a single symbol.

    Args:
        symbol: stock ticker
        daily_df: pandas DataFrame with open/high/low/close/volume,
                  indexed by date, sorted ascending. Need >= 21 bars.
        news_count_60min: NewsAPI/RSS hits mentioning the ticker recently.
        composite_threshold: trigger z-score (default 4.0σ)
        min_avg_volume: minimum 20-day avg volume to consider (liquidity)
        min_price: minimum bar price (penny-stock filter)
        max_price: maximum bar price (whale-stock filter)

    Returns:
        AnomalyScore with `triggered` True if all gates passed.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    if daily_df is None or len(daily_df) < 21:
        return AnomalyScore(
            symbol=symbol, timestamp=now_iso,
            last_price=float("nan"), pct_move_today=float("nan"),
            composite_z=float("nan"), volume_z=float("nan"),
            price_z=float("nan"), range_z=float("nan"),
            news_velocity=news_count_60min, momentum_confirmed=False,
            triggered=False, skip_reason="insufficient_history",
        )

    # Last bar = "today"; baseline = preceding 20 bars.
    last = daily_df.iloc[-1]
    baseline = daily_df.iloc[-21:-1]

    last_price = float(last["close"])
    last_open = float(last["open"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_volume = float(last["volume"])
    pct_move = (last_price - last_open) / max(abs(last_open), 1e-9)

    # 1) Volume z-score vs 20d baseline
    vol_mean = float(baseline["volume"].mean())
    vol_std = float(baseline["volume"].std())
    volume_z = _safe_zscore(last_volume, vol_mean, vol_std)

    # 2) Price-move z-score: bar return / ATR%
    atr_value = _atr(daily_df.iloc[:-1], period=14)
    atr_pct = (atr_value / max(last_open, 1e-9)) if not np.isnan(atr_value) else 0.02
    # Normalize: pct_move expressed in ATR-units, then express as z-score
    # against the historical distribution of (daily_return / ATR%).
    base_returns = baseline["close"].pct_change().dropna()
    base_returns_in_atr = base_returns / max(atr_pct, 1e-9)
    price_in_atr_today = pct_move / max(atr_pct, 1e-9)
    pr_mean = float(base_returns_in_atr.mean()) if len(base_returns_in_atr) else 0.0
    pr_std = float(base_returns_in_atr.std()) if len(base_returns_in_atr) else 1.0
    price_z = _safe_zscore(price_in_atr_today, pr_mean, pr_std or 1.0)

    # 3) Range expansion z-score
    today_range = last_high - last_low
    base_ranges = baseline["high"] - baseline["low"]
    range_mean = float(base_ranges.mean())
    range_std = float(base_ranges.std())
    range_z = _safe_zscore(today_range, range_mean, range_std)

    # 4) News velocity — caller supplies. Heuristic mapping to z-equivalent:
    #    0 articles = -0.7, 1 = 0, 5+ = ~2.5σ.
    news_z = (news_count_60min - 1.0) / 1.5

    composite_z = (
        W_VOLUME * volume_z
        + W_PRICE * price_z
        + W_RANGE * range_z
        + W_NEWS * news_z
    )

    # Momentum confirmation — last DEFAULT_MOMENTUM_BARS bars, count greens.
    last_n = daily_df.tail(DEFAULT_MOMENTUM_BARS)
    greens = int(((last_n["close"] > last_n["open"]).sum()))
    momentum_confirmed = greens >= DEFAULT_MOMENTUM_GREENS

    # Quality / safety gates
    skip_reason: str | None = None
    if vol_mean < min_avg_volume:
        skip_reason = f"low_avg_volume_{int(vol_mean):,}"
    elif last_price < min_price:
        skip_reason = f"price_too_low_{last_price:.2f}"
    elif last_price > max_price:
        skip_reason = f"price_too_high_{last_price:.2f}"
    elif pct_move <= 0:
        skip_reason = "no_positive_move"

    triggered = bool(
        composite_z >= composite_threshold
        and momentum_confirmed
        and skip_reason is None
    )

    return AnomalyScore(
        symbol=symbol,
        timestamp=now_iso,
        last_price=last_price,
        pct_move_today=pct_move,
        composite_z=round(composite_z, 3),
        volume_z=round(volume_z, 3),
        price_z=round(price_z, 3),
        range_z=round(range_z, 3),
        news_velocity=int(news_count_60min),
        momentum_confirmed=momentum_confirmed,
        triggered=triggered,
        skip_reason=skip_reason,
    )


# --- Universe scan -------------------------------------------------------

def scan_universe(
    client,
    symbols: list[str],
    fetch_news: bool = True,
    bars_lookback: int = 30,
    composite_threshold: float = DEFAULT_COMPOSITE_THRESHOLD,
) -> list[AnomalyScore]:
    """Scan a list of symbols for anomalies.

    Args:
        client: AlpacaClient instance (provides get_bars).
        symbols: tickers to scan.
        fetch_news: query NewsAPI/RSS for last-window mentions per symbol.
                    Disable for backtesting / cost-conscious runs.
        bars_lookback: number of daily bars to fetch (need >= 21).
        composite_threshold: trigger threshold; relaxes for backtest sweeps.

    Returns:
        Sorted list of AnomalyScore (highest composite first).
    """
    if not symbols:
        return []

    bars = client.get_bars(symbols, timeframe="1Day", limit=bars_lookback)

    news_velocity: dict[str, int] = {}
    if fetch_news:
        try:
            from lib.news_sentiment import check_batch_sentiment
            results = check_batch_sentiment(symbols)
            for sym, res in results.items():
                # article_count is total over the lookback the sentiment
                # module uses; treat as our velocity proxy for now.
                news_velocity[sym] = int(getattr(res, "article_count", 0) or 0)
        except Exception as e:
            log_event(
                "anomaly_detector", "news_fetch_failed",
                {"error": str(e)[:200]}, result="degraded",
            )

    scores: list[AnomalyScore] = []
    for sym in symbols:
        df = bars.get(sym)
        nv = news_velocity.get(sym, 0)
        scores.append(
            compute_features(
                sym, df, news_count_60min=nv,
                composite_threshold=composite_threshold,
            )
        )

    def _sort_key(s: AnomalyScore) -> float:
        return -999.0 if np.isnan(s.composite_z) else s.composite_z

    scores.sort(key=_sort_key, reverse=True)
    return scores


# --- Reporting & persistence --------------------------------------------

def print_anomaly_report(scores: list[AnomalyScore], top: int = 15) -> None:
    """Pretty-print the ranked anomaly report."""
    print("=" * 84)
    print("  ANOMALY DETECTOR — TOP CANDIDATES")
    print("=" * 84)
    print(
        f"  {'Sym':<6} {'Move':>8} {'Vol-z':>7} {'Px-z':>7} {'Rng-z':>7} "
        f"{'News':>5} {'Comp':>7} {'Mom':>4} {'Hit':>4}  Notes"
    )
    print("  " + "-" * 80)
    for s in scores[:top]:
        if isinstance(s.composite_z, float) and np.isnan(s.composite_z):
            print(f"  {s.symbol:<6} insufficient history")
            continue
        hit = "✓" if s.triggered else " "
        mom = "✓" if s.momentum_confirmed else " "
        notes = s.skip_reason or ""
        print(
            f"  {s.symbol:<6} {s.pct_move_today * 100:>7.2f}% "
            f"{s.volume_z:>7.2f} {s.price_z:>7.2f} {s.range_z:>7.2f} "
            f"{s.news_velocity:>5d} {s.composite_z:>7.2f} "
            f" {mom:<3} {hit:<4} {notes}"
        )
    print("  " + "-" * 80)
    triggered = [s for s in scores if s.triggered]
    print(f"  Triggered: {len(triggered)} of {len(scores)} symbols")
    print()


def persist_scores(scores: list[AnomalyScore]) -> int:
    """Append all triggered scores to anomaly_log.jsonl. Returns N persisted."""
    triggered = [s for s in scores if s.triggered]
    if not triggered:
        return 0
    ANOMALY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ANOMALY_LOG_PATH, "a") as f:
        for s in triggered:
            f.write(json.dumps(s.to_dict()) + "\n")
    for s in triggered:
        log_event(
            "anomaly_detector", "triggered", s.to_dict(), result="info"
        )
    return len(triggered)


# --- Default watchlist ---------------------------------------------------

# Sane default: equities with high retail interest + meme/squeeze history.
# Override via `python main.py anomaly --watchlist sym1,sym2,sym3`.
DEFAULT_WATCHLIST = [
    # Mega-cap tech (FDA/AI/earnings catalysts hit hard here)
    "NVDA", "AMD", "TSLA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "AVGO",
    # Crypto-sensitive (squeeze + crypto narrative)
    "COIN", "MARA", "RIOT", "MSTR", "HOOD",
    # Squeeze / meme history
    "GME", "AMC", "BBBY", "SOFI", "PLTR", "DKNG", "RIVN", "LCID", "NIO",
    # Biotech (FDA decisions = canonical skyrocket events)
    "MRNA", "BNTX", "CRSP", "NTLA", "BEAM", "EDIT", "VRTX",
    # Small/mid caps with options-chain depth
    "U", "AFRM", "PYPL", "SHOP", "RBLX", "SNAP", "PINS",
]
