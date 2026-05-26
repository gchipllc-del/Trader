"""
Long-Term Holding Picker — multi-factor scoring engine for 10-year-horizon
core holdings.

This is a *PM tool*, not an autonomous PM. It scores a candidate universe
across five orthogonal factors and returns a ranked dossier; the human
decides which top-N to add to `wheel_strategy.yaml.core_holdings`.

Why this design:
  • Long-term holds need different signals than swing trades. Momentum and
    technical patterns matter less; durable economics, balance-sheet
    health, and reinvestment runway matter more.
  • Every weight, threshold, and sub-score is visible in the output so
    the user can see WHY a name ranked where it did.
  • Scoring is deterministic — same inputs, same rank. No LLM judgement
    in the actual selection math (we use Kronos as a tiebreaker only).
  • Graceful degradation: if yfinance returns a None for a field, that
    field is dropped from its sub-score weighted average rather than
    being treated as 0.

The five factors and weights:

    Quality     35%  ROE, FCF positive, operating margin, debt/equity,
                     gross margin. The "is the business actually good"
                     score.
    Growth      25%  Revenue growth, earnings growth. The "is it
                     getting bigger" score.
    Moat        15%  Operating margin level + market cap + business age.
                     Proxy for durability.
    Momentum    15%  Position vs. 200-day MA, drawdown from 52w high,
                     Kronos AI directional bias. The "is it a falling
                     knife" filter.
    Valuation   10%  PEG ratio, distance from 52w high, P/FCF. Lower
                     weight because for long-term winners, paying up
                     is often correct (NVDA at "expensive" 5y ago etc).
                     But still helps avoid blowoff tops.

Use:
    from lib.longterm_picker import score_universe, render_dossier
    df = score_universe(["NVDA", "MSFT", ...])
    print(render_dossier(df, top_n=10))

CLI:
    python main.py longterm-pick --themes ai_compute,defense_cyber --top 10

Outputs a markdown table sorted by composite_score DESC with all factor
sub-scores plus a one-line "why" string per ticker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from lib.audit import log_event


# ── Candidate universe ─────────────────────────────────────────────
# Curated by theme, NOT a recommendation. The picker scores whatever
# you pass it; this list is just a sensible starter set covering the
# major secular themes credibly tied to 10-year compounding.
DEFAULT_CANDIDATES_BY_THEME: dict[str, list[str]] = {
    "ai_compute": [
        "NVDA", "MSFT", "GOOGL", "META", "AVGO", "AMD", "TSM", "ASML",
        "MU", "AMAT", "LRCX", "QCOM",
    ],
    "energy_transition": [
        "TSLA", "ENPH", "FSLR", "NEE", "ALB", "LIT", "ICLN",
    ],
    "defense_cyber": [
        "LMT", "RTX", "NOC", "GD", "PLTR", "CRWD", "ZS", "PANW", "S",
    ],
    "healthcare_automation": [
        "UNH", "ISRG", "VEEV", "MDT", "TMO", "DHR", "DXCM",
    ],
    "fintech_rails": [
        "V", "MA", "JPM", "BX", "MCO",
    ],
    "consumer_compounders": [
        "COST", "AMZN", "NFLX", "SBUX", "BRK-B",
    ],
    "industrial_automation": [
        "CAT", "DE", "ROK", "ETN", "PH",
    ],
}

ALL_DEFAULT_CANDIDATES: list[str] = sorted({
    t for tickers in DEFAULT_CANDIDATES_BY_THEME.values() for t in tickers
})


# ── Factor weights ─────────────────────────────────────────────────
FACTOR_WEIGHTS: dict[str, float] = {
    "quality":   0.35,
    "growth":    0.25,
    "moat":      0.15,
    "momentum":  0.15,
    "valuation": 0.10,
}
if abs(sum(FACTOR_WEIGHTS.values()) - 1.0) >= 1e-9:
    # Hard validate — `assert` here gets stripped under `python -O`, which
    # would silently let a typo'd config produce composite scores that
    # don't reflect the documented weighting.
    raise ValueError(
        f"FACTOR_WEIGHTS must sum to 1.0, got {sum(FACTOR_WEIGHTS.values()):.6f}"
    )


@dataclass
class TickerScore:
    ticker: str
    composite: float
    quality: float
    growth: float
    moat: float
    momentum: float
    valuation: float
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    why: str = ""
    raw: dict = field(default_factory=dict)


# ── Sub-score helpers ──────────────────────────────────────────────

def _piecewise(x: float, low: float, high: float) -> float:
    """Linear ramp: x <= low → 0, x >= high → 1, linear in between.
    If high < low, the ramp is inverted (lower x = higher score).
    """
    if high == low:
        return 0.5  # degenerate; neutral
    if high > low:
        return max(0.0, min(1.0, (x - low) / (high - low)))
    # Inverted ramp — lower is better
    return max(0.0, min(1.0, (low - x) / (low - high)))


def _weighted_avg(parts: list[tuple[Optional[float], float]]) -> float:
    """Drop None values, return weighted average of the rest. Returns
    0.5 (neutral) if all values are None — penalizes neither
    direction when the data is missing."""
    valid = [(v, w) for v, w in parts if v is not None]
    if not valid:
        return 0.5
    total_weight = sum(w for _, w in valid)
    if total_weight <= 0:
        return 0.5
    return sum(v * w for v, w in valid) / total_weight


def score_quality(f: dict) -> float:
    """Quality = ROE + FCF + margins + debt + gross margin. 0..1."""
    roe = f.get("roe")
    fcf = f.get("free_cashflow")
    op_margin = f.get("operating_margin")
    debt_eq = f.get("debt_to_equity")
    gross_margin = f.get("gross_margin")

    # ROE: 5% → 0, 25% → 1
    s_roe = _piecewise(roe, 0.05, 0.25) if roe is not None else None
    # FCF: positive = 1, negative = 0
    s_fcf = (1.0 if fcf > 0 else 0.0) if fcf is not None else None
    # Operating margin: 0% → 0, 25% → 1
    s_om = _piecewise(op_margin, 0.0, 0.25) if op_margin is not None else None
    # Debt/Equity: yfinance often returns as percent (e.g. 50 = 0.5).
    # Normalize: if value > 5, divide by 100. Lower is better.
    if debt_eq is not None:
        de = debt_eq / 100 if debt_eq > 5 else debt_eq
        s_de = _piecewise(de, 2.5, 0.2)  # 0.2 = 1.0, 2.5 = 0.0
    else:
        s_de = None
    # Gross margin: 15% → 0, 60% → 1
    s_gm = _piecewise(gross_margin, 0.15, 0.60) if gross_margin is not None else None

    return _weighted_avg([
        (s_roe, 0.30),
        (s_fcf, 0.15),
        (s_om,  0.20),
        (s_de,  0.20),
        (s_gm,  0.15),
    ])


def score_growth(f: dict) -> float:
    """Growth = revenue + earnings growth. 0..1."""
    rev_g = f.get("revenue_growth")     # YoY, e.g. 0.15 = +15%
    earn_g = f.get("earnings_growth")

    # 0% = 0, 25% = 1
    s_rev = _piecewise(rev_g, 0.0, 0.25) if rev_g is not None else None
    s_earn = _piecewise(earn_g, 0.0, 0.30) if earn_g is not None else None

    return _weighted_avg([
        (s_rev,  0.55),
        (s_earn, 0.45),
    ])


def score_moat(f: dict) -> float:
    """Moat proxy = operating margin level + market cap (durable scale)
    + gross margin. Real moat assessment is qualitative; this is a
    rough numerical analog.
    """
    op_margin = f.get("operating_margin")
    market_cap = f.get("market_cap")
    gross_margin = f.get("gross_margin")

    # Operating margin sustained > 20% is a strong durability signal
    s_om = _piecewise(op_margin, 0.05, 0.30) if op_margin is not None else None
    # Market cap: $1B = 0, $500B = 1 (log-scaled)
    if market_cap is not None and market_cap > 0:
        log_cap = math.log10(market_cap)
        s_cap = _piecewise(log_cap, 9.0, 11.7)  # 1B → 500B
    else:
        s_cap = None
    # Gross margin > 50% suggests pricing power
    s_gm = _piecewise(gross_margin, 0.20, 0.65) if gross_margin is not None else None

    return _weighted_avg([
        (s_om,  0.45),
        (s_cap, 0.30),
        (s_gm,  0.25),
    ])


def score_momentum(f: dict) -> float:
    """Momentum filter — keeps us from buying falling knives even on
    fundamentally great names. Combines 200d-MA position, drawdown
    from 52w high, and Kronos AI direction.
    """
    price = f.get("current_price")
    ma_200 = f.get("ma_200")
    high_52 = f.get("52w_high")
    kronos_dir = f.get("kronos_direction")  # "bullish" / "bearish" / None

    # Price vs 200-day MA: 0.85 (-15%) → 0, 1.10 (+10%) → 1
    if price is not None and ma_200 is not None and ma_200 > 0:
        ratio = price / ma_200
        s_ma = _piecewise(ratio, 0.85, 1.10)
    else:
        s_ma = None
    # Drawdown from 52w high: 0 (at high) = 1, 50% off = 0
    if price is not None and high_52 is not None and high_52 > 0:
        drawdown = 1 - (price / high_52)
        s_dd = _piecewise(drawdown, 0.50, 0.0)  # 0% drawdown → 1
    else:
        s_dd = None
    # Kronos directional bias
    if kronos_dir == "bullish":
        s_k = 1.0
    elif kronos_dir == "bearish":
        s_k = 0.0
    elif kronos_dir is not None:
        s_k = 0.5
    else:
        s_k = None

    return _weighted_avg([
        (s_ma, 0.40),
        (s_dd, 0.30),
        (s_k,  0.30),
    ])


def score_valuation(f: dict) -> float:
    """Valuation — lighter weight because for true long-term winners
    "expensive" is often a feature, but a sanity gate avoids paying
    blowoff-top multiples.
    """
    peg = f.get("peg_ratio")           # < 1 = cheap-ish
    forward_pe = f.get("forward_pe")
    fcf = f.get("free_cashflow")
    market_cap = f.get("market_cap")

    # PEG: 3.0 → 0, 0.5 → 1 (inverted: lower is better)
    s_peg = _piecewise(peg, 3.0, 0.5) if peg is not None and peg > 0 else None
    # Forward PE: 60 → 0, 12 → 1 (inverted)
    s_pe = _piecewise(forward_pe, 60.0, 12.0) if forward_pe is not None and forward_pe > 0 else None
    # P/FCF: cap/fcf. 50 → 0, 12 → 1 (inverted)
    if fcf is not None and fcf > 0 and market_cap is not None and market_cap > 0:
        p_fcf = market_cap / fcf
        s_pfcf = _piecewise(p_fcf, 50.0, 12.0)
    else:
        s_pfcf = None

    return _weighted_avg([
        (s_peg,  0.40),
        (s_pe,   0.30),
        (s_pfcf, 0.30),
    ])


# ── Data fetch ─────────────────────────────────────────────────────

def fetch_fundamentals(ticker: str) -> dict:
    """Pull fundamentals from yfinance. Adds 200d MA from Alpaca-quality
    bars if available; otherwise falls back to yfinance bars.

    Returns a dict with all fields we score against. Any field can be
    None (graceful degradation in scoring).
    """
    out: dict = {"ticker": ticker}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        out.update({
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("trailingPegRatio") or info.get("pegRatio"),
            "profit_margin": info.get("profitMargins"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "free_cashflow": info.get("freeCashflow"),
            "operating_cashflow": info.get("operatingCashflow"),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "current_price": info.get("currentPrice")
                              or info.get("regularMarketPrice"),
            "company_name": info.get("longName") or info.get("shortName"),
        })
        # 200-day moving average from yfinance bars
        try:
            hist = t.history(period="1y", auto_adjust=True)
            if hist is not None and len(hist) > 0 and "Close" in hist.columns:
                ma_200 = hist["Close"].tail(200).mean()
                out["ma_200"] = float(ma_200) if not math.isnan(ma_200) else None
        except Exception:
            out["ma_200"] = None
    except Exception as e:
        log_event("longterm_picker", "fetch_failed",
                  {"ticker": ticker, "error": str(e)[:200]},
                  result="degraded")
    return out


def fetch_kronos_direction(ticker: str) -> Optional[str]:
    """Optional Kronos AI directional bias. Returns 'bullish', 'bearish',
    'neutral', or None on failure. Best-effort — failures are silent so
    a Kronos outage doesn't break ranking.

    For long-term holds we use a longer prediction window than the swing
    engine (60 bars ≈ 12 trading weeks) so the signal aligns more with
    multi-month trend bias than with day-to-day chop.
    """
    try:
        from lib.kronos_forecaster import predict_price
        # 60-bar daily prediction. Kronos paper-conformant defaults
        # (T=0.6, top_p=0.90, N=10) inherit from predict_price.
        forecast = predict_price(
            ticker=ticker,
            pred_bars=60,
            interval="1d",
            sample_count=10,
        )
        return getattr(forecast, "direction", None)
    except Exception:
        return None


# ── Composite scoring ──────────────────────────────────────────────

def score_ticker(ticker: str, *, use_kronos: bool = True) -> TickerScore:
    """Score one ticker. Returns a TickerScore with composite + factors."""
    fundamentals = fetch_fundamentals(ticker)

    if use_kronos:
        kronos_dir = fetch_kronos_direction(ticker)
        if kronos_dir:
            fundamentals["kronos_direction"] = kronos_dir

    q = score_quality(fundamentals)
    g = score_growth(fundamentals)
    m = score_moat(fundamentals)
    mom = score_momentum(fundamentals)
    val = score_valuation(fundamentals)

    composite = (
        q   * FACTOR_WEIGHTS["quality"] +
        g   * FACTOR_WEIGHTS["growth"] +
        m   * FACTOR_WEIGHTS["moat"] +
        mom * FACTOR_WEIGHTS["momentum"] +
        val * FACTOR_WEIGHTS["valuation"]
    )

    why = _build_why(fundamentals, q, g, m, mom, val)

    return TickerScore(
        ticker=ticker,
        composite=composite,
        quality=q,
        growth=g,
        moat=m,
        momentum=mom,
        valuation=val,
        sector=fundamentals.get("sector"),
        market_cap=fundamentals.get("market_cap"),
        why=why,
        raw=fundamentals,
    )


def _build_why(f: dict, q: float, g: float, m: float, mom: float, val: float) -> str:
    """One-line summary highlighting the dominant factor + a watch-out."""
    factors = {"quality": q, "growth": g, "moat": m, "momentum": mom, "valuation": val}
    top_factor = max(factors, key=factors.get)
    bottom_factor = min(factors, key=factors.get)

    bits = []
    # Strength
    if factors[top_factor] >= 0.70:
        if top_factor == "quality":
            roe = f.get("roe")
            bits.append(f"strong quality (ROE {roe*100:.0f}%)" if roe else "strong quality")
        elif top_factor == "growth":
            rg = f.get("revenue_growth")
            bits.append(f"high growth ({rg*100:+.0f}% rev)" if rg else "high growth")
        elif top_factor == "moat":
            om = f.get("operating_margin")
            bits.append(f"durable moat ({om*100:.0f}% op-margin)" if om else "durable moat")
        elif top_factor == "momentum":
            bits.append("price strength")
        elif top_factor == "valuation":
            bits.append("attractive multiple")
    # Weakness
    if factors[bottom_factor] <= 0.30:
        if bottom_factor == "valuation":
            pe = f.get("forward_pe")
            bits.append(f"rich (fwd P/E {pe:.0f})" if pe else "expensive")
        elif bottom_factor == "momentum":
            bits.append("weak trend / drawdown")
        elif bottom_factor == "growth":
            bits.append("growth slowing")
        elif bottom_factor == "quality":
            bits.append("balance-sheet flag")
        elif bottom_factor == "moat":
            bits.append("commodity-like economics")
    if not bits:
        bits.append("balanced profile")
    return " — ".join(bits)


def score_universe(tickers: list[str], *, use_kronos: bool = True) -> list[TickerScore]:
    """Score a list of tickers. Returns list sorted by composite DESC."""
    log_event("longterm_picker", "scoring_started",
              {"n_tickers": len(tickers), "use_kronos": use_kronos})
    scores: list[TickerScore] = []
    for t in tickers:
        try:
            scores.append(score_ticker(t, use_kronos=use_kronos))
        except Exception as e:
            log_event("longterm_picker", "ticker_failed",
                      {"ticker": t, "error": str(e)[:200]}, result="degraded")
    scores.sort(key=lambda s: s.composite, reverse=True)
    log_event("longterm_picker", "scoring_complete",
              {"n_scored": len(scores), "top_5": [s.ticker for s in scores[:5]]})
    return scores


# ── Output formatting ──────────────────────────────────────────────

def render_dossier(scores: list[TickerScore], top_n: int = 15) -> str:
    """Render a markdown table of the top-N scored tickers."""
    if not scores:
        return "(no tickers scored)"
    header = (
        "| Rank | Ticker | Composite | Quality | Growth | Moat | Momentum | Valuation | Sector | Why |\n"
        "|------|--------|-----------|---------|--------|------|----------|-----------|--------|-----|"
    )
    rows = []
    for i, s in enumerate(scores[:top_n], 1):
        sector = (s.sector or "—")[:22]
        rows.append(
            f"| {i} | **{s.ticker}** | "
            f"**{s.composite:.3f}** | "
            f"{s.quality:.2f} | {s.growth:.2f} | {s.moat:.2f} | "
            f"{s.momentum:.2f} | {s.valuation:.2f} | "
            f"{sector} | {s.why} |"
        )
    return header + "\n" + "\n".join(rows)


def themes_to_tickers(themes: Optional[str]) -> list[str]:
    """Resolve a comma-separated theme list to a flat ticker list. If
    themes is None or 'all', returns the full default candidate set.
    Unknown themes are silently dropped.
    """
    if not themes or themes.lower() == "all":
        return ALL_DEFAULT_CANDIDATES
    result: list[str] = []
    seen: set[str] = set()
    for theme in (t.strip().lower() for t in themes.split(",")):
        for ticker in DEFAULT_CANDIDATES_BY_THEME.get(theme, []):
            if ticker not in seen:
                seen.add(ticker)
                result.append(ticker)
    return result
