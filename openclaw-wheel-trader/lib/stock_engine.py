"""
Phase 1: Stock Trading Engine — for building portfolio before options.

Uses the same analysis framework (trend + zones + candlestick signals)
to buy stocks at support with bullish confirmation. Sells at resistance
or trailing stop. Graduates to options (Phase 2) once portfolio reaches
the CSP collateral threshold.

This is the stepping stone to the full Wheel Strategy.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from lib.audit import log_event
from lib.alpaca_client import AlpacaClient
from lib.screener import _load_strategy_config
from lib.zones import detect_zones, get_nearest_support, get_nearest_resistance
from lib.trend import analyze_trend, multi_timeframe_analysis
from lib.candlestick import get_latest_signal, scan_patterns
from lib.iv_rank import calculate_historical_volatility
from lib.memory_palace import diary_write, remember_trade_decision, get_current_regime
from lib.circuit_breaker import check_paper_mode, check_daily_loss, CircuitBreakerTripped
from lib.quant_screener import screen_universe, print_screening_report
from lib.momentum import analyze_momentum

POSITIONS_PATH = Path(__file__).parent.parent / "data" / "positions.json"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
STRATEGY_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"

# Phase thresholds
PHASE_2_THRESHOLD = 5000   # Start selling CSPs on cheap stocks
PHASE_3_THRESHOLD = 10000  # Full Wheel on bigger tickers


def _load_settings() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_strategy() -> dict:
    with open(STRATEGY_PATH) as f:
        return yaml.safe_load(f)


def _load_positions() -> list[dict]:
    if not POSITIONS_PATH.exists():
        return []
    with open(POSITIONS_PATH) as f:
        return json.load(f)


def _save_positions(positions: list[dict]):
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)


def get_current_phase(portfolio_value: float) -> int:
    """Determine trading phase based on portfolio size."""
    if portfolio_value >= PHASE_3_THRESHOLD:
        return 3
    elif portfolio_value >= PHASE_2_THRESHOLD:
        return 2
    return 1


def score_stock_buy(
    ticker: str,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    current_price: float,
    max_position_value: float,
) -> dict | None:
    """
    Score a stock as a buy candidate using trend + level + signal + momentum.

    Composite score 0-13:
      - Trend alignment (0-3)
      - Support level quality (0-3)
      - Candlestick signal (0-3)
      - Momentum (0-4): RSI, MACD, volume surge, ROC

    Parameters are read from wheel_strategy.yaml so Hermes can tune them.
    """
    if len(daily_df) < 30 or len(weekly_df) < 10:
        return None

    strategy = _load_strategy()
    stock_cfg = strategy.get("stock_params", {})
    min_score = stock_cfg.get("min_composite_score", 3)
    stop_loss_pct = stock_cfg.get("stop_loss_pct", 0.05)
    default_target_pct = stock_cfg.get("default_target_pct", 0.10)
    allow_momentum_only = stock_cfg.get("allow_momentum_only", False)
    support_distance_tiers = stock_cfg.get("support_distance_tiers", [0.03, 0.05, 0.08])

    # --- Trend Score (0-3) ---
    mtf = multi_timeframe_analysis(weekly_df, daily_df)
    trend_score = mtf["alignment_score"]

    weekly_trend = mtf["weekly"]
    # Don't buy into strong downtrends (strength 3 = confirmed)
    if weekly_trend.direction == "downtrend" and weekly_trend.strength >= 3:
        return None

    # --- Level Score (0-3) ---
    zones = detect_zones(daily_df, current_price)
    nearest_support = get_nearest_support(zones, current_price)

    level_score = 0
    zone_level = 0.0
    zone_touches = 0

    if nearest_support:
        distance_to_zone = abs(current_price - nearest_support.level) / nearest_support.level
        tiers = support_distance_tiers
        if distance_to_zone < tiers[0]:
            level_score = 3
        elif distance_to_zone < tiers[1]:
            level_score = 2
        elif distance_to_zone < tiers[2]:
            level_score = 1

        zone_level = nearest_support.level
        zone_touches = nearest_support.touches

    # --- Signal Score (0-3) ---
    signal_score = 0
    pattern_name = None

    bullish_patterns = [
        "hammer", "bullish_engulfing", "morning_star",
        "dragonfly_doji", "bullish_harami", "tweezers_bottom",
    ]
    signal = get_latest_signal(daily_df, "bullish", bullish_patterns)
    if signal:
        signal_score = signal.strength
        pattern_name = signal.pattern

    # --- Momentum Score (0-4) ---
    momentum = analyze_momentum(daily_df)
    momentum_score = momentum.momentum_score if momentum else 0
    momentum_details = {}
    if momentum:
        momentum_details = {
            "rsi": momentum.rsi,
            "rsi_signal": momentum.rsi_signal,
            "macd_cross": momentum.macd_cross,
            "volume_surge": momentum.volume_surge,
            "roc_5d": momentum.roc_5d,
        }

    # --- Composite ---
    composite = trend_score + level_score + signal_score + momentum_score

    # Entry logic — more permissive in growth mode
    not_downtrend = weekly_trend.direction != "downtrend"
    has_signal = signal_score >= 1
    momentum_only_min = stock_cfg.get("momentum_only_min_score", 3)
    has_strong_momentum = momentum_score >= momentum_only_min

    if allow_momentum_only:
        # Growth mode: strong momentum (default >= 3/4) can substitute for candlestick signal
        tradeable = composite >= min_score and not_downtrend and (has_signal or has_strong_momentum)
    else:
        # Conservative: require candlestick signal
        tradeable = composite >= min_score and not_downtrend and has_signal

    # Calculate position size
    shares = calculate_position_size(current_price, max_position_value)
    if shares < 1:
        return None

    # Nearest resistance for profit target
    nearest_resistance = get_nearest_resistance(zones, current_price)
    target_price = nearest_resistance.level if nearest_resistance else current_price * (1 + default_target_pct)

    # Dynamic stop: tighter when momentum is strong, wider when weak
    if momentum_score >= 3:
        actual_stop_pct = stop_loss_pct * 0.8  # Tighter stop, trust the momentum
    else:
        actual_stop_pct = stop_loss_pct

    return {
        "ticker": ticker,
        "trade_type": "stock_buy",
        "current_price": round(current_price, 2),
        "shares": shares,
        "position_value": round(current_price * shares, 2),
        "trend_score": trend_score,
        "level_score": level_score,
        "signal_score": signal_score,
        "momentum_score": momentum_score,
        "composite_score": composite,
        "zone_level": zone_level,
        "zone_touches": zone_touches,
        "pattern": pattern_name,
        "momentum": momentum_details,
        "target_price": round(target_price, 2),
        "stop_loss": round(current_price * (1 - actual_stop_pct), 2),
        "tradeable": tradeable,
        "weekly_trend": weekly_trend.direction,
    }


def calculate_position_size(
    price: float,
    max_position_value: float,
    min_shares: int = 1,
) -> int:
    """
    Calculate how many shares to buy.
    Never exceed max_position_value per position.
    """
    if price <= 0:
        return 0
    shares = int(max_position_value / price)
    return max(shares, 0)


def scan_for_stocks(
    client: AlpacaClient,
    daily_data: dict[str, pd.DataFrame],
    weekly_data: dict[str, pd.DataFrame],
    portfolio_value: float,
) -> list[dict]:
    """
    Scan tickers for stock buy candidates.

    Five-gate system:
      Gate 1: Quantitative screen (Sharpe, drawdown, volatility)
      Gate 2: Technical screen (trend + zones + candlestick signals)
      Gate 3: Momentum screen (RSI, MACD, volume, ROC)
      Gate 4: Kronos AI — bearish prediction vetoes the trade
      Gate 5: News sentiment — strongly negative news vetoes the trade

    Returns ranked list of tradeable candidates.
    """
    settings = _load_settings()
    strategy = _load_strategy()
    max_position_pct = settings["circuit_breakers"]["max_position_pct"]
    max_position_value = portfolio_value * max_position_pct
    max_concurrent = strategy.get("stock_params", {}).get("max_concurrent_positions", 5)

    # Gate 1: Quantitative screening
    quant_scores = screen_universe(
        daily_data,
        max_price=max_position_value,
        exclude_avoid=True,
    )

    print("\n  --- Quant Screening ---")
    print_screening_report(quant_scores)

    quant_passed = {s.ticker for s in quant_scores if s.verdict != "AVOID"}

    # Check existing positions — respect max concurrent
    positions = _load_positions()
    held_tickers = set(
        p["ticker"] for p in positions
        if p.get("status") in ("open", "assigned") and p.get("type") == "stock"
    )
    open_count = len(held_tickers)

    if open_count >= max_concurrent:
        log_event("stock_engine", "max_positions_reached", {"open": open_count, "max": max_concurrent})
        print(f"  Max concurrent positions reached ({open_count}/{max_concurrent})")
        return []

    slots_available = max_concurrent - open_count

    # Gate 2+3: Technical + Momentum screening (only on quant-passed tickers)
    candidates = []
    for ticker in quant_passed:
        if ticker in held_tickers:
            continue

        daily_df = daily_data.get(ticker)
        if daily_df is None or len(daily_df) < 30:
            continue

        current_price = daily_df["close"].iloc[-1]
        weekly_df = weekly_data.get(ticker)
        if weekly_df is None or len(weekly_df) < 10:
            continue

        candidate = score_stock_buy(
            ticker, daily_df, weekly_df, current_price, max_position_value,
        )
        if candidate and candidate["tradeable"]:
            # Attach quant score for ranking
            qs = next((s for s in quant_scores if s.ticker == ticker), None)
            if qs:
                candidate["quant_score"] = qs.quant_score
                candidate["sharpe"] = qs.sharpe_ratio
                candidate["max_drawdown"] = qs.max_drawdown
            candidates.append(candidate)

    if not candidates:
        log_event("stock_engine", "no_candidates_after_technical", {})
        return []

    # Gate 4: Kronos AI price prediction — veto bearish forecasts
    kronos_cfg = strategy.get("kronos", {})
    if kronos_cfg.get("enabled", False):
        candidates = _apply_kronos_gate(candidates, kronos_cfg)

    # Gate 5: News sentiment — veto strongly negative news
    news_cfg = strategy.get("news_sentiment", {})
    if news_cfg.get("enabled", False):
        candidates = _apply_news_gate(candidates, news_cfg)

    # Gate 6: Bayesian multi-signal forecast (combines all signals into calibrated win prob)
    bayes_cfg = strategy.get("bayesian", {})
    if bayes_cfg.get("enabled", True):
        candidates = _apply_bayesian_gate(candidates, bayes_cfg)

    # Gate 6.5: Earnings proximity — hard veto if too close, soft penalty if within window.
    # Options get absolute veto (CLAUDE.md); stock swings take binary-event haircut instead.
    earnings_cfg = strategy.get("earnings_proximity", {})
    if earnings_cfg.get("enabled", True):
        candidates = _apply_earnings_proximity_gate(candidates, earnings_cfg)

    # Gate 7: Correlation check — avoid loading up on correlated stocks
    corr_cfg = strategy.get("correlation", {})
    if corr_cfg.get("enabled", True):
        candidates = _apply_correlation_gate(candidates, held_tickers, daily_data, corr_cfg)

    # Gate 8: Kelly position sizing (overrides shares count with optimal size)
    kelly_cfg = strategy.get("kelly", {})
    if kelly_cfg.get("enabled", True):
        candidates = _apply_kelly_sizing(candidates, portfolio_value, kelly_cfg)

    # Sort by: bayesian win probability (primary), then composite score (secondary)
    candidates.sort(
        key=lambda c: (
            c.get("bayesian_win_prob", c["composite_score"] / 13.0),
            c["composite_score"],
            c.get("quant_score", 0),
        ),
        reverse=True,
    )

    # Limit to available slots
    candidates = candidates[:slots_available]

    log_event("stock_engine", "scan_complete", {
        "quant_passed": len(quant_passed),
        "technical_passed": len(candidates),
        "slots_available": slots_available,
        "top": candidates[0]["ticker"] if candidates else "none",
    })

    return candidates


def _apply_kronos_gate(candidates: list[dict], kronos_cfg: dict) -> list[dict]:
    """
    Gate 4: Kronos AI price prediction.

    If Kronos says bearish (expected return < veto_threshold), skip the stock.
    If Kronos says bullish, attach the forecast for confidence boosting.
    Non-blocking — if Kronos fails, candidate passes through.

    Performance:
        Uses ``predict_prices_batch`` to run all candidate tickers through
        Kronos in a single ``predict_batch`` call. On a cold-cache N-ticker
        scan this is typically 3-10× faster than the per-ticker loop because
        the model's autoregressive overhead is paid once per batch instead
        of once per ticker.

        If the batch call itself raises, we fall back to the original
        per-ticker loop so a single bad ticker never kills a screen.
    """
    if not candidates:
        return []

    veto_threshold = kronos_cfg.get("veto_return_threshold", -0.02)
    pred_bars = kronos_cfg.get("pred_bars", 10)

    # Use the screening_gate preset — Gate 4 is a binary "is this bearish?"
    # decision, so N=10 paths is wasted compute.
    from lib.kronos_forecaster import PAPER_PRESETS
    preset = PAPER_PRESETS["screening_gate"]

    tickers = [c["ticker"] for c in candidates]
    forecasts_by_ticker: dict[str, object] = {}

    try:
        from lib.kronos_forecaster import predict_prices_batch
        forecasts = predict_prices_batch(
            tickers=tickers,
            pred_bars=pred_bars,
            interval="1d",
            lookback=200,
            sample_count=preset["sample_count"],
            temperature=preset["T"],
            top_p=preset["top_p"],
        )
        for f in forecasts:
            forecasts_by_ticker[f.ticker] = f
        log_event("stock_engine", "kronos_batch_ok", {
            "input": len(tickers),
            "returned": len(forecasts),
        })
    except Exception as e:
        # Batch path failed — fall back to per-ticker so a transient batch
        # error doesn't tank the whole scan. We keep the same non-blocking
        # semantics for the per-ticker path below.
        log_event("stock_engine", "kronos_batch_failed_fallback", {
            "error": str(e)[:200],
            "tickers": len(tickers),
        }, result="failed")
        forecasts_by_ticker = {}

        from lib.kronos_forecaster import predict_price
        for candidate in candidates:
            ticker = candidate["ticker"]
            try:
                forecast = predict_price(
                    ticker=ticker,
                    pred_bars=pred_bars,
                    interval="1d",
                    lookback=200,
                    sample_count=preset["sample_count"],
                    temperature=preset["T"],
                    top_p=preset["top_p"],
                )
                forecasts_by_ticker[ticker] = forecast
            except Exception as inner_err:
                log_event("stock_engine", "kronos_error", {
                    "ticker": ticker,
                    "error": str(inner_err)[:200],
                })
                # Leave forecast missing — handled below

    # Distribute forecasts to candidates and apply veto.
    filtered = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        forecast = forecasts_by_ticker.get(ticker)

        if forecast is None:
            # No forecast (fetch error or batch+per-ticker both failed).
            # Non-blocking: let the candidate through.
            candidate["kronos_direction"] = None
            candidate["kronos_expected_return"] = None
            candidate["kronos_confidence"] = None
            filtered.append(candidate)
            continue

        candidate["kronos_direction"] = forecast.direction
        candidate["kronos_expected_return"] = forecast.expected_return
        candidate["kronos_confidence"] = forecast.confidence

        if forecast.expected_return < veto_threshold:
            print(f"  🔴 {ticker}: Kronos VETO — predicted {forecast.expected_return:+.1%} "
                  f"({forecast.direction}, conf={forecast.confidence:.2f})")
            log_event("stock_engine", "kronos_veto", {
                "ticker": ticker,
                "expected_return": forecast.expected_return,
                "direction": forecast.direction,
            })
            continue

        emoji = "🟢" if forecast.direction == "bullish" else "🟡"
        print(f"  {emoji} {ticker}: Kronos {forecast.direction} "
              f"({forecast.expected_return:+.1%}, conf={forecast.confidence:.2f})")
        filtered.append(candidate)

    return filtered


def _apply_news_gate(candidates: list[dict], news_cfg: dict) -> list[dict]:
    """
    Gate 5: News sentiment filter.

    If recent news is strongly bearish (sentiment < veto_threshold), skip.
    Non-blocking — if news check fails, candidate passes through.
    """
    veto_threshold = news_cfg.get("veto_sentiment_threshold", 0.25)

    filtered = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        try:
            from lib.news_sentiment import check_stock_sentiment
            result = check_stock_sentiment(ticker)

            candidate["news_sentiment"] = result.sentiment
            candidate["news_signal"] = result.signal
            candidate["news_confidence"] = result.confidence

            if result.sentiment < veto_threshold and result.confidence > 0.3:
                # Strong bearish news with decent confidence — skip
                print(f"  📰 {ticker}: News VETO — {result.signal} sentiment "
                      f"({result.sentiment:.2f}, {result.article_count} articles)")
                if result.headlines:
                    print(f"      Top: {result.headlines[0][:80]}")
                log_event("stock_engine", "news_veto", {
                    "ticker": ticker,
                    "sentiment": result.sentiment,
                    "signal": result.signal,
                    "articles": result.article_count,
                })
                continue
            else:
                if result.article_count > 0:
                    print(f"  📰 {ticker}: News {result.signal} "
                          f"(sentiment={result.sentiment:.2f}, {result.article_count} articles)")
                filtered.append(candidate)

        except Exception as e:
            # News check failed — let candidate through
            candidate["news_sentiment"] = None
            candidate["news_signal"] = None
            candidate["news_confidence"] = None
            log_event("stock_engine", "news_error", {
                "ticker": ticker,
                "error": str(e)[:200],
            })
            filtered.append(candidate)

    return filtered


def _apply_bayesian_gate(candidates: list[dict], bayes_cfg: dict) -> list[dict]:
    """
    Gate 6: Bayesian multi-signal aggregation.

    Combines composite score + Kronos + news + pattern + momentum into a
    calibrated win probability. Vetoes setups below min_win_prob threshold.
    """
    min_win_prob = bayes_cfg.get("min_win_prob", 0.58)

    filtered = []
    for candidate in candidates:
        try:
            from lib.bayesian_forecaster import forecast_stock
            forecast = forecast_stock(
                ticker=candidate["ticker"],
                composite_score=candidate["composite_score"],
                trend_score=candidate["trend_score"],
                level_score=candidate["level_score"],
                signal_score=candidate["signal_score"],
                momentum_score=candidate["momentum_score"],
                pattern=candidate.get("pattern"),
                zone_touches=candidate.get("zone_touches", 0),
                weekly_direction=candidate.get("weekly_trend", "sideways"),
                kronos_expected_return=candidate.get("kronos_expected_return"),
                kronos_confidence=candidate.get("kronos_confidence", 0.5),
                news_sentiment=candidate.get("news_sentiment"),
                news_confidence=candidate.get("news_confidence", 0.5),
            )

            candidate["bayesian_win_prob"] = forecast.win_probability
            candidate["bayesian_confidence"] = forecast.confidence
            candidate["bayesian_sources"] = forecast.sources
            candidate["bayesian_summary"] = forecast.evidence_summary

            if forecast.win_probability < min_win_prob:
                print(f"  🎯 {candidate['ticker']}: Bayesian VETO — win_prob "
                      f"{forecast.win_probability:.0%} < {min_win_prob:.0%}")
                log_event("stock_engine", "bayesian_veto", {
                    "ticker": candidate["ticker"],
                    "win_prob": forecast.win_probability,
                })
                continue

            emoji = "🟢" if forecast.win_probability >= 0.70 else "🟡"
            print(f"  {emoji} {candidate['ticker']}: Bayesian "
                  f"{forecast.win_probability:.0%} (conf {forecast.confidence:.2f}) — "
                  f"{forecast.evidence_summary}")
            filtered.append(candidate)

        except Exception as e:
            candidate["bayesian_win_prob"] = None
            log_event("stock_engine", "bayesian_error", {
                "ticker": candidate["ticker"],
                "error": str(e)[:200],
            })
            filtered.append(candidate)  # Let through on error

    return filtered


def _apply_earnings_proximity_gate(candidates: list[dict], earnings_cfg: dict) -> list[dict]:
    """
    Gate 6.5: Earnings proximity filter for stock swings.

    Stock swings carry binary risk through earnings reports. Unlike options
    (which get an absolute veto per CLAUDE.md), stock buys use a two-tier model:
      - Hard veto if earnings within hard_veto_days (too close to manage)
      - Soft penalty if earnings within soft_warn_days (downsize via Kelly)

    Fails open when Finnhub unavailable — no data = no block, no penalty.
    Sets candidate["earnings_penalty"] which Kelly multiplies into sizing.
    """
    hard_veto_days = earnings_cfg.get("hard_veto_days", 2)
    soft_warn_days = earnings_cfg.get("soft_warn_days", 14)
    soft_penalty = earnings_cfg.get("soft_penalty", 0.5)

    filtered = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        try:
            from lib.earnings_filter import days_to_next_earnings
            days = days_to_next_earnings(ticker, lookahead_days=max(soft_warn_days, 60))

            if days is None:
                # No data — fail open (no block, no penalty)
                candidate["earnings_days"] = None
                candidate["earnings_penalty"] = 1.0
                filtered.append(candidate)
                continue

            candidate["earnings_days"] = days

            if days <= hard_veto_days:
                print(f"  📅 {ticker}: Earnings VETO — {days}d away (≤ {hard_veto_days}d cutoff)")
                log_event("stock_engine", "earnings_veto", {
                    "ticker": ticker, "days_until_earnings": days,
                })
                continue

            if days <= soft_warn_days:
                candidate["earnings_penalty"] = soft_penalty
                print(f"  📅 {ticker}: Earnings in {days}d — sizing down to "
                      f"{int(soft_penalty * 100)}%")
                log_event("stock_engine", "earnings_soft_penalty", {
                    "ticker": ticker,
                    "days_until_earnings": days,
                    "penalty": soft_penalty,
                })
            else:
                candidate["earnings_penalty"] = 1.0

            filtered.append(candidate)

        except Exception as e:
            # Anything goes wrong — fail open
            candidate["earnings_days"] = None
            candidate["earnings_penalty"] = 1.0
            log_event("stock_engine", "earnings_proximity_error", {
                "ticker": ticker, "error": str(e)[:200],
            })
            filtered.append(candidate)

    return filtered


def _apply_correlation_gate(
    candidates: list[dict],
    held_tickers: set,
    daily_data: dict,
    corr_cfg: dict,
) -> list[dict]:
    """
    Gate 7: Correlation check.

    Prevents loading up on correlated stocks (e.g., F + NIO = double EV exposure).
    Softer than a hard veto — reduces position size rather than blocking entirely.
    """
    threshold = corr_cfg.get("threshold", 0.70)
    hard_veto = corr_cfg.get("hard_veto", False)

    if not held_tickers:
        return candidates  # Nothing to correlate with

    filtered = []
    for candidate in candidates:
        try:
            from lib.correlation import check_portfolio_correlation
            result = check_portfolio_correlation(
                new_ticker=candidate["ticker"],
                held_tickers=list(held_tickers),
                daily_data=daily_data,
                threshold=threshold,
            )

            candidate["correlation_check"] = result

            if result["correlated"]:
                if hard_veto:
                    print(f"  🔗 {candidate['ticker']}: Correlation VETO — {result['reason']}")
                    log_event("stock_engine", "correlation_veto", {
                        "ticker": candidate["ticker"],
                        "conflicts": [c["ticker"] for c in result["conflicts"]],
                    })
                    continue
                else:
                    # Soft veto: flag + shrink position
                    print(f"  ⚠️  {candidate['ticker']}: Correlated — {result['reason']} (sizing down)")
                    candidate["correlation_penalty"] = 0.5  # Will halve Kelly sizing
            filtered.append(candidate)

        except Exception as e:
            log_event("stock_engine", "correlation_error", {
                "ticker": candidate["ticker"],
                "error": str(e)[:200],
            })
            filtered.append(candidate)

    return filtered


def _apply_kelly_sizing(candidates: list[dict], portfolio_value: float, kelly_cfg: dict) -> list[dict]:
    """
    Gate 8: Kelly position sizing.

    Replaces the naive max_position_pct sizing with mathematically-optimal
    Kelly sizing based on win probability and reward/risk ratio.
    """
    fraction = kelly_cfg.get("fraction", 0.25)

    for candidate in candidates:
        try:
            from lib.kelly import kelly_position_size

            # Use Bayesian win prob if available, else composite-derived
            composite_score = candidate["composite_score"]

            sizing = kelly_position_size(
                portfolio_value=portfolio_value,
                current_price=candidate["current_price"],
                target_price=candidate["target_price"],
                stop_loss=candidate["stop_loss"],
                composite_score=composite_score,
                kronos_expected_return=candidate.get("kronos_expected_return"),
                fraction=fraction,
            )

            # If Bayesian gave us a more accurate win_prob, override Kelly's composite-derived one
            bayesian_wp = candidate.get("bayesian_win_prob")
            if bayesian_wp is not None and bayesian_wp > 0:
                # Recompute Kelly with Bayesian prob
                from lib.kelly import fractional_kelly_stock
                reward_pct = sizing.get("reward_pct", 0)
                risk_pct = sizing.get("risk_pct", 0)
                if reward_pct > 0 and risk_pct > 0:
                    frac_k = fractional_kelly_stock(bayesian_wp, reward_pct, risk_pct, fraction)
                    strategy = _load_strategy()
                    max_pct = strategy.get("stock_params", {}).get("max_position_pct", 0.30)
                    pct = min(frac_k, max_pct)
                    # Apply correlation + earnings penalties if flagged
                    pct *= candidate.get("correlation_penalty", 1.0)
                    pct *= candidate.get("earnings_penalty", 1.0)

                    sizing["win_prob"] = round(bayesian_wp, 4)
                    sizing["fractional_kelly"] = round(frac_k, 4)
                    sizing["pct_of_portfolio"] = round(pct, 4)
                    sizing["position_value"] = round(portfolio_value * pct, 2)
                    sizing["shares"] = int(portfolio_value * pct / candidate["current_price"])

            # Override the screener's default sizing with Kelly's
            if sizing.get("shares", 0) > 0:
                candidate["kelly_sizing"] = sizing
                candidate["shares"] = sizing["shares"]
                candidate["position_value"] = sizing["position_value"]
                print(f"  💰 {candidate['ticker']}: Kelly sized to "
                      f"{sizing['shares']} shares "
                      f"({sizing['pct_of_portfolio']*100:.1f}% of portfolio, "
                      f"R/R {sizing.get('reward_to_risk', 0)}x)")
            else:
                print(f"  ⚠️  {candidate['ticker']}: Kelly says no trade — {sizing.get('reason', 'unknown')}")
                # Keep original sizing if Kelly says 0
                candidate["kelly_sizing"] = sizing

        except Exception as e:
            log_event("stock_engine", "kelly_sizing_error", {
                "ticker": candidate["ticker"],
                "error": str(e)[:200],
            })

    return candidates


def execute_stock_buy(
    candidate: dict,
    client: AlpacaClient,
    portfolio_value: float,
    current_daily_pnl: float = 0,
) -> dict | None:
    """
    Execute a stock purchase through circuit breakers.

    Uses market orders for simplicity (stocks are liquid).
    """
    ticker = candidate["ticker"]
    shares = candidate["shares"]
    price = candidate["current_price"]

    # Safety checks
    try:
        check_paper_mode()
        check_daily_loss(current_daily_pnl)
    except CircuitBreakerTripped as e:
        log_event("stock_engine", "blocked", {"ticker": ticker, "error": str(e)})
        diary_write("strategy_agent", f"{ticker}|STOCK_BLOCKED|{e}")
        return None

    # Position size check
    position_value = price * shares
    settings = _load_settings()
    max_pct = settings["circuit_breakers"]["max_position_pct"]
    if position_value / portfolio_value > max_pct:
        shares = int(portfolio_value * max_pct / price)
        if shares < 1:
            log_event("stock_engine", "too_expensive", {"ticker": ticker, "price": price})
            return None

    reasoning = (
        f"Buying {shares} shares of {ticker} at ${price:.2f}. "
        f"Composite score {candidate['composite_score']}/13 "
        f"(trend:{candidate['trend_score']} level:{candidate['level_score']} "
        f"signal:{candidate['signal_score']}). "
        f"Support zone at {candidate['zone_level']} ({candidate['zone_touches']} touches). "
        f"Pattern: {candidate['pattern'] or 'none'}. "
        f"Target: ${candidate['target_price']:.2f}  Stop: ${candidate['stop_loss']:.2f}."
    )

    log_event("stock_engine", "executing", {
        "ticker": ticker, "shares": shares, "price": price,
        "score": candidate["composite_score"],
    }, result="pending")

    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        trading_client = client._get_trading_client()
        client.limiter.wait_if_needed()

        order = trading_client.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))

        response = {
            "id": str(order.id),
            "status": str(order.status),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": str(order.side),
        }

        log_event("stock_engine", "executed", {
            "ticker": ticker, "order_id": response["id"],
            "status": response["status"],
        }, result="success")

    except Exception as e:
        log_event("stock_engine", "failed", {"ticker": ticker, "error": str(e)}, result="failed")
        diary_write("strategy_agent", f"{ticker}|STOCK_FAILED|{e}")
        return None

    # Track position
    positions = _load_positions()
    positions.append({
        "ticker": ticker,
        "type": "stock",
        "status": "open",
        "shares": shares,
        "entry_price": price,
        "target_price": candidate["target_price"],
        "stop_loss": candidate["stop_loss"],
        "order_id": response["id"],
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "composite_score": candidate["composite_score"],
    })
    _save_positions(positions)

    # Memory
    remember_trade_decision(
        ticker=ticker, trade_type="stock_buy",
        details={"shares": shares, "price": price, "score": candidate["composite_score"]},
        reasoning=reasoning,
    )
    diary_write("strategy_agent",
        f"{ticker}|STOCK_BUY|{shares}sh@${price:.2f}|"
        f"score_{candidate['composite_score']}/13|{candidate['pattern'] or 'no_pattern'}")

    # Calibration — record prediction for accuracy tracking
    try:
        from lib.stock_calibration import record_prediction
        record_prediction(
            ticker=ticker,
            composite_score=candidate["composite_score"],
            kronos_direction=candidate.get("kronos_direction"),
            kronos_expected_return=candidate.get("kronos_expected_return"),
            news_sentiment=candidate.get("news_sentiment"),
            pattern=candidate.get("pattern"),
            momentum_score=candidate.get("momentum_score", 0),
            entry_price=price,
            target_price=candidate.get("target_price", 0),
            stop_loss=candidate.get("stop_loss", 0),
            bayesian_win_prob=candidate.get("bayesian_win_prob"),
            bayesian_sources=candidate.get("bayesian_sources"),
        )
    except Exception:
        pass  # Non-critical — don't block trades on calibration errors

    return response


def check_stock_exits(
    client: AlpacaClient,
    daily_data: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    Check open stock positions for exit signals:
    - Hit target price (take profit at resistance)
    - Hit stop loss (cut losses)
    - Trailing stop (lock in profits as price rises)
    - Bearish reversal signal at resistance
    - Momentum death (MACD bearish cross + RSI overbought)
    """
    positions = _load_positions()
    stock_positions = [p for p in positions if p.get("type") == "stock" and p.get("status") == "open"]

    if not stock_positions:
        return []

    strategy = _load_strategy()
    stock_cfg = strategy.get("stock_params", {})
    trailing_stop_pct = stock_cfg.get("trailing_stop_pct", 0.0)
    partial_exit_threshold = stock_cfg.get("partial_exit_threshold", 0.15)
    partial_exit_fraction = stock_cfg.get("partial_exit_fraction", 0.5)
    enable_scale_out = stock_cfg.get("enable_scale_out", True)

    broker_positions = client.get_positions()
    broker_map = {p["symbol"]: p for p in broker_positions}
    positions_changed = False

    exits = []
    for pos in stock_positions:
        ticker = pos.get("ticker", "")
        target = pos.get("target_price", 0)
        stop = pos.get("stop_loss", 0)

        bp = broker_map.get(ticker)
        if not bp:
            continue

        current_price = float(bp["current_price"])
        entry_price = pos.get("entry_price", 0)
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        # Trailing stop: ratchet stop up as price rises
        if trailing_stop_pct > 0 and current_price > entry_price:
            trailing_stop = current_price * (1 - trailing_stop_pct)
            if trailing_stop > stop:
                pos["stop_loss"] = round(trailing_stop, 2)
                stop = pos["stop_loss"]
                positions_changed = True

        exit_signal = None

        # Scale-out: sell part of position at +15% gain (lock gains, let runner ride)
        # Only fires once per position (tracked via partial_exit_taken flag)
        partial_exit_taken = pos.get("partial_exit_taken", False)
        if (enable_scale_out
                and not partial_exit_taken
                and pnl_pct >= partial_exit_threshold
                and pos.get("shares", 0) >= 2):  # Need at least 2 shares to scale out
            shares_to_sell = max(1, int(pos["shares"] * partial_exit_fraction))
            exit_signal = {
                "ticker": ticker,
                "action": "scale_out",
                "reason": f"Scale-out at +{pnl_pct:.1%}: selling {shares_to_sell}/{pos['shares']} shares",
                "pnl_pct": pnl_pct,
                "partial_shares": shares_to_sell,
            }

        # Stop loss hit
        elif current_price <= stop:
            exit_signal = {
                "ticker": ticker,
                "action": "stop_loss",
                "reason": f"Stop loss hit: ${current_price:.2f} <= ${stop:.2f}",
                "pnl_pct": pnl_pct,
            }

        # Target hit
        elif current_price >= target:
            exit_signal = {
                "ticker": ticker,
                "action": "take_profit",
                "reason": f"Target reached: ${current_price:.2f} >= ${target:.2f}",
                "pnl_pct": pnl_pct,
            }

        # Momentum death: bearish MACD cross + RSI overbought
        elif ticker in daily_data and pnl_pct > 0.02:
            df = daily_data[ticker]
            mom = analyze_momentum(df)
            if mom and mom.macd_cross == "bearish_cross" and mom.rsi > 70:
                exit_signal = {
                    "ticker": ticker,
                    "action": "momentum_exit",
                    "reason": f"Momentum death: bearish MACD cross + RSI {mom.rsi:.0f}",
                    "pnl_pct": pnl_pct,
                }

        # Bearish reversal at resistance
        if exit_signal is None and ticker in daily_data:
            df = daily_data[ticker]
            bearish_signal = get_latest_signal(df, "bearish", [
                "shooting_star", "bearish_engulfing", "evening_star",
                "gravestone_doji", "bearish_harami",
            ])
            zones_list = detect_zones(df, current_price)
            resistance = get_nearest_resistance(zones_list, entry_price)

            if bearish_signal and resistance and abs(current_price - resistance.level) / resistance.level < 0.03:
                exit_signal = {
                    "ticker": ticker,
                    "action": "bearish_reversal",
                    "reason": f"Bearish {bearish_signal.pattern} at resistance {resistance.level:.2f}",
                    "pnl_pct": pnl_pct,
                }

        if exit_signal:
            exits.append(exit_signal)
            log_event("stock_engine", "exit_signal", exit_signal)

    # Save trailing stop updates
    if positions_changed:
        _save_positions(positions)

    return exits


TRADE_HISTORY_PATH = Path(__file__).parent.parent / "data" / "trade_history.json"


def _load_trade_history() -> list[dict]:
    if not TRADE_HISTORY_PATH.exists():
        return []
    with open(TRADE_HISTORY_PATH) as f:
        return json.load(f)


def _save_trade_history(history: list[dict]):
    TRADE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def execute_stock_sell(
    ticker: str,
    client: AlpacaClient,
    reason: str = "signal",
) -> dict | None:
    """Sell all shares of a stock position."""
    positions = _load_positions()
    pos = None
    for p in positions:
        if p.get("ticker") == ticker and p.get("type") == "stock" and p.get("status") == "open":
            pos = p
            break

    if not pos:
        return None

    shares = pos.get("shares", 0)
    if shares < 1:
        return None

    # Get current price for P/L calculation
    broker_positions = client.get_positions()
    broker_map = {p["symbol"]: p for p in broker_positions}
    exit_price = float(broker_map[ticker]["current_price"]) if ticker in broker_map else pos.get("entry_price", 0)

    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        trading_client = client._get_trading_client()
        client.limiter.wait_if_needed()

        order = trading_client.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))

        response = {
            "id": str(order.id),
            "status": str(order.status),
            "symbol": order.symbol,
            "qty": str(order.qty),
        }

        # Calculate realized P/L
        entry_price = pos.get("entry_price", 0)
        realized_pnl = (exit_price - entry_price) * shares
        hold_duration = ""
        if pos.get("opened_at"):
            opened = datetime.fromisoformat(pos["opened_at"])
            hold_duration = str(datetime.now(timezone.utc) - opened)

        # Update position
        pos["status"] = "closed"
        pos["closed_at"] = datetime.now(timezone.utc).isoformat()
        pos["close_reason"] = reason
        pos["exit_price"] = exit_price
        pos["realized_pnl"] = round(realized_pnl, 2)
        _save_positions(positions)

        # Record to trade history (Hermes needs this!)
        history = _load_trade_history()
        history.append({
            "ticker": ticker,
            "type": "stock",
            "side": "sell",
            "shares": shares,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_pnl": round(realized_pnl, 2),
            "pnl_pct": round((exit_price - entry_price) / entry_price, 4) if entry_price > 0 else 0,
            "composite_score": pos.get("composite_score", 0),
            "close_reason": reason,
            "opened_at": pos.get("opened_at", ""),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "hold_duration": hold_duration,
        })
        _save_trade_history(history)

        log_event("stock_engine", "sold", {
            "ticker": ticker, "shares": shares, "reason": reason,
            "realized_pnl": round(realized_pnl, 2),
            "exit_price": exit_price,
        }, result="success")

        diary_write("strategy_agent",
            f"{ticker}|STOCK_SOLD|{shares}sh@${exit_price:.2f}|{reason}|pnl_${realized_pnl:+.2f}")

        # Calibration — record outcome for accuracy tracking
        try:
            from lib.stock_calibration import record_outcome
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            record_outcome(
                ticker=ticker,
                outcome="win" if realized_pnl > 0 else "loss",
                pnl_pct=pnl_pct,
                close_reason=reason,
            )
        except Exception:
            pass  # Non-critical

        return response

    except Exception as e:
        log_event("stock_engine", "sell_failed", {"ticker": ticker, "error": str(e)}, result="failed")
        return None


def execute_partial_stock_sell(
    ticker: str,
    shares_to_sell: int,
    client: AlpacaClient,
    reason: str = "scale_out",
) -> dict | None:
    """
    Sell a fraction of a stock position. Unlike execute_stock_sell (full close),
    this keeps the position open with reduced share count and marks it as
    partial_exit_taken=True so we only scale out once.

    Purpose: Lock in gains on big winners while letting a runner ride with
    the trailing stop.
    """
    positions = _load_positions()
    pos = None
    for p in positions:
        if p.get("ticker") == ticker and p.get("type") == "stock" and p.get("status") == "open":
            pos = p
            break

    if not pos:
        return None

    total_shares = pos.get("shares", 0)
    if shares_to_sell >= total_shares:
        # Not a partial — route to full sell
        return execute_stock_sell(ticker, client, reason)

    if shares_to_sell < 1:
        return None

    # Get current price for P/L
    broker_positions = client.get_positions()
    broker_map = {bp["symbol"]: bp for bp in broker_positions}
    exit_price = float(broker_map[ticker]["current_price"]) if ticker in broker_map else pos.get("entry_price", 0)

    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        trading_client = client._get_trading_client()
        client.limiter.wait_if_needed()

        order = trading_client.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=shares_to_sell,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))

        response = {
            "id": str(order.id),
            "status": str(order.status),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "partial": True,
        }

        # Update position — reduce shares, mark partial taken
        entry_price = pos.get("entry_price", 0)
        realized_pnl = (exit_price - entry_price) * shares_to_sell
        pos["shares"] = total_shares - shares_to_sell
        pos["partial_exit_taken"] = True
        pos["partial_exit_price"] = exit_price
        pos["partial_exit_shares"] = shares_to_sell
        pos["partial_exit_pnl"] = round(realized_pnl, 2)
        pos["partial_exit_at"] = datetime.now(timezone.utc).isoformat()
        _save_positions(positions)

        # Record partial sale in trade history (important for Hermes)
        history = _load_trade_history()
        history.append({
            "ticker": ticker,
            "type": "stock",
            "side": "partial_sell",
            "shares": shares_to_sell,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_pnl": round(realized_pnl, 2),
            "pnl_pct": round((exit_price - entry_price) / entry_price, 4) if entry_price > 0 else 0,
            "composite_score": pos.get("composite_score", 0),
            "close_reason": reason,
            "opened_at": pos.get("opened_at", ""),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "is_partial": True,
            "remaining_shares": pos["shares"],
        })
        _save_trade_history(history)

        log_event("stock_engine", "partial_sold", {
            "ticker": ticker,
            "shares_sold": shares_to_sell,
            "remaining": pos["shares"],
            "realized_pnl": round(realized_pnl, 2),
            "exit_price": exit_price,
        }, result="success")

        diary_write("strategy_agent",
            f"{ticker}|SCALE_OUT|{shares_to_sell}/{total_shares}sh@${exit_price:.2f}|"
            f"locked_${realized_pnl:+.2f}|letting_{pos['shares']}sh_ride")

        return response

    except Exception as e:
        log_event("stock_engine", "partial_sell_failed", {
            "ticker": ticker, "error": str(e)
        }, result="failed")
        return None


def run_stock_scan_and_execute(
    client: AlpacaClient,
    daily_data: dict[str, pd.DataFrame],
    weekly_data: dict[str, pd.DataFrame],
    portfolio_value: float,
    max_trades: int = 3,
) -> list[dict]:
    """
    Full stock trading pipeline: scan → score → execute.
    Also checks existing positions for exit signals (stop, target, scale-out).
    """
    strategy = _load_strategy()
    max_trades = strategy.get("stock_params", {}).get("max_trades_per_scan", max_trades)

    log_event("stock_engine", "pipeline_started", {"portfolio": portfolio_value, "max_trades": max_trades})

    results = []

    # Check exits first — free up capital for new trades
    exits = check_stock_exits(client, daily_data)
    freed_capital = False
    for exit_signal in exits:
        ticker = exit_signal["ticker"]
        action = exit_signal.get("action")
        print(f"  🔔 Exit signal: {ticker} — {exit_signal['reason']}")

        # Scale-out is a partial sale (keeps position open, smaller size)
        if action == "scale_out":
            shares = exit_signal.get("partial_shares", 0)
            resp = execute_partial_stock_sell(ticker, shares, client, "scale_out")
            if resp:
                results.append({"action": "scale_out", **exit_signal, "order": resp})
                freed_capital = True
                print(f"  💰 Partial sold {shares}x {ticker} — runner continues")
        else:
            # Full close (stop loss, target, momentum death, bearish reversal)
            resp = execute_stock_sell(ticker, client, action)
            if resp:
                results.append({"action": "sell", **exit_signal, "order": resp})
                freed_capital = True

    # After exits, wait briefly for settlement then re-fetch portfolio value
    # Paper account settles instantly; live has T+2 but buying power updates fast
    if freed_capital:
        import time
        time.sleep(2)  # Brief pause for order settlement
        try:
            account = client.get_account()
            old_pv = portfolio_value
            portfolio_value = account["portfolio_value"]
            cash_available = float(account.get("cash", 0))
            log_event("stock_engine", "capital_refreshed", {
                "old_portfolio_value": old_pv,
                "new_portfolio_value": portfolio_value,
                "cash_available": cash_available,
                "exits": len(exits),
            })
            print(f"  ♻️  Capital recycled: ${cash_available:,.2f} cash available after {len(exits)} exits")
        except Exception:
            pass  # Use original portfolio_value if refresh fails

    # Scan for new buys
    candidates = scan_for_stocks(client, daily_data, weekly_data, portfolio_value)

    if not candidates:
        log_event("stock_engine", "no_candidates", {})
        return results

    # Print momentum details for top candidates
    for c in candidates[:5]:
        mom = c.get("momentum", {})
        mom_str = ""
        if mom:
            mom_str = (f" RSI:{mom.get('rsi', '-')} MACD:{mom.get('macd_cross', '-')} "
                      f"Vol:{mom.get('volume_surge', '-')}x ROC5d:{mom.get('roc_5d', 0):+.1%}")
        print(f"  📊 {c['ticker']:5s} score={c['composite_score']}/13 "
              f"(T:{c['trend_score']} L:{c['level_score']} S:{c['signal_score']} M:{c['momentum_score']})"
              f"{mom_str}")

    # Execute top candidates
    # If we freed capital from exits, allow up to max_trades + freed_slots
    # so that one exit can be immediately replaced by one high-conviction entry
    effective_max_trades = max_trades
    if freed_capital:
        # Give ourselves extra slots equal to number of full exits (not scale-outs)
        full_exits = sum(1 for e in exits if e.get("action") != "scale_out")
        effective_max_trades = max_trades + full_exits
        if full_exits > 0:
            print(f"  🚀 Fast-recycle: allowing up to {effective_max_trades} buys this cycle (redeploying {full_exits} freed slots)")

    executed = 0
    for candidate in candidates:
        if executed >= effective_max_trades:
            break

        resp = execute_stock_buy(candidate, client, portfolio_value)
        if resp:
            results.append({"action": "buy", "candidate": candidate, "order": resp})
            executed += 1

    log_event("stock_engine", "pipeline_complete", {
        "exits": len(exits), "buys": executed,
        "scale_outs": sum(1 for e in exits if e.get("action") == "scale_out"),
        "freed_slots_redeployed": executed if freed_capital else 0,
    })

    return results
