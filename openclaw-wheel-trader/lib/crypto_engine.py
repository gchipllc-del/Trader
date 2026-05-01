"""
Crypto Engine — Tier 2 (2026-04-25).

A deliberately simple 24/7 trading path for BTC/USD, ETH/USD, SOL/USD via
Alpaca. Mirrors the stock engine's structure (scan -> screen -> Kelly size
-> circuit breakers -> execute) but cuts the parts that don't apply to
crypto:

    - No earnings filter (crypto has no earnings)
    - No PDT rule (crypto is exempt)
    - No bayesian/kronos heavy machinery (those are tuned on stock data —
      using them on crypto without recalibration would mislead Hermes)
    - Allows fractional shares (Alpaca supports notional crypto orders)

Strategy logic is intentionally simple — Hermes can tune as resolved
trades accumulate:
    - Trend: price above 20-day MA, MA rising, recent ROC positive
    - Momentum: RSI in [min_rsi, max_rsi] (avoid oversold + overbought)
    - Sizing: half-Kelly with the same calibration curve as stocks
    - Exit: 7% stop / 15% target / 4% trailing

Risk management (shared with stocks):
    - Daily loss circuit breaker (lib/circuit_breaker.check_daily_loss)
    - Sector concentration NOT applied (crypto is its own asset class —
      diversification math is different from stocks)
    - Per-position cap: crypto_params.max_position_pct
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import yaml

import numpy as np
import pandas as pd

from lib.alpaca_client import AlpacaClient
from lib.audit import log_event
from lib.circuit_breaker import check_paper_mode, check_daily_loss, CircuitBreakerTripped
from lib.kelly import kelly_position_size
from lib.order_gate import (
    OrderIntent,
    step1_propose,
    step2_validate,
    step3_execute,
    submit_close,
)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"
# File-locked positions store (Wave 3 #15). POSITIONS_PATH is module-local
# so tests can monkeypatch it.
from lib.positions_store import (
    POSITIONS_PATH,
    load_positions as _store_load,
    save_positions as _store_save,
    mutate_positions as _store_mutate,
)


def mutate_positions():
    """Module-local mutate that honors crypto_engine.POSITIONS_PATH overrides."""
    return _store_mutate(POSITIONS_PATH)


def _load_strategy() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_positions() -> list[dict]:
    """Locked snapshot of positions.json (Wave 3 #15 — see positions_store)."""
    return _store_load(POSITIONS_PATH)


def _save_positions(positions: list[dict]) -> None:
    """Atomic-overwrite under exclusive lock. Prefer mutate_positions()
    over a load + save pair to avoid the read-then-write race."""
    _store_save(positions, POSITIONS_PATH)


# ── Indicators ───────────────────────────────────────────────────

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Standard RSI (Wilder smoothing). Returns 50 on insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    diffs = np.diff(closes)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard MACD: (line, signal, histogram). Returns NaN-prefixed arrays."""
    s = pd.Series(closes)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    line = (ema_fast - ema_slow).values
    sig = pd.Series(line).ewm(span=signal, adjust=False).mean().values
    hist = line - sig
    return line, sig, hist


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average True Range. Returns 0 on insufficient data."""
    if len(closes) < period + 1:
        return 0.0
    tr = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1]),
    ])
    return float(pd.Series(tr).rolling(period).mean().iloc[-1])


def _bollinger_pctb(closes: np.ndarray, period: int = 20, stdev: float = 2.0) -> float:
    """Bollinger %B = (close - lower) / (upper - lower). 0=at lower, 1=at upper."""
    if len(closes) < period:
        return 0.5
    s = pd.Series(closes)
    ma = s.rolling(period).mean().iloc[-1]
    sd = s.rolling(period).std().iloc[-1]
    upper = ma + stdev * sd
    lower = ma - stdev * sd
    if upper == lower:
        return 0.5
    return float((closes[-1] - lower) / (upper - lower))


def _trend_score(df: pd.DataFrame) -> tuple[int, dict]:
    """
    Multi-feature trend score, 0-10. Tier A (2026-04-25) — adapted from
    Intelligent-Trading-Bot's feature engineering pattern.

    Each component is a 0/1 flag; sum is the score. The OLD score (0-3)
    used: above_ma20, ma20_rising, roc_7d_positive. The NEW expansion
    keeps those three and adds 7 confirming features so a "10" represents
    genuine, multi-confirmed trend with healthy momentum and volume.

    Components (each +1):
      1. Close above MA20                         (basic trend)
      2. MA20 rising over 5 days                  (trend slope)
      3. 7-day ROC positive                       (short-term momentum)
      4. 14-day ROC positive                      (medium-term confirmation)
      5. MACD line > signal line                  (trend confirmation)
      6. MACD histogram increasing                (momentum acceleration)
      7. Bollinger %B in [0.40, 0.85]             (healthy zone, not extended)
      8. 5d avg volume > 20d avg volume           (genuine vs drift)
      9. Close > MA20 by ≥ 1 ATR                  (volatility-normalized strength)
     10. RSI in [40, 65]                          (no overbought/oversold)
    """
    closes = df["close"].values
    if len(closes) < 30:
        return 0, {"reason": "insufficient_history"}

    highs = df["high"].values if "high" in df.columns else closes
    lows = df["low"].values if "low" in df.columns else closes
    volumes = df["volume"].values if "volume" in df.columns else np.ones_like(closes)

    score = 0
    d: dict = {}

    # 1+2: MA20 trend
    s = pd.Series(closes)
    ma20 = s.rolling(20).mean().values
    d["close"] = float(closes[-1])
    d["ma20"] = float(ma20[-1])
    if closes[-1] > ma20[-1]:
        score += 1
        d["above_ma20"] = True
    if ma20[-1] > ma20[-5]:
        score += 1
        d["ma20_rising"] = True

    # 3+4: ROC
    roc7 = (closes[-1] - closes[-7]) / closes[-7] if closes[-7] > 0 else 0
    roc14 = (closes[-1] - closes[-14]) / closes[-14] if len(closes) >= 14 and closes[-14] > 0 else 0
    d["roc_7d"] = round(float(roc7), 4)
    d["roc_14d"] = round(float(roc14), 4)
    if roc7 > 0:
        score += 1
        d["roc_7d_positive"] = True
    if roc14 > 0:
        score += 1
        d["roc_14d_positive"] = True

    # 5+6: MACD
    try:
        macd_line, macd_sig, macd_hist = _macd(closes)
        d["macd_hist"] = round(float(macd_hist[-1]), 6)
        if macd_line[-1] > macd_sig[-1]:
            score += 1
            d["macd_bullish"] = True
        if len(macd_hist) >= 3 and macd_hist[-1] > macd_hist[-3]:
            score += 1
            d["macd_hist_rising"] = True
    except Exception:
        pass

    # 7: Bollinger %B
    pctb = _bollinger_pctb(closes)
    d["bb_pctb"] = round(float(pctb), 4)
    if 0.40 <= pctb <= 0.85:
        score += 1
        d["bb_healthy"] = True

    # 8: Volume thrust
    try:
        v5 = volumes[-5:].mean()
        v20 = volumes[-20:].mean()
        d["vol_5d"] = round(float(v5), 2)
        d["vol_20d"] = round(float(v20), 2)
        if v5 > v20:
            score += 1
            d["volume_thrust"] = True
    except Exception:
        pass

    # 9: ATR-normalized strength
    atr = _atr(highs, lows, closes)
    d["atr"] = round(float(atr), 4)
    if atr > 0 and (closes[-1] - ma20[-1]) >= atr:
        score += 1
        d["atr_strong"] = True

    # 10: RSI band
    rsi_val = _rsi(closes)
    d["rsi"] = round(float(rsi_val), 2)
    if 40 <= rsi_val <= 65:
        score += 1
        d["rsi_healthy"] = True

    return score, d


# ── Scan ─────────────────────────────────────────────────────────

def scan_crypto(
    client: AlpacaClient,
    portfolio_value: float,
    current_daily_pnl: float = 0.0,
) -> list[dict]:
    """
    Run the crypto scan + screening pipeline.

    Returns a ranked list of candidate dicts ready for execution. Does NOT
    place orders — caller handles that via execute_crypto_buy().
    """
    strategy = _load_strategy()
    crypto_cfg = strategy.get("crypto_params", {})
    tickers = strategy.get("tickers_crypto", [])

    if not crypto_cfg.get("enabled", True) or not tickers:
        log_event("crypto_engine", "scan_skipped", {
            "enabled": crypto_cfg.get("enabled"),
            "tickers": len(tickers),
        })
        return []

    print("\n" + "=" * 50)
    print("  CRYPTO SCAN")
    print("=" * 50)
    print(f"  Portfolio: ${portfolio_value:.2f}  Daily P/L: ${current_daily_pnl:.2f}")
    print(f"  Universe: {tickers}")

    # Pre-trade circuit breakers (paper mode + daily loss)
    try:
        check_paper_mode()
        check_daily_loss(current_daily_pnl, portfolio_value=portfolio_value)
    except CircuitBreakerTripped as e:
        print(f"  HALTED: {e}")
        log_event("crypto_engine", "halted", {"error": str(e)})
        return []

    # Skip already-held positions (max_concurrent gate)
    positions = _load_positions()
    held = {
        p["ticker"] for p in positions
        if p.get("status") in ("open", "assigned") and p.get("type") == "crypto"
    }
    max_concurrent = crypto_cfg.get("max_concurrent", 2)
    if len(held) >= max_concurrent:
        print(f"  Max crypto positions reached ({len(held)}/{max_concurrent})")
        return []

    # Fetch bars
    bars = client.get_crypto_bars(tickers, timeframe="1Day", days_back=120)
    if not bars:
        print("  No crypto data returned.")
        return []

    candidates: list[dict] = []
    # Tier A 2026-04-25: trend_score is now 0-10 (was 0-3). Default 6
    # = 60% of features bullish. RSI band kept as hard gate too.
    min_trend = crypto_cfg.get("min_trend_score", 6)
    min_rsi = crypto_cfg.get("min_rsi", 40)
    max_rsi = crypto_cfg.get("max_rsi", 65)

    # LLM ensemble forecast (Tier A.2). Optional — gracefully degrades to
    # zero-weight if no providers respond or module missing. Each
    # candidate gets an llm_probability (probability the asset gains
    # crypto_params.target_pct over the holding window).
    try:
        from lib.crypto_llm import forecast_crypto_target
    except Exception:
        forecast_crypto_target = None

    base_target_pct = crypto_cfg.get("target_pct", 0.03)
    base_stop_pct = crypto_cfg.get("stop_loss_pct", 0.015)

    # Cost-edge gate (Tier B B4) — skip trades whose net edge after spread
    # and slippage is below min_net_edge.
    min_net_edge = float(crypto_cfg.get("min_net_edge", 0.025))
    spread_est = float(crypto_cfg.get("spread_estimate", 0.002))
    slippage_est = float(crypto_cfg.get("slippage_estimate", 0.001))
    round_trip_cost = (spread_est + slippage_est) * 2  # entry + exit

    # ATR-adaptive switches (Tier B B6)
    atr_adaptive = crypto_cfg.get("atr_adaptive_enabled", True)
    vol_high = float(crypto_cfg.get("volatility_high", 0.04))
    vol_low = float(crypto_cfg.get("volatility_low", 0.015))
    atr_high_mult = float(crypto_cfg.get("atr_high_multiplier", 1.5))
    atr_low_mult = float(crypto_cfg.get("atr_low_multiplier", 0.7))

    # Dynamic Kelly bounds (Tier B B5)
    k_base = float(crypto_cfg.get("kelly_fraction_base", 0.30))
    k_high = float(crypto_cfg.get("kelly_fraction_high_conviction", 0.65))
    k_low = float(crypto_cfg.get("kelly_fraction_low_conviction", 0.20))

    for sym, df in bars.items():
        if sym in held:
            print(f"  {sym}: already held, skip")
            continue
        if df is None or df.empty:
            continue

        trend_score, trend_details = _trend_score(df)
        rsi = trend_details.get("rsi", _rsi(df["close"].values))
        current_price = float(df["close"].iloc[-1])

        passes_trend = trend_score >= min_trend
        passes_rsi = min_rsi <= rsi <= max_rsi

        # Compact feature display for visibility into why a candidate passed
        flags = []
        for k in ("above_ma20","ma20_rising","macd_bullish","macd_hist_rising",
                  "bb_healthy","volume_thrust","atr_strong","rsi_healthy"):
            if trend_details.get(k):
                flags.append(k.split("_")[0][:3])

        line = (
            f"  {sym:<10} ${current_price:>10.2f}  trend={trend_score}/10  "
            f"RSI={rsi:.1f}  feats=[{','.join(flags) or '-'}]  "
            f"{'PASS' if (passes_trend and passes_rsi) else 'skip'}"
        )
        print(line)

        if not (passes_trend and passes_rsi):
            continue

        # ── B6: ATR-adaptive target/stop sizing ─────────────────
        atr = float(trend_details.get("atr", 0))
        atr_pct = atr / current_price if current_price > 0 else 0
        if atr_adaptive and atr_pct > 0:
            if atr_pct >= vol_high:
                vol_mult = atr_high_mult
                vol_label = "HIGH-VOL"
            elif atr_pct <= vol_low:
                vol_mult = atr_low_mult
                vol_label = "LOW-VOL"
            else:
                vol_mult = 1.0
                vol_label = "normal-vol"
        else:
            vol_mult = 1.0
            vol_label = "fixed"
        target_pct_eff = base_target_pct * vol_mult
        stop_pct_eff = base_stop_pct * vol_mult
        if vol_mult != 1.0:
            print(f"     📐 {sym} vol_mult={vol_mult} ({vol_label}, atr/price={atr_pct:.4f}) "
                  f"→ target {target_pct_eff:.4f}, stop {stop_pct_eff:.4f}")

        target_price = current_price * (1.0 + target_pct_eff)
        stop_loss = current_price * (1.0 - stop_pct_eff)

        # ── B4: Cost-edge gate ──────────────────────────────────
        net_edge = target_pct_eff - round_trip_cost
        if net_edge < min_net_edge:
            print(f"     ⚠ {sym} net edge ({net_edge:.4f}) < min ({min_net_edge:.4f}) — skip")
            log_event("crypto_engine", "edge_gate_veto", {
                "symbol": sym, "target_pct": target_pct_eff,
                "round_trip_cost": round_trip_cost, "net_edge": net_edge,
                "min_net_edge": min_net_edge,
            })
            continue

        # LLM forecast on top of technical score
        llm_prob = None
        llm_reasoning = None
        if forecast_crypto_target is not None:
            try:
                llm_result = forecast_crypto_target(
                    symbol=sym,
                    df=df,
                    target_pct=target_pct,
                    horizon_days=30,
                    trend_details=trend_details,
                )
                if llm_result:
                    llm_prob = llm_result.get("probability")
                    llm_reasoning = llm_result.get("reasoning", "")[:200]
                    print(f"     🤖 LLM: prob={llm_prob:.2f}  ({llm_reasoning[:80]}...)")
            except Exception as e:
                log_event("crypto_engine", "llm_forecast_failed",
                          {"symbol": sym, "error": str(e)[:200]})

        # Composite for kelly sizing — keep 0-13 scale matching stock kelly:
        # baseline 3 + trend_score (0-10 scaled to 0-7) + llm boost (0-3)
        scaled_trend = round(trend_score * 0.7, 1)  # 0-10 -> 0-7
        llm_boost = 0
        if llm_prob is not None:
            # llm_prob in [0,1]; >0.6 is bullish. Boost = 0 at p=0.5, 3 at p=1.0
            llm_boost = max(0, int((llm_prob - 0.5) * 6))
        composite = 3 + scaled_trend + llm_boost  # 3 to ~13

        # ── B5: Dynamic Kelly fraction by conviction ────────────
        # High conviction: composite >= 10 AND llm_prob >= 0.70
        # Low conviction:  composite < 7 OR (llm_prob is not None AND llm_prob < 0.55)
        # Otherwise: base
        if composite >= 10 and (llm_prob or 0) >= 0.70:
            dyn_kelly = k_high
            conviction_label = "HIGH"
        elif composite < 7 or (llm_prob is not None and llm_prob < 0.55):
            dyn_kelly = k_low
            conviction_label = "LOW"
        else:
            dyn_kelly = k_base
            conviction_label = "BASE"

        sizing = kelly_position_size(
            portfolio_value=portfolio_value,
            current_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            composite_score=composite,
            max_position_pct=crypto_cfg.get("max_position_pct", 0.20),
            fraction=dyn_kelly,  # B5: per-trade conviction-adapted
        )
        print(f"     💪 {sym} conviction={conviction_label}  kelly_frac={dyn_kelly}  "
              f"composite={composite}  llm_prob={llm_prob}")

        # Crypto uses NOTIONAL (fractional) sizing, not integer shares.
        # Kelly's `position_value` is `int(shares) * price` which floors to 0
        # when crypto unit price > the kelly $ allocation (e.g. BTC at $77k
        # vs a $313 budget). Compute notional from the *percent* directly.
        pct = float(sizing.get("pct_of_portfolio", 0) or 0)
        if pct <= 0:
            print(f"     {sym}: kelly skip — {sizing.get('reason','no_size')}")
            continue
        notional = pct * portfolio_value
        if notional < 1.0:
            continue  # Alpaca's crypto minimum is $1 notional

        candidates.append({
            "ticker": sym,
            "type": "crypto",
            "current_price": current_price,
            "target_price": round(target_price, 2),
            "stop_loss": round(stop_loss, 2),
            "trailing_stop_pct": crypto_cfg.get("trailing_stop_pct", 0.04),
            "composite_score": composite,
            "trend_score": trend_score,
            "trend_details": trend_details,
            "rsi": round(rsi, 2),
            "llm_probability": llm_prob,
            "llm_reasoning": llm_reasoning,
            "kelly_sizing": sizing,
            "notional": round(notional, 2),
        })

    # Rank by composite then trend
    candidates.sort(key=lambda c: (c["composite_score"], c["trend_score"]), reverse=True)

    # Limit to per-scan cap and remaining slots
    max_per_scan = crypto_cfg.get("max_trades_per_scan", 1)
    slots = max_concurrent - len(held)
    candidates = candidates[: min(max_per_scan, slots)]

    log_event("crypto_engine", "scan_complete", {
        "passed": len(candidates),
        "tickers": [c["ticker"] for c in candidates],
        "held": list(held),
    })

    return candidates


# ── Execute ──────────────────────────────────────────────────────

def execute_crypto_buy(
    candidate: dict,
    client: AlpacaClient,
    portfolio_value: float = 0.0,
    current_daily_pnl: float = 0.0,
    current_open_orders: int = 0,
    dry_run: bool = False,
) -> dict | None:
    """
    Place a notional (fractional) market buy on Alpaca for a crypto candidate
    via the order_gate pipeline (propose → validate → execute).

    portfolio_value/current_daily_pnl/current_open_orders feed step2_validate
    so the same circuit breakers that protect stocks also protect crypto.
    """
    sym = candidate["ticker"]
    notional = float(candidate["notional"])
    current_price = float(candidate["current_price"])
    score = int(candidate.get("composite_score", 0))

    log_event("crypto_engine", "execute_attempt", {
        "ticker": sym,
        "notional": notional,
        "price": current_price,
        "dry_run": dry_run,
    })

    if dry_run:
        print(f"     DRY RUN — would buy {sym} notional ${notional:.2f} (~{notional/current_price:.6f} units)")
        return {"dry_run": True, "ticker": sym, "notional": notional}

    intent = OrderIntent(
        ticker=sym,
        side="buy",
        order_type="market",
        asset_type="crypto",
        quantity=notional / current_price if current_price > 0 else 0.0,
        notional=notional,
        limit_price=current_price,
        reason=f"crypto_scan_score_{score}",
        composite_score=score,
    )

    try:
        intent = step1_propose(intent)
        # Crypto candidates pre-clear strategy gates (composite ≥ 10 in scan_crypto).
        # Use min_composite_score=0 here so the scoring scale difference between
        # equity (0-9) and crypto (0-10+) doesn't artificially fail validation —
        # the entry score floor is enforced upstream in scan_crypto.
        step2_validate(
            intent,
            portfolio_value=portfolio_value,
            current_daily_pnl=current_daily_pnl,
            current_open_orders=current_open_orders,
            min_composite_score=0,
        )
        response = step3_execute(intent, client)

        order_id = response.get("id", "")

        # Wave 2 #9: poll for terminal status before recording the position.
        # The submit response often shows pending_new with filled_qty=0;
        # blindly recording it leaves a zombie entry if the order is later
        # rejected (e.g., insufficient buying power, market closed).
        try:
            final = client.wait_for_fill(order_id, timeout_seconds=10)
        except Exception as e:
            log_event("crypto_engine", "wait_for_fill_failed",
                      {"ticker": sym, "order_id": order_id,
                       "error": str(e)[:200]}, result="degraded")
            final = response

        status = final.get("status", "unknown")
        filled_qty = float(final.get("filled_qty") or 0)
        filled_avg_raw = final.get("filled_avg_price")
        filled_avg = float(filled_avg_raw) if filled_avg_raw else current_price

        if filled_qty <= 0 or status in ("rejected", "canceled", "cancelled",
                                          "expired", "suspended"):
            log_event("crypto_engine", "execute_no_fill", {
                "ticker": sym, "order_id": str(order_id), "status": status,
                "filled_qty": filled_qty,
            }, result="failed")
            print(f"     NO FILL: {sym} order_id={order_id} status={status}")
            return None

        # Append under exclusive lock to avoid clobbering concurrent
        # stock_engine / monitor writes (Wave 3 #15).
        with mutate_positions() as positions:
            positions.append({
                "ticker": sym,
                "type": "crypto",
                "status": "open",
                "shares": filled_qty,
                "entry_price": filled_avg,
                "target_price": candidate["target_price"],
                "stop_loss": candidate["stop_loss"],
                "trailing_stop_pct": candidate["trailing_stop_pct"],
                "order_id": str(order_id),
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "composite_score": score,
                "notional_at_entry": notional,
            })

        log_event("crypto_engine", "executed", {
            "ticker": sym,
            "notional": notional,
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg,
            "order_id": str(order_id),
            "status": str(status),
        }, result="success")

        print(f"     EXECUTED: {sym} order_id={order_id} status={status} "
              f"~{filled_qty:.6f} @ ${filled_avg:.2f}")

        return {
            "ticker": sym,
            "order_id": str(order_id),
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg,
            "status": str(status),
        }
    except (CircuitBreakerTripped, ValueError) as e:
        log_event("crypto_engine", "execute_blocked", {
            "ticker": sym,
            "reason": str(e)[:200],
        }, result="blocked")
        print(f"     BLOCKED: {sym} — {e}")
        return None
    except Exception as e:
        log_event("crypto_engine", "execute_failed", {
            "ticker": sym,
            "error": str(e)[:200],
        }, result="failed")
        print(f"     FAILED: {sym} — {e}")
        return None


# ── Top-level entry point ────────────────────────────────────────

def scan_and_trade_crypto(
    client: AlpacaClient,
    portfolio_value: float,
    current_daily_pnl: float = 0.0,
    dry_run: bool = False,
) -> dict:
    """Single entry-point used by main.py and run_crypto_scan.sh."""
    candidates = scan_crypto(client, portfolio_value, current_daily_pnl)

    if not candidates:
        print("  No crypto trades this cycle.")
        return {"executed": 0, "candidates": []}

    print("\n" + "-" * 50)
    print(f"  EXECUTING {len(candidates)} CRYPTO TRADE(S)")
    print("-" * 50)

    # Snapshot open-order count once for the cycle so each candidate's
    # validation reflects the same broker state.
    try:
        current_open_orders = len(client.get_open_orders() or [])
    except Exception:
        current_open_orders = 0

    results = []
    executed = 0
    for cand in candidates:
        res = execute_crypto_buy(
            cand, client,
            portfolio_value=portfolio_value,
            current_daily_pnl=current_daily_pnl,
            current_open_orders=current_open_orders,
            dry_run=dry_run,
        )
        if res:
            executed += 1
            results.append(res)
            current_open_orders += 1

    print(f"\n  Crypto cycle complete: {executed} executed")
    return {"executed": executed, "candidates": candidates, "results": results}


# ── 24/7 Monitor (Tier B) ────────────────────────────────────────

def monitor_crypto_positions(client: AlpacaClient, dry_run: bool = False) -> dict:
    """
    24/7 crypto position checker — fires every 60s via launchd.

    For each open crypto position, checks (in order):
      1. Stop loss hit → close full position
      2. Target hit → close full position
      3. Trailing stop hit → close full position
      4. Partial-exit threshold hit + not yet partialed → close half
      5. Update high-water mark for trailing-stop ratchet

    The stock monitor (lib/monitor.py) handles stocks during market
    hours. This runs round-the-clock and ONLY touches positions with
    type=="crypto" — no overlap, no race conditions.
    """
    positions = _load_positions()
    open_crypto = [
        p for p in positions
        if p.get("status") == "open" and p.get("type") == "crypto"
    ]

    if not open_crypto:
        return {"checked": 0, "exits": 0}

    # Pull live prices for held tickers
    held_tickers = list({p["ticker"] for p in open_crypto})
    bars = client.get_crypto_bars(held_tickers, timeframe="1Day", days_back=2)

    exits: list[dict] = []
    partials: list[dict] = []

    # Build a price map (latest close per symbol)
    prices: dict[str, float] = {}
    for sym, df in bars.items():
        if df is not None and not df.empty:
            prices[sym] = float(df["close"].iloc[-1])

    now = datetime.now(timezone.utc).isoformat()
    changed = False

    for pos in open_crypto:
        sym = pos["ticker"]
        price = prices.get(sym)
        if price is None:
            continue

        entry = float(pos.get("entry_price", 0))
        stop = float(pos.get("stop_loss", 0))
        target = float(pos.get("target_price", 0))
        trailing_pct = float(pos.get("trailing_stop_pct", 0.01))
        high_water = float(pos.get("high_water_mark", entry))

        # Update high-water mark
        if price > high_water:
            high_water = price
            pos["high_water_mark"] = high_water
            changed = True

        gain_pct = (price - entry) / entry if entry > 0 else 0.0
        trail_stop_price = high_water * (1.0 - trailing_pct)

        exit_reason = None
        if price <= stop:
            exit_reason = "stop_loss"
        elif price >= target:
            exit_reason = "target_hit"
        elif gain_pct > 0 and price <= trail_stop_price:
            exit_reason = "trailing_stop"

        if exit_reason:
            exits.append({
                "ticker": sym,
                "reason": exit_reason,
                "exit_price": price,
                "entry_price": entry,
                "gain_pct": round(gain_pct, 4),
                "shares": float(pos.get("shares", 0)),
            })
            if not dry_run:
                _close_crypto_position(pos, price, exit_reason, client)
            changed = True
            continue

        # Partial exit at threshold (half off, mark partialed)
        partial_thr = float(pos.get("partial_exit_threshold",
                                    _load_strategy()
                                    .get("crypto_params", {})
                                    .get("partial_exit_threshold", 0.015)))
        if (gain_pct >= partial_thr) and not pos.get("partial_exited"):
            partials.append({
                "ticker": sym,
                "gain_pct": round(gain_pct, 4),
                "exit_price": price,
            })
            if not dry_run:
                _partial_exit_crypto(pos, price, client)
                pos["partial_exited"] = True
                pos["partial_exit_at"] = now
                changed = True

    if changed and not dry_run:
        _save_positions(positions)

    log_event("crypto_engine", "monitor_check", {
        "checked": len(open_crypto),
        "exits": len(exits),
        "partials": len(partials),
        "exit_details": exits,
        "partial_details": partials,
    })

    if exits or partials:
        print(f"  [crypto-monitor] {len(exits)} exit(s), {len(partials)} partial(s)")
        for e in exits:
            print(f"    🔴 {e['ticker']} {e['reason']} @ ${e['exit_price']:.2f} ({e['gain_pct']*100:+.2f}%)")
        for p in partials:
            print(f"    🟡 {p['ticker']} partial @ ${p['exit_price']:.2f} ({p['gain_pct']*100:+.2f}%)")

    return {"checked": len(open_crypto), "exits": len(exits), "partials": len(partials)}


def _close_crypto_position(pos: dict, price: float, reason: str, client: AlpacaClient) -> None:
    """Submit a market sell for the full crypto position via order_gate.submit_close."""
    sym = pos["ticker"]
    qty = float(pos.get("shares", 0))
    if qty <= 0:
        return

    intent = OrderIntent(
        ticker=sym,
        side="sell",
        order_type="market",
        asset_type="crypto",
        quantity=qty,
        limit_price=price,
        reason=f"crypto_close_{reason}",
    )
    try:
        response = submit_close(intent, client)
        order_id = response.get("id", "")

        pos["status"] = "closed"
        pos["closed_at"] = datetime.now(timezone.utc).isoformat()
        pos["exit_price"] = price
        pos["close_reason"] = reason
        pos["close_order_id"] = str(order_id)
        pnl = (price - float(pos.get("entry_price", 0))) * qty
        pos["realized_pnl"] = round(pnl, 4)

        log_event("crypto_engine", "position_closed", {
            "ticker": sym,
            "reason": reason,
            "exit_price": price,
            "qty": qty,
            "realized_pnl": pos["realized_pnl"],
            "order_id": str(order_id),
        }, result="success")
    except Exception as e:
        log_event("crypto_engine", "close_failed",
                  {"ticker": pos.get("ticker"), "error": str(e)[:200]},
                  result="failed")


def _partial_exit_crypto(pos: dict, price: float, client: AlpacaClient) -> None:
    """Sell half the position at +partial_threshold via order_gate.submit_close."""
    sym = pos["ticker"]
    qty = float(pos.get("shares", 0))
    sell_qty = qty * 0.5
    if sell_qty <= 0:
        return

    intent = OrderIntent(
        ticker=sym,
        side="sell",
        order_type="market",
        asset_type="crypto",
        quantity=sell_qty,
        limit_price=price,
        reason="crypto_partial_exit",
    )
    try:
        response = submit_close(intent, client)
        order_id = response.get("id", "")

        partial_pnl = (price - float(pos.get("entry_price", 0))) * sell_qty
        pos["shares"] = qty - sell_qty
        pos["partial_pnl_realized"] = round(partial_pnl, 4)

        log_event("crypto_engine", "partial_exit", {
            "ticker": sym,
            "exit_price": price,
            "sold_qty": sell_qty,
            "remaining_qty": pos["shares"],
            "partial_pnl": pos["partial_pnl_realized"],
            "order_id": str(order_id),
        }, result="success")
    except Exception as e:
        log_event("crypto_engine", "partial_exit_failed",
                  {"ticker": pos.get("ticker"), "error": str(e)[:200]},
                  result="failed")
