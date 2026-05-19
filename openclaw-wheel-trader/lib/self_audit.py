"""
Self-audit — pipeline funnel monitor for the trading bot.

The bot's audit log captures every stage of every trade attempt
(pipeline_started → executing → propose → validate → execute → fill).
When a single stage drops the pass-through to 0% across multiple
attempts, that's an obvious bug signal — exactly the pattern that
masked the 2026-05-13 contracts/shares conflation bug for half a day
before manual diagnosis.

This module computes the funnel over a recent window and flags
suspicious drops as ``self_audit.alert`` events with severity. Hooks
into monitor.run_monitoring_check so the bot watches itself once per
cycle.

Anomaly patterns flagged:

  * **PIPELINE_STARVED**     — many ``pipeline_started`` events, zero
                               ``step3_execute`` (or ``executed``)
  * **CONSENSUS_BLOCK**      — many ``consensus_rejected`` /
                               ``vetoed_by_*`` of same reason
  * **BREAKER_LOOP**         — same circuit-breaker tripping >N times
                               for the same ticker (config drift)
  * **MONITOR_TIMEOUTS**     — repeated ``check_timeout`` events
                               (network or upstream hang)
  * **SDK_FAILURES**         — repeated ``network_retry`` /
                               ``broker_reconcile_failed`` events
  * **DEGRADED_CYCLES**      — high ratio of ``result=degraded`` events

The audit is read-only: it never modifies state and never blocks a
trade. It only writes its own findings as audit events so an operator
(or a downstream Telegram alert hook) can act.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lib.audit import log_event

AUDIT_LOG = Path(__file__).parent.parent / "logs" / "audit_log.jsonl"


@dataclass
class FunnelStat:
    """Counts for one pipeline stage."""
    started: int = 0
    executing: int = 0
    proposed: int = 0
    validated: int = 0
    breaker_tripped: int = 0
    consensus_rejected: int = 0
    bayesian_vetoed: int = 0
    executed: int = 0
    blocked: int = 0


@dataclass
class AuditAlert:
    """A single pipeline anomaly the bot detected about itself."""
    severity: str          # "info" | "warn" | "critical"
    code: str              # PIPELINE_STARVED / CONSENSUS_BLOCK / ...
    summary: str           # Human-readable one-liner
    evidence: dict = field(default_factory=dict)


def _recent_events(hours: float = 4.0) -> list[dict]:
    """Stream-load audit events from the last ``hours``. Returns a list
    of parsed dicts (lossy on malformed lines — those are skipped)."""
    if not AUDIT_LOG.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    events: list[dict] = []
    # Read backward-ish: full file scan is cheapest at this size (~MB);
    # for huge logs we'd seek to tail, but the bot rotates daily.
    try:
        with open(AUDIT_LOG) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    ts = e.get("timestamp", "")
                    if not ts:
                        continue
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt >= cutoff:
                        events.append(e)
                except (ValueError, json.JSONDecodeError):
                    continue
    except OSError:
        return []
    return events


def compute_funnel(hours: float = 4.0) -> tuple[dict[str, FunnelStat], list[dict]]:
    """Compute per-asset-class pipeline counts and return the raw
    events too (so callers can mine them further).

    Asset class is derived from the event_type prefix:
      stock_engine.* → stocks
      csp_engine.*   → csp
      cc_engine.*    → cc
      crypto_engine.* → crypto
    """
    events = _recent_events(hours)
    funnels: dict[str, FunnelStat] = defaultdict(FunnelStat)

    # Track the last engine that emitted an event so we can attribute
    # cross-cutting events (order_gate, circuit_breaker) to the right
    # asset class. The bot's logs interleave but each pipeline burst
    # is sequential — pipeline_started → engine work → order_gate →
    # pipeline_complete. Keep a sliding "current asset" pointer.
    current_cls: str | None = None

    for e in events:
        et = e.get("event_type", "")
        action = e.get("action", "")
        result = e.get("result", "")
        cls = _classify_asset(et) or current_cls
        if cls is not None:
            current_cls = cls
        else:
            continue
        f = funnels[cls]
        if action == "pipeline_started":
            f.started += 1
        elif action == "executing":
            f.executing += 1
        elif action == "scan_complete" and result == "success":
            # Most engines also emit a scan_complete; treat as a "started"
            # signal if no explicit pipeline_started was logged.
            pass
        elif action == "step1_proposed":
            f.proposed += 1
        elif action == "step2_validated":
            f.validated += 1
        elif action == "step2_breaker_tripped":
            f.breaker_tripped += 1
        elif action == "consensus_rejected":
            f.consensus_rejected += 1
        elif action in ("bayesian_veto", "bear_veto"):
            f.bayesian_vetoed += 1
        elif action in ("step3_executed", "executed", "stock_buy_executed",
                         "csp_executed", "cc_executed"):
            f.executed += 1
        elif action == "blocked":
            f.blocked += 1

    return dict(funnels), events


def _classify_asset(event_type: str) -> str | None:
    """Map event_type to an asset class bucket for funnel grouping.

    Returns None for cross-cutting event types (order_gate,
    circuit_breaker) — callers fall back to the last-known asset class.
    """
    if event_type == "stock_engine":
        return "stocks"
    if event_type == "csp_engine":
        return "csp"
    if event_type == "cc_engine":
        return "cc"
    if event_type == "crypto_engine":
        return "crypto"
    return None


def detect_alerts(
    funnels: dict[str, FunnelStat],
    events: list[dict],
    *,
    min_attempts_to_alert: int = 5,
    consensus_concentration_threshold: float = 0.8,
    network_failure_threshold: int = 10,
    timeout_threshold: int = 2,
) -> list[AuditAlert]:
    """Look at the funnel + raw events for anomaly patterns.

    Each threshold is configurable so the operator can tune sensitivity.
    Defaults are tuned for the bot's current trade volume (~10 attempts
    per active hour).
    """
    alerts: list[AuditAlert] = []

    # 1. PIPELINE_STARVED — many starts, zero executes
    for cls, f in funnels.items():
        attempts = max(f.started, f.executing, f.proposed)
        if attempts >= min_attempts_to_alert and f.executed == 0:
            # Distinguish by where the drop happened
            if f.breaker_tripped >= attempts * 0.8:
                cause = f"circuit breaker tripped on {f.breaker_tripped}/{attempts}"
                code = "BREAKER_LOOP"
                sev = "critical"
            elif f.consensus_rejected >= attempts * 0.8:
                cause = f"consensus rejected {f.consensus_rejected}/{attempts}"
                code = "CONSENSUS_BLOCK"
                sev = "warn"
            elif f.bayesian_vetoed >= attempts * 0.8:
                cause = f"Bayesian/bear veto on {f.bayesian_vetoed}/{attempts}"
                code = "BAYESIAN_VETO_LOOP"
                sev = "warn"
            else:
                cause = "unknown — check audit log between propose and execute"
                code = "PIPELINE_STARVED"
                sev = "critical"
            alerts.append(AuditAlert(
                severity=sev, code=code,
                summary=(
                    f"{cls}: {attempts} attempt(s) → 0 executed. {cause}."
                ),
                evidence={
                    "asset_class": cls,
                    "attempts": attempts,
                    "executed": f.executed,
                    "breaker_tripped": f.breaker_tripped,
                    "consensus_rejected": f.consensus_rejected,
                    "bayesian_vetoed": f.bayesian_vetoed,
                    "blocked": f.blocked,
                },
            ))

    # 2. BREAKER_LOOP — same ticker repeatedly hitting the same breaker
    breaker_by_ticker_reason: Counter = Counter()
    for e in events:
        if e.get("action") != "step2_breaker_tripped":
            continue
        details = e.get("details") or {}
        ticker = details.get("ticker") or details.get("hash", "?")[:8]
        reason = (details.get("reason") or "")[:40]
        breaker_by_ticker_reason[(ticker, reason)] += 1
    for (ticker, reason), count in breaker_by_ticker_reason.most_common(5):
        if count >= 3:
            alerts.append(AuditAlert(
                severity="warn", code="BREAKER_LOOP",
                summary=(
                    f"{ticker}: tripped same breaker {count}× — "
                    f"likely config drift. Reason: {reason}"
                ),
                evidence={"ticker": ticker, "count": count, "reason": reason},
            ))

    # 3. MONITOR_TIMEOUTS — repeated check_timeout events indicate
    # upstream (Alpaca / data provider) hangs that the SDK isn't
    # signalling cleanly.
    timeout_count = sum(1 for e in events
                        if e.get("event_type") == "monitor"
                        and e.get("action") == "check_timeout")
    if timeout_count >= timeout_threshold:
        alerts.append(AuditAlert(
            severity="critical", code="MONITOR_TIMEOUTS",
            summary=(
                f"Monitor timed out {timeout_count}× in window — "
                f"upstream API likely unhealthy"
            ),
            evidence={"timeout_count": timeout_count},
        ))

    # 4. SDK_FAILURES — network retries piling up
    network_retry_count = sum(1 for e in events
                              if e.get("action") in ("network_retry",
                                                     "broker_reconcile_failed",
                                                     "option_price_fetch_failed"))
    if network_retry_count >= network_failure_threshold:
        alerts.append(AuditAlert(
            severity="warn", code="SDK_FAILURES",
            summary=(
                f"{network_retry_count} network retries/failures in window "
                f"— transient outage or rate-limit pressure"
            ),
            evidence={"network_retry_count": network_retry_count},
        ))

    # 5. DEGRADED_CYCLES — high ratio of result=degraded events
    total = len(events)
    degraded = sum(1 for e in events if e.get("result") == "degraded")
    if total >= 50 and degraded / total >= 0.20:
        alerts.append(AuditAlert(
            severity="warn", code="DEGRADED_CYCLES",
            summary=(
                f"{degraded}/{total} events ({degraded/total:.0%}) "
                f"flagged 'degraded' in window — multiple data sources unhealthy"
            ),
            evidence={"total": total, "degraded": degraded},
        ))

    return alerts


# ============================================================
# Additional self-audit categories beyond the pipeline funnel
# ============================================================

def check_state_reconciliation(broker_client) -> list[AuditAlert]:
    """Compare ``positions.json`` open positions against the broker's
    authoritative position list. Flag any drift.

    Catches: assignments not yet reflected locally, manual broker
    activity, stale positions.json after a crash, broker-side fills the
    bot didn't record. Distinct from ``wheel_state`` preflight (which
    only runs at scan time) — this is continuous.

    Core-holding awareness: tickers in ``wheel_strategy.yaml.core_holdings``
    are treated specially — when local says held and broker shows zero,
    the alert is upgraded to ``CORE_HOLDING_MISSING`` (critical) instead
    of ``STATE_DRIFT_BROKER`` (warn). Core holdings are forever-holds;
    their absence at the broker is operationally urgent, not "drift".
    """
    alerts: list[AuditAlert] = []
    try:
        from lib.positions_store import load_positions
        positions_path = Path(__file__).parent.parent / "data" / "positions.json"
        local = load_positions(positions_path) or []
        local_open = [p for p in local if p.get("status") == "open"]
        broker_positions = broker_client.get_positions() or []
    except Exception as e:
        # Can't read either side — log degraded, no alert.
        log_event("self_audit", "state_recon_unavailable",
                  {"error": str(e)[:200]}, result="degraded")
        return []

    # Load core_holdings from config so we can distinguish forever-holds
    # from transient screener trades. Best-effort: if config can't load,
    # treat all as transient.
    core_holdings: set[str] = set()
    try:
        import yaml
        cfg_path = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"
        with open(cfg_path) as f:
            strategy = yaml.safe_load(f) or {}
        core_holdings = {str(t).upper() for t in (strategy.get("core_holdings") or [])}
    except Exception:
        pass

    # Build (symbol, side-or-stock) → qty maps. Side matters because
    # short put and long stock can share an underlying ticker.
    def _key(p: dict) -> tuple[str, str]:
        sym = str(p.get("symbol") or p.get("ticker") or "").upper()
        return sym, str(p.get("side", "")).lower()

    broker_map: dict[tuple[str, str], float] = {}
    for bp in broker_positions:
        try:
            broker_map[_key(bp)] = float(bp.get("qty", 0) or 0)
        except (TypeError, ValueError):
            continue

    # Local positions reference share/contract counts under varying keys
    # — "shares" for stocks, "contracts" for options. Normalize.
    def _local_qty(p: dict) -> float:
        return float(p.get("shares", 0) or p.get("contracts", 0)
                     or p.get("quantity", 0) or 0)

    # Broker-only: shares we own but no local row for. Could be a
    # post-assignment ghost or a manual broker buy.
    local_tickers = {(str(p.get("ticker", "")).upper(), str(p.get("type", "stock")))
                     for p in local_open}
    for (sym, side), qty in broker_map.items():
        # Skip option symbols (15+ chars suffix) — those need OCC parsing,
        # which the wheel_state pre-flight handles separately.
        if len(sym) > 15:
            continue
        if (sym, "stock") not in local_tickers and qty != 0:
            alerts.append(AuditAlert(
                severity="warn", code="STATE_DRIFT_LOCAL",
                summary=(
                    f"Broker holds {qty:g} sh of {sym} but no local "
                    f"position record — possible silent assignment or "
                    f"manual broker action"
                ),
                evidence={"symbol": sym, "broker_qty": qty},
            ))

    # Local-only: positions.json claims open but broker shows nothing.
    # Could be a phantom from a crashed write or an order that filled
    # then was closed broker-side.
    for p in local_open:
        ticker = str(p.get("ticker", "")).upper()
        ptype = p.get("type", "stock")
        local_qty = _local_qty(p)
        if ptype != "stock":
            # Options/CSPs handled by wheel_state pre-flight.
            continue
        # Look up broker qty by ticker (stock symbol is just ticker).
        broker_qty = broker_map.get((ticker, "long"), 0) or broker_map.get((ticker, ""), 0)
        if local_qty > 0 and broker_qty == 0:
            is_core = ticker in core_holdings
            alerts.append(AuditAlert(
                severity="critical" if is_core else "warn",
                code="CORE_HOLDING_MISSING" if is_core else "STATE_DRIFT_BROKER",
                summary=(
                    (f"⚠️ CORE HOLDING MISSING: {ticker} (forever-hold) — "
                     f"local says {local_qty:g} sh, broker shows 0. "
                     f"Replenish ASAP via manual buy.")
                    if is_core else
                    f"{ticker}: local says {local_qty:g} sh open, broker shows 0 "
                    f"— phantom local position or unrecorded close"
                ),
                evidence={"ticker": ticker, "local_qty": local_qty,
                          "is_core_holding": is_core},
            ))
        elif local_qty > 0 and broker_qty > 0 and abs(local_qty - broker_qty) >= 1:
            alerts.append(AuditAlert(
                severity="warn", code="STATE_DRIFT_QTY",
                summary=(
                    f"{ticker}: local has {local_qty:g} sh, broker has "
                    f"{broker_qty:g} — partial fill or split not reconciled"
                ),
                evidence={"ticker": ticker, "local_qty": local_qty,
                          "broker_qty": broker_qty},
            ))

    return alerts


def auto_reconcile_phantom_positions(
    broker_client,
    alerts: list[AuditAlert],
) -> dict:
    """Auto-resolve state drift between positions.json and the broker.

    Handles TWO directions:

      A. STATE_DRIFT_BROKER (local says open, broker shows 0):
         The broker sold but local didn't get updated. Look up the
         most recent FILLED sell for that ticker and close the local
         record with the broker's real filled_avg_price → realistic
         P&L. If no sell is found, close flat with P&L=0 and a
         needs_review flag.

      B. STATE_DRIFT_LOCAL (broker has shares, local has no record):
         The bot bought at the broker but local didn't get updated.
         Look up the most recent FILLED buy and ADD a new open
         position to positions.json with the broker's filled_avg
         as entry price. Tagged backfilled_from_broker=True.

    Skips:
      • Core holdings (``CORE_HOLDING_MISSING``) — those are critical,
        require human attention, never auto-modify.
      • Options / CSPs — wheel_state preflight handles those.

    Both directions audit-logged. Returns counts + ticker lists.
    """
    phantom_alerts = [
        a for a in alerts
        if a.code == "STATE_DRIFT_BROKER"
    ]
    broker_only_alerts = [
        a for a in alerts
        if a.code == "STATE_DRIFT_LOCAL"
    ]
    if not phantom_alerts and not broker_only_alerts:
        return {"reconciled": 0, "no_sell_found": 0,
                "backfilled": 0, "tickers": []}

    from lib.positions_store import mutate_positions
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    reconciled = 0
    no_sell = 0
    tickers: list[str] = []
    now_iso = _dt.now(_tz.utc).isoformat()

    # Build a (ticker -> latest FILLED sell within 7 days) map in one
    # broker query, rather than firing N separate ones.
    fills_by_ticker: dict[str, dict] = {}
    try:
        # AlpacaClient wraps the SDK — pull recent orders for the
        # tickers we care about.
        target_tickers = sorted({
            str(a.evidence.get("ticker", "")).upper()
            for a in phantom_alerts
        })
        # The wrapper exposes get_orders via _get_trading_client; use
        # it directly to filter by symbols + status. Fall back to no
        # data if the SDK call raises.
        try:
            tc = broker_client._get_trading_client()
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            since = _dt.now(_tz.utc) - _td(days=7)
            broker_client.limiter.wait_if_needed()
            orders = tc.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=target_tickers,
                after=since,
                limit=200,
            )) or []
            # Pick the latest FILLED sell per ticker
            for o in orders:
                if str(o.side).split(".")[-1].lower() not in ("sell",):
                    continue
                if str(o.status).split(".")[-1].lower() != "filled":
                    continue
                sym = str(o.symbol).upper()
                cur = fills_by_ticker.get(sym)
                if cur is None or o.submitted_at > cur["submitted_at"]:
                    fills_by_ticker[sym] = {
                        "filled_avg_price": float(o.filled_avg_price or 0),
                        "qty": float(o.qty or 0),
                        "submitted_at": o.submitted_at,
                    }
        except Exception as e:
            log_event("self_audit", "auto_reconcile_order_lookup_failed",
                      {"error": str(e)[:200]}, result="degraded")
    except Exception:
        pass

    # Now mutate positions.json under the file lock; for each phantom
    # find its open record and close it.
    targets = {a.evidence.get("ticker", "").upper() for a in phantom_alerts}
    with mutate_positions() as positions:
        for p in positions:
            if p.get("status") != "open":
                continue
            ticker = str(p.get("ticker", "")).upper()
            if ticker not in targets:
                continue
            if p.get("type") != "stock":
                continue

            entry = float(p.get("entry_price", 0) or 0)
            qty = float(p.get("shares", 0) or p.get("quantity", 0) or 0)
            sell_info = fills_by_ticker.get(ticker)

            if sell_info and sell_info["filled_avg_price"] > 0:
                exit_price = sell_info["filled_avg_price"]
                p["status"] = "closed"
                p["closed_at"] = (
                    sell_info["submitted_at"].isoformat()
                    if hasattr(sell_info["submitted_at"], "isoformat")
                    else str(sell_info["submitted_at"])
                )
                p["exit_price"] = round(exit_price, 4)
                p["close_reason"] = "auto_reconcile_broker_sold"
                p["realized_pnl"] = round((exit_price - entry) * qty, 2)
                reconciled += 1
                tickers.append(ticker)
                log_event("self_audit", "auto_reconciled", {
                    "ticker": ticker,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "qty": qty,
                    "realized_pnl": p["realized_pnl"],
                    "method": "matched_broker_sell",
                }, result="success")
            else:
                # No matching sell at broker — close flat with no P&L
                # assumption. Safe default, audit-flagged for review.
                p["status"] = "closed"
                p["closed_at"] = now_iso
                p["exit_price"] = entry
                p["close_reason"] = "auto_reconcile_no_sell_found"
                p["realized_pnl"] = 0.0
                no_sell += 1
                tickers.append(ticker)
                log_event("self_audit", "auto_reconciled", {
                    "ticker": ticker,
                    "entry_price": entry,
                    "qty": qty,
                    "method": "no_sell_found_flat_close",
                    "needs_review": True,
                }, result="degraded")

    # ── Direction B: backfill broker-only positions into local ─────────
    # Broker has shares we don't track locally (bot bought but local
    # write was skipped). Look up the most recent buy, add a local
    # record with the broker's real entry data.
    backfilled = 0
    if broker_only_alerts:
        broker_only_tickers = sorted({
            str(a.evidence.get("symbol", "")).upper()
            for a in broker_only_alerts
        })

        # Pull the latest filled buy per ticker
        buys_by_ticker: dict[str, dict] = {}
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            tc = broker_client._get_trading_client()
            broker_client.limiter.wait_if_needed()
            buy_orders = tc.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=broker_only_tickers,
                limit=200,
            )) or []
            for o in buy_orders:
                if str(o.side).split(".")[-1].lower() != "buy":
                    continue
                if str(o.status).split(".")[-1].lower() != "filled":
                    continue
                sym = str(o.symbol).upper()
                cur = buys_by_ticker.get(sym)
                if cur is None or o.submitted_at > cur["submitted_at"]:
                    buys_by_ticker[sym] = {
                        "filled_avg_price": float(o.filled_avg_price or 0),
                        "qty": float(o.qty or 0),
                        "submitted_at": o.submitted_at,
                        "order_id": str(o.id),
                    }
        except Exception as e:
            log_event("self_audit", "auto_backfill_lookup_failed",
                      {"error": str(e)[:200]}, result="degraded")

        # Pull the broker's current qty for each (canonical source)
        broker_qtys: dict[str, float] = {}
        try:
            for bp in broker_client.get_positions() or []:
                sym = str(bp.get("symbol") or bp.get("ticker") or "").upper()
                broker_qtys[sym] = float(bp.get("qty", 0) or 0)
        except Exception:
            pass

        with mutate_positions() as positions:
            for sym in broker_only_tickers:
                qty = broker_qtys.get(sym, 0)
                if qty <= 0:
                    continue
                buy = buys_by_ticker.get(sym)
                if buy and buy["filled_avg_price"] > 0:
                    entry_price = buy["filled_avg_price"]
                    opened_at = (
                        buy["submitted_at"].isoformat()
                        if hasattr(buy["submitted_at"], "isoformat")
                        else str(buy["submitted_at"])
                    )
                    order_id = buy["order_id"]
                else:
                    # No buy found — use broker's avg_entry as best guess
                    entry_price = broker_qtys.get(sym + "_avg", 0) or 0.0
                    opened_at = now_iso
                    order_id = "auto_backfill_unknown"

                new_pos = {
                    "ticker": sym,
                    "type": "stock",
                    "status": "open",
                    "shares": int(qty),
                    "entry_price": round(entry_price, 4),
                    "order_id": str(order_id),
                    "opened_at": opened_at,
                    "composite_score": 0,
                    "backfilled_from_broker": True,
                    "backfill_reason": "auto_reconcile_state_drift_local",
                }
                positions.append(new_pos)
                backfilled += 1
                tickers.append(sym)
                log_event("self_audit", "auto_backfilled", {
                    "ticker": sym,
                    "qty": qty,
                    "entry_price": entry_price,
                    "method": "matched_broker_buy" if buy else "no_buy_found",
                }, result="success" if buy else "degraded")

    return {
        "reconciled": reconciled,
        "no_sell_found": no_sell,
        "backfilled": backfilled,
        "tickers": tickers,
    }


def check_config_sanity() -> list[AuditAlert]:
    """Verify config files satisfy internal invariants. Catches Hermes
    or manual edits that produced an inconsistent state."""
    import yaml
    config_dir = Path(__file__).parent.parent / "config"
    alerts: list[AuditAlert] = []

    try:
        with open(config_dir / "wheel_strategy.yaml") as f:
            strategy = yaml.safe_load(f) or {}
        with open(config_dir / "settings.yaml") as f:
            settings = yaml.safe_load(f) or {}
    except Exception as e:
        log_event("self_audit", "config_unreadable",
                  {"error": str(e)[:200]}, result="degraded")
        return [AuditAlert(
            severity="critical", code="CONFIG_UNREADABLE",
            summary=f"Could not load config files: {str(e)[:120]}",
            evidence={"error": str(e)[:200]},
        )]

    def _bad(code: str, summary: str, **ev):
        alerts.append(AuditAlert(severity="warn", code=code,
                                 summary=summary, evidence=ev))

    # CSP / CC delta + DTE ranges
    for trade_type in ("csp", "cc"):
        cfg = strategy.get(trade_type, {}) or {}
        dmin, dmax = cfg.get("delta_min"), cfg.get("delta_max")
        if dmin is not None and dmax is not None and dmin >= dmax:
            _bad("CONFIG_DELTA_RANGE",
                 f"{trade_type}.delta_min ({dmin}) >= delta_max ({dmax})",
                 trade_type=trade_type, delta_min=dmin, delta_max=dmax)
        tmin, tmax = cfg.get("dte_min"), cfg.get("dte_max")
        if tmin is not None and tmax is not None and tmin > tmax:
            _bad("CONFIG_DTE_RANGE",
                 f"{trade_type}.dte_min ({tmin}) > dte_max ({tmax})",
                 trade_type=trade_type, dte_min=tmin, dte_max=tmax)
        ymin = cfg.get("min_annualized_return")
        ymax = cfg.get("max_annualized_return")
        if ymin is not None and ymax is not None and ymin >= ymax:
            _bad("CONFIG_YIELD_RANGE",
                 f"{trade_type}.min_annualized_return ({ymin}) >= max ({ymax})",
                 trade_type=trade_type, min=ymin, max=ymax)

    # Half-Kelly cap
    kelly_frac = float(strategy.get("kelly", {}).get("fraction", 0.50))
    if kelly_frac > 0.50:
        _bad("CONFIG_KELLY_OVER_HALF",
             f"kelly.fraction is {kelly_frac} — full-Kelly risks ruin "
             f"on a single bad streak; cap is 0.5",
             kelly_fraction=kelly_frac)

    # Circuit breaker invariants
    cb = settings.get("circuit_breakers", {}) or {}
    max_pos_pct = float(cb.get("max_position_pct", 0))
    if max_pos_pct > 0.30:
        _bad("CONFIG_POSITION_CAP_TOO_HIGH",
             f"circuit_breakers.max_position_pct is {max_pos_pct} "
             f"(>0.30 — single position dominates portfolio)",
             max_position_pct=max_pos_pct)

    # CAR ceiling sanity
    car_pct = float(strategy.get("risk", {}).get("max_capital_at_risk_pct", 0.80))
    if car_pct > 1.0 or car_pct <= 0:
        _bad("CONFIG_CAR_PCT_INVALID",
             f"risk.max_capital_at_risk_pct = {car_pct} (must be in (0, 1.0])",
             car_pct=car_pct)

    # Confirmation score range
    conf = strategy.get("confirmation", {}) or {}
    score_min = conf.get("min_composite_score")
    if score_min is not None and score_min > 13:
        _bad("CONFIG_SCORE_FLOOR_UNREACHABLE",
             f"confirmation.min_composite_score = {score_min} — "
             f"universe rarely scores >13/13 (caused 2026-04-27 starvation)",
             min_composite_score=score_min)

    return alerts


def check_cron_health() -> list[AuditAlert]:
    """Check launchd for jobs that have missed their scheduled window.

    For each known job, ``launchctl list <label>`` returns:
      * PID = currently running (or "-" if not)
      * LastExitStatus = exit code of the last run
    A job that hasn't fired within 2× its expected interval is suspicious.

    We don't actually have last-firing timestamps from launchd
    (LastExitStatus is what we get), so we use the audit log as a
    proxy: when did each cron last write to the log?
    """
    import subprocess
    alerts: list[AuditAlert] = []

    # Map launchd label → expected max interval (seconds) AND
    # audit-log event filter that fires when the cron runs. Event-type
    # names verified against actual audit_log.jsonl entries — easy to
    # get wrong, so when adding a new entry here run:
    #   grep -E "event_type.*<engine>" logs/audit_log.jsonl | head
    # to confirm the actual ``(event_type, action)`` shape.
    expected = {
        # Wheel monitor every 3 min → alert at 10 min stale
        "ai.openclaw.monitor": (600, ("monitor", "check_complete")),
        # Hermes daily after market close → alert at 36h stale
        "ai.openclaw.hermes": (130_000, ("hermes", "optimization_complete")),
        # Crypto SCAN (separate from monitor) every 2h → alert at 5h stale
        "ai.openclaw.crypto": (18_000, ("crypto_engine", "scan_complete")),
    }

    # Look up the most-recent matching audit event per label.
    events = _recent_events(hours=48)  # 48h window to catch daily jobs
    now = datetime.now(timezone.utc)

    for label, (max_stale_seconds, (et, action)) in expected.items():
        # Confirm launchd knows about this job
        try:
            result = subprocess.run(
                ["launchctl", "list", label],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                alerts.append(AuditAlert(
                    severity="critical", code="CRON_UNREGISTERED",
                    summary=f"launchd has no job named {label}",
                    evidence={"label": label},
                ))
                continue
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue  # launchctl unavailable — skip silently

        # Find most recent matching event
        matching = [e for e in events
                    if e.get("event_type") == et and e.get("action") == action]
        if not matching:
            alerts.append(AuditAlert(
                severity="warn", code="CRON_NO_RECENT_RUN",
                summary=(
                    f"{label}: no '{et}:{action}' event in last 48h — "
                    f"job may not have fired"
                ),
                evidence={"label": label, "event": f"{et}:{action}"},
            ))
            continue

        latest = max(matching, key=lambda e: e.get("timestamp", ""))
        try:
            latest_dt = datetime.fromisoformat(
                latest.get("timestamp", "").replace("Z", "+00:00")
            )
            stale_seconds = (now - latest_dt).total_seconds()
            if stale_seconds > max_stale_seconds:
                sev = "critical" if stale_seconds > max_stale_seconds * 2 else "warn"
                alerts.append(AuditAlert(
                    severity=sev, code="CRON_STALE",
                    summary=(
                        f"{label}: last fired {stale_seconds/60:.0f} min ago "
                        f"(expected ≤{max_stale_seconds/60:.0f} min)"
                    ),
                    evidence={"label": label, "stale_minutes": round(stale_seconds/60, 1),
                              "max_minutes": max_stale_seconds/60},
                ))
        except (ValueError, TypeError):
            continue

    return alerts


def check_pnl_reconciliation(broker_client, *, tolerance_dollars: float = 50.0) -> list[AuditAlert]:
    """Compare summed claimed P&L in positions.json against broker
    account equity delta vs baseline. Flag drift > ``tolerance_dollars``.
    """
    alerts: list[AuditAlert] = []
    try:
        from lib.positions_store import load_positions
        positions_path = Path(__file__).parent.parent / "data" / "positions.json"
        local = load_positions(positions_path) or []
    except Exception:
        return []

    # Sum claimed unrealized + realized P&L from local positions
    claimed_pnl = 0.0
    for p in local:
        for field_name in ("unrealized_pnl", "realized_pnl", "net_profit"):
            val = p.get(field_name)
            if val is not None:
                try:
                    claimed_pnl += float(val)
                except (TypeError, ValueError):
                    pass

    # Compare to broker equity delta from baseline
    try:
        account = broker_client.get_account()
        equity = float(account.get("portfolio_value", 0) or 0)
    except Exception:
        return []

    baseline_path = Path(__file__).parent.parent / "data" / "baseline_equity.json"
    try:
        with open(baseline_path) as f:
            baseline = float(json.load(f).get("baseline_equity", equity))
    except (FileNotFoundError, json.JSONDecodeError):
        return []  # No baseline → can't reconcile, skip

    actual_delta = equity - baseline
    drift = abs(claimed_pnl - actual_delta)
    if drift > tolerance_dollars and baseline > 0:
        sev = "critical" if drift > tolerance_dollars * 4 else "warn"
        alerts.append(AuditAlert(
            severity=sev, code="PNL_DRIFT",
            summary=(
                f"P&L reconciliation drift ${drift:,.2f} "
                f"(local claims {claimed_pnl:+,.2f}, "
                f"broker delta from baseline {actual_delta:+,.2f})"
            ),
            evidence={"claimed_pnl": round(claimed_pnl, 2),
                      "broker_delta": round(actual_delta, 2),
                      "drift": round(drift, 2),
                      "tolerance": tolerance_dollars},
        ))

    return alerts


def check_memory_consistency() -> list[AuditAlert]:
    """Every position with a ``decision_drawer_id`` should resolve to
    a real MemPalace drawer. Orphans indicate a broken learning loop
    where the bot's "past outcomes" injection won't find anything to
    inject.
    """
    alerts: list[AuditAlert] = []
    try:
        from lib.positions_store import load_positions
        positions_path = Path(__file__).parent.parent / "data" / "positions.json"
        local = load_positions(positions_path) or []
    except Exception:
        return []

    orphan_ids: list[str] = []
    for p in local:
        for fld in ("decision_drawer_id", "cc_decision_drawer_id"):
            did = p.get(fld)
            if not did:
                continue
            # Check if the drawer exists. Cheap lookup via search_memory
            # — but search_memory may not return by ID; need a tighter check.
            try:
                from lib.memory_palace import search_memory
                hits = search_memory(p.get("ticker", "") or "X",
                                     wing=f"wing_{(p.get('ticker','x')).lower()}",
                                     n_results=20)
                if not any(h.get("drawer_id") == did for h in hits):
                    orphan_ids.append(did)
            except Exception:
                pass  # MemPalace not available → no audit; that's fine

    if orphan_ids:
        sev = "warn" if len(orphan_ids) < 5 else "critical"
        alerts.append(AuditAlert(
            severity=sev, code="MEM_ORPHAN_IDS",
            summary=(
                f"{len(orphan_ids)} position(s) reference drawer_ids that "
                f"MemPalace doesn't have — learning loop will be blind to them"
            ),
            evidence={"orphan_count": len(orphan_ids),
                      "sample_ids": orphan_ids[:5]},
        ))

    return alerts


def check_heartbeat_freshness(*, max_stale_minutes: int = 10) -> list[AuditAlert]:
    """Alert if the heartbeat file hasn't been touched recently.

    Belt-and-braces against the existing ``_check_for_missed_cycles``
    watchdog inside monitor.py — that one drives the kill switch; this
    one fires a much earlier warning, well before the kill threshold,
    so the operator can intervene in time.
    """
    heartbeat_path = Path(__file__).parent.parent / "data" / "heartbeat.json"
    try:
        with open(heartbeat_path) as f:
            last_iso = json.load(f).get("last_check_at", "")
        last_dt = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return []  # File never written yet — let the watchdog handle it

    stale_minutes = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
    if stale_minutes > max_stale_minutes:
        sev = "critical" if stale_minutes > max_stale_minutes * 3 else "warn"
        return [AuditAlert(
            severity=sev, code="HEARTBEAT_STALE",
            summary=(
                f"Monitor heartbeat is {stale_minutes:.0f} min old "
                f"(threshold {max_stale_minutes} min) — cycle may be hung"
            ),
            evidence={"stale_minutes": round(stale_minutes, 1),
                      "last_check_at": last_iso,
                      "threshold_minutes": max_stale_minutes},
        )]
    return []


def run_self_audit(hours: float = 4.0, *, broker_client=None,
                    auto_reconcile: bool | None = None) -> dict:
    """Compute the funnel, detect anomalies, log everything. The return
    dict goes into the monitor's summary so the operator dashboard can
    surface it.

    ``broker_client``: optional AlpacaClient. When provided, runs the
    state-reconciliation and P&L-reconciliation checks (those need
    live broker data). Skipped when None.

    ``auto_reconcile``: when True, automatically closes phantom positions
    (positions.json says open, broker shows 0) by matching against the
    broker's recent FILLED sell orders. When None, reads
    ``settings.yaml.auto_reconcile_phantoms`` (default True). Set to
    False to disable.

    Mostly idempotent and read-only. The ONE state-changing path is
    auto-reconcile when explicitly enabled — every change is audit-
    logged.
    """
    funnels, events = compute_funnel(hours)
    alerts = detect_alerts(funnels, events)

    # Cross-cutting checks. Each is exception-isolated — a failure in
    # one category never blocks the others or the trade loop.
    for check_fn, args in [
        (check_config_sanity, ()),
        (check_cron_health, ()),
        (check_memory_consistency, ()),
        (check_heartbeat_freshness, ()),
    ]:
        try:
            alerts.extend(check_fn(*args))
        except Exception as e:
            log_event("self_audit", "check_failed",
                      {"check": check_fn.__name__, "error": str(e)[:200]},
                      result="degraded")

    # Broker-dependent checks
    auto_reconcile_result: dict = {"reconciled": 0, "no_sell_found": 0,
                                    "backfilled": 0, "tickers": []}
    if broker_client is not None:
        for check_fn in (check_state_reconciliation, check_pnl_reconciliation):
            try:
                alerts.extend(check_fn(broker_client))
            except Exception as e:
                log_event("self_audit", "check_failed",
                          {"check": check_fn.__name__, "error": str(e)[:200]},
                          result="degraded")

        # Auto-reconcile phantoms if enabled. Resolves the
        # STATE_DRIFT_BROKER pattern where broker sold but local
        # positions.json still says open — instead of re-alerting every
        # 3 min for hours (current behavior), we close the local record
        # with the broker's actual filled exit price.
        try:
            if auto_reconcile is None:
                import yaml
                cfg_path = Path(__file__).parent.parent / "config" / "settings.yaml"
                with open(cfg_path) as f:
                    settings = yaml.safe_load(f) or {}
                auto_reconcile = bool(
                    settings.get("auto_reconcile_phantoms", True)
                )
            if auto_reconcile:
                auto_reconcile_result = auto_reconcile_phantom_positions(
                    broker_client, alerts,
                )
                total_actions = (auto_reconcile_result["reconciled"]
                                 + auto_reconcile_result["no_sell_found"]
                                 + auto_reconcile_result["backfilled"])
                if total_actions > 0:
                    log_event("self_audit", "auto_reconcile_summary",
                              auto_reconcile_result, result="success")
        except Exception as e:
            log_event("self_audit", "auto_reconcile_failed",
                      {"error": str(e)[:200]}, result="degraded")

    # Always log the audit even when clean — gives the operator a
    # heartbeat that the self-audit is itself running.
    log_event("self_audit", "completed", {
        "window_hours": hours,
        "n_events_scanned": len(events),
        "funnels": {cls: f.__dict__ for cls, f in funnels.items()},
        "n_alerts": len(alerts),
        "alert_codes": [a.code for a in alerts],
    })

    # One audit row per individual alert so the operator can grep by code.
    for a in alerts:
        log_event("self_audit", "alert", {
            "code": a.code,
            "severity": a.severity,
            "summary": a.summary,
            **a.evidence,
        }, result=a.severity)

    return {
        "window_hours": hours,
        "funnels": {cls: f.__dict__ for cls, f in funnels.items()},
        "alerts": [
            {"code": a.code, "severity": a.severity, "summary": a.summary,
             "evidence": a.evidence}
            for a in alerts
        ],
        "auto_reconcile": auto_reconcile_result,
    }
