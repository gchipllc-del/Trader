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


def run_self_audit(hours: float = 4.0) -> dict:
    """Compute the funnel, detect anomalies, log everything. The return
    dict goes into the monitor's summary so the operator dashboard can
    surface it.

    Idempotent and read-only: this function never modifies positions,
    config, or any other state.
    """
    funnels, events = compute_funnel(hours)
    alerts = detect_alerts(funnels, events)

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
    }
