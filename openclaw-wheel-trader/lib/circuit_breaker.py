"""
Circuit Breakers — hard limits that cannot be bypassed.
If any breaker trips, trading halts until conditions clear or human intervenes.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lib.audit import log_event

CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"

# Settings cache to avoid re-reading the YAML file on every breaker call
# (audit finding 2026-05-01 #6: under high order volume the disk-read
# inside check_daily_loss became a meaningful bottleneck and risked stale
# reads under concurrent edits). 60-second TTL is short enough that an
# operator config change still propagates quickly.
_SETTINGS_CACHE: dict = {"data": None, "loaded_at": 0.0}
_SETTINGS_TTL_SECONDS = 60


def _load_settings() -> dict:
    now = time.time()
    if (_SETTINGS_CACHE["data"] is not None
            and now - _SETTINGS_CACHE["loaded_at"] < _SETTINGS_TTL_SECONDS):
        return _SETTINGS_CACHE["data"]
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)
    _SETTINGS_CACHE["data"] = data
    _SETTINGS_CACHE["loaded_at"] = now
    return data


def invalidate_settings_cache() -> None:
    """Force the next _load_settings() call to re-read from disk.
    Call this after a programmatic settings change (e.g. Hermes optimizer
    rewriting strategy.yaml — though that uses a separate file)."""
    _SETTINGS_CACHE["data"] = None
    _SETTINGS_CACHE["loaded_at"] = 0.0


class CircuitBreakerTripped(Exception):
    """Raised when a circuit breaker condition is violated."""
    pass


def check_paper_mode(settings: dict | None = None):
    """CRITICAL: Ensure we're in paper mode unless explicitly migrated."""
    if settings is None:
        settings = _load_settings()
    if settings.get("mode") != "paper" and not settings.get("live_migration_approved"):
        log_event("circuit_breaker", "paper_mode_violation", {
            "mode": settings.get("mode"),
            "approved": settings.get("live_migration_approved"),
        }, result="blocked")
        raise CircuitBreakerTripped(
            "BLOCKED: Live trading not approved. Set live_migration_approved: true in settings.yaml"
        )


def check_daily_loss(
    current_daily_pnl: float,
    settings: dict | None = None,
    portfolio_value: float | None = None,
) -> bool:
    """Check if daily loss limit has been breached.

    The effective limit is the MORE RESTRICTIVE of:
      - max_daily_loss      (fixed dollar floor, always applied)
      - max_daily_loss_pct  (equity-relative, applied iff portfolio_value given)

    Backward compatible: callers that don't pass portfolio_value get the
    legacy dollar-only behaviour, which is what the existing test suite
    expects. New callers (stock_engine, order_gate) pass the live equity
    so the breaker auto-scales as the bankroll grows.
    """
    if settings is None:
        settings = _load_settings()
    cb = settings["circuit_breakers"]
    dollar_floor = cb["max_daily_loss"]

    pct_limit_dollar = None
    pct_limit = cb.get("max_daily_loss_pct")
    if pct_limit is not None and portfolio_value is not None and portfolio_value > 0:
        pct_limit_dollar = portfolio_value * pct_limit  # pct is negative, so this is negative

    # Pick the tighter (less negative = stricter) of the two
    if pct_limit_dollar is not None:
        max_loss = max(dollar_floor, pct_limit_dollar)
        limit_source = "pct" if max_loss == pct_limit_dollar else "dollar_floor"
    else:
        max_loss = dollar_floor
        limit_source = "dollar_floor"

    if current_daily_pnl <= max_loss:
        log_event("circuit_breaker", "daily_loss_breached", {
            "current_pnl": current_daily_pnl,
            "max_loss": max_loss,
            "limit_source": limit_source,
            "portfolio_value": portfolio_value,
            "dollar_floor": dollar_floor,
            "pct_limit": pct_limit,
        }, result="blocked")
        raise CircuitBreakerTripped(
            f"HALTED: Daily P/L ${current_daily_pnl:.2f} breached limit ${max_loss:.2f} ({limit_source})"
        )
    return True


def check_position_size(order_value: float, portfolio_value: float, settings: dict | None = None) -> bool:
    """Ensure no single position exceeds max allocation."""
    if settings is None:
        settings = _load_settings()
    max_pct = settings["circuit_breakers"]["max_position_pct"]
    position_pct = order_value / portfolio_value if portfolio_value > 0 else 1.0

    if position_pct > max_pct:
        log_event("circuit_breaker", "position_size_exceeded", {
            "order_value": order_value,
            "portfolio_value": portfolio_value,
            "position_pct": round(position_pct, 4),
            "max_pct": max_pct,
        }, result="blocked")
        raise CircuitBreakerTripped(
            f"BLOCKED: Position {position_pct:.1%} exceeds {max_pct:.0%} limit"
        )
    return True


def check_open_orders(current_open_count: int, settings: dict | None = None) -> bool:
    """Limit concurrent pending orders.

    ``max_open_orders`` in settings.yaml is the inclusive ceiling — a value
    of 8 allows up to 8 concurrent orders and blocks the 9th. The previous
    ``>=`` comparison blocked at the ceiling itself, which silently capped
    the user one slot short of what the config advertised.
    """
    if settings is None:
        settings = _load_settings()
    max_orders = settings["circuit_breakers"]["max_open_orders"]

    if current_open_count > max_orders:
        log_event("circuit_breaker", "max_open_orders", {
            "current": current_open_count,
            "max": max_orders,
        }, result="blocked")
        raise CircuitBreakerTripped(
            f"BLOCKED: {current_open_count} open orders, max is {max_orders}"
        )
    return True


def check_cooldown(last_loss_time: datetime | None, settings: dict | None = None) -> bool:
    """Enforce cooldown period after a losing trade."""
    if last_loss_time is None:
        return True

    if settings is None:
        settings = _load_settings()
    cooldown_min = settings["circuit_breakers"]["cooldown_after_loss_minutes"]
    elapsed = (datetime.now(timezone.utc) - last_loss_time).total_seconds() / 60

    if elapsed < cooldown_min:
        remaining = cooldown_min - elapsed
        log_event("circuit_breaker", "cooldown_active", {
            "last_loss": last_loss_time.isoformat(),
            "remaining_minutes": round(remaining, 1),
        }, result="blocked")
        raise CircuitBreakerTripped(
            f"BLOCKED: Cooldown active. {remaining:.0f} minutes remaining."
        )
    return True


def check_contracts_per_order(contracts: int, settings: dict | None = None) -> bool:
    """Limit contracts per single order."""
    if settings is None:
        settings = _load_settings()
    max_contracts = settings["circuit_breakers"]["max_contracts_per_order"]

    if contracts > max_contracts:
        log_event("circuit_breaker", "max_contracts_exceeded", {
            "requested": contracts,
            "max": max_contracts,
        }, result="blocked")
        raise CircuitBreakerTripped(
            f"BLOCKED: {contracts} contracts exceeds max {max_contracts}"
        )
    return True


def check_consecutive_losses(settings: dict | None = None) -> bool:
    """
    Streak-based kill switch (adapted from NoFx, github.com/NoFxAiOS/nofx).

    After N consecutive losing trades, refuse new entries for a cooldown
    period. This catches regime-mismatch days where the bot is fighting
    a trend it doesn't understand — even if the per-trade stop is doing
    its job, a streak of losses signals the bot's edge has temporarily
    inverted and capital preservation > new entries.

    Reads recent trade outcomes from the calibration log (most recent N
    resolved entries, where N = max_consecutive_losses + 1). Trips if
    ALL of the last N were losses. Cooldown duration is
    `streak_cooldown_hours` from settings.

    Defaults: 3 consecutive losses → 24h cooldown. Returns True when
    not tripped. Disabled when settings entry is missing OR set to 0.
    """
    if settings is None:
        settings = _load_settings()
    cb = settings.get("circuit_breakers", {})
    n_threshold = int(cb.get("max_consecutive_losses", 0) or 0)
    cooldown_hours = float(cb.get("streak_cooldown_hours", 24) or 24)

    if n_threshold <= 0:
        return True  # Feature disabled

    # Read the calibration log (lightweight) — this is the same outcome
    # source used by stock_calibration / agent_accuracy.
    try:
        from lib.stock_calibration import _load_calibration as _load_cal
        entries = _load_cal() or []
    except Exception:
        # If we can't read history, default to allowing trades.
        # Better to keep trading than to over-trip on infra errors.
        return True

    resolved = [e for e in entries if e.get("outcome") in ("win", "loss")]
    if len(resolved) < n_threshold:
        return True  # Not enough history to trip

    recent = resolved[-n_threshold:]
    all_losses = all(e.get("outcome") == "loss" for e in recent)
    if not all_losses:
        return True

    # Streak detected — check if we're still inside the cooldown window.
    last_loss_iso = recent[-1].get("resolved_at") or recent[-1].get("timestamp")
    if not last_loss_iso:
        return True  # Can't determine timing; don't block
    try:
        last_loss_time = datetime.fromisoformat(last_loss_iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return True
    elapsed_hours = (datetime.now(timezone.utc) - last_loss_time).total_seconds() / 3600
    if elapsed_hours >= cooldown_hours:
        # Cooldown expired — let the bot trade again. The streak still
        # shows in history but it's stale.
        return True

    remaining_hours = cooldown_hours - elapsed_hours
    log_event("circuit_breaker", "consecutive_loss_streak", {
        "n_losses": n_threshold,
        "tickers": [e.get("ticker") for e in recent],
        "last_loss_at": last_loss_iso,
        "cooldown_hours": cooldown_hours,
        "remaining_hours": round(remaining_hours, 2),
    }, result="blocked")
    raise CircuitBreakerTripped(
        f"BLOCKED: {n_threshold} consecutive losses "
        f"({', '.join(e.get('ticker','?') for e in recent)}) — "
        f"streak cooldown {remaining_hours:.1f}h remaining "
        f"(regime-mismatch protection; resumes after cooldown)"
    )


def check_hard_floor(current_equity: float, settings: dict | None = None) -> bool:
    """Unified hard-floor: halt entries when equity is at or below
    (hard_floor + hard_floor_buffer) from ``unified_goals.json``.

    Exits remain allowed — callers wire this only on ``order_intent="entry"``.

    Self-clearing: if currently halted, equity recovered above
    floor + 2*buffer, AND ``daily_reset_cooldown_hours`` have elapsed since
    halt, this method silently clears the halt instead of tripping.
    """
    try:
        from tradingcore.unified_goals import (
            load_goals, set_halt, clear_halt
        )
    except ImportError as e:
        # tradingcore not on path — treat as no-op rather than crash trading
        log_event("circuit_breaker", "hard_floor_unavailable",
                  {"error": str(e)[:200]}, result="degraded")
        return True

    goals = load_goals()
    tb = goals.get("traderbot", {})
    floor = float(tb.get("hard_floor", 1500.0))
    buffer_amt = float(tb.get("hard_floor_buffer", 10.0))
    effective_floor = floor + buffer_amt
    cooldown_hours = float(tb.get("daily_reset_cooldown_hours", 24))

    # Self-clear path
    if tb.get("halt_state") == "halted" and tb.get("halt_reason", "").startswith("hard_floor_breach"):
        halted_at = tb.get("halted_at")
        if halted_at and current_equity > effective_floor + 2 * buffer_amt:
            try:
                halted_dt = datetime.fromisoformat(halted_at.replace("Z", "+00:00"))
                elapsed_hours = (datetime.now(timezone.utc) - halted_dt).total_seconds() / 3600.0
                if elapsed_hours >= cooldown_hours:
                    clear_halt("traderbot")
                    log_event("circuit_breaker", "hard_floor_cleared",
                              {"equity": current_equity, "effective_floor": effective_floor,
                               "elapsed_hours": round(elapsed_hours, 2)},
                              result="success")
                    return True
            except (ValueError, TypeError) as e:
                # A corrupted halted_at would silently prevent the
                # self-clear forever. Surface it so the operator can
                # fix the field instead of being stuck halted.
                log_event("circuit_breaker", "halted_at_parse_failed",
                          {"halted_at": halted_at, "error": str(e)[:200]},
                          result="degraded")

    if current_equity <= effective_floor:
        reason = f"hard_floor_breach:{current_equity:.2f}<={effective_floor:.2f}"
        set_halt("traderbot", reason)
        log_event("circuit_breaker", "hard_floor_breach",
                  {"equity": current_equity, "floor": floor, "buffer": buffer_amt,
                   "effective_floor": effective_floor},
                  result="blocked")
        raise CircuitBreakerTripped(
            f"BLOCKED: hard floor breach — equity ${current_equity:.2f} "
            f"<= effective floor ${effective_floor:.2f} "
            f"(${floor:.2f} + ${buffer_amt:.2f} buffer). "
            f"Entries halted; exits still allowed. "
            f"Auto-clears after equity > ${effective_floor + 2*buffer_amt:.2f} "
            f"and {cooldown_hours:.0f}h cooldown."
        )
    return True


def run_all_checks(
    order_value: float,
    portfolio_value: float,
    current_daily_pnl: float,
    current_open_orders: int,
    contracts: int,
    last_loss_time: datetime | None = None,
    order_intent: str = "entry",
) -> bool:
    """
    Run every circuit breaker check. Raises CircuitBreakerTripped on any failure.
    Call this before ANY order execution.

    ``order_intent`` — "entry" (default) runs hard_floor check; "exit" skips it
    so positions can always be closed even when entries are halted.
    """
    settings = _load_settings()

    check_paper_mode(settings)
    check_daily_loss(current_daily_pnl, settings, portfolio_value=portfolio_value)
    check_position_size(order_value, portfolio_value, settings)
    check_open_orders(current_open_orders, settings)
    check_contracts_per_order(contracts, settings)
    check_cooldown(last_loss_time, settings)
    check_consecutive_losses(settings)
    if order_intent == "entry":
        check_hard_floor(portfolio_value, settings)

    log_event("circuit_breaker", "all_checks_passed", {
        "order_value": order_value,
        "daily_pnl": current_daily_pnl,
        "open_orders": current_open_orders,
        "contracts": contracts,
        "order_intent": order_intent,
    }, result="success")


def check_stock_buys_enabled() -> bool:
    """Refuse stock-buy entries when the directional layer is gated off.

    Wheel CSP/CC trades are NOT gated — only the momentum-driven stock buys.
    Call this from execute_stock_buy() ONLY, not from CSP/CC paths.
    """
    try:
        from tradingcore.unified_goals import is_stock_buys_enabled
    except ImportError:
        return True  # fail open

    enabled, reason = is_stock_buys_enabled()
    if enabled:
        return True

    log_event("circuit_breaker", "stock_buys_gated", {
        "reason": reason,
    }, result="blocked")
    raise CircuitBreakerTripped(
        f"BLOCKED: stock-buys gate disabled — {reason}. "
        "Wheel CSP/CC trades still allowed; momentum stock buys refused. "
        "Run `python main.py stock-buys-gate` for details or "
        "`python lib/stock_buys_gate.py --reset` to force-enable."
    )
