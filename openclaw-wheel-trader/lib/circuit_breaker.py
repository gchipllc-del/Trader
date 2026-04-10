"""
Circuit Breakers — hard limits that cannot be bypassed.
If any breaker trips, trading halts until conditions clear or human intervenes.
"""

from datetime import datetime, timezone
from pathlib import Path

import yaml

from lib.audit import log_event

CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


def _load_settings() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


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


def check_daily_loss(current_daily_pnl: float, settings: dict | None = None) -> bool:
    """Check if daily loss limit has been breached."""
    if settings is None:
        settings = _load_settings()
    max_loss = settings["circuit_breakers"]["max_daily_loss"]

    if current_daily_pnl <= max_loss:
        log_event("circuit_breaker", "daily_loss_breached", {
            "current_pnl": current_daily_pnl,
            "max_loss": max_loss,
        }, result="blocked")
        raise CircuitBreakerTripped(
            f"HALTED: Daily P/L ${current_daily_pnl:.2f} breached limit ${max_loss}"
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
    """Limit concurrent pending orders."""
    if settings is None:
        settings = _load_settings()
    max_orders = settings["circuit_breakers"]["max_open_orders"]

    if current_open_count >= max_orders:
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


def run_all_checks(
    order_value: float,
    portfolio_value: float,
    current_daily_pnl: float,
    current_open_orders: int,
    contracts: int,
    last_loss_time: datetime | None = None,
) -> bool:
    """
    Run every circuit breaker check. Raises CircuitBreakerTripped on any failure.
    Call this before ANY order execution.
    """
    settings = _load_settings()

    check_paper_mode(settings)
    check_daily_loss(current_daily_pnl, settings)
    check_position_size(order_value, portfolio_value, settings)
    check_open_orders(current_open_orders, settings)
    check_contracts_per_order(contracts, settings)
    check_cooldown(last_loss_time, settings)

    log_event("circuit_breaker", "all_checks_passed", {
        "order_value": order_value,
        "daily_pnl": current_daily_pnl,
        "open_orders": current_open_orders,
        "contracts": contracts,
    }, result="success")
    return True
