"""
Wheel state machine — global pre-flight check.

The per-trade agents (strategy/risk/compliance/bear) are myopic: each
decision is reviewed in isolation against ``positions.json`` records.
That works for green-field trades but doesn't catch state drift between
``positions.json`` and the broker's actual position view. Real failure
modes we've seen:

  * 2026-04-28 BAC double-buy — a stale ``positions.json`` row let the
    scanner re-buy a position the broker already held.
  * "Phantom assignment" — broker shows long shares but ``positions.json``
    still has the CSP marked open, leaving the bot unaware it should now
    be selling CCs.
  * Uncovered short calls — a CC remains on the books after the
    underlying shares were called away or sold elsewhere.

This module reads the broker's authoritative position list, classifies
every underlying into a wheel stage, and raises ``IllegalWheelState``
on any inconsistency. Call ``classify_book()`` once per scan cycle
**before** any csp/cc/stock engine runs. If it raises, the scan halts
and an operator alert fires — much better than silently compounding
the drift with another bad trade.

The five legal stages per underlying:

    flat                    — no positions, scanner is free to open a CSP
    short_put               — short put(s) outstanding, no shares yet
    long_shares             — shares held (often via assignment), no CC yet
    long_shares_with_cc     — full wheel in motion, CC sold against shares
    long_shares_with_csp    — shares held + new CSP at a lower strike on
                              same ticker (legal pyramid). Don't open more.

Anything else — long puts, long calls, short equity, uncovered short
calls, more contracts shorted than shares held — is illegal.

Pattern inspired by alpacahq/options-wheel core/state_manager.py
(2026-04 release). Adapted to a deterministic check that can run as a
pre-flight rather than a per-trade decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lib.audit import log_event


class IllegalWheelState(Exception):
    """Raised when the broker's positions don't form a legal wheel state.

    The scan should HALT and alert the operator — placing more orders
    on top of an illegal state is exactly how the BAC double-buy
    happened. Attribute ``per_underlying`` carries the diagnostic map
    so the operator can see what's wrong without re-running anything.
    """

    def __init__(self, message: str, per_underlying: dict[str, "WheelState"]):
        super().__init__(message)
        self.per_underlying = per_underlying


WheelStage = Literal[
    "flat",
    "short_put",
    "long_shares",
    "long_shares_with_cc",
    "long_shares_with_csp",
    "illegal",
]


@dataclass
class WheelLeg:
    """One position row from the broker, normalized to wheel vocabulary."""
    symbol: str
    underlying: str
    qty: float                       # signed: negative for short
    market_value: float              # signed: negative for short
    kind: Literal["stock", "put", "call"]
    strike: float | None = None      # only for options
    expiration: str | None = None    # ISO YYYY-MM-DD


@dataclass
class WheelState:
    """Aggregate wheel state for one underlying."""
    underlying: str
    stage: WheelStage
    legs: list[WheelLeg] = field(default_factory=list)
    illegal_reason: str | None = None
    # Capital metrics for the risk-budget check downstream
    short_put_collateral: float = 0.0   # Σ strike × 100 × |qty|
    long_share_value: float = 0.0       # Σ market_value of long stock legs
    short_call_count: int = 0           # number of short call contracts
    long_share_count: int = 0           # number of shares held long


# ── OCC symbol parsing ───────────────────────────────────────────────

# OCC option symbol format: <ROOT><YYMMDD><C|P><STRIKE*1000 zero-padded to 8>
# Example: SPY250620C00500000 → SPY, exp 2025-06-20, call, strike 500.00
_OCC_SUFFIX_LEN = 15  # YYMMDD + C/P + 8-digit strike


def parse_occ_symbol(symbol: str) -> tuple[str, str, Literal["put", "call"], float] | None:
    """Decode an OCC option symbol.

    Returns ``(underlying, expiration_iso, option_type, strike)`` or
    ``None`` if ``symbol`` doesn't look like OCC (i.e. it's a stock or
    crypto symbol).
    """
    if len(symbol) <= _OCC_SUFFIX_LEN:
        return None
    suffix = symbol[-_OCC_SUFFIX_LEN:]
    date_part, side_part, strike_part = suffix[:6], suffix[6], suffix[7:]
    if not (date_part.isdigit() and side_part in ("C", "P") and strike_part.isdigit()):
        return None
    underlying = symbol[:-_OCC_SUFFIX_LEN]
    # YYMMDD → 20YY-MM-DD (good through year 2099)
    yy, mm, dd = date_part[:2], date_part[2:4], date_part[4:6]
    expiration = f"20{yy}-{mm}-{dd}"
    option_type: Literal["put", "call"] = "put" if side_part == "P" else "call"
    strike = int(strike_part) / 1000.0
    return underlying, expiration, option_type, strike


def _normalize_leg(broker_pos: dict) -> WheelLeg:
    """Broker row → WheelLeg with kind/strike/expiration filled in."""
    symbol = broker_pos["symbol"]
    qty = float(broker_pos.get("qty", 0))
    market_value = float(broker_pos.get("market_value", 0))
    side = str(broker_pos.get("side", "")).lower()
    if side == "short" and qty > 0:
        qty = -qty
        market_value = -abs(market_value)

    parsed = parse_occ_symbol(symbol)
    if parsed is None:
        return WheelLeg(
            symbol=symbol, underlying=symbol, qty=qty, market_value=market_value,
            kind="stock",
        )
    underlying, expiration, option_type, strike = parsed
    return WheelLeg(
        symbol=symbol, underlying=underlying, qty=qty, market_value=market_value,
        kind=option_type, strike=strike, expiration=expiration,
    )


# ── Classification ───────────────────────────────────────────────────

def _classify_one(underlying: str, legs: list[WheelLeg]) -> WheelState:
    """Reduce a list of legs for one underlying to a single WheelState."""
    state = WheelState(underlying=underlying, stage="flat", legs=legs)

    long_shares = 0
    short_shares = 0
    short_puts: list[WheelLeg] = []
    short_calls: list[WheelLeg] = []
    long_options: list[WheelLeg] = []

    for leg in legs:
        if leg.kind == "stock":
            if leg.qty > 0:
                long_shares += leg.qty
                state.long_share_value += leg.market_value
            else:
                short_shares += abs(leg.qty)
        elif leg.kind == "put":
            if leg.qty < 0:
                short_puts.append(leg)
                # collateral the broker holds against this short put
                state.short_put_collateral += (leg.strike or 0) * 100 * abs(leg.qty)
            else:
                long_options.append(leg)
        elif leg.kind == "call":
            if leg.qty < 0:
                short_calls.append(leg)
            else:
                long_options.append(leg)

    state.long_share_count = int(long_shares)
    state.short_call_count = sum(int(abs(l.qty)) for l in short_calls)

    # ── Illegal-state detection ──────────────────────────────────────

    if short_shares > 0:
        state.stage = "illegal"
        state.illegal_reason = (
            f"short equity position ({short_shares} shares) — wheel strategy "
            f"never shorts stock; investigate manual order or unexpected fill"
        )
        return state

    if long_options:
        state.stage = "illegal"
        opt_summary = ", ".join(
            f"{l.kind}@{l.strike}/{l.expiration}" for l in long_options
        )
        state.illegal_reason = (
            f"long option leg(s) present ({opt_summary}) — wheel strategy "
            f"only sells options; investigate"
        )
        return state

    contracts_short_call = sum(int(abs(l.qty)) for l in short_calls)
    if contracts_short_call * 100 > long_shares:
        state.stage = "illegal"
        state.illegal_reason = (
            f"uncovered short call(s): {contracts_short_call} contract(s) but "
            f"only {long_shares} shares held (need {contracts_short_call * 100})"
        )
        return state

    # ── Legal-state classification ───────────────────────────────────

    has_short_put = len(short_puts) > 0
    has_short_call = len(short_calls) > 0
    has_shares = long_shares > 0

    if not has_short_put and not has_short_call and not has_shares:
        state.stage = "flat"
    elif has_shares and has_short_call and has_short_put:
        # Legal but uncommon: shares + CC + new CSP at lower strike
        state.stage = "long_shares_with_cc"  # treat CC as dominant signal
    elif has_shares and has_short_call:
        state.stage = "long_shares_with_cc"
    elif has_shares and has_short_put:
        state.stage = "long_shares_with_csp"
    elif has_shares:
        state.stage = "long_shares"
    elif has_short_put:
        state.stage = "short_put"
    elif has_short_call:
        # Defensive: we ruled out uncovered short calls above, so reaching
        # here would mean zero shares + short call simultaneously, which
        # the uncovered check already rejected. Belt-and-braces.
        state.stage = "illegal"
        state.illegal_reason = "short call with no underlying shares"

    return state


def classify_book(
    broker_positions: list[dict],
    *,
    raise_on_illegal: bool = True,
) -> dict[str, WheelState]:
    """Classify every underlying in the broker's position list.

    Args:
        broker_positions: Result of ``AlpacaClient.get_positions()`` —
            each dict has ``symbol``, ``qty``, ``market_value``, ``side``.
        raise_on_illegal: If True (default), raise ``IllegalWheelState``
            when any underlying lands in stage="illegal". Disable for
            read-only diagnostics (dashboard rendering, etc.).

    Returns:
        Map of ``underlying → WheelState``. Empty dict if the broker
        holds no positions.

    Raises:
        IllegalWheelState: if ``raise_on_illegal`` and at least one
            underlying is in an illegal state. The exception carries
            the full classification dict so callers can log everything.
    """
    if not broker_positions:
        return {}

    # Group legs by underlying
    legs_by_underlying: dict[str, list[WheelLeg]] = {}
    for raw in broker_positions:
        leg = _normalize_leg(raw)
        legs_by_underlying.setdefault(leg.underlying, []).append(leg)

    book: dict[str, WheelState] = {
        u: _classify_one(u, legs) for u, legs in legs_by_underlying.items()
    }

    illegal = {u: s for u, s in book.items() if s.stage == "illegal"}
    if illegal:
        log_event("wheel_state", "illegal_state_detected", {
            "underlyings": list(illegal.keys()),
            "reasons": {u: s.illegal_reason for u, s in illegal.items()},
        }, result="failed")
        if raise_on_illegal:
            summary = "; ".join(
                f"{u}: {s.illegal_reason}" for u, s in illegal.items()
            )
            raise IllegalWheelState(
                f"Illegal wheel state on {len(illegal)} underlying(s) — {summary}",
                per_underlying=book,
            )

    log_event("wheel_state", "classified", {
        "underlyings": len(book),
        "stages": {s.stage: 1 for s in book.values()},
    })
    return book


# ── Capital-at-risk aggregation ──────────────────────────────────────

def total_capital_at_risk(book: dict[str, WheelState]) -> dict:
    """Sum collateral + long-share market value across the entire book.

    The result feeds the global capital-at-risk gate in ``risk_agent``.

    Returns:
        ``{"short_put_collateral", "long_share_value", "total"}``
    """
    short_put_collateral = sum(s.short_put_collateral for s in book.values())
    long_share_value = sum(s.long_share_value for s in book.values())
    return {
        "short_put_collateral": short_put_collateral,
        "long_share_value": long_share_value,
        "total": short_put_collateral + long_share_value,
    }
