"""
Feature Status — one-command visibility into the 5 GitHub-sourced upgrades.

Each function gathers live state for one feature and returns a dict with
two keys we render to the operator:
    title    : human-readable feature name
    status   : "active" | "dormant" | "pending_data" | "error"
    fields   : list of (label, value) tuples for the terminal table
    note     : optional one-line context

`gather_all()` runs all five gatherers under exception isolation — any
single gatherer failure shows as `status="error"` without breaking the
others.

This is read-only / pure observability. Never modifies state.

Call from main.py:
    from lib.feature_status import gather_all, render_report
    print(render_report(gather_all()))
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parent.parent


def _safe_call(fn, *args, **kwargs) -> dict[str, Any]:
    """Run a gatherer with exception isolation."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {
            "title": fn.__name__.replace("_status_", "").replace("_", " ").title(),
            "status": "error",
            "fields": [("error", str(e)[:200])],
            "note": "gather failed — see logs",
        }


# ── Feature 1: Weighted bull/bear agents ─────────────────────────────

def status_agent_weights() -> dict[str, Any]:
    from lib.agent_accuracy import get_agent_weights, _read_log
    w = get_agent_weights()
    rows = _read_log()
    resolved = [r for r in rows if r.get("outcome") in ("win", "loss")]
    last_resolution = None
    if resolved:
        last_resolution = sorted(
            resolved, key=lambda r: r.get("resolved_at", ""), reverse=True,
        )[0].get("resolved_at", "")

    is_default = w.get("is_default", True)
    status = "pending_data" if is_default else "active"
    fields = [
        ("bull weight", w["bull_weight"]),
        ("bear weight", w["bear_weight"]),
        ("bull accuracy",
            f"{w.get('bull_accuracy', 'n/a')}  (n={w.get('n_bull', 0)})"),
        ("bear accuracy",
            f"{w.get('bear_accuracy', 'n/a')}  (n={w.get('n_bear', 0)})"),
        ("total decisions logged", len(rows)),
        ("resolved", len(resolved)),
    ]
    if last_resolution:
        fields.append(("last resolved", last_resolution[:19]))

    note = w.get("reason", "")
    if is_default and not note:
        note = "weights stay at 0.5/0.5 until 5+ resolved trades per agent"
    return {
        "title": "#1 Weighted bull/bear agents (ai-hedge-fund pattern)",
        "status": status,
        "fields": fields,
        "note": note,
    }


# ── Feature 2: Streak kill switch ────────────────────────────────────

def status_streak_killer() -> dict[str, Any]:
    import yaml
    settings = yaml.safe_load(open(_ROOT / "config" / "settings.yaml"))
    cb = settings.get("circuit_breakers", {})
    threshold = int(cb.get("max_consecutive_losses", 0) or 0)
    cooldown_hours = float(cb.get("streak_cooldown_hours", 24) or 24)

    if threshold == 0:
        return {
            "title": "#2 Streak kill switch (NoFx pattern)",
            "status": "dormant",
            "fields": [("max_consecutive_losses", "0 (disabled)")],
            "note": "set max_consecutive_losses > 0 to enable",
        }

    # Inspect recent outcomes for current streak
    from lib.stock_calibration import _load_calibration
    entries = _load_calibration() or []
    resolved = [e for e in entries if e.get("outcome") in ("win", "loss")]
    current_streak = 0
    streak_kind = None
    for e in reversed(resolved):
        outcome = e.get("outcome")
        if streak_kind is None:
            streak_kind = outcome
            current_streak = 1
        elif outcome == streak_kind:
            current_streak += 1
        else:
            break

    # Cooldown active?
    cooldown_active = False
    remaining_hours = 0.0
    if streak_kind == "loss" and current_streak >= threshold:
        last_loss = resolved[-1].get("resolved_at") or resolved[-1].get("timestamp")
        if last_loss:
            try:
                last_t = datetime.fromisoformat(last_loss.replace("Z", "+00:00"))
                elapsed = (datetime.now(timezone.utc) - last_t).total_seconds() / 3600
                remaining_hours = max(0.0, cooldown_hours - elapsed)
                cooldown_active = remaining_hours > 0
            except Exception:
                pass

    fields = [
        ("threshold (consecutive losses)", threshold),
        ("cooldown window", f"{cooldown_hours}h"),
        ("current streak", f"{current_streak} × {streak_kind or 'none'}"),
        ("cooldown active", "YES — pausing trades" if cooldown_active else "no"),
    ]
    if cooldown_active:
        fields.append(("hours remaining", round(remaining_hours, 2)))

    status = "tripped" if cooldown_active else "armed"
    note = ""
    if cooldown_active:
        note = "Bot will refuse new entries until cooldown expires"
    elif streak_kind == "loss" and current_streak == threshold - 1:
        note = f"⚠ one more loss = cooldown trip"
    return {
        "title": "#2 Streak kill switch (NoFx pattern)",
        "status": status,
        "fields": fields,
        "note": note,
    }


# ── Feature 3: Vol-aware sizing ──────────────────────────────────────

def status_vol_sizing() -> dict[str, Any]:
    import yaml
    strategy = yaml.safe_load(open(_ROOT / "config" / "wheel_strategy.yaml"))
    cfg = strategy.get("vol_aware_sizing", {})
    enabled = bool(cfg.get("enabled", True))

    fields = [
        ("enabled", enabled),
        ("current window", f"{cfg.get('current_window', 20)} bars"),
        ("baseline window", f"{cfg.get('baseline_window', 60)} bars"),
        ("floor / ceiling", f"{cfg.get('floor', 0.5)} → {cfg.get('ceiling', 1.0)}"),
        ("kronos forecast", cfg.get("use_kronos_forecast", False)),
    ]

    # Scan recent audit events for vol-sizing tags. We look at the
    # in-process audit log for any "executing" events with [VOL×...]
    # but those are stdout-only — instead, look at any
    # vol_sizing_failed / step1_proposed for the meta dict if present.
    # Simpler: count recent positions whose kelly_sizing has vol_meta.
    try:
        with open(_ROOT / "data" / "positions.json") as _f:
            positions = json.load(_f)
        recent = sorted(
            positions, key=lambda p: p.get("opened_at", ""), reverse=True,
        )[:10]
        with_vol = [p for p in recent if p.get("kelly_sizing", {}).get("vol_multiplier") is not None]
        downsized = [p for p in with_vol if p["kelly_sizing"].get("vol_multiplier", 1.0) < 1.0]
        fields.append(("recent positions with vol meta", f"{len(with_vol)}/{len(recent)}"))
        fields.append(("recent downsized by vol", len(downsized)))
    except Exception:
        fields.append(("recent positions with vol meta", "n/a"))

    status = "active" if enabled else "dormant"
    return {
        "title": "#3 Vol-aware sizing (Adaptive-Vol-Regime pattern)",
        "status": status,
        "fields": fields,
        "note": "every Kelly-sized trade is multiplied by sqrt(baseline/current_vol)",
    }


# ── Feature 4: BM25 memory retrieval ─────────────────────────────────

def status_bm25_memory() -> dict[str, Any]:
    from lib.memory_palace import PALACE_DIR

    drawers_path = PALACE_DIR / "drawers.jsonl"
    drawer_count = 0
    if drawers_path.exists():
        try:
            with open(drawers_path) as f:
                for line in f:
                    if line.strip():
                        drawer_count += 1
        except Exception:
            pass

    # Detect upstream tiers
    has_chroma = False
    has_vec = False
    try:
        import chromadb  # noqa: F401
        has_chroma = (PALACE_DIR / "chroma").exists()
    except ImportError:
        pass
    try:
        from lib import memory_vec
        has_vec = memory_vec.semantic_search_available()
    except Exception:
        pass

    active_tier = (
        "ChromaDB" if has_chroma else
        "sqlite-vec" if has_vec else
        "BM25 (JSONL)"
    )

    fields = [
        ("drawers indexed", drawer_count),
        ("active retrieval tier", active_tier),
        ("BM25 path always available", drawer_count > 0),
        ("BM25 k1 / b", "1.5 / 0.75 (paper default)"),
    ]
    note = (
        "BM25 fires as fallback when higher tiers miss/aren't installed; "
        "for trade memories it often beats vector similarity on typed content"
    )
    if drawer_count == 0:
        status = "pending_data"
        note = "no drawers yet — populate via diary_write() or remember_*()"
    else:
        status = "active"

    return {
        "title": "#4 BM25 memory retrieval (TradingAgents v0.2.0 pattern)",
        "status": status,
        "fields": fields,
        "note": note,
    }


# ── Feature 5: Sector rotation ───────────────────────────────────────

def status_sector_rotation() -> dict[str, Any]:
    import yaml
    from lib.sector_rotation import get_sleeve_preferences

    strategy = yaml.safe_load(open(_ROOT / "config" / "wheel_strategy.yaml"))
    cfg = strategy.get("sector_rotation", {})
    enabled = bool(cfg.get("enabled", True))
    activation = float(cfg.get("activation_bankroll", 5000.0))

    # Try to get current bankroll + regime
    bankroll = None
    try:
        from lib.alpaca_client import AlpacaClient
        client = AlpacaClient()
        acct = client.get_account()
        bankroll = float(acct.get("portfolio_value", 0) or 0)
    except Exception:
        pass
    regime = "unknown"
    try:
        from lib.memory_palace import get_current_regime
        regime = get_current_regime() or "unknown"
    except Exception:
        pass

    fields = [
        ("enabled", enabled),
        ("activation bankroll", f"${activation:,.0f}"),
        ("current bankroll", f"${bankroll:,.2f}" if bankroll else "n/a"),
        ("current regime", regime),
        ("max bonus swing", f"±{cfg.get('max_bonus_swing', 0.25):.0%}"),
    ]

    is_dormant = (not enabled) or (bankroll is None) or (bankroll < activation)
    status = "dormant" if is_dormant else "active"

    if not is_dormant:
        prefs = get_sleeve_preferences(regime, cfg)
        fields.append(("active sleeve weights", str(prefs)))
        note = f"actively biasing candidates by sleeve for {regime} regime"
    else:
        gap = activation - (bankroll or 0)
        note = (
            f"sleeves classified but scoring untouched; "
            f"${gap:,.0f} more bankroll to activate"
            if bankroll is not None else
            "dormant (could not fetch bankroll)"
        )

    return {
        "title": "#5 Sector rotation (FinRL-X Adaptive Rotation pattern)",
        "status": status,
        "fields": fields,
        "note": note,
    }


# ── Aggregator + renderer ────────────────────────────────────────────

def gather_all() -> list[dict[str, Any]]:
    return [
        _safe_call(status_agent_weights),
        _safe_call(status_streak_killer),
        _safe_call(status_vol_sizing),
        _safe_call(status_bm25_memory),
        _safe_call(status_sector_rotation),
    ]


_STATUS_BADGE = {
    "active":       "[ACTIVE]      ",
    "armed":        "[ARMED]       ",
    "dormant":      "[DORMANT]     ",
    "pending_data": "[PENDING DATA]",
    "tripped":      "[TRIPPED ⚠]   ",
    "error":        "[ERROR ✗]     ",
}


def render_report(features: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("=" * 75)
    lines.append("  OPENCLAW FEATURE STATUS — 5 GitHub-sourced upgrades")
    lines.append("=" * 75)
    for feat in features:
        status = feat.get("status", "error")
        badge = _STATUS_BADGE.get(status, f"[{status}]")
        lines.append("")
        lines.append(f"  {badge}  {feat['title']}")
        for label, value in feat.get("fields", []):
            lines.append(f"      {label:<35} {value}")
        if feat.get("note"):
            lines.append(f"      → {feat['note']}")
    lines.append("")
    lines.append("=" * 75)
    return "\n".join(lines)
