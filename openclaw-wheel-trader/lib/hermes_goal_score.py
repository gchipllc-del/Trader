"""Goal-driven scoring for Hermes — gives the optimizer an explicit
'distance to target' metric so every tuning experiment can be evaluated
against actual goal progress (not just generic PnL).

From the video 2 framework: criterion #3 is "well-defined goal" — success
must be measurable in concrete numbers. Hermes uses this module to compute:

  - goal_distance_pct : fraction of the remaining gap to target
  - velocity_per_day  : average $/day toward target over the rolling window
  - days_to_target_at_velocity : straight-line ETA
  - drawdown_from_peak_pct : current equity vs the peak since the anchor

Reads the unified_goals.json file (cross-bot truth source) and the local
baseline_equity.json + trade_history.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "data" / "baseline_equity.json"
TRADE_HISTORY_PATH = ROOT / "data" / "trade_history.json"
POSITIONS_PATH = ROOT / "data" / "positions.json"


def _load_json(p: Path, default):
    if not p.exists():
        return default
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _load_unified_goals() -> dict:
    try:
        from tradingcore.unified_goals import load_goals
        return load_goals()
    except Exception:
        return {}


def _current_equity_fallback() -> float | None:
    """If unified_goals doesn't have a fresh equity value, fall back to
    the local baseline_equity.json."""
    data = _load_json(BASELINE_PATH, {})
    eq = data.get("baseline_equity")
    if eq is None:
        return None
    try:
        return float(eq)
    except (TypeError, ValueError):
        return None


def compute_goal_metrics(window_days: int = 30) -> dict:
    """End-to-end metrics Hermes uses to score itself.

    All values are JSON-serializable so they slot into the experiment
    ledger as baseline/post snapshots without further transformation.
    """
    goals = _load_unified_goals()
    tb = goals.get("traderbot", {}) if isinstance(goals, dict) else {}

    anchor = float(tb.get("anchor", 1500.0))
    target = float(tb.get("target", 5000.0))
    current = tb.get("current_equity")
    if current is None or float(current) <= 0:
        current = _current_equity_fallback()
    if current is None:
        current = anchor
    current = float(current)

    # Goal distance: 0.0 = at target, 1.0 = at anchor (no progress)
    gap_total = max(target - anchor, 1e-9)
    gap_remaining = max(target - current, 0.0)
    goal_distance_pct = min(1.0, gap_remaining / gap_total)
    progress_pct = 1.0 - goal_distance_pct

    # Per-day velocity. 2026-05-23 fix: trade_history.json is sparse
    # (~13 entries) while positions.json carries the actual closed-trade
    # outcomes (~1000+ rows from auto_reconcile flows). Prefer positions
    # when both exist; fall back to trade_history.
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    positions_raw = _load_json(POSITIONS_PATH, [])
    positions = (
        positions_raw.get("positions", positions_raw)
        if isinstance(positions_raw, dict) else positions_raw
    )
    pos_window = []
    for p in positions or []:
        if p.get("status") != "closed":
            continue
        closed_ts = p.get("closed_at")
        if not closed_ts:
            continue
        try:
            dt = datetime.fromisoformat(str(closed_ts).replace("Z", "+00:00"))
            if dt >= cutoff:
                pos_window.append(p)
        except (ValueError, TypeError):
            continue

    if pos_window:
        window = pos_window
        pnls = [float(p.get("realized_pnl") or 0.0) for p in window]
        source = "positions.json"
    else:
        history = _load_json(TRADE_HISTORY_PATH, [])
        if isinstance(history, dict):
            history = history.get("trades", [])
        window = []
        for t in history:
            closed_ts = (
                t.get("closed_at") or t.get("completed_at") or t.get("opened_at")
            )
            if not closed_ts:
                continue
            try:
                dt = datetime.fromisoformat(str(closed_ts).replace("Z", "+00:00"))
                if dt >= cutoff:
                    window.append(t)
            except (ValueError, TypeError):
                continue
        pnls = [
            float(t.get("total_pnl") or t.get("realized_pnl") or 0.0)
            for t in window
        ]
        source = "trade_history.json"

    pnl_window = sum(pnls)
    wins = sum(1 for v in pnls if v > 0)
    losses = sum(1 for v in pnls if v < 0)
    win_rate = (wins / (wins + losses)) if (wins + losses) > 0 else None
    velocity_per_day = pnl_window / max(window_days, 1)

    if velocity_per_day > 0 and gap_remaining > 0:
        days_to_target = gap_remaining / velocity_per_day
    else:
        days_to_target = None

    # Drawdown-from-peak — peak equity since the anchor date. Walk
    # cumulative realized PnL across ALL closed positions (not just the
    # window) and track the high-water mark.
    peak = anchor
    cum = anchor
    if positions:
        sorted_pos = sorted(
            [p for p in positions if p.get("status") == "closed"
             and p.get("closed_at")],
            key=lambda p: p.get("closed_at", ""),
        )
        for p in sorted_pos:
            try:
                cum += float(p.get("realized_pnl") or 0.0)
                if cum > peak:
                    peak = cum
            except (TypeError, ValueError):
                continue
    if peak <= 0:
        peak = max(current, anchor)
    drawdown_from_peak = max(0.0, (peak - current) / peak) if peak > 0 else 0.0

    return {
        "anchor": round(anchor, 2),
        "current_equity": round(current, 2),
        "target": round(target, 2),
        "gap_remaining": round(gap_remaining, 2),
        "goal_distance_pct": round(goal_distance_pct, 6),
        "progress_pct": round(progress_pct, 6),
        "rolling_window_days": window_days,
        "rolling_pnl": round(pnl_window, 2),
        "rolling_30d_pnl": round(pnl_window, 2),  # alias for the ledger
        "n_trades_window": len(window),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "velocity_per_day": round(velocity_per_day, 4),
        "pnl_source": source,
        "days_to_target_at_velocity": (
            round(days_to_target, 1) if days_to_target is not None else None
        ),
        "peak_equity": round(peak, 2),
        "drawdown_from_peak_pct": round(drawdown_from_peak, 6),
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
    }


def render(m: dict) -> str:
    """Operator-friendly one-screen summary."""
    days = m.get("days_to_target_at_velocity")
    days_str = f"{days:.0f} days" if days is not None else "n/a (no velocity)"
    lines = [
        "=" * 70,
        "HERMES GOAL SCORECARD",
        "=" * 70,
        f"  Anchor:        ${m['anchor']:.2f}",
        f"  Current:       ${m['current_equity']:.2f}",
        f"  Target:        ${m['target']:.2f}",
        f"  Gap remaining: ${m['gap_remaining']:.2f}  "
        f"({m['goal_distance_pct']:.1%} of total gap)",
        f"  Progress:      {m['progress_pct']:.2%}",
        "",
        f"  Rolling {m['rolling_window_days']}d:",
        f"    PnL:          ${m['rolling_pnl']:+.2f}",
        f"    Trades:       {m['n_trades_window']} "
        f"({m['wins']}W / {m['losses']}L, "
        f"WR={(m['win_rate']*100 if m.get('win_rate') is not None else 0):.1f}%)",
        f"    Velocity:     ${m['velocity_per_day']:+.4f}/day",
        f"    Days to $:    {days_str}",
        "",
        f"  Peak equity:   ${m['peak_equity']:.2f}",
        f"  Drawdown:      {m['drawdown_from_peak_pct']:.2%}",
        "",
    ]
    return "\n".join(lines)


__all__ = ["compute_goal_metrics", "render"]


if __name__ == "__main__":
    print(render(compute_goal_metrics()))
