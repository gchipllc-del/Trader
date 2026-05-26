"""Hermes scientific-method wrapper — overlays the video-2 self-improving
discipline on top of the existing ``agents.hermes_optimizer``.

What this adds on top of the legacy optimizer:

  1. CLOSE prior open experiments first. Each open experiment had a
     baseline goal-score; we compare against current goal-score and
     either keep the change (new baseline) or roll it back (revert the
     YAML).
  2. ONE-VARIABLE-AT-A-TIME. Even when the legacy diagnosis returns
     N recommendations, we apply only the highest-priority one per
     cycle. Other recommendations queue for future cycles. This makes
     every change attributable.
  3. MODE GATE. settings.yaml :: hermes_mode ∈ {"review", "live"}.
     - review: write a markdown report, NO YAML changes
     - live:   apply one change + open a ledger experiment
  4. RECENT-FAIL FILTER. Skip parameters that were rolled back within
     the last 72h (don't re-propose a recently-failed experiment).
  5. WEEKLY MARKDOWN REVIEW. Writes data/hermes_reviews/weekly_*.md
     summarizing experiments closed in the last 7 days, current goal
     velocity, and the next proposed change.

This module never deletes or modifies anything the legacy optimizer
wrote — it consumes its recommendations and arbitrates which one fires.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import yaml

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"
STRATEGY_PATH = ROOT / "config" / "wheel_strategy.yaml"
REVIEWS_DIR = ROOT / "data" / "hermes_reviews"

HermesMode = Literal["review", "live"]


# ─── Mode read ────────────────────────────────────────────────────────

def get_mode() -> HermesMode:
    """Read hermes_mode from settings.yaml. Defaults to 'review' (safe)."""
    try:
        with open(SETTINGS_PATH) as f:
            s = yaml.safe_load(f) or {}
        m = str(s.get("hermes_mode", "review")).lower().strip()
        return "live" if m == "live" else "review"
    except OSError:
        return "review"


def set_mode(mode: HermesMode) -> None:
    """Persist hermes_mode to settings.yaml."""
    mode = "live" if mode == "live" else "review"
    with open(SETTINGS_PATH) as f:
        s = yaml.safe_load(f) or {}
    s["hermes_mode"] = mode
    with open(SETTINGS_PATH, "w") as f:
        yaml.safe_dump(s, f, default_flow_style=False, sort_keys=False)


# ─── Prior-experiment closing ─────────────────────────────────────────

def close_prior_experiments(
    min_age_hours: float = 24.0,
    keep_threshold_delta: float = 0.001,
) -> list[dict]:
    """Look at every currently-open experiment older than `min_age_hours`,
    compare its baseline goal-distance to right now, and close it as kept
    or rolled_back. If rolled_back, revert the parameter in the YAML.

    Returns the list of experiments closed this cycle.
    """
    from lib.hermes_ledger import list_open_experiments, close_experiment
    from lib.hermes_goal_score import compute_goal_metrics

    now = datetime.now(timezone.utc)
    open_exps = list_open_experiments()
    closed: list[dict] = []
    current_metrics = compute_goal_metrics()

    for exp in open_exps:
        try:
            opened_dt = datetime.fromisoformat(
                exp["opened_at"].replace("Z", "+00:00")
            )
        except (KeyError, ValueError):
            continue
        age_h = (now - opened_dt).total_seconds() / 3600.0
        if age_h < min_age_hours:
            # Not ripe yet — leave open
            continue
        result = close_experiment(
            exp["experiment_id"],
            post_metrics=current_metrics,
            keep_threshold_delta=keep_threshold_delta,
        )
        if result is None:
            continue
        if result["status"] == "rolled_back":
            _revert_param(result["param"], result["old_value"])
        closed.append(result)
    return closed


def _revert_param(param: str, old_value) -> None:
    """Restore a parameter in wheel_strategy.yaml to its prior value.

    For params that have a settings.yaml mirror (currently:
    max_position_pct → circuit_breakers.max_position_pct), also sync the
    rollback over so the circuit breaker reads match the strategy. The
    legacy apply_adjustments writes both files on forward; without this
    parallel update the rollback leaves settings.yaml stale.
    """
    try:
        with open(STRATEGY_PATH) as f:
            strategy = yaml.safe_load(f) or {}
    except OSError:
        return
    sp = strategy.setdefault("stock_params", {})
    sp[param] = old_value
    with open(STRATEGY_PATH, "w") as f:
        yaml.safe_dump(strategy, f, default_flow_style=False, sort_keys=False)

    # Mirror to settings.yaml for params that live in both
    SETTINGS_MIRRORED = {"max_position_pct"}
    if param in SETTINGS_MIRRORED:
        try:
            with open(SETTINGS_PATH) as f:
                settings = yaml.safe_load(f) or {}
            settings.setdefault("circuit_breakers", {})[param] = old_value
            with open(SETTINGS_PATH, "w") as f:
                yaml.safe_dump(
                    settings, f, default_flow_style=False, sort_keys=False
                )
        except OSError:
            pass


# ─── Single-variable change selection ─────────────────────────────────

def goal_aware_recommendations(metrics: dict) -> list[dict]:
    """Inject defensive / aggressive recommendations from goal-level
    signals that the per-trade diagnose() can't see.

    Two main triggers:

      DRAWDOWN — if drawdown_from_peak > 20%, the bot is bleeding from
      a high-water mark. Recommend cutting position size + tightening
      stops to preserve remaining equity. This is the bot's #1 capital-
      preservation lever.

      VELOCITY — if velocity is strongly positive AND drawdown is
      contained, push for more aggression (larger positions, more
      trades per scan).

    Returns recs in the same shape as agents.hermes_optimizer.diagnose().
    """
    out: list[dict] = []
    dd = metrics.get("drawdown_from_peak_pct", 0.0) or 0.0
    vel = metrics.get("velocity_per_day", 0.0) or 0.0
    n_trades = metrics.get("n_trades_window", 0)
    wr = metrics.get("win_rate")

    # Defensive: significant drawdown
    if dd >= 0.20 and n_trades >= 10:
        out.append({
            "param": "max_position_pct",
            "direction": "decrease",
            "confidence": min(1.0, dd * 2),  # 20% dd → 0.4 conf, 50% dd → 1.0
            "reason": (
                f"goal_aware: drawdown {dd:.0%} from peak — "
                "reduce position size to preserve equity"
            ),
        })
    if dd >= 0.30 and n_trades >= 10:
        out.append({
            "param": "stop_loss_pct",
            "direction": "decrease",  # tighter stops in drawdown
            "confidence": min(1.0, dd * 1.5),
            "reason": (
                f"goal_aware: drawdown {dd:.0%} — tighten stops to "
                "limit further bleed"
            ),
        })

    # Aggressive: strong velocity + WR + low drawdown
    if (
        vel > 5.0  # > $5/day on the current equity scale
        and dd < 0.10
        and wr is not None and wr >= 0.55
        and n_trades >= 20
    ):
        out.append({
            "param": "max_trades_per_scan",
            "direction": "increase",
            "confidence": 0.6,
            "reason": (
                f"goal_aware: velocity ${vel:+.2f}/day, WR {wr:.0%}, "
                f"drawdown {dd:.0%} — bot is healthy, push more trades"
            ),
        })

    return out


def pick_one_change(recommendations: list[dict]) -> dict | None:
    """From the optimizer's recommendation list, pick the single best
    candidate to apply this cycle. Filters out recently-rolled-back
    params.

    Priority order: highest absolute confidence (if present), then alpha
    order on param name for determinism.
    """
    from lib.hermes_ledger import recently_rolled_back

    eligible = []
    for rec in recommendations:
        param = rec.get("param")
        if not param or param == "none":
            continue
        if rec.get("direction") == "hold":
            continue
        if recently_rolled_back(param):
            # Skip — same change just failed
            continue
        eligible.append(rec)
    if not eligible:
        return None
    # Highest confidence first; tie-break by param name
    eligible.sort(
        key=lambda r: (-(r.get("confidence", 0.0) or 0.0), r.get("param", "")),
    )
    return eligible[0]


# ─── Main cycle ───────────────────────────────────────────────────────

def run_scientific_cycle(
    lookback_days: int = 14,
    force_mode: HermesMode | None = None,
) -> dict:
    """Closed-loop optimization cycle following the video-2 framework.

    Steps:
      1. Close prior open experiments by comparing to current goal-score
         (this also rolls back parameters whose experiments regressed).
      2. Compute baseline goal-metrics for any new experiment opened
         this cycle.
      3. Run the legacy optimizer in dry-run mode to get recommendations.
      4. Pick exactly ONE change (highest confidence, not recently failed).
      5. If mode == 'live' and a change was picked: apply that one
         parameter change and open an experiment in the ledger.
      6. Return a structured cycle report (also used for the weekly md).
    """
    from agents.hermes_optimizer import (
        run_optimization, apply_adjustments,
    )
    from lib.hermes_ledger import open_experiment, list_open_experiments
    from lib.hermes_goal_score import compute_goal_metrics

    mode = force_mode or get_mode()

    closed = close_prior_experiments()
    baseline_metrics = compute_goal_metrics()

    # Always run optimizer in dry_run to get recommendations — we
    # control what (if anything) actually applies.
    legacy_report = run_optimization(
        lookback_days=lookback_days,
        dry_run=True,
    )
    recs = legacy_report.get("recommendations", []) or []
    # Layer in goal-aware recs (drawdown defense, velocity-driven aggression)
    goal_recs = goal_aware_recommendations(baseline_metrics)
    # Goal-aware recs go first so they win the priority sort on ties
    all_recs = goal_recs + recs
    pick = pick_one_change(all_recs)

    applied_change: dict | None = None
    new_experiment: dict | None = None

    if mode == "live" and pick is not None:
        # Apply ONLY the picked recommendation
        with open(STRATEGY_PATH) as f:
            strategy_before = yaml.safe_load(f) or {}
        before_value = (
            (strategy_before.get("stock_params") or {}).get(pick["param"])
        )
        changes = apply_adjustments([pick])
        if changes and pick["param"] in changes:
            ch = changes[pick["param"]]
            applied_change = {
                "param": pick["param"],
                "old": ch["old"],
                "new": ch["new"],
                "reason": ch["reason"],
            }
            new_experiment = open_experiment(
                param=pick["param"],
                old_value=ch["old"],
                new_value=ch["new"],
                reason=pick["reason"],
                baseline_metrics=baseline_metrics,
                expected_direction="up",
            )

    open_after = list_open_experiments()

    return {
        "cycle_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "lookback_days": lookback_days,
        "baseline_metrics": baseline_metrics,
        "closed_experiments": closed,
        "legacy_report_summary": {
            "trades_reviewed": legacy_report.get("review", {}).get("total_trades"),
            "skip_reason": legacy_report.get("skip_reason"),
            "n_recommendations": len(recs),
        },
        "pick": pick,
        "applied_change": applied_change,
        "new_experiment": new_experiment,
        "open_experiments_count": len(open_after),
    }


# ─── Weekly markdown review ───────────────────────────────────────────

def write_weekly_review(report: dict) -> Path:
    """Write a human-readable weekly review to data/hermes_reviews/."""
    from lib.hermes_ledger import history, stats

    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    week_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = REVIEWS_DIR / f"weekly_{week_tag}.md"

    bm = report.get("baseline_metrics", {})
    pick = report.get("pick")
    applied = report.get("applied_change")
    closed = report.get("closed_experiments", []) or []
    s = stats()
    recent = history(limit=8)

    lines = []
    lines.append(f"# Hermes Weekly Review — {week_tag}")
    lines.append("")
    lines.append(f"Mode: **{report.get('mode')}**")
    lines.append("")
    lines.append("## Goal scorecard")
    lines.append("")
    lines.append(f"- Current equity: **${bm.get('current_equity', 0):.2f}**")
    lines.append(f"- Target: ${bm.get('target', 0):.2f}")
    lines.append(f"- Gap remaining: ${bm.get('gap_remaining', 0):.2f} "
                 f"({bm.get('goal_distance_pct', 0):.1%} of total)")
    lines.append(f"- 30d velocity: ${bm.get('velocity_per_day', 0):+.4f}/day")
    days = bm.get("days_to_target_at_velocity")
    days_s = f"{days:.0f} days" if days is not None else "n/a (no velocity)"
    lines.append(f"- ETA at current velocity: {days_s}")
    lines.append(f"- 30d WR: "
                 f"{((bm.get('win_rate') or 0) * 100):.1f}%  "
                 f"({bm.get('wins', 0)}W / {bm.get('losses', 0)}L)")
    lines.append(f"- Drawdown from peak: {bm.get('drawdown_from_peak_pct', 0):.2%}")
    lines.append("")
    lines.append("## Experiments closed this cycle")
    lines.append("")
    if not closed:
        lines.append("_None — all experiments still open or none pending._")
    else:
        for e in closed:
            verdict = "✓ kept" if e["status"] == "kept" else "✗ rolled back"
            lines.append(
                f"- **{e['param']}** {e['old_value']} → {e['new_value']} — "
                f"{verdict} ({e.get('verdict', '?')})"
            )
    lines.append("")
    lines.append("## Next proposed change")
    lines.append("")
    if applied:
        lines.append(
            f"Applied (mode=live): **{applied['param']}** "
            f"{applied['old']} → {applied['new']}"
        )
        lines.append(f"Reason: {applied['reason']}")
    elif pick:
        lines.append(
            f"Picked (would apply if mode=live): **{pick.get('param')}** → "
            f"{pick.get('direction')}"
        )
        lines.append(f"Reason: {pick.get('reason')}")
    else:
        lines.append("_No change picked this cycle._")
    lines.append("")
    lines.append("## Ledger stats (lifetime)")
    lines.append("")
    counts = s.get("counts", {})
    lines.append(f"- Total experiments: {s.get('total', 0)}")
    lines.append(f"- Kept: {counts.get('kept', 0)}  "
                 f"Rolled back: {counts.get('rolled_back', 0)}  "
                 f"Open: {counts.get('open', 0)}  "
                 f"Expired: {counts.get('expired', 0)}")
    kr = s.get("keep_rate")
    if kr is not None:
        lines.append(f"- Keep rate: {kr:.1%}")
    lines.append("")
    lines.append("## Recent experiment history")
    lines.append("")
    if recent:
        lines.append("| When | Param | Δ | Verdict |")
        lines.append("|---|---|---|---|")
        for e in recent:
            when = (e.get("opened_at") or "")[:19].replace("T", " ")
            verdict = e.get("verdict") or e.get("status") or "?"
            lines.append(
                f"| {when} | {e.get('param')} | "
                f"{e.get('old_value')} → {e.get('new_value')} | {verdict} |"
            )
    else:
        lines.append("_No ledger entries yet._")
    lines.append("")
    path.write_text("\n".join(lines))
    return path


def render_cycle(report: dict) -> str:
    """One-screen console summary of a scientific-cycle run."""
    bm = report.get("baseline_metrics", {})
    lines = []
    lines.append("=" * 70)
    lines.append(f"HERMES SCIENTIFIC CYCLE  —  mode={report.get('mode')}")
    lines.append("=" * 70)
    lines.append(
        f"Goal: ${bm.get('current_equity', 0):.2f} / ${bm.get('target', 0):.0f}"
        f"  (gap ${bm.get('gap_remaining', 0):.2f}, "
        f"velocity ${bm.get('velocity_per_day', 0):+.4f}/day)"
    )
    legacy = report.get("legacy_report_summary", {})
    lines.append(
        f"Optimizer: {legacy.get('trades_reviewed', 0)} trades reviewed, "
        f"{legacy.get('n_recommendations', 0)} recommendations  "
        f"({legacy.get('skip_reason') or 'ok'})"
    )
    closed = report.get("closed_experiments", []) or []
    if closed:
        lines.append("")
        lines.append(f"Closed {len(closed)} experiment(s):")
        for e in closed:
            mark = "✓" if e["status"] == "kept" else "✗"
            lines.append(
                f"  {mark} {e['param']}: {e['old_value']} → {e['new_value']}  "
                f"({e.get('verdict')})"
            )
    pick = report.get("pick")
    applied = report.get("applied_change")
    if applied:
        lines.append("")
        lines.append(
            f"APPLIED: {applied['param']}  "
            f"{applied['old']} → {applied['new']}"
        )
        lines.append(f"  reason: {applied['reason']}")
    elif pick:
        lines.append("")
        lines.append(
            f"PICKED (review mode, no change applied): "
            f"{pick.get('param')} → {pick.get('direction')}"
        )
        lines.append(f"  reason: {pick.get('reason')}")
    else:
        lines.append("")
        lines.append("No change picked this cycle.")
    lines.append(f"Open experiments after cycle: {report.get('open_experiments_count', 0)}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "get_mode", "set_mode",
    "close_prior_experiments", "pick_one_change",
    "run_scientific_cycle", "write_weekly_review", "render_cycle",
]
