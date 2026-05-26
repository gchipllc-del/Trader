"""Post-Earnings Announcement Drift (PEAD) signal.

Adapted from Mato Conti's video (2026-05-26) walking through the
academic foundations:

  Ball & Brown (1968) — "An Empirical Evaluation of Accounting Income
    Numbers" — original observation that stock prices keep drifting
    in the direction of an earnings surprise for weeks after the
    announcement.

  Bernard & Thomas (1989) — "Post-Earnings Announcement Drift:
    Delayed Price Response or Risk Premium?" — formalized the
    Standardized Unexpected Earnings (SUE) decile sort. Top minus
    bottom decile produced positive abnormal returns in 41 of 48
    quarters; average drift duration ~60 days.

  Livnat & Mendenhall (2006) — refined surprise methodology using
    analyst-forecast-based SUE (which we approximate with a simple
    relative surprise here since we don't have analyst dispersion).

The pattern:
  Positive earnings surprise → stock drifts UP for ~60 days afterward
  Negative earnings surprise → stock drifts DOWN for ~60 days afterward

How this changes traderbot behavior:
  - Currently: earnings_filter VETOES any CSP/CC entry within ~14 days
    of earnings. Reasonable for IV-crush avoidance.
  - PEAD says: AFTER the announcement (post-event window), if surprise
    was positive, the underlying tends to drift up — making it a
    structurally favorable window to write CSPs or take stock entries.

This module is OBSERVE-ONLY in this commit. Surfaces as:
  - lib.pead_signal.pead_score(ticker) → score in [-1, +1]
  - CLI: `python main.py pead --ticker NVDA`
  - Universe scan: `python main.py pead-scan`

Wiring into the live composite is a follow-up (PEAD score as a new
contribution to composite, weighted moderately — 1.0 — because the
60-day decay reduces day-to-day signal-to-noise compared to short-term
momentum).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

# PEAD canonical parameters from Bernard & Thomas (1989) and refined
# in Livnat & Mendenhall (2006). 60-day post-announcement drift window.
PEAD_DRIFT_DAYS = 60
PEAD_BLACKOUT_DAYS = 1   # don't trade the announcement day itself

# Surprise classification thresholds (relative — actual vs estimate)
LARGE_BEAT_THRESHOLD = 0.10    # ≥ +10% above estimate
SMALL_BEAT_THRESHOLD = 0.02    # ≥ +2%
SMALL_MISS_THRESHOLD = -0.02   # ≤ -2%
LARGE_MISS_THRESHOLD = -0.10   # ≤ -10%

PeadKind = Literal[
    "large_beat", "small_beat", "in_line",
    "small_miss", "large_miss", "no_data",
]


def _classify_surprise(surprise_pct: float | None) -> PeadKind:
    if surprise_pct is None:
        return "no_data"
    if surprise_pct >= LARGE_BEAT_THRESHOLD:
        return "large_beat"
    if surprise_pct >= SMALL_BEAT_THRESHOLD:
        return "small_beat"
    if surprise_pct <= LARGE_MISS_THRESHOLD:
        return "large_miss"
    if surprise_pct <= SMALL_MISS_THRESHOLD:
        return "small_miss"
    return "in_line"


def _surprise_pct(estimate: float | None, actual: float | None) -> float | None:
    """Relative surprise = (actual - estimate) / |estimate|.

    Returns None if either input is missing or estimate is too close
    to zero (relative math becomes unstable). Falls back to a sign-only
    score in that degenerate case.
    """
    if estimate is None or actual is None:
        return None
    if abs(estimate) < 0.01:
        # Estimate too close to zero — relative undefined. Use sign-only.
        if actual > estimate + 0.01:
            return SMALL_BEAT_THRESHOLD  # tag as small beat
        if actual < estimate - 0.01:
            return SMALL_MISS_THRESHOLD  # tag as small miss
        return 0.0
    return (actual - estimate) / abs(estimate)


def _drift_score(kind: PeadKind, days_since: int) -> float:
    """Translate a (surprise classification, days-since-announcement)
    into a signed score in [-1, +1].

    Score decays linearly across the 60-day drift window so a fresh
    beat (day 2) is worth more than a stale beat (day 50).
    """
    if kind == "no_data" or kind == "in_line":
        return 0.0
    if days_since < PEAD_BLACKOUT_DAYS:
        return 0.0  # IV-crush window — don't act on day 0
    if days_since > PEAD_DRIFT_DAYS:
        return 0.0  # drift exhausted
    decay = max(0.0, 1.0 - (days_since / PEAD_DRIFT_DAYS))
    magnitude_map = {
        "large_beat": +1.0,
        "small_beat": +0.5,
        "small_miss": -0.5,
        "large_miss": -1.0,
    }
    return decay * magnitude_map[kind]


def pead_score(ticker: str, days_back: int = 90) -> dict:
    """Compute a PEAD score for `ticker` based on its most recent
    earnings report.

    Returns a dict with the surprise %, kind, days_since, score, and
    a human-readable rationale string. Score is in [-1, +1].
    """
    try:
        from lib.finnhub_client import get_earnings_calendar_with_status
    except ImportError as e:
        return {"ticker": ticker, "score": 0.0, "kind": "no_data",
                "reason": f"finnhub_unavailable: {e}"}

    # Pull recent past + near-future. Most events in the calendar API
    # are forward-looking; for PEAD we want the most recent PAST event.
    # The Finnhub helper supports past lookups via negative days_ahead.
    try:
        events, _ok = get_earnings_calendar_with_status(
            ticker, days_ahead=-abs(days_back)
        )
    except Exception as e:
        return {"ticker": ticker, "score": 0.0, "kind": "no_data",
                "reason": f"earnings_lookup_failed: {str(e)[:120]}"}

    if not events:
        return {"ticker": ticker, "score": 0.0, "kind": "no_data",
                "reason": "no_past_earnings_in_window"}

    # Sort and pick the most recent event in the past (or today)
    today = datetime.now(timezone.utc).date()
    past = []
    for e in events:
        try:
            event_date = datetime.fromisoformat(e.date).date()
        except (AttributeError, ValueError, TypeError):
            continue
        if event_date <= today:
            past.append((event_date, e))
    if not past:
        return {"ticker": ticker, "score": 0.0, "kind": "no_data",
                "reason": "no_past_earnings_in_window"}
    past.sort(key=lambda x: x[0], reverse=True)
    event_date, evt = past[0]
    days_since = (today - event_date).days

    surprise = _surprise_pct(evt.eps_estimate, evt.eps_actual)
    kind = _classify_surprise(surprise)
    score = _drift_score(kind, days_since)

    return {
        "ticker": ticker,
        "event_date": event_date.isoformat(),
        "days_since": days_since,
        "eps_estimate": evt.eps_estimate,
        "eps_actual": evt.eps_actual,
        "surprise_pct": round(surprise, 4) if surprise is not None else None,
        "kind": kind,
        "score": round(score, 4),
        "in_drift_window": (
            PEAD_BLACKOUT_DAYS <= days_since <= PEAD_DRIFT_DAYS
        ),
        "reason": (
            f"{kind} ({surprise:+.1%} surprise) "
            f"{days_since}d after announcement"
            if surprise is not None
            else f"{kind} ({days_since}d since announcement, no eps data)"
        ),
    }


def universe_scan(tickers: list[str]) -> list[dict]:
    """Compute PEAD scores across a universe, return ALL results sorted
    by |score| desc so the operator sees the strongest drift signals.
    """
    out = []
    for t in tickers:
        try:
            out.append(pead_score(t))
        except Exception as e:
            out.append({
                "ticker": t, "score": 0.0,
                "kind": "no_data", "reason": f"error: {str(e)[:120]}",
            })
    out.sort(key=lambda r: -abs(r.get("score") or 0))
    return out


def render(result: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"PEAD SIGNAL — {result.get('ticker', '?')}")
    lines.append("=" * 70)
    if result.get("kind") == "no_data":
        lines.append(f"NO DATA  ({result.get('reason', '?')})")
        return "\n".join(lines + [""])
    lines.append(f"Event date:       {result.get('event_date', '?')}")
    lines.append(f"Days since:       {result.get('days_since', '?')}")
    est = result.get("eps_estimate")
    act = result.get("eps_actual")
    if est is not None and act is not None:
        lines.append(f"EPS estimate:     ${est:.2f}")
        lines.append(f"EPS actual:       ${act:.2f}")
    sp = result.get("surprise_pct")
    if sp is not None:
        lines.append(f"Surprise:         {sp:+.2%}")
    lines.append(f"Classification:   {result.get('kind', '?')}")
    lines.append(f"In drift window:  {result.get('in_drift_window', False)}")
    score = result.get("score", 0.0)
    direction = "LONG" if score > 0.10 else "SHORT" if score < -0.10 else "FLAT"
    lines.append(f"Score:            {score:+.4f}  → bias {direction}")
    lines.append(f"Reason:           {result.get('reason', '')}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "PEAD_DRIFT_DAYS", "PEAD_BLACKOUT_DAYS",
    "pead_score", "universe_scan", "render",
]


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    print(render(pead_score(ticker)))
