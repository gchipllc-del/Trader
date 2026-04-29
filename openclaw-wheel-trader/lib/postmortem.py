"""
Daily Postmortem — counterfactual analysis of today's trades.

Diagnostic, not prescriptive: for every closed trade today, replay the minute
bars from entry to exit and compute "what if" P/L for several alternative
exit rules. Surfaces patterns like:

    - Trailing stop triggered too early; we'd have made $X more if held to close
    - 5% stop would have rescued $Y on the loser without breaking winners
    - Tighter 3% target would have locked in $Z extra on the winner

Then a missed-opportunities sweep: anomaly_detector flags symbols that ran
today but we didn't trade — why? Cross-check the audit log.

Output is print-only in v1. Phase 2 hooks (Telegram digest, memory-palace
lesson, Hermes parameter feedback) are stubbed but disabled.

CLI:
    python main.py postmortem            # today
    python main.py postmortem --date 2026-04-26
    python main.py postmortem --watchlist NVDA,TSLA,COIN
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from lib.audit import log_event

ROOT = Path(__file__).parent.parent
TRADE_HISTORY_PATH = ROOT / "data" / "trade_history.json"
POSITIONS_PATH = ROOT / "data" / "positions.json"
AUDIT_PATH = ROOT / "logs" / "audit_log.jsonl"
POSTMORTEM_LOG_PATH = ROOT / "data" / "postmortem_log.jsonl"


# --- Data classes --------------------------------------------------------

@dataclass
class Counterfactual:
    label: str
    pnl: float
    delta_vs_actual: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeReplay:
    ticker: str
    side: str
    shares: int
    entry_price: float
    exit_price: float
    actual_pnl: float
    pnl_pct: float
    opened_at: str
    completed_at: str
    close_reason: str
    counterfactuals: list[Counterfactual] = field(default_factory=list)

    @property
    def best_counterfactual(self) -> Optional[Counterfactual]:
        if not self.counterfactuals:
            return None
        return max(self.counterfactuals, key=lambda c: c.pnl)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["counterfactuals"] = [c.to_dict() for c in self.counterfactuals]
        return d


@dataclass
class MissedOpportunity:
    symbol: str
    pct_move: float
    open_price: float
    close_price: float
    volume: float
    composite_z: Optional[float] = None
    skip_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PostmortemReport:
    date: str
    actual_pnl: float
    best_alt_pnl: float
    best_alt_label: str
    closed_trades: list[TradeReplay] = field(default_factory=list)
    missed: list[MissedOpportunity] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "actual_pnl": self.actual_pnl,
            "best_alt_pnl": self.best_alt_pnl,
            "best_alt_label": self.best_alt_label,
            "closed_trades": [t.to_dict() for t in self.closed_trades],
            "missed": [m.to_dict() for m in self.missed],
            "lessons": self.lessons,
        }


# --- Trade-history reader ------------------------------------------------

def load_closed_trades_for(target_date: date) -> list[dict]:
    """Read trade_history.json and filter to trades that closed on target_date."""
    if not TRADE_HISTORY_PATH.exists():
        return []
    try:
        with open(TRADE_HISTORY_PATH) as f:
            history = json.load(f)
    except Exception:
        return []

    target_iso = target_date.isoformat()
    out = []
    for record in history:
        completed = (record.get("completed_at") or "")[:10]
        if completed == target_iso:
            out.append(record)
    return out


# --- Counterfactual exit replays ----------------------------------------

def _replay_stop(
    bars: pd.DataFrame, entry_price: float, shares: int, stop_pct: float
) -> float:
    """Walk bars from entry forward; trigger stop if low ever <= entry*(1-stop)."""
    if bars is None or bars.empty:
        return 0.0
    stop_level = entry_price * (1.0 - stop_pct)
    for _, bar in bars.iterrows():
        if float(bar["low"]) <= stop_level:
            return (stop_level - entry_price) * shares
    # Never stopped — exit at last bar's close
    return (float(bars.iloc[-1]["close"]) - entry_price) * shares


def _replay_target(
    bars: pd.DataFrame, entry_price: float, shares: int, target_pct: float
) -> float:
    """Walk bars; exit at target if high ever >= entry*(1+target)."""
    if bars is None or bars.empty:
        return 0.0
    target_level = entry_price * (1.0 + target_pct)
    for _, bar in bars.iterrows():
        if float(bar["high"]) >= target_level:
            return (target_level - entry_price) * shares
    return (float(bars.iloc[-1]["close"]) - entry_price) * shares


def _replay_hold_to_close(
    bars: pd.DataFrame, entry_price: float, shares: int
) -> float:
    if bars is None or bars.empty:
        return 0.0
    return (float(bars.iloc[-1]["close"]) - entry_price) * shares


def compute_counterfactuals(
    record: dict, bars: pd.DataFrame
) -> list[Counterfactual]:
    """Run all counterfactual exits for a single closed-trade record.

    bars: minute bars covering the trade window (entry → exit). For long-only
          stock trades; pass `None` to skip and return empty list.
    """
    if bars is None or bars.empty:
        return []

    entry = float(record.get("entry_price", 0))
    shares = int(record.get("shares", 0))
    actual_pnl = float(record.get("realized_pnl", 0.0))
    trade_type = str(record.get("type", "stock")).lower()

    if entry <= 0 or shares <= 0:
        return []
    # Phase-1 bot is long-only stocks; skip options/CSP/CC trades whose
    # P/L mechanics differ. trade_history's `side` field describes the
    # *exit* action (sell-to-close on a long → side="sell"), so we infer
    # long-vs-short from `type` instead.
    if trade_type != "stock":
        return []

    cfs: list[Counterfactual] = []

    # Hold to close
    pnl_hold = _replay_hold_to_close(bars, entry, shares)
    cfs.append(Counterfactual(
        "hold_to_close", round(pnl_hold, 2),
        round(pnl_hold - actual_pnl, 2),
    ))

    # Stops at 2%, 3.5%, 5%, 7%
    for pct in (0.02, 0.035, 0.05, 0.07):
        pnl = _replay_stop(bars, entry, shares, pct)
        cfs.append(Counterfactual(
            f"stop_{pct*100:.1f}%", round(pnl, 2),
            round(pnl - actual_pnl, 2),
        ))

    # Targets at 3%, 5%, 8%, 12%
    for pct in (0.03, 0.05, 0.08, 0.12):
        pnl = _replay_target(bars, entry, shares, pct)
        cfs.append(Counterfactual(
            f"target_{pct*100:.0f}%", round(pnl, 2),
            round(pnl - actual_pnl, 2),
        ))

    # No trade
    cfs.append(Counterfactual(
        "no_trade", 0.0, round(0.0 - actual_pnl, 2),
    ))

    return cfs


# --- Bar fetcher ---------------------------------------------------------

def _fetch_trade_bars(client, ticker: str, opened_at: str,
                      completed_at: str) -> pd.DataFrame | None:
    """Fetch daily bars covering the trade window. Returns None on failure.

    Uses daily bars rather than minute bars because Alpaca's get_bars()
    paginates forward from a 400-day-back start; minute-bar pagination
    doesn't reliably reach the trade's specific date window. Daily bars
    cover any trade window in a single request and the replay is still
    directionally useful for stop/target counterfactuals.
    """
    try:
        bars_dict = client.get_bars([ticker], timeframe="1Day", limit=400)
        df = bars_dict.get(ticker)
        if df is None or df.empty:
            return None

        # Normalize trade-window timestamps; tz-strip to match index
        try:
            o = pd.Timestamp(opened_at)
            c = pd.Timestamp(completed_at)
            if df.index.tz is None:
                if o.tz is not None:
                    o = o.tz_convert("UTC").tz_localize(None)
                if c.tz is not None:
                    c = c.tz_convert("UTC").tz_localize(None)
            # Slice to trade window; expand by 1 day on each end to be safe
            window = df.loc[
                (df.index >= o - pd.Timedelta(days=1))
                & (df.index <= c + pd.Timedelta(days=1))
            ]
            if window is not None and not window.empty:
                return window
        except Exception:
            pass

        # Fallback: return last 30 bars (won't replay the right window but
        # at least gives a sample for diagnostic visibility)
        return df.tail(30)
    except Exception:
        return None


# --- Missed-opportunity scan --------------------------------------------

def find_missed_opportunities(
    client,
    watchlist: list[str],
    target_date: date,
    move_threshold: float = 0.05,
) -> list[MissedOpportunity]:
    """Symbols in watchlist that moved >= move_threshold today and weren't traded."""
    if not watchlist:
        return []

    bars_dict = client.get_bars(watchlist, timeframe="1Day", limit=3)

    # Symbols traded today
    traded_today = set()
    closed = load_closed_trades_for(target_date)
    for r in closed:
        traded_today.add(str(r.get("ticker", "")).upper())
    # Also exclude open positions with opened_at == today
    if POSITIONS_PATH.exists():
        try:
            positions = json.load(open(POSITIONS_PATH))
            target_iso = target_date.isoformat()
            for p in positions:
                if str(p.get("opened_at", ""))[:10] == target_iso:
                    traded_today.add(str(p.get("ticker", "")).upper())
        except Exception:
            pass

    missed: list[MissedOpportunity] = []
    for sym in watchlist:
        if sym.upper() in traded_today:
            continue
        df = bars_dict.get(sym)
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        op = float(last["open"])
        cl = float(last["close"])
        if op <= 0:
            continue
        pct_move = (cl - op) / op
        if abs(pct_move) >= move_threshold:
            missed.append(MissedOpportunity(
                symbol=sym,
                pct_move=round(pct_move, 4),
                open_price=op,
                close_price=cl,
                volume=float(last["volume"]),
            ))
    missed.sort(key=lambda m: -abs(m.pct_move))
    return missed


# --- Lesson generation ---------------------------------------------------

def _generate_lessons(
    closed_trades: list[TradeReplay], missed: list[MissedOpportunity]
) -> list[str]:
    """Heuristic lesson extraction. Cheap rules; LLM analysis is Phase 2."""
    lessons: list[str] = []

    if not closed_trades and not missed:
        lessons.append("Quiet day — no closed trades, no >5% missed movers.")
        return lessons

    # Stop-tightness pattern: was wider stop a strict improvement?
    wider_better = 0
    for t in closed_trades:
        cf_5 = next(
            (c for c in t.counterfactuals if c.label == "stop_5.0%"), None
        )
        if cf_5 and cf_5.delta_vs_actual > 0:
            wider_better += 1
    if closed_trades and wider_better >= max(2, len(closed_trades) // 2):
        lessons.append(
            f"Wider stop (5%) beat actual on {wider_better}/{len(closed_trades)} "
            "closed trades — consider relaxing trailing_stop_pct."
        )

    # Hold-to-close pattern
    hold_better = 0
    hold_delta_total = 0.0
    for t in closed_trades:
        cf = next(
            (c for c in t.counterfactuals if c.label == "hold_to_close"), None
        )
        if cf and cf.delta_vs_actual > 0:
            hold_better += 1
            hold_delta_total += cf.delta_vs_actual
    if closed_trades and hold_better >= max(2, len(closed_trades) // 2):
        lessons.append(
            f"Hold-to-close beat actual on {hold_better}/{len(closed_trades)} "
            f"trades (+${hold_delta_total:.2f} total) — exits may be premature."
        )

    # Missed-mover concentration
    if len(missed) >= 3:
        names = ", ".join(m.symbol for m in missed[:5])
        biggest = missed[0]
        lessons.append(
            f"{len(missed)} watchlist names ran >5% today (top: {names}). "
            f"Biggest missed: {biggest.symbol} {biggest.pct_move*100:+.1f}%."
        )

    if not lessons:
        lessons.append("No strong signal — trades performed roughly to plan.")
    return lessons


# --- Top-level driver ----------------------------------------------------

def generate_report(
    client,
    target_date: date | None = None,
    watchlist: list[str] | None = None,
) -> PostmortemReport:
    """Build the full daily postmortem."""
    if target_date is None:
        target_date = date.today()

    # Pull closed trades + replay each
    closed_records = load_closed_trades_for(target_date)
    replays: list[TradeReplay] = []
    for r in closed_records:
        bars = _fetch_trade_bars(
            client, r["ticker"], r.get("opened_at", ""), r.get("completed_at", "")
        )
        cfs = compute_counterfactuals(r, bars)
        replays.append(TradeReplay(
            ticker=r["ticker"],
            side=str(r.get("side", "buy")),
            shares=int(r.get("shares", 0)),
            entry_price=float(r.get("entry_price", 0)),
            exit_price=float(r.get("exit_price", 0)),
            actual_pnl=float(r.get("realized_pnl", 0)),
            pnl_pct=float(r.get("pnl_pct", 0)),
            opened_at=r.get("opened_at", ""),
            completed_at=r.get("completed_at", ""),
            close_reason=r.get("close_reason", ""),
            counterfactuals=cfs,
        ))

    # Missed opportunities
    if watchlist is None:
        try:
            from lib.anomaly_detector import DEFAULT_WATCHLIST
            watchlist = DEFAULT_WATCHLIST
        except Exception:
            watchlist = []
    missed = find_missed_opportunities(client, watchlist, target_date)

    actual_pnl = sum(t.actual_pnl for t in replays)
    # Best alternative across all trades, summed by counterfactual label
    by_label: dict[str, float] = {}
    for t in replays:
        for cf in t.counterfactuals:
            by_label[cf.label] = by_label.get(cf.label, 0.0) + cf.pnl
    if by_label:
        best_label = max(by_label, key=by_label.get)
        best_pnl = by_label[best_label]
    else:
        best_label = "n/a"
        best_pnl = actual_pnl

    lessons = _generate_lessons(replays, missed)

    return PostmortemReport(
        date=target_date.isoformat(),
        actual_pnl=round(actual_pnl, 2),
        best_alt_pnl=round(best_pnl, 2),
        best_alt_label=best_label,
        closed_trades=replays,
        missed=missed,
        lessons=lessons,
    )


# --- Output --------------------------------------------------------------

def print_report(report: PostmortemReport) -> None:
    print("=" * 80)
    print(f"  DAILY POSTMORTEM — {report.date}")
    print("=" * 80)

    print(f"  Actual P/L today: ${report.actual_pnl:+,.2f}")
    if report.best_alt_label != "n/a":
        delta = report.best_alt_pnl - report.actual_pnl
        print(f"  Best alternative: ${report.best_alt_pnl:+,.2f} "
              f"({report.best_alt_label}) — Δ ${delta:+,.2f}")
    print()

    if report.closed_trades:
        print(f"  CLOSED TRADES ({len(report.closed_trades)}):")
        for t in report.closed_trades:
            print(f"    {t.ticker:<6} {t.side:<4} {t.shares}sh "
                  f"{t.entry_price:.2f}→{t.exit_price:.2f} = "
                  f"${t.actual_pnl:+.2f} ({t.pnl_pct*100:+.2f}%) "
                  f"reason={t.close_reason}")
            best = t.best_counterfactual
            if best and best.delta_vs_actual > 0.5:
                print(f"        ↑ best alt: {best.label} → ${best.pnl:+.2f} "
                      f"(Δ ${best.delta_vs_actual:+.2f})")
        print()

    if report.missed:
        print(f"  MISSED MOVERS (>5% on watchlist, not traded):")
        for m in report.missed[:10]:
            arrow = "↑" if m.pct_move > 0 else "↓"
            print(f"    {m.symbol:<6} {arrow} {m.pct_move*100:+.2f}% "
                  f"${m.open_price:.2f}→${m.close_price:.2f} "
                  f"vol={int(m.volume):,}")
        print()

    print("  LESSONS:")
    for lesson in report.lessons:
        print(f"    • {lesson}")
    print("=" * 80)


def persist_report(report: PostmortemReport) -> None:
    """Append the report to data/postmortem_log.jsonl for trend analysis."""
    POSTMORTEM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POSTMORTEM_LOG_PATH, "a") as f:
        f.write(json.dumps(report.to_dict()) + "\n")
    log_event("postmortem", "report_persisted", {
        "date": report.date,
        "actual_pnl": report.actual_pnl,
        "best_alt_pnl": report.best_alt_pnl,
        "trades": len(report.closed_trades),
        "missed": len(report.missed),
    })
