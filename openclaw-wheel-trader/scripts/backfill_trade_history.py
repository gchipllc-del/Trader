#!/usr/bin/env python3
"""Rebuild data/trade_history.json from data/positions.json.

WHY THIS EXISTS
  Today's investigation found that ``self_audit.auto_reconcile_phantom_
  positions`` closes positions in positions.json but never appends to
  trade_history.json. As a result, trade_history had ~20 entries while
  positions.json had 934 closed records — most of them duplicate-counted
  by the auto-reconcile feedback loop (now fixed via 15-min grace +
  3-cycle confirmation counter).

  The dashboard, Hermes goal scorer, postmortem, and calibration ALL
  read trade_history.json. With 20 entries instead of the ~16 real
  unique trades, every downstream analysis was based on wildly wrong
  data.

WHAT THIS SCRIPT DOES
  1. Loads data/positions.json
  2. Filters to closed positions with non-zero realized P&L (skips
     zombie no_sell_found_flat_close, duplicate_purge records)
  3. Deduplicates by (ticker, entry_price, exit_price) — same trade
     if same prices, keep the earliest closed_at as canonical
  4. Maps each position-record into the trade_history schema
  5. Backs up the existing trade_history to trade_history.backup.json
  6. Writes the deduped real trades to trade_history.json

POST-BACKFILL HOOK NEEDED
  Going forward, self_audit.auto_reconcile_phantom_positions MUST
  ALSO write matched_broker_sell closes to trade_history.json so
  this never drifts again. That fix is in a sibling commit.

RUN
  cd openclaw-wheel-trader
  python scripts/backfill_trade_history.py [--dry-run]

The dry-run mode prints the summary without writing anything.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).parent.parent
POSITIONS_PATH = ROOT / "data" / "positions.json"
HISTORY_PATH = ROOT / "data" / "trade_history.json"
BACKUP_PATH = ROOT / "data" / "trade_history.backup.json"


def _hold_duration(opened_at: str, closed_at: str) -> str:
    """Compute a hold duration string in the format trade_history uses."""
    try:
        o = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
        c = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
        delta = c - o
        hours = delta.total_seconds() / 3600
        if hours < 24:
            return f"{hours:.1f}h"
        days = hours / 24
        return f"{days:.1f}d"
    except Exception:
        return "unknown"


def _position_to_history_entry(p: dict) -> dict:
    """Map a positions.json closed-entry into the trade_history schema."""
    entry = float(p.get("entry_price", 0) or 0)
    exit_p = float(p.get("exit_price", 0) or 0)
    pnl_pct = (exit_p - entry) / entry if entry > 0 else 0.0
    return {
        "ticker": p.get("ticker"),
        "type": p.get("type", "stock"),
        "side": "sell",  # all closes are sells
        "shares": int(p.get("shares", 0) or 0),
        "entry_price": round(entry, 4),
        "exit_price": round(exit_p, 4),
        "realized_pnl": round(float(p.get("realized_pnl", 0) or 0), 2),
        "pnl_pct": round(pnl_pct, 4),
        "composite_score": p.get("composite_score", 0),
        "close_reason": p.get("close_reason", ""),
        "opened_at": p.get("opened_at", ""),
        "completed_at": p.get("closed_at", ""),
        "hold_duration": _hold_duration(p.get("opened_at", ""), p.get("closed_at", "")),
        "backfilled_from_positions": True,
        "backfill_date": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    positions = json.loads(POSITIONS_PATH.read_text())
    closed = [
        p for p in positions
        if p.get("status") == "closed"
           and abs(float(p.get("realized_pnl", 0) or 0)) > 0.01
           and p.get("close_reason") != "duplicate_purge"
    ]

    # Dedupe by (ticker, entry_price, exit_price) at 2-decimal precision —
    # keep the earliest closed_at as the canonical record. Auto-reconcile
    # loop created multiple records for the same actual broker sell, and
    # the same fill can appear at 4dp / 2dp across code paths.
    seen: dict[tuple, dict] = {}
    for p in closed:
        key = (
            p.get("ticker"),
            round(float(p.get("entry_price", 0) or 0), 2),
            round(float(p.get("exit_price", 0) or 0), 2),
        )
        if key not in seen:
            seen[key] = p
        else:
            existing = seen[key]
            # Keep the one with the earlier closed_at (likely canonical)
            if (p.get("closed_at", "") < existing.get("closed_at", "")):
                seen[key] = p

    unique_real_trades = list(seen.values())
    new_history = [_position_to_history_entry(p) for p in unique_real_trades]
    # Sort by completed_at chronologically for readable inspection
    new_history.sort(key=lambda x: x.get("completed_at", ""))

    total_pnl = sum(t["realized_pnl"] for t in new_history)
    wins = sum(1 for t in new_history if t["realized_pnl"] > 0)
    losses = sum(1 for t in new_history if t["realized_pnl"] < 0)
    wr = wins / (wins + losses) * 100 if (wins + losses) else 0

    print("=" * 60)
    print("  TRADE HISTORY BACKFILL")
    print("=" * 60)
    print(f"  Raw closed entries (positions.json):   {len(positions)}")
    print(f"  With non-zero P&L (skip zombies):      {len(closed)}")
    print(f"  Unique real trades (deduped):          {len(new_history)}")
    print(f"  Wins:  {wins}  ({wr:.1f}% WR)")
    print(f"  Losses: {losses}")
    print(f"  Cumulative real P&L:  ${total_pnl:+,.2f}")
    print()

    # Existing trade_history check
    if HISTORY_PATH.exists():
        try:
            existing = json.loads(HISTORY_PATH.read_text())
            print(f"  Existing trade_history.json: {len(existing)} entries")
            print(f"  (will be backed up to {BACKUP_PATH.name})")
        except Exception as e:
            print(f"  Existing trade_history.json: UNREADABLE ({e})")

    print()

    if dry_run:
        print("  --dry-run: not writing anything")
        return 0

    # Backup existing history
    if HISTORY_PATH.exists():
        BACKUP_PATH.write_text(HISTORY_PATH.read_text())
        print(f"  Backed up existing → {BACKUP_PATH}")

    HISTORY_PATH.write_text(json.dumps(new_history, indent=2))
    print(f"  Wrote {len(new_history)} entries → {HISTORY_PATH}")
    print()
    print("  Verify with:")
    print("    python -c \"import json; "
          "h=json.load(open('data/trade_history.json')); "
          "print(len(h), 'entries,', "
          "sum(t['realized_pnl'] for t in h), 'cumulative P&L')\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
