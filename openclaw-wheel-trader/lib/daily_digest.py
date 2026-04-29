"""
Daily Digest — unified end-of-day status snapshot for traderbot.

Pulls today's:
  - Portfolio value, cash, daily P/L vs baseline
  - Closed trades (count, win rate, total P/L)
  - Open positions (count, unrealized P/L)
  - Postmortem headline (best counterfactual delta if positive)
  - Anomaly triggers fired today
  - Hermes recommendations (read from cron_hermes.log if present)

Output goes to stdout. If TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are present
in env, also sends a one-shot Telegram message via lib.monitor.send_alert.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRADE_HISTORY_PATH = ROOT / "data" / "trade_history.json"
POSITIONS_PATH = ROOT / "data" / "positions.json"
ANOMALY_LOG_PATH = ROOT / "data" / "anomaly_log.jsonl"
POSTMORTEM_LOG_PATH = ROOT / "data" / "postmortem_log.jsonl"
BASELINE_PATH = ROOT / "data" / "baseline_equity.json"


@dataclass
class DigestSection:
    title: str
    lines: list[str]

    def render(self, prefix: str = "  ") -> str:
        out = [f"{prefix}{self.title}"]
        for ln in self.lines:
            out.append(f"{prefix}  {ln}")
        return "\n".join(out)


def _today() -> str:
    return date.today().isoformat()


def _portfolio_section(client) -> DigestSection:
    try:
        acct = client.get_account()
        portfolio = float(acct.get("portfolio_value", 0))
        cash = float(acct.get("cash", 0))
    except Exception as e:
        return DigestSection("PORTFOLIO", [f"(unable to read account: {e})"])

    baseline = portfolio
    if BASELINE_PATH.exists():
        try:
            baseline = float(json.load(open(BASELINE_PATH)).get("baseline_equity", portfolio))
        except Exception:
            pass

    daily_pnl = portfolio - baseline
    daily_pct = (daily_pnl / baseline) if baseline > 0 else 0.0
    return DigestSection("PORTFOLIO", [
        f"Equity: ${portfolio:,.2f}   Cash: ${cash:,.2f}",
        f"vs ${baseline:,.2f} baseline = ${daily_pnl:+,.2f} ({daily_pct*100:+.2f}%)",
    ])


def _closed_today() -> list[dict]:
    if not TRADE_HISTORY_PATH.exists():
        return []
    try:
        history = json.load(open(TRADE_HISTORY_PATH))
    except Exception:
        return []
    today = _today()
    return [r for r in history if (r.get("completed_at") or "")[:10] == today]


def _trades_section() -> DigestSection:
    closed = _closed_today()
    if not closed:
        return DigestSection("TRADES TODAY", ["No closed trades today."])

    pnl = sum(float(r.get("realized_pnl", 0)) for r in closed)
    wins = sum(1 for r in closed if float(r.get("realized_pnl", 0)) > 0)
    win_rate = wins / len(closed) if closed else 0.0
    lines = [
        f"{len(closed)} closed   wins {wins}/{len(closed)} ({win_rate*100:.0f}%)   "
        f"P/L ${pnl:+,.2f}",
    ]
    for r in closed[:5]:
        sign = "+" if float(r.get("realized_pnl", 0)) >= 0 else ""
        lines.append(
            f"  {r.get('ticker','?'):<6} {r.get('side','?')[:4]:<4} "
            f"{int(r.get('shares',0))}sh  P/L {sign}${float(r.get('realized_pnl', 0)):.2f} "
            f"({r.get('close_reason','?')})"
        )
    if len(closed) > 5:
        lines.append(f"  … {len(closed)-5} more")
    return DigestSection("TRADES TODAY", lines)


def _open_section() -> DigestSection:
    if not POSITIONS_PATH.exists():
        return DigestSection("OPEN POSITIONS", ["No positions file."])
    try:
        positions = json.load(open(POSITIONS_PATH))
    except Exception:
        return DigestSection("OPEN POSITIONS", ["(positions file unreadable)"])

    open_pos = [p for p in positions if p.get("status") == "open"]
    if not open_pos:
        return DigestSection("OPEN POSITIONS", ["None."])

    lines = [f"{len(open_pos)} open"]
    for p in open_pos[:8]:
        ticker = p.get("ticker", "?")
        shares = int(p.get("shares", 0))
        entry = float(p.get("entry_price", 0))
        lines.append(
            f"  {ticker:<6} {shares}sh @ ${entry:.2f}   "
            f"opened {(p.get('opened_at','') or '')[:10]}"
        )
    if len(open_pos) > 8:
        lines.append(f"  … {len(open_pos)-8} more")
    return DigestSection("OPEN POSITIONS", lines)


def _postmortem_section() -> DigestSection:
    if not POSTMORTEM_LOG_PATH.exists():
        return DigestSection("POSTMORTEM", ["(none persisted yet)"])

    today = _today()
    last = None
    with open(POSTMORTEM_LOG_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("date") == today:
                last = rec  # take latest match
    if last is None:
        return DigestSection("POSTMORTEM", ["(no postmortem persisted today)"])

    lines = [
        f"Actual ${last.get('actual_pnl', 0):+,.2f}   "
        f"Best alt ${last.get('best_alt_pnl', 0):+,.2f} "
        f"({last.get('best_alt_label', 'n/a')})",
    ]
    for lesson in last.get("lessons", [])[:3]:
        lines.append(f"  • {lesson}")
    return DigestSection("POSTMORTEM", lines)


def _anomaly_section() -> DigestSection:
    if not ANOMALY_LOG_PATH.exists():
        return DigestSection("ANOMALIES TODAY", ["(none)"])
    today = _today()
    triggered: list[dict] = []
    with open(ANOMALY_LOG_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = (rec.get("timestamp", "") or "")[:10]
            if ts == today and rec.get("triggered"):
                triggered.append(rec)
    if not triggered:
        return DigestSection("ANOMALIES TODAY", ["(none triggered)"])

    triggered.sort(key=lambda r: -(r.get("composite_z") or 0))
    lines = [f"{len(triggered)} triggered"]
    for r in triggered[:5]:
        lines.append(
            f"  {r.get('symbol'):<6} composite {r.get('composite_z', 0):.1f}σ  "
            f"move {r.get('pct_move_today', 0)*100:+.2f}%  "
            f"vol-z {r.get('volume_z', 0):.1f}"
        )
    return DigestSection("ANOMALIES TODAY", lines)


def _hermes_section() -> DigestSection:
    """Read the latest Hermes run, if any. cron_hermes.log appended by
    scripts/run_hermes.sh; we read the last 50 lines and grab the
    summary block."""
    log_path = ROOT / "logs" / "cron_hermes.log"
    if not log_path.exists():
        return DigestSection("HERMES", ["(no log found)"])
    try:
        lines = log_path.read_text().splitlines()
    except Exception:
        return DigestSection("HERMES", ["(log unreadable)"])
    tail = lines[-30:] if len(lines) > 30 else lines
    if not any("hermes" in ln.lower() or "recommend" in ln.lower() for ln in tail):
        return DigestSection("HERMES", ["No recent recommendations."])
    summary = [ln.strip() for ln in tail if ln.strip()][-5:]
    return DigestSection("HERMES (last log lines)", summary or ["(empty)"])


# --- Driver --------------------------------------------------------------

def build_digest(client) -> str:
    """Assemble the digest string. Returns plain text (no markdown)."""
    sections = [
        _portfolio_section(client),
        _trades_section(),
        _open_section(),
        _postmortem_section(),
        _anomaly_section(),
        _hermes_section(),
    ]

    header = (
        f"=== DAILY DIGEST — {_today()} "
        f"({datetime.now().strftime('%H:%M')}) ===\n"
    )
    body = "\n\n".join(s.render() for s in sections)
    return header + body + "\n"


def send_telegram_digest(text: str) -> bool:
    """Send digest via Telegram (truncated to 3500 chars). Returns sent?"""
    try:
        from lib.monitor import send_alert
    except Exception:
        return False
    snippet = text[:3500]
    try:
        send_alert(snippet)
        return True
    except Exception:
        return False
