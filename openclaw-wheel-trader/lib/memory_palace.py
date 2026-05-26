"""
Trading Memory Palace — MemPalace integration for the Wheel Strategy bot.

Architecture (from github.com/milla-jovovich/mempalace):
  Wings   = Tickers + Strategy + Market
  Rooms   = Specific topics (trade decisions, zone history, regime analysis)
  Halls   = Memory types (facts, events, discoveries, preferences, advice)
  Closets = Summaries pointing to raw drawers
  Drawers = Verbatim trade reasoning and market observations

Knowledge Graph:
  Temporal entity-relationship triples in SQLite.
  "AAPL → entered_csp → 170P_2024-06-21 (valid_from: 2024-05-01)"
  "AAPL → assigned → 100_shares (valid_from: 2024-06-21)"
  "market → regime → bull (valid_from: 2024-01-15, ended: 2024-03-01)"

Agent Diaries:
  Each governance agent (strategy, risk, compliance) keeps a persistent
  diary of its observations and decisions across sessions.

Requires: pip install chromadb mempalace
Falls back to SQLite-only mode if ChromaDB unavailable.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal

# Try ChromaDB for semantic search, fall back gracefully
try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

PALACE_DIR = Path(__file__).parent.parent / "data" / "palace"
KG_DB = PALACE_DIR / "knowledge_graph.db"
DIARY_DIR = PALACE_DIR / "diaries"


# ============================================================
# PALACE STRUCTURE
# ============================================================

# Wings for the trading bot
WINGS = {
    # Per-ticker wings
    "wing_aapl": {"type": "ticker", "keywords": ["aapl", "apple"]},
    "wing_msft": {"type": "ticker", "keywords": ["msft", "microsoft"]},
    "wing_nvda": {"type": "ticker", "keywords": ["nvda", "nvidia"]},
    "wing_amd": {"type": "ticker", "keywords": ["amd"]},
    "wing_amzn": {"type": "ticker", "keywords": ["amzn", "amazon"]},
    "wing_meta": {"type": "ticker", "keywords": ["meta"]},
    "wing_googl": {"type": "ticker", "keywords": ["googl", "google", "alphabet"]},
    "wing_spy": {"type": "ticker", "keywords": ["spy", "s&p", "sp500"]},
    # Strategy wings
    "wing_wheel": {"type": "strategy", "keywords": ["wheel", "csp", "covered call", "assignment"]},
    "wing_market": {"type": "market", "keywords": ["market", "regime", "macro", "vix", "breadth"]},
}

# Halls — same in every wing (from MemPalace spec)
HALLS = [
    "hall_facts",        # Decisions locked in: "sold AAPL 170P for $3.50"
    "hall_events",       # Sessions/milestones: "assigned on AAPL at 170"
    "hall_discoveries",  # Insights: "AAPL forms strong support at 165 zone"
    "hall_preferences",  # Bot tuning: "prefer 30-delta puts over 25-delta"
    "hall_advice",       # Lessons: "don't sell puts through earnings"
]


def init_palace():
    """Create the palace directory structure."""
    PALACE_DIR.mkdir(parents=True, exist_ok=True)
    DIARY_DIR.mkdir(parents=True, exist_ok=True)

    # Init ChromaDB collection if available
    if HAS_CHROMA:
        client = chromadb.PersistentClient(path=str(PALACE_DIR / "chroma"))
        client.get_or_create_collection(
            name="trading_drawers",
            metadata={"hnsw:space": "cosine"},
        )

    # Init knowledge graph
    _init_kg_db()

    return True


# ============================================================
# DRAWERS — Verbatim memory storage + semantic search
# ============================================================

@dataclass
class Drawer:
    """A single memory unit in the palace."""
    wing: str
    hall: str
    room: str
    content: str          # Verbatim text — the actual reasoning
    metadata: dict        # ticker, strike, date, composite_score, etc.
    created_at: str = ""
    drawer_id: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.drawer_id:
            import hashlib
            h = hashlib.sha256(
                f"{self.wing}:{self.room}:{self.content[:100]}:{self.created_at}".encode()
            ).hexdigest()[:12]
            self.drawer_id = h


def add_drawer(drawer: Drawer) -> str:
    """
    Store a memory in the palace. Uses ChromaDB if available,
    falls back to SQLite.
    """
    init_palace()

    if HAS_CHROMA:
        client = chromadb.PersistentClient(path=str(PALACE_DIR / "chroma"))
        collection = client.get_or_create_collection("trading_drawers")
        collection.add(
            ids=[drawer.drawer_id],
            documents=[drawer.content],
            metadatas=[{
                "wing": drawer.wing,
                "hall": drawer.hall,
                "room": drawer.room,
                "created_at": drawer.created_at,
                **{k: str(v) for k, v in drawer.metadata.items()},
            }],
        )

    # Always mirror to JSONL so BM25 retrieval and feature_status counters
    # have a tier-independent view of every drawer.
    fallback_file = PALACE_DIR / "drawers.jsonl"
    with open(fallback_file, "a") as f:
        f.write(json.dumps(asdict(drawer)) + "\n")

    # Mirror into sqlite-vec semantic index if available. No-op if the deps
    # aren't installed (lib.memory_vec short-circuits). This gives us a
    # ChromaDB-independent semantic search path alongside the existing one.
    try:
        from lib import memory_vec
        memory_vec.index_drawer(
            drawer.drawer_id, drawer.content,
            wing=drawer.wing, hall=drawer.hall, room=drawer.room,
        )
    except Exception:
        pass  # never block writes on vector indexing

    return drawer.drawer_id


def search_memory(
    query: str,
    wing: str | None = None,
    hall: str | None = None,
    room: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """
    Semantic search across the palace.
    Filters by wing/hall/room if provided (the +34% retrieval boost).

    Preference order:
      1. ChromaDB (if installed and has data) — dense vector similarity
      2. sqlite-vec + sentence-transformers (if deps installed)
      3. BM25 over JSONL drawers — deterministic ranking, always available
         (adapted from TradingAgents v0.2.0)
      4. JSONL substring fallback — last resort, simple match
    """
    if HAS_CHROMA:
        init_palace()
        client = chromadb.PersistentClient(path=str(PALACE_DIR / "chroma"))
        collection = client.get_or_create_collection("trading_drawers")

        where_filter = {}
        if wing:
            where_filter["wing"] = wing
        if hall:
            where_filter["hall"] = hall
        if room:
            where_filter["room"] = room

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None,
        )

        memories = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                memories.append({
                    "content": doc,
                    "metadata": meta,
                    "relevance": round(1 - dist, 4),
                    "drawer_id": results["ids"][0][i] if results["ids"] else "",
                })
        if memories:
            return memories

    # Second preference: sqlite-vec semantic search.
    try:
        from lib import memory_vec
        if memory_vec.semantic_search_available():
            hits = memory_vec.search(query, k=n_results, wing=wing)
            if hits:
                return [{
                    "content": h["text"],
                    "metadata": {"wing": h["wing"], "hall": h["hall"], "room": h["room"]},
                    "relevance": h["similarity"],
                    "drawer_id": h["drawer_id"],
                } for h in hits]
    except Exception:
        pass

    # Third preference: BM25 over the JSONL drawer log. Deterministic
    # ranking (TradingAgents v0.2.0 pattern), zero deps, strong baseline
    # on short typed content like trade memories. Beats substring matching
    # on relevance ordering while staying offline.
    try:
        from lib.memory_bm25 import bm25_search
        bm25_hits = bm25_search(
            query=query,
            jsonl_path=PALACE_DIR / "drawers.jsonl",
            wing=wing,
            k=n_results,
        )
        if bm25_hits:
            return bm25_hits
    except Exception:
        # Best-effort — if BM25 errors (e.g. corrupt file), fall through
        # to the simpler substring fallback rather than failing search.
        pass

    return _search_fallback(query, wing, n_results)


def _search_fallback(query: str, wing: str | None, n: int) -> list[dict]:
    """Simple keyword search when ChromaDB isn't available."""
    fallback_file = PALACE_DIR / "drawers.jsonl"
    if not fallback_file.exists():
        return []

    results = []
    query_lower = query.lower()
    with open(fallback_file) as f:
        for line in f:
            drawer = json.loads(line.strip())
            if wing and drawer.get("wing") != wing:
                continue
            if query_lower in drawer.get("content", "").lower():
                results.append({
                    "content": drawer["content"],
                    "metadata": drawer.get("metadata", {}),
                    "relevance": 0.5,
                    "drawer_id": drawer.get("drawer_id", ""),
                })

    return results[:n]


# ============================================================
# KNOWLEDGE GRAPH — Temporal entity-relationship triples
# ============================================================

def _init_kg_db():
    """Initialize the knowledge graph SQLite database.

    Uses IF NOT EXISTS so running it on every call is safe and cheap.
    We don't cache an "initialized" flag because tests monkeypatch KG_DB
    to temp dirs, and a stale flag would skip schema creation in the new DB.
    """
    KG_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(KG_DB))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_object ON triples(object)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predicate ON triples(predicate)")
        conn.commit()
    finally:
        conn.close()


def kg_add(subject: str, predicate: str, obj: str,
           valid_from: str | None = None, metadata: dict | None = None):
    """
    Add a fact to the knowledge graph.

    Examples:
        kg_add("AAPL", "entered_csp", "170P_2024-06-21", valid_from="2024-05-01")
        kg_add("market", "regime", "bull", valid_from="2024-01-15")
        kg_add("AAPL_170P", "premium_collected", "$3.50")
        kg_add("strategy_agent", "vetoed_by", "risk_agent", metadata={"reason": "sector_concentration"})
    """
    _init_kg_db()
    conn = sqlite3.connect(str(KG_DB))
    conn.execute(
        "INSERT INTO triples (subject, predicate, object, valid_from, metadata) VALUES (?, ?, ?, ?, ?)",
        (subject, predicate, obj,
         valid_from or datetime.now(timezone.utc).isoformat(),
         json.dumps(metadata) if metadata else None),
    )
    conn.commit()
    conn.close()


def kg_invalidate(subject: str, predicate: str, obj: str, ended: str | None = None):
    """Mark a fact as no longer current (but keep it for history)."""
    _init_kg_db()
    end_time = ended or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(KG_DB))
    conn.execute(
        "UPDATE triples SET valid_to = ? WHERE subject = ? AND predicate = ? AND object = ? AND valid_to IS NULL",
        (end_time, subject, predicate, obj),
    )
    conn.commit()
    conn.close()


def kg_query(subject: str, as_of: str | None = None, current_only: bool = True) -> list[dict]:
    """
    Query all facts about an entity.
    If current_only, returns only facts without a valid_to date.
    If as_of is given, returns facts that were valid at that point in time.
    """
    _init_kg_db()
    conn = sqlite3.connect(str(KG_DB))
    conn.row_factory = sqlite3.Row

    if as_of:
        rows = conn.execute(
            "SELECT * FROM triples WHERE subject = ? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)",
            (subject, as_of, as_of),
        ).fetchall()
    elif current_only:
        rows = conn.execute(
            "SELECT * FROM triples WHERE subject = ? AND valid_to IS NULL",
            (subject,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM triples WHERE subject = ?",
            (subject,),
        ).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def kg_timeline(subject: str) -> list[dict]:
    """Get chronological story of an entity — all facts ordered by time."""
    _init_kg_db()
    conn = sqlite3.connect(str(KG_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM triples WHERE subject = ? OR object = ? ORDER BY valid_from ASC",
        (subject, subject),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# AGENT DIARIES — Persistent per-agent memory
# ============================================================

def diary_write(agent_name: str, entry: str):
    """
    Write to an agent's diary. Each agent keeps its own memory.
    
    Examples:
        diary_write("strategy_agent", "AAPL|CSP_170P|score_8/9|zone_at_168|3_touches|hammer_confirmed")
        diary_write("risk_agent", "VETOED|NVDA|sector_tech_at_28pct|max_30pct|too_close")
        diary_write("compliance_agent", "CLEAR|AAPL|no_wash_sale|last_loss_45d_ago")
    """
    DIARY_DIR.mkdir(parents=True, exist_ok=True)
    diary_file = DIARY_DIR / f"{agent_name}.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry": entry,
    }

    with open(diary_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def diary_read(agent_name: str, last_n: int = 20) -> list[dict]:
    """Read the last N entries from an agent's diary."""
    diary_file = DIARY_DIR / f"{agent_name}.jsonl"
    if not diary_file.exists():
        return []

    entries = []
    with open(diary_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    return entries[-last_n:]


# ============================================================
# TRADING-SPECIFIC MEMORY HELPERS
# ============================================================

def remember_trade_decision(
    ticker: str,
    trade_type: str,        # "csp", "cc", "close", "roll"
    details: dict,
    reasoning: str,
) -> str:
    """
    Store a complete trade decision in the palace.
    Creates both a drawer (verbatim) and KG triple (structured).

    Returns the drawer_id of the verbatim reasoning entry. Callers should
    persist this on the position record so the outcome (assignment,
    early-close, called-away) can later be linked back via
    ``record_trade_outcome`` — that's how the bot turns historical
    decisions into a learning signal.
    """
    wing = f"wing_{ticker.lower()}"
    if wing not in WINGS:
        wing = "wing_market"

    # Verbatim memory in drawer
    drawer = Drawer(
        wing=wing,
        hall="hall_facts",
        room=f"{ticker.lower()}-{trade_type}",
        content=reasoning,
        metadata={
            "ticker": ticker,
            "trade_type": trade_type,
            "outcome_status": "pending",  # set to "resolved" when outcome is recorded
            **details,
        },
    )
    drawer_id = add_drawer(drawer)

    # Structured fact in KG
    obj_str = f"{details.get('strike', '')}_{details.get('expiration', '')}"
    kg_add(ticker, f"entered_{trade_type}", obj_str,
           metadata={**details, "decision_drawer_id": drawer_id})

    return drawer_id


def record_trade_outcome(
    *,
    ticker: str,
    decision_drawer_id: str,
    realized_return_pct: float,
    holding_days: int,
    exit_reason: str,        # "csp_expired" | "csp_assigned" | "cc_expired" |
                              # "cc_called_away" | "early_close" | "stop_loss" | "rolled"
    final_pnl_dollars: float | None = None,
    alpha_pct: float | None = None,
    extras: dict | None = None,
) -> str:
    """Resolve a previously-remembered trade decision with its outcome.

    Writes a structured outcome drawer in ``hall_outcomes`` referencing
    the original decision's drawer_id, plus a KG triple ``outcome``
    pointing at the same. This is the half of the loop the bot was
    missing — without it, prior decisions stay forever "pending" in
    memory and never inform future agents. Pattern from
    TauricResearch/TradingAgents v0.2.4 deferred-resolution memory log.

    Returns the outcome drawer_id.
    """
    wing = f"wing_{ticker.lower()}"
    if wing not in WINGS:
        wing = "wing_market"

    won = realized_return_pct >= 0
    content = (
        f"OUTCOME: {ticker} {exit_reason} after {holding_days} days. "
        f"Realized return {realized_return_pct:+.2%}"
        + (f" (alpha {alpha_pct:+.2%})" if alpha_pct is not None else "")
        + (f", P&L ${final_pnl_dollars:,.2f}" if final_pnl_dollars is not None else "")
        + f". Decision={decision_drawer_id}."
    )
    metadata = {
        "ticker": ticker,
        "decision_drawer_id": decision_drawer_id,
        "exit_reason": exit_reason,
        "realized_return_pct": round(realized_return_pct, 4),
        "holding_days": int(holding_days),
        "won": won,
    }
    if alpha_pct is not None:
        metadata["alpha_pct"] = round(alpha_pct, 4)
    if final_pnl_dollars is not None:
        metadata["final_pnl_dollars"] = round(final_pnl_dollars, 2)
    if extras:
        metadata.update(extras)

    outcome_id = add_drawer(Drawer(
        wing=wing,
        hall="hall_outcomes",
        room=f"{ticker.lower()}-outcomes",
        content=content,
        metadata=metadata,
    ))

    kg_add(ticker, "outcome",
           f"{exit_reason}_{realized_return_pct:+.2%}",
           metadata={"decision_drawer_id": decision_drawer_id,
                     "outcome_drawer_id": outcome_id,
                     "holding_days": int(holding_days)})

    return outcome_id


def reflect_on_outcome(
    *,
    decision_drawer_id: str,
    outcome_drawer_id: str,
    ticker: str,
    decision_reasoning: str,
    outcome_summary: str,
) -> str | None:
    """Generate an LLM reflection on a closed trade and store it in
    ``hall_advice`` so future agent prompts can read the lesson.

    Returns the reflection drawer_id, or None if the LLM is disabled,
    unavailable, or the call fails (silent fallback — reflection is
    advisory, must never block trading).

    Uses an Anthropic Haiku-class model for cost. One call per closed
    trade, capped at ~200 tokens. Skipped silently if the API key isn't
    set — reflections are advisory, never block trading.
    """
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    system = (
        "You are a senior options-wheel trader reviewing a closed position. "
        "Given the original reasoning and the realized outcome, write 2-4 "
        "sentences of plain prose:\n"
        "  1. Was the original call correct?\n"
        "  2. What specifically held or failed?\n"
        "  3. One concrete lesson for the next time a similar setup appears.\n"
        "Be concrete. No platitudes. Reference numbers if they're in the data."
    )
    user = (
        f"TICKER: {ticker}\n\n"
        f"ORIGINAL DECISION REASONING:\n{decision_reasoning}\n\n"
        f"OUTCOME:\n{outcome_summary}\n\n"
        "Write the reflection now."
    )

    model = os.environ.get("REFLECTION_MODEL", "claude-haiku-4-5")
    try:
        from anthropic import Anthropic
        client = Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        if not text:
            return None
    except Exception as e:
        try:
            from lib.audit import log_event
            log_event("memory_palace", "reflect_failed",
                      {"error": str(e)[:200], "ticker": ticker},
                      result="degraded")
        except Exception:
            pass
        return None

    wing = f"wing_{ticker.lower()}"
    if wing not in WINGS:
        wing = "wing_market"

    return add_drawer(Drawer(
        wing=wing,
        hall="hall_advice",
        room=f"{ticker.lower()}-lessons",
        content=text,
        metadata={
            "ticker": ticker,
            "decision_drawer_id": decision_drawer_id,
            "outcome_drawer_id": outcome_drawer_id,
            "kind": "reflection",
            "llm_model": model,
        },
    ))


def get_past_outcomes(
    ticker: str,
    *,
    n_same_ticker: int = 5,
    n_cross_ticker_lessons: int = 3,
) -> str:
    """Format prior outcomes and reflections for injection into a new
    decision's LLM prompt.

    Returns plain prose suitable to drop into a system prompt under a
    "Prior outcomes on this ticker" section. Empty string if nothing
    relevant exists yet (bot's first trade on this ticker, etc.).

    Pattern from TradingAgents v0.2.4 — past_context section of the
    Portfolio Manager prompt. Without injection, MemPalace stores
    knowledge that no agent ever reads.
    """
    wing = f"wing_{ticker.lower()}"

    # Same-ticker outcomes — recent first. Query is just the ticker
    # because that string appears in every outcome drawer's content
    # ("OUTCOME: {TICKER} ..."), making the JSONL keyword fallback
    # work correctly when ChromaDB isn't available.
    same_outcomes = search_memory(
        ticker, wing=wing, hall="hall_outcomes",
        n_results=n_same_ticker,
    )
    # Cross-ticker reflections — lessons that may transfer
    cross_lessons = search_memory(
        ticker, hall="hall_advice",
        n_results=n_cross_ticker_lessons,
    )

    if not same_outcomes and not cross_lessons:
        return ""

    lines: list[str] = []
    if same_outcomes:
        lines.append(f"Recent {ticker} outcomes:")
        for o in same_outcomes[:n_same_ticker]:
            m = o.get("metadata", {})
            ret = m.get("realized_return_pct")
            days = m.get("holding_days")
            reason = m.get("exit_reason", "")
            ret_str = f"{float(ret):+.1%}" if ret is not None else "?"
            days_str = f"{days}d" if days is not None else "?d"
            lines.append(f"  • {reason} in {days_str} → {ret_str}")

    if cross_lessons:
        lines.append("")
        lines.append("Relevant lessons from prior trades:")
        for l in cross_lessons[:n_cross_ticker_lessons]:
            content = (l.get("content") or "").strip().replace("\n", " ")
            if content:
                lines.append(f"  • {content[:240]}")

    return "\n".join(lines)


def prior_loss_rate(ticker: str, lookback_n: int = 10) -> tuple[int, int, float]:
    """Compute the loss rate on the last ``lookback_n`` resolved trades
    for ``ticker``. Returns ``(losses, total, rate)``.

    Cheap deterministic signal — bear_agent can read this without
    needing an LLM to weight the bear case higher when a ticker has
    been losing.
    """
    facts = kg_query(ticker, current_only=False)
    outcomes = [f for f in facts if f.get("predicate") == "outcome"]
    if not outcomes:
        return 0, 0, 0.0
    # Newest first if timestamps available
    outcomes.sort(key=lambda f: f.get("created_at", ""), reverse=True)
    sample = outcomes[:lookback_n]
    losses = sum(1 for f in sample
                 if (f.get("object") or "").split("_")[-1].startswith("-"))
    total = len(sample)
    return losses, total, (losses / total if total else 0.0)


def remember_zone_observation(ticker: str, zone_type: str, level: float, observation: str):
    """Store a support/resistance zone observation."""
    wing = f"wing_{ticker.lower()}"
    add_drawer(Drawer(
        wing=wing,
        hall="hall_discoveries",
        room=f"{ticker.lower()}-zones",
        content=observation,
        metadata={"ticker": ticker, "zone_type": zone_type, "level": level},
    ))


def remember_regime_change(new_regime: str, evidence: str):
    """Record a market regime change."""
    # Invalidate old regime
    current = kg_query("market", current_only=True)
    for fact in current:
        if fact["predicate"] == "regime":
            kg_invalidate("market", "regime", fact["object"])

    # Add new regime
    kg_add("market", "regime", new_regime, metadata={"evidence": evidence})

    add_drawer(Drawer(
        wing="wing_market",
        hall="hall_events",
        room="regime-changes",
        content=evidence,
        metadata={"new_regime": new_regime},
    ))


def recall_ticker_history(ticker: str) -> dict:
    """
    Get everything the bot remembers about a ticker:
    - KG facts (positions, assignments, premiums)
    - Recent memories (trade reasoning, zone observations)
    - Agent diary entries mentioning this ticker
    """
    kg_facts = kg_query(ticker, current_only=False)
    timeline = kg_timeline(ticker)

    wing = f"wing_{ticker.lower()}"
    memories = search_memory(ticker, wing=wing, n_results=10)

    # Check agent diaries for mentions
    agent_mentions = {}
    for agent in ["strategy_agent", "risk_agent", "compliance_agent"]:
        entries = diary_read(agent, last_n=50)
        mentions = [e for e in entries if ticker.upper() in e.get("entry", "").upper()]
        if mentions:
            agent_mentions[agent] = mentions[-5:]  # Last 5 mentions

    return {
        "ticker": ticker,
        "kg_facts": kg_facts,
        "timeline": timeline,
        "memories": memories,
        "agent_mentions": agent_mentions,
    }


def get_current_regime() -> str | None:
    """What market regime does the bot think we're in?"""
    facts = kg_query("market", current_only=True)
    for f in facts:
        if f["predicate"] == "regime":
            return f["object"]
    return None
