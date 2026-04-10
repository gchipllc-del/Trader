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
    else:
        # Fallback: append to JSONL
        fallback_file = PALACE_DIR / "drawers.jsonl"
        with open(fallback_file, "a") as f:
            f.write(json.dumps(asdict(drawer)) + "\n")

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
    """
    if not HAS_CHROMA:
        return _search_fallback(query, wing, n_results)

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
                "relevance": round(1 - dist, 4),  # Convert distance to similarity
                "drawer_id": results["ids"][0][i] if results["ids"] else "",
            })

    return memories


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

_kg_initialized = False


def _init_kg_db():
    """Initialize the knowledge graph SQLite database (idempotent, runs schema once)."""
    global _kg_initialized
    KG_DB.parent.mkdir(parents=True, exist_ok=True)
    if not _kg_initialized:
        conn = sqlite3.connect(str(KG_DB))
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
        conn.close()
        _kg_initialized = True


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
):
    """
    Store a complete trade decision in the palace.
    Creates both a drawer (verbatim) and KG triple (structured).
    """
    wing = f"wing_{ticker.lower()}"
    if wing not in WINGS:
        wing = "wing_market"

    # Verbatim memory in drawer
    add_drawer(Drawer(
        wing=wing,
        hall="hall_facts",
        room=f"{ticker.lower()}-{trade_type}",
        content=reasoning,
        metadata={
            "ticker": ticker,
            "trade_type": trade_type,
            **details,
        },
    ))

    # Structured fact in KG
    obj_str = f"{details.get('strike', '')}_{details.get('expiration', '')}"
    kg_add(ticker, f"entered_{trade_type}", obj_str, metadata=details)


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
