"""
Semantic search over memory_palace drawers using sqlite-vec + sentence-transformers.

Design goals:
    * NO new infrastructure — vectors live in the same SQLite file as the
      existing knowledge graph (data/palace/knowledge_graph.db).
    * NO external services — all-MiniLM-L6-v2 runs locally on CPU (~90MB model,
      auto-cached to ~/.cache/huggingface/ on first use).
    * NO breakage if deps are missing — this module is opt-in; callers probe
      `semantic_search_available()` first and fall back to the existing
      ChromaDB / JSONL paths in lib/memory_palace.py.

Why this over ChromaDB?
    ChromaDB works but is heavy (duckdb+hnswlib+tenant layer). For the
    drawer corpus (currently <10k rows), sqlite-vec is strictly lighter
    and co-locates vectors with the existing KG tables — one file, one backup.

Why this over the awesome-deepseek or RAG-Anything picks?
    Those are for document/news RAG (SEC filings, earnings transcripts,
    RSS corpora). This module is purpose-built for the trade-reasoning
    drawer corpus (short text, exact-match today). Different job.

Usage:
    from lib import memory_vec
    memory_vec.index_drawer(drawer_id, "AAPL 170P CSP — sold at hammer at 168 support zone...")
    hits = memory_vec.search("what happened last time we sold puts at support", k=5)
    # -> [{"drawer_id": "...", "similarity": 0.83, "text": "..."}]
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from lib.memory_palace import KG_DB

# Lazy imports + availability probe — keep this module importable even when
# deps are missing so callers can branch cleanly.
try:
    import sqlite_vec  # type: ignore
    _HAS_SQLITE_VEC = True
except Exception:
    _HAS_SQLITE_VEC = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False


_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, ~90MB, CPU-friendly
_EMBED_DIM = 384
_model: "SentenceTransformer | None" = None  # lazy-loaded


def semantic_search_available() -> bool:
    """True iff both deps are importable. Safe to call at import time."""
    return _HAS_SQLITE_VEC and _HAS_ST


def _get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not installed")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _connect() -> sqlite3.Connection:
    """Open the palace KG DB with sqlite-vec extension loaded + table ensured."""
    if not _HAS_SQLITE_VEC:
        raise RuntimeError("sqlite-vec not installed")
    KG_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(KG_DB))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # vec0 virtual table — embedding as FLOAT[384]. One row per drawer.
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS drawer_vec "
        f"USING vec0(drawer_id TEXT PRIMARY KEY, embedding FLOAT[{_EMBED_DIM}])"
    )
    # Sidecar table to store original text + light metadata so search results
    # are self-sufficient (no JOIN back into a separate drawers store required).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS drawer_meta ("
        "  drawer_id TEXT PRIMARY KEY,"
        "  wing TEXT,"
        "  hall TEXT,"
        "  room TEXT,"
        "  text TEXT NOT NULL,"
        "  created_at TEXT DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    return conn


def _encode(text: str) -> bytes:
    """Encode text to a 384-dim float32 vector, L2-normalized, packed as bytes."""
    import numpy as np  # sentence-transformers pulls it in anyway
    vec = _get_model().encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype="float32").tobytes()


def index_drawer(
    drawer_id: str,
    text: str,
    *,
    wing: str | None = None,
    hall: str | None = None,
    room: str | None = None,
) -> None:
    """
    Store (or replace) the embedding for a drawer.

    Safe to call repeatedly with the same drawer_id — INSERT OR REPLACE.
    No-op if semantic search isn't available (so callers can just always
    call this and rely on the feature flag at read time).
    """
    if not semantic_search_available():
        return
    if not text or not text.strip():
        return
    conn = _connect()
    try:
        # vec0 virtual tables don't support INSERT OR REPLACE on the PK; we
        # emulate upsert with DELETE + INSERT inside a single transaction.
        conn.execute("DELETE FROM drawer_vec WHERE drawer_id = ?", (drawer_id,))
        conn.execute(
            "INSERT INTO drawer_vec(drawer_id, embedding) VALUES (?, ?)",
            (drawer_id, _encode(text)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO drawer_meta(drawer_id, wing, hall, room, text) "
            "VALUES (?, ?, ?, ?, ?)",
            (drawer_id, wing, hall, room, text),
        )
        conn.commit()
    finally:
        conn.close()


def search(query: str, k: int = 5,
           wing: str | None = None) -> list[dict[str, Any]]:
    """
    Top-k drawers most similar to `query`. Empty list on any failure.

    Returns each result as:
        {"drawer_id": str, "similarity": float, "text": str,
         "wing": str | None, "hall": str | None, "room": str | None}

    `similarity` ranges [0, 1] — computed as 1 - L2^2/2 for normalized vectors
    (equivalent to cosine similarity, cheaper to compute).
    """
    if not semantic_search_available() or not query.strip():
        return []
    try:
        conn = _connect()
    except Exception:
        return []
    try:
        qvec = _encode(query)
        # Pull a few extra rows if we'll filter by wing in Python.
        limit = max(k * 4, k) if wing else k
        # sqlite-vec's vec0 virtual table requires a `k = N` clause (not LIMIT)
        # to trigger kNN search; k must be an integer literal in the SQL, not a bind
        # parameter, so we interpolate it after a bounds check.
        if not isinstance(limit, int) or limit < 1 or limit > 1000:
            return []
        sql = (
            "SELECT v.drawer_id, v.distance, m.text, m.wing, m.hall, m.room "
            "FROM drawer_vec v LEFT JOIN drawer_meta m ON m.drawer_id = v.drawer_id "
            f"WHERE v.embedding MATCH ? AND k = {limit} ORDER BY v.distance"
        )
        rows = conn.execute(sql, (qvec,)).fetchall()
    except Exception:
        return []
    finally:
        conn.close()

    out = []
    for drawer_id, dist, text, w, h, r in rows:
        if wing and w != wing:
            continue
        # For L2-normalized vectors, L2^2 ∈ [0, 4], so similarity ∈ [-1, 1].
        # We clamp to [0, 1] since we don't expect true anti-correlation here.
        sim = max(0.0, min(1.0, 1.0 - (dist * dist) / 2.0))
        out.append({
            "drawer_id": drawer_id,
            "similarity": round(sim, 4),
            "text": text or "",
            "wing": w,
            "hall": h,
            "room": r,
        })
        if len(out) >= k:
            break
    return out


def drawer_count() -> int:
    """How many drawers are currently indexed. Returns 0 if unavailable."""
    if not semantic_search_available():
        return 0
    try:
        conn = _connect()
    except Exception:
        return 0
    try:
        return int(conn.execute("SELECT COUNT(*) FROM drawer_vec").fetchone()[0])
    except Exception:
        return 0
    finally:
        conn.close()
