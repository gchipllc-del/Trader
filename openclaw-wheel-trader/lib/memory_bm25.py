"""
BM25 Memory Retrieval — deterministic ranking over the JSONL fallback.

Adapted from TradingAgents v0.2.0 (github.com/tauricresearch/tradingagents,
Feb 2026 release notes: "Replaced ChromaDB with BM25 for memory retrieval,
offering deterministic ranking over vector similarity").

Why BM25 for trade memory:
  • Deterministic — same query + same corpus = identical ranking. Vector
    similarity drifts with model updates and re-indexing.
  • Offline — no embeddings, no models, no GPU. Always available.
  • Strong baseline — for typed/structured content (which trade memories
    largely are — tickers, regimes, scores, outcomes), term-frequency
    ranking is competitive with dense embeddings.
  • Fast on small corpora (rebuilds index in memory each search). Below
    ~10k drawers, latency is sub-100ms.

The standard BM25 formula:

    score(D, Q) = Σ IDF(qi) × f(qi, D) × (k1 + 1)
                                ──────────────────────────────────
                                f(qi, D) + k1 × (1 - b + b × |D| / avgdl)

where IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)

Tuning constants are paper defaults (k1=1.5, b=0.75). These are robust
for general English; we don't tune per-corpus.

Public API:
    bm25_search(query, jsonl_path, wing=None, k=5) -> list[dict]

Returns the standard memory-search shape:
    [{"content": str, "metadata": dict, "relevance": float, "drawer_id": str}, ...]

`relevance` here is the raw BM25 score normalized into [0, 1] by dividing
by the top score for the query — so the highest-ranked result is always
1.0, and the relative ordering carries the meaning. This matches the
shape callers already use from ChromaDB / sqlite-vec.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable


# Tokenization: lowercase, strip punctuation, split on whitespace.
# Intentionally simple — no stemming. Trade memories use exact tickers
# ("NVDA") and structured tags ("BUY|score_7/13|...") that stemming
# would mangle.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_$/]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


# BM25 tuning constants (paper defaults).
K1 = 1.5
B = 0.75


def _load_drawers(jsonl_path: Path, wing: str | None) -> list[dict]:
    """Load JSONL drawers, optionally filtered by wing."""
    out: list[dict] = []
    if not jsonl_path.exists():
        return out
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if wing and d.get("wing") != wing:
                    continue
                out.append(d)
    except OSError:
        return []
    return out


def _build_index(drawers: list[dict]) -> tuple[list[list[str]], dict[str, int], float]:
    """Return (per_doc_tokens, term_doc_freq, avgdl).

    per_doc_tokens[i]  : tokens for drawer i
    term_doc_freq[t]   : how many drawers contain term t
    avgdl              : average document length
    """
    per_doc_tokens: list[list[str]] = []
    term_doc_freq: dict[str, int] = {}
    total_len = 0
    for d in drawers:
        toks = _tokenize(d.get("content", ""))
        per_doc_tokens.append(toks)
        total_len += len(toks)
        seen = set(toks)
        for t in seen:
            term_doc_freq[t] = term_doc_freq.get(t, 0) + 1
    avgdl = (total_len / len(drawers)) if drawers else 1.0
    return per_doc_tokens, term_doc_freq, avgdl


def _score_doc(
    doc_tokens: list[str],
    query_terms: Iterable[str],
    term_doc_freq: dict[str, int],
    n_docs: int,
    avgdl: float,
) -> float:
    """Compute BM25 score for a single doc against the query terms."""
    if not doc_tokens:
        return 0.0
    doc_len = len(doc_tokens)
    # Per-doc term frequency
    tf: dict[str, int] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1

    score = 0.0
    norm = K1 * (1 - B + B * doc_len / avgdl)
    for qt in query_terms:
        f = tf.get(qt, 0)
        if f == 0:
            continue
        n_qt = term_doc_freq.get(qt, 0)
        # +0.5 smoothing in standard BM25 formulation
        idf = math.log((n_docs - n_qt + 0.5) / (n_qt + 0.5) + 1)
        score += idf * (f * (K1 + 1)) / (f + norm)
    return score


def bm25_search(
    query: str,
    jsonl_path: Path,
    wing: str | None = None,
    k: int = 5,
) -> list[dict]:
    """Run BM25 retrieval against the JSONL drawer log.

    Returns the standard memory-search dict shape with `relevance`
    normalized so the top hit is 1.0 and the rest are proportional.
    Returns [] when the file is missing/empty or no terms hit.

    O(N) per search — fine for the trade-memory corpus size we operate at.
    For larger corpora the index could be cached, but the simpler design
    is preferred until cache invalidation becomes an actual problem.
    """
    drawers = _load_drawers(Path(jsonl_path), wing)
    if not drawers:
        return []

    query_terms = list(set(_tokenize(query)))
    if not query_terms:
        return []

    per_doc_tokens, term_doc_freq, avgdl = _build_index(drawers)
    n_docs = len(drawers)

    scored: list[tuple[float, dict]] = []
    for i, drawer in enumerate(drawers):
        score = _score_doc(
            per_doc_tokens[i], query_terms, term_doc_freq, n_docs, avgdl
        )
        if score > 0:
            scored.append((score, drawer))

    if not scored:
        return []

    scored.sort(key=lambda x: -x[0])
    top_k = scored[:k]
    top_score = top_k[0][0] if top_k else 1.0

    out: list[dict] = []
    for score, drawer in top_k:
        normalized = round(score / top_score, 4) if top_score > 0 else 0.0
        out.append({
            "content": drawer.get("content", ""),
            "metadata": drawer.get("metadata", {}),
            "relevance": normalized,
            "raw_bm25": round(score, 4),
            "drawer_id": drawer.get("drawer_id", ""),
        })
    return out
