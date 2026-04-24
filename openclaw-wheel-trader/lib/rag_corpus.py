"""
Document RAG layer over SEC filings, earnings transcripts, and news corpora,
powered by HKUDS/RAG-Anything (MIT).

Why this exists:
    The bot already has:
      * lib/memory_palace.py — short verbatim trade reasoning (drawers)
      * lib/memory_vec.py    — sqlite-vec semantic search over those drawers
      * lib/news_sentiment.py (legacy) — crude keyword sentiment on headlines

    None of those handle LONG documents (10-Ks, earnings transcripts, Congress
    filings, insider transactions). RAG-Anything does: it parses PDFs/tables/
    images into a unified graph+vector index and answers queries with
    LightRAG's retrieval pipeline.

When to use:
    * Earnings week: index the latest 10-K/10-Q + transcript for every position
      and ask targeted questions before selling options into the report.
    * Insider flow: index Form 4 filings for the watchlist and query for
      cluster buying / selling patterns.
    * Macro: index Fed minutes + ECB/BoJ statements and query regime shifts.

Status: FEATURE-FLAGGED SKELETON. Enable with `rag.enabled: true` in
config/wheel_strategy.yaml. Install with:
    pip install raganything

The module stays import-safe when raganything isn't installed so tests and
default runs aren't affected.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lib.audit import log_event

CONFIG_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"
CORPUS_DIR = Path(__file__).parent.parent / "data" / "rag_corpus"
WORKING_DIR = CORPUS_DIR / "lightrag_storage"

try:
    from raganything import RAGAnything, RAGAnythingConfig  # type: ignore
    _HAS_RAG = True
except Exception:
    _HAS_RAG = False


@dataclass
class RAGResult:
    """Result of a corpus query."""
    query: str
    answer: str
    mode: str                 # "hybrid" | "local" | "global" | "vlm"
    sources: list[str]        # file paths or entity names referenced
    available: bool = True


def rag_available() -> bool:
    """True iff raganything is installed. Config flag checked separately in enabled()."""
    return _HAS_RAG


def _load_cfg() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return (yaml.safe_load(f) or {}).get("rag", {}) or {}
    except Exception:
        return {}


def enabled() -> bool:
    """True iff rag is both installed AND enabled in config AND has a DeepSeek key."""
    cfg = _load_cfg()
    return (
        _HAS_RAG
        and bool(cfg.get("enabled", False))
        and bool(os.environ.get("DEEPSEEK_API_KEY"))
    )


_rag_instance: "RAGAnything | None" = None


def _get_instance() -> "RAGAnything | None":
    """Lazy-init singleton. None if not available."""
    global _rag_instance
    if not enabled():
        return None
    if _rag_instance is not None:
        return _rag_instance
    try:
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        WORKING_DIR.mkdir(parents=True, exist_ok=True)
        cfg = RAGAnythingConfig(
            working_dir=str(WORKING_DIR),
            mineru_parse_method="auto",
            enable_image_processing=False,     # off by default (needs VLM)
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        # Delegate LLM + embedding to DeepSeek / local sentence-transformers
        # to match the rest of the bot. RAGAnything expects callable funcs.
        from openai import OpenAI
        llm_client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
        def llm_fn(prompt, system_prompt=None, history_messages=None, **kwargs):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for m in (history_messages or []):
                messages.append(m)
            messages.append({"role": "user", "content": prompt})
            resp = llm_client.chat.completions.create(
                model=_load_cfg().get("model", "deepseek-chat"),
                messages=messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content

        # Embeddings via sentence-transformers (local, no API)
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        def embed_fn(texts: list[str]):
            import numpy as np
            vecs = _embed_model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype="float32")

        _rag_instance = RAGAnything(
            config=cfg,
            llm_model_func=llm_fn,
            embedding_func=embed_fn,
        )
        log_event("rag_corpus", "initialized", {"working_dir": str(WORKING_DIR)})
        return _rag_instance
    except Exception as e:
        log_event("rag_corpus", "init_failed", {"error": str(e)[:300]})
        return None


def ingest_file(path: str | Path) -> bool:
    """Parse and index a single file (PDF, Markdown, docx, txt...). Safe no-op if disabled."""
    instance = _get_instance()
    if instance is None:
        return False
    try:
        p = str(Path(path).resolve())
        asyncio.run(instance.process_document_complete(
            file_path=p,
            output_dir=str(CORPUS_DIR / "parsed"),
        ))
        log_event("rag_corpus", "ingested", {"path": p})
        return True
    except Exception as e:
        log_event("rag_corpus", "ingest_failed", {"path": str(path), "error": str(e)[:300]})
        return False


def ingest_directory(dir_path: str | Path, pattern: str = "**/*") -> int:
    """Ingest every matching file under a directory. Returns count successfully indexed."""
    if not enabled():
        return 0
    root = Path(dir_path)
    if not root.exists():
        return 0
    n_ok = 0
    for p in sorted(root.glob(pattern)):
        if p.is_file() and p.suffix.lower() in {".pdf", ".md", ".txt", ".docx", ".html"}:
            if ingest_file(p):
                n_ok += 1
    return n_ok


def query(question: str, mode: str = "hybrid") -> RAGResult:
    """
    Ask the corpus a question.

    mode:
        * "hybrid" — best default, combines graph + vector retrieval
        * "local"  — vector-search only (faster, less context)
        * "global" — graph-community-based (better for thematic questions)
        * "vlm"    — visual, requires image processing (not enabled by default)
    """
    instance = _get_instance()
    if instance is None:
        return RAGResult(query=question, answer="", mode=mode, sources=[], available=False)
    try:
        answer = asyncio.run(instance.aquery(question, mode=mode))
        return RAGResult(
            query=question,
            answer=str(answer)[:4000],
            mode=mode,
            sources=[],  # TODO: surface retrieved sources when upstream API stabilizes
            available=True,
        )
    except Exception as e:
        log_event("rag_corpus", "query_failed", {"query": question[:100], "error": str(e)[:300]})
        return RAGResult(query=question, answer="", mode=mode, sources=[], available=False)


def stats() -> dict[str, Any]:
    """Summary of the corpus state for the dashboard."""
    if not _HAS_RAG:
        return {"available": False, "reason": "raganything not installed"}
    cfg = _load_cfg()
    if not cfg.get("enabled"):
        return {"available": False, "reason": "rag.enabled is false in config"}
    parsed_dir = CORPUS_DIR / "parsed"
    return {
        "available": enabled(),
        "working_dir": str(WORKING_DIR),
        "parsed_files": len(list(parsed_dir.glob("*"))) if parsed_dir.exists() else 0,
    }
