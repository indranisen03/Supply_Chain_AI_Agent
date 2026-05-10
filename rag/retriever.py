"""
Hybrid RAG retriever — FAISS (dense) + BM25 (sparse) via EnsembleRetriever.

When OpenAI embeddings are unavailable, falls back to BM25-only retrieval
so the pipeline runs on Anthropic keys alone.

Weights: BM25=0.6, FAISS=0.4  (BM25 prioritised for exact supplier/contract ID matching)
Top-K: 3 chunks passed to LLM to stay within token budget.
"""

import logging
import pickle
from pathlib import Path
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import (
    ANTHROPIC_API_KEY,
    ENSEMBLE_BM25_WEIGHT,
    ENSEMBLE_FAISS_WEIGHT,
    FAISS_INDEX_PATH,
    OPENAI_API_KEY,
    RAG_TOP_K,
)
from rag.document_loader import load_documents

logger = logging.getLogger(__name__)

_RETRIEVER_SINGLETON = None
_BM25_PICKLE = Path(str(FAISS_INDEX_PATH) + "_bm25.pkl")
_FAISS_INDEX_FILE = Path(str(FAISS_INDEX_PATH) + "_store")

_HAS_OPENAI = bool(OPENAI_API_KEY) and OPENAI_API_KEY not in ("none", "sk-...")


def _build_bm25_only(docs: List[Document]):
    """BM25-only retriever — no embeddings required."""
    r = BM25Retriever.from_documents(docs)
    r.k = RAG_TOP_K
    with open(_BM25_PICKLE, "wb") as f:
        pickle.dump(r, f)
    logger.info("BM25-only retriever built and cached.")
    return r


def _build_ensemble(docs: List[Document]):
    """Full FAISS + BM25 ensemble (requires OpenAI embeddings)."""
    from langchain_community.vectorstores import FAISS
    from langchain.retrievers import EnsembleRetriever
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    faiss_store = FAISS.from_documents(docs, embeddings)
    faiss_store.save_local(str(_FAISS_INDEX_FILE))
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": RAG_TOP_K})

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = RAG_TOP_K
    with open(_BM25_PICKLE, "wb") as f:
        pickle.dump(bm25, f)

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25],
        weights=[ENSEMBLE_FAISS_WEIGHT, ENSEMBLE_BM25_WEIGHT],
    )


def get_retriever(force_rebuild: bool = False):
    """Return singleton retriever. BM25-only when no OpenAI key."""
    global _RETRIEVER_SINGLETON
    if _RETRIEVER_SINGLETON is not None and not force_rebuild:
        return _RETRIEVER_SINGLETON

    # Try loading BM25 cache first
    if not force_rebuild and _BM25_PICKLE.exists():
        try:
            with open(_BM25_PICKLE, "rb") as f:
                r = pickle.load(f)
            r.k = RAG_TOP_K
            logger.info("Loaded BM25 retriever from cache.")
            _RETRIEVER_SINGLETON = r
            return _RETRIEVER_SINGLETON
        except Exception as exc:
            logger.warning("Cache load failed, rebuilding: %s", exc)

    docs = load_documents(force_reload=force_rebuild)

    if _HAS_OPENAI:
        try:
            _RETRIEVER_SINGLETON = _build_ensemble(docs)
            logger.info("FAISS+BM25 ensemble retriever built.")
        except Exception as exc:
            logger.warning("FAISS build failed, falling back to BM25: %s", exc)
            _RETRIEVER_SINGLETON = _build_bm25_only(docs)
    else:
        logger.info("No OpenAI key — using BM25-only retriever.")
        _RETRIEVER_SINGLETON = _build_bm25_only(docs)

    return _RETRIEVER_SINGLETON


def retrieve(query: str, retriever=None) -> str:
    """Retrieve top-K chunks and return as formatted string."""
    r = retriever or get_retriever()
    docs = r.invoke(query)
    parts = []
    for i, doc in enumerate(docs[:RAG_TOP_K]):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i+1}: {src}]\n{doc.page_content}")
    return "\n\n".join(parts)
