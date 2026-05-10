"""
BM25 retriever for policy document search.

Top-K: 3 chunks passed to LLM to stay within token budget.
"""

import logging
import pickle
from pathlib import Path
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import DATA_DIR, RAG_TOP_K
from rag.document_loader import load_documents

logger = logging.getLogger(__name__)

_RETRIEVER_SINGLETON = None
_BM25_PICKLE = DATA_DIR / "bm25_retriever.pkl"


def get_retriever(force_rebuild: bool = False):
    """Return singleton BM25 retriever, building from documents if needed."""
    global _RETRIEVER_SINGLETON
    if _RETRIEVER_SINGLETON is not None and not force_rebuild:
        return _RETRIEVER_SINGLETON

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
    r = BM25Retriever.from_documents(docs)
    r.k = RAG_TOP_K
    with open(_BM25_PICKLE, "wb") as f:
        pickle.dump(r, f)
    logger.info("BM25 retriever built and cached.")
    _RETRIEVER_SINGLETON = r
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
