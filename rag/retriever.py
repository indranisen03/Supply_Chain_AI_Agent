"""
Hybrid RAG retriever — FAISS (dense) + BM25 (sparse) via EnsembleRetriever.

Weights: BM25=0.6, FAISS=0.4  (BM25 prioritised for exact supplier/contract ID matching)
Top-K: 3 chunks passed to LLM to stay within token budget.

Usage
─────
    from rag.retriever import get_retriever
    retriever = get_retriever()
    docs = retriever.invoke("Stellantis emergency procurement approval threshold")
"""

import logging
import pickle
from pathlib import Path
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings

from config import (
    ENSEMBLE_BM25_WEIGHT,
    ENSEMBLE_FAISS_WEIGHT,
    FAISS_INDEX_PATH,
    OPENAI_API_KEY,
    RAG_TOP_K,
)
from rag.document_loader import load_documents

logger = logging.getLogger(__name__)

_RETRIEVER_SINGLETON: EnsembleRetriever | None = None
_FAISS_INDEX_FILE = Path(str(FAISS_INDEX_PATH) + "_store")
_BM25_PICKLE = Path(str(FAISS_INDEX_PATH) + "_bm25.pkl")


def _build_retriever(docs: List[Document]) -> EnsembleRetriever:
    """Build FAISS + BM25 ensemble from document list."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

    # FAISS vectorstore
    faiss_store = FAISS.from_documents(docs, embeddings)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": RAG_TOP_K})

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = RAG_TOP_K

    # Persist FAISS index
    faiss_store.save_local(str(_FAISS_INDEX_FILE))

    # Persist BM25 (not natively serializable, use pickle)
    with open(_BM25_PICKLE, "wb") as f:
        pickle.dump(bm25_retriever, f)

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[ENSEMBLE_FAISS_WEIGHT, ENSEMBLE_BM25_WEIGHT],
    )


def _load_cached_retriever() -> EnsembleRetriever | None:
    """Load FAISS + BM25 from disk cache if available."""
    if not _FAISS_INDEX_FILE.exists() or not _BM25_PICKLE.exists():
        return None
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
        faiss_store = FAISS.load_local(str(_FAISS_INDEX_FILE), embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": RAG_TOP_K})
        with open(_BM25_PICKLE, "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = RAG_TOP_K
        logger.info("Loaded RAG retriever from disk cache.")
        return EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[ENSEMBLE_FAISS_WEIGHT, ENSEMBLE_BM25_WEIGHT],
        )
    except Exception as exc:
        logger.warning("Cache load failed, rebuilding: %s", exc)
        return None


def get_retriever(force_rebuild: bool = False) -> EnsembleRetriever:
    """
    Return the singleton EnsembleRetriever. Builds from scratch on first call
    (or if force_rebuild=True), then returns the cached instance.
    FAISS index is loaded once at startup and shared across all agent calls.
    """
    global _RETRIEVER_SINGLETON
    if _RETRIEVER_SINGLETON is not None and not force_rebuild:
        return _RETRIEVER_SINGLETON

    if not force_rebuild:
        cached = _load_cached_retriever()
        if cached is not None:
            _RETRIEVER_SINGLETON = cached
            return _RETRIEVER_SINGLETON

    logger.info("Building RAG index from source documents...")
    docs = load_documents(force_reload=force_rebuild)
    _RETRIEVER_SINGLETON = _build_retriever(docs)
    logger.info("RAG index ready. %d chunks indexed.", len(docs))
    return _RETRIEVER_SINGLETON


def retrieve(query: str, retriever: EnsembleRetriever | None = None) -> str:
    """Convenience: retrieve top-K chunks and return as formatted string."""
    r = retriever or get_retriever()
    docs = r.invoke(query)
    parts = []
    for i, doc in enumerate(docs[:RAG_TOP_K]):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i+1}: {src}]\n{doc.page_content}")
    return "\n\n".join(parts)
