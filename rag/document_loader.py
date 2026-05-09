"""
RAG document loader.

Downloads 4 real Stellantis / FAR public PDFs (or HTML), chunks them with
RecursiveCharacterTextSplitter (chunk_size=500, overlap=50), and returns a
list of LangChain Document objects ready for FAISS + BM25 indexing.

Falls back to curated stub text if a download fails, so the demo runs
offline without crashing.
"""

import logging
import re
import time
from pathlib import Path
from typing import List

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCS_DIR, RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE, RAG_DOC_URLS

logger = logging.getLogger(__name__)

# ── Stub content (used when PDF download fails) ───────────────────────────────
_STUBS: dict[str, str] = {
    "stellantis_code_of_conduct": (
        "Stellantis Code of Conduct (Feb 2026): All supplier engagements must comply with "
        "ethical sourcing standards. Suppliers are required to maintain transparency in "
        "delivery schedules and communicate disruptions within 48 hours. Force majeure "
        "clauses apply to natural disasters and government-mandated shutdowns. Stellantis "
        "reserves the right to dual-source any part with lead time > 30 days or on-time "
        "delivery rate < 80%. Suppliers must maintain minimum safety stock of 15 days of "
        "consumption. Emergency procurement above $50,000 USD requires VP-level approval."
    ),
    "global_responsible_purchasing": (
        "Stellantis Global Responsible Purchasing Guidelines: Purchase orders must be "
        "issued at least 14 days before required delivery for standard parts and 21 days "
        "for custom assemblies. Preferred suppliers with reliability > 90% qualify for "
        "blanket order agreements. Cost variances > 15% from baseline require procurement "
        "committee review. ESG scoring is mandatory for all Tier-1 suppliers. Dual-source "
        "qualification required for sole-source parts with annual value > $1M. "
        "Order quantities must include a 20% safety buffer for Q4 demand surges."
    ),
    "supplier_management_principles": (
        "Stellantis Supplier Management Principles (FCA Heritage): Supplier scorecards "
        "are reviewed quarterly. On-time delivery < 85% triggers a corrective action plan. "
        "Lead times must be contractually bounded; any extension beyond 120% of contracted "
        "lead time constitutes a material breach. Emergency orders carry a 10% premium. "
        "Safety stock for critical parts (Tier-1 powertrain, ADAS) must cover 30 days of "
        "forward demand. Suppliers must notify Stellantis of sub-tier disruptions within "
        "24 hours. Approved Vendor List (AVL) review occurs annually."
    ),
    "far_part_12": (
        "FAR Part 12 — Commercial Items: Clauses 52.212-4 (Contract Terms and Conditions) "
        "require delivery in accordance with the schedule in the contract. Force majeure "
        "excuses include acts of God, government acts, fires, floods, epidemics, strikes. "
        "Warranty: supplies must conform to specifications for 1 year post-delivery. "
        "Inspection and acceptance at destination within 30 days. Changes clause: any "
        "modification to quantity or delivery schedule requires bilateral contract "
        "modification. Liquidated damages apply for late delivery when schedule is critical. "
        "FAR 52.212-5 requires flow-down of anti-trafficking, labor, and DFARS clauses "
        "to sub-contractors supplying automotive assemblies."
    ),
}


def _download_pdf(url: str, save_path: Path) -> bytes:
    """Download a file with retry logic. Returns raw bytes."""
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            save_path.write_bytes(resp.content)
            return resp.content
        except Exception as exc:
            logger.warning("Download attempt %d failed for %s: %s", attempt + 1, url, exc)
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url} after 3 attempts")


def _extract_text_from_pdf(path: Path) -> str:
    """Extract text from PDF using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(texts)
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", path, exc)
        return ""


def _extract_text_from_html(content: bytes) -> str:
    """Strip HTML tags from bytes content."""
    text = content.decode("utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_documents(force_reload: bool = False) -> List[Document]:
    """
    Load and chunk all 4 RAG documents.
    Returns LangChain Document objects with source metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )
    all_docs: List[Document] = []

    for doc_key, url in RAG_DOC_URLS.items():
        save_path = DOCS_DIR / f"{doc_key}.pdf"
        text = ""

        # Try to use cached file first
        if save_path.exists() and not force_reload:
            logger.info("Loading cached: %s", save_path.name)
            text = _extract_text_from_pdf(save_path)

        # Try downloading
        if not text:
            try:
                logger.info("Downloading %s ...", url)
                raw_bytes = _download_pdf(url, save_path)
                if url.endswith(".pdf"):
                    text = _extract_text_from_pdf(save_path)
                else:
                    text = _extract_text_from_html(raw_bytes)
            except Exception as exc:
                logger.warning("Using stub for %s: %s", doc_key, exc)

        # Fall back to stub
        if not text or len(text.strip()) < 100:
            text = _STUBS.get(doc_key, f"[Stub] {doc_key}: No content available.")
            logger.info("Using stub content for: %s", doc_key)

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": doc_key,
                        "url": url,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
            )
        logger.info("Loaded %d chunks from %s", len(chunks), doc_key)

    logger.info("Total RAG documents: %d chunks", len(all_docs))
    return all_docs
