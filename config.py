"""Central configuration — all env vars resolved once at import time."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KAGGLE_DIR = DATA_DIR / "kaggle"
DOCS_DIR = DATA_DIR / "docs"
AUDIT_DIR = BASE_DIR / "audit"

AUDIT_TRAIL_PATH = AUDIT_DIR / "audit_trail.jsonl"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"

# Dataset 1 — SKU catalog (harshsingh2209/supply-chain-analysis)
KAGGLE_CSV_PATH = KAGGLE_DIR / "supply_chain_data.csv"
SKU_CATALOG_PATH = KAGGLE_CSV_PATH  # alias

# Dataset 2 — Order history (shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
DATACO_CSV_PATH = KAGGLE_DIR / "DataCoSupplyChainDataset.csv"

# Ensure directories exist
for d in [DATA_DIR, KAGGLE_DIR, DOCS_DIR, AUDIT_DIR, FAISS_INDEX_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM Models ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# GPT-4o-mini: Sensing, Historical Trend, Simulation, Summarizer (cost-efficient)
MODEL_MINI = "gpt-4o-mini"
# GPT-4o: Optimization (complex reasoning)
MODEL_LARGE = "gpt-4o"
# Claude claude-sonnet-4-20250514: Validation Judge (independent second opinion)
MODEL_JUDGE = "claude-sonnet-4-5"

# ── LangSmith ──────────────────────────────────────────────────────────────────
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "stellantis-supply-chain")

if LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# ── HITL Thresholds ────────────────────────────────────────────────────────────
HITL_AUTO_MAX_USD = 10_000.0      # < $10k → AUTO
HITL_SOFT_MAX_USD = 50_000.0      # $10k–$50k → SOFT
# > $50k → HARD
SOFT_HITL_TIMEOUT_HOURS = float(os.getenv("SOFT_HITL_TIMEOUT_HOURS", "12"))

# ── Validation Judge ───────────────────────────────────────────────────────────
JUDGE_PASS_THRESHOLD = 7.0        # overall_score must exceed this
JUDGE_CONFIDENCE_THRESHOLD = 0.85

# ── Cost Guardrail ─────────────────────────────────────────────────────────────
MAX_COST_PER_RUN_USD = float(os.getenv("MAX_COST_PER_RUN_USD", "0.50"))

# ── RAG ────────────────────────────────────────────────────────────────────────
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 3
ENSEMBLE_BM25_WEIGHT = 0.6
ENSEMBLE_FAISS_WEIGHT = 0.4

# ── Safety Stock & Thresholds ──────────────────────────────────────────────────
DEFAULT_SAFETY_STOCK = 300
STOCKOUT_RISK_THRESHOLD = 0.50    # 50% → trigger decision window

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── RAG Document URLs ──────────────────────────────────────────────────────────
RAG_DOC_URLS = {
    "stellantis_code_of_conduct": (
        "https://www.stellantis.com/content/dam/stellantis-corporate/group/governance/"
        "code-of-conduct/Stellantis_CoC_EN.pdf"
    ),
    "global_responsible_purchasing": (
        "https://www.stellantis.com/content/dam/stellantis-corporate/group/governance/"
        "corporate-regulations/global-responsible-purchasing-guidelines.pdf"
    ),
    "supplier_management_principles": (
        "https://www.stellantis.com/content/dam/stellantis-corporate/archives/fca/"
        "corporate-regulations/Supplier_Management_Principles.pdf"
    ),
    "far_part_12": "https://www.acquisition.gov/far/part-12",
}

# ── Retry ──────────────────────────────────────────────────────────────────────
MAX_AGENT_RETRIES = 2
