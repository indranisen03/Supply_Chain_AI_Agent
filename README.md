# Stellantis Supply Chain AI Agent

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)
![Claude](https://img.shields.io/badge/Claude-Sonnet_4.6-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![SOC2](https://img.shields.io/badge/SOC2-Audit_Trail-gold)

A production-grade multi-agent AI system that monitors supply chain inventory across a warehouse network and autonomously generates purchase order recommendations — with tiered human-in-the-loop oversight, an independent LLM judge, RAG-grounded policy compliance, and a full SOC2 audit trail.


---
## Objective
**Situation**: 
Automotive supply chains lose significant production time to reactive procurement meaning orders placed after stockout risk is already critical, with no auditable reasoning trail.

**Task**: 
Idea is to build a working AI system that monitors SKU inventory across warehouses, surfaces purchase order recommendations before risk materializes, and routes decisions through a tiered                human in the loop approval gate with a full audit trail.

**Action**: 
Designed and built a 6-agent LangGraph pipeline with hybrid RAG grounded in real Stellantis procurement documents, a tiered HITL gate (AUTO / SOFT / HARD), an independent Claude judge for                blind validation, and a two-page Streamlit dashboard — all within a SOC2-compliant JSONL audit trail.

**Result**: 
System runs end-to-end in under 60 seconds, generates RAG-grounded PO recommendations with confidence scores, triggers the correct HITL tier based on order value, and produces a downloadable audit log per run.

**Data**: 
Simulated using a Kaggle supply chain dataset (DataCo); RAG grounded in real Stellantis public procurement documents. Production deployment would swap the CSV for live SAP/ERP feeds.

---
## What It Does

The system continuously monitors SKU inventory levels across warehouses. When a stockout risk is detected, a 6-agent AI pipeline activates to:

1. **Sense** current inventory, disruptions, and supplier status
2. **Analyse** 3 years of historical demand and supplier performance trends
3. **Simulate** three what-if scenarios (order today vs. wait 7 vs. 14 days)
4. **Recommend** a purchase order grounded in Stellantis procurement policy documents
5. **Validate** the recommendation with an independent Claude judge (blind evaluation)
6. **Summarise** the full decision trail in plain English for procurement teams

Every decision routes through a tiered approval gate — orders under $10k auto-approve, larger orders require human sign-off before proceeding.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│               STELLANTIS SUPPLY CHAIN AI PIPELINE                │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐ │
│  │ Agent 1  │─▶│ Agent 2  │─▶│ Agent 3  │─▶│    Agent 4      │ │
│  │ Sensing  │  │Historical│  │Simulation│  │  Optimization   │ │
│  │          │  │  Trend   │  │          │  │   + RAG         │ │
│  └──────────┘  └──────────┘  └──────────┘  └───────┬─────────┘ │
│                                                      │            │
│  ┌───────────────────────────────────────────────────▼─────────┐ │
│  │                    TIERED HITL GATE                          │ │
│  │  AUTO  (< $10k)      — log and proceed immediately           │ │
│  │  SOFT  ($10k–$50k)   — auto-approves after 12hr timeout      │ │
│  │  HARD  (> $50k)      — full block, explicit approval required │ │
│  └──────────────────────────────────┬────────────────────────────┘ │
│                                      │                              │
│  ┌───────────────────────────────────▼────────────────────────────┐ │
│  │   Agent 5 — Validation Judge (Claude Sonnet 4.5)               │ │
│  │   Blind evaluation: quantity_justified / timeline_realistic     │ │
│  │   / reasoning_grounded  →  score out of 10, PASS / FAIL        │ │
│  └───────────────────────────────────┬────────────────────────────┘ │
│                                       │                              │
│  ┌────────────────────────────────────▼───────────────────────────┐ │
│  │   Agent 6 — Executive Summarizer (Claude Haiku 4.5)            │ │
│  │   5–6 sentence plain-English summary for procurement teams      │ │
│  └────────────────────────────────────┬───────────────────────────┘ │
│                                        │                             │
│  ┌─────────────────────────────────────▼──────────────────────────┐ │
│  │              SOC2 AUDIT TRAIL  (append-only JSONL)             │ │
│  │   Every tool call, verification check, human decision,         │ │
│  │   judge score, token count, cost, and integrity hash           │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘

RAG:           FAISS + BM25 EnsembleRetriever (weights 0.4 / 0.6)
               Stellantis Code of Conduct, Purchasing Guidelines, FAR Part 12
Observability: LangSmith tracing (opt-in via LANGCHAIN_TRACING_V2=true)
Self-healing:  Each agent wrapped in retry logic with LLM debug reasoning
```

---

## Dashboard

A two-page Streamlit app gives procurement teams full visibility into every run.

**Overview page**
- Live urgency cards (Critical / At Risk / Healthy) from Kaggle supply chain data
- Scenario presets: Normal, Port Strike, High Value, Critical Stockout
- SKU and warehouse selectors
- Live pipeline animation — each agent lights up as it runs

**Decision Explorer page**
- Single-row PO summary strip: order qty, value, supplier, required-by date, confidence tier, HITL tier
- Pipeline panel with per-agent status (Risk Found / Warning / Complete) and verification pill counts
- Four tabs: Verification Checks, Scenario Simulation chart, RAG Policy Sources, SOC2 Audit log
- HITL banner with Approve (green) / Reject (red) buttons for SOFT and HARD tier runs
- Fixed bottom strip: cumulative cost, tokens, latency, and judge verdict

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 0.2 — StateGraph, MemorySaver, interrupt/resume |
| LLMs | Claude Haiku 4.5 (Agents 1–3, 6) · Claude Sonnet 4.6 (Agent 4) · Claude Sonnet 4.5 (Judge) |
| RAG | FAISS vector store + BM25 EnsembleRetriever |
| Structured output | Pydantic v2 models for PO and judge output |
| UI | Streamlit 1.45 — live streaming via `st.empty()`, JS button styling |
| API | FastAPI — run, resume, status, and audit endpoints |
| Audit | Append-only SOC2 JSONL with integrity hash per event |
| Observability | LangSmith tracing (optional) |
| Data | Kaggle supply chain datasets + synthetic data generator |
| Evals | RAGAS faithfulness + context precision + E2E objective checks |
| Deploy | Docker + docker-compose |

---

## Project Structure

```
Supply_Chain_AI_Agent/
├── agents/
│   ├── _llm.py               # Unified LLM factory (Claude / GPT-4o routing)
│   ├── state.py              # Shared TypedDict + Pydantic models
│   ├── sensing.py            # Agent 1 — inventory level, disruption detection
│   ├── historical_trend.py   # Agent 2 — 3-year trend analysis from Kaggle CSV
│   ├── simulation.py         # Agent 3 — three what-if scenario model
│   ├── optimization.py       # Agent 4 — RAG-grounded PO recommendation
│   ├── validation.py         # Agent 5 — Claude blind judge (3 dimensions)
│   └── summarizer.py         # Agent 6 — plain-English executive summary
├── tools/
│   └── supply_chain_tools.py # Simulated SAP tools + per-run session cache
├── rag/
│   ├── document_loader.py    # PDF download, chunking, and stubs
│   └── retriever.py          # FAISS + BM25 EnsembleRetriever
├── audit/
│   └── soc2_logger.py        # Append-only SOC2 JSONL logger
├── api/
│   └── main.py               # FastAPI — run / resume / status / audit endpoints
├── ui/
│   └── dashboard.py          # Streamlit two-page dashboard
├── data/
│   ├── data_mapper.py        # Kaggle CSV loader and SKU catalog builder
│   ├── generate_synthetic_data.py
│   └── kaggle/               # CSV files (gitignored)
├── evals/
│   └── ragas_eval.py         # RAGAS + E2E evaluations
├── graph.py                  # LangGraph StateGraph, HITL gate, retry wrapper
├── config.py                 # Central config — all env vars resolved at import
├── demo.py                   # CLI demo script (3 scenarios)
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Anthropic API key (required)
- OpenAI API key (optional — system falls back to Claude-only if absent)

### 1. Clone and configure

```bash
git clone git@github.com:indranisen03/Supply_Chain_AI_Agent.git
cd Supply_Chain_AI_Agent
cp .env.example .env
# Add your ANTHROPIC_API_KEY (and optionally OPENAI_API_KEY) to .env
```

### 2. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run ui/dashboard.py
```

Open `http://localhost:8501`. Select a scenario preset and click **Run procurement analysis**.

### 4. Run the full stack (API + UI)

```bash
# Terminal 1 — FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit
streamlit run ui/dashboard.py --server.port 8501
```

API docs at `http://localhost:8000/docs`.

### 5. Docker

```bash
docker-compose up --build
```

---

## Scenario Presets

| Preset | SKU | Warehouse | Inventory | Expected Outcome |
|---|---|---|---|---|
| Normal | SKU-0000 | WH-Detroit | 350 units | AUTO tier, healthy reorder |
| Port Strike | SKU-0026 | WH-Chicago | 5 units | SOFT/HARD, disruption detected |
| High Value | SKU-0047 | WH-Dallas | 8 units | HARD tier, explicit approval |
| Critical Stockout | SKU-0000 | WH-Detroit | 10 units | SOFT/HARD, urgent order |

---

## HITL Tiers

| Tier | PO Value | Behaviour |
|---|---|---|
| AUTO | < $10,000 | Proceeds immediately, logged |
| SOFT | $10k – $50k | Auto-approves after 12-hour timeout; overrideable |
| HARD | > $50,000 | Pipeline blocked until explicit human approval |

---

## Confidence Scoring

The Validation agent scores the PO recommendation across three dimensions and reports a three-tier confidence result:

| Tier | Threshold | Meaning |
|---|---|---|
| PASS | ≥ 0.85 | High confidence — proceed |
| WARN | 0.65 – 0.85 | Moderate — human review recommended |
| FAIL | < 0.65 | Low confidence — order flagged |

---

## SOC2 Audit Trail

Every run appends structured events to `audit/audit_trail.jsonl`:

```json
{
  "run_id": "uuid",
  "timestamp": "2026-05-10T09:14:02.341Z",
  "agent": "OPTIMIZATION",
  "action": "agent_complete",
  "output_summary": { "quantity": 800, "hitl_tier": "HARD" },
  "verification_passed": true,
  "cost_usd": 0.0014,
  "tokens_used": 480,
  "integrity_hash": "a3f92b1c..."
}
```

The audit log is excluded from version control (runtime-generated). It can be downloaded as CSV directly from the dashboard SOC2 Audit tab.

---

## Evaluations

```bash
python -m evals.ragas_eval
```

Checks:
- `qty_coverage` — recommended qty ≥ forecast + safety stock
- `timeline_valid` — required-by date ≥ today + lead time days
- `cost_per_run` — < $0.50 per full pipeline run
- `judge_score` — > 7.0 for PASS verdict
- RAGAS `faithfulness` and `context_precision` on RAG-grounded reasoning

---

## License

MIT License — personal portfolio project.
