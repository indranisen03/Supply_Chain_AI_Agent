# Supply Chain AI Agent

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.74-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)
![Claude](https://img.shields.io/badge/Claude-Sonnet_4.6-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![SOC2](https://img.shields.io/badge/SOC2-Audit_Trail-gold)

A multi-agent supply chain AI procurement recommendation system with tiered HITL oversight, role-based approval authority, an independent LLM judge, RAG-grounded policy compliance, procedural memory, and a full SOC2 audit trail.

---

## Objective

**Situation:**
Automotive companies lose millions when parts are reordered too late, causing production delays with little visibility into how procurement decisions were made.

**Task:**
Build a working AI system that monitors SKU inventory across warehouses, surfaces purchase order recommendations before risk materialises, and routes decisions through a tiered human-in-the-loop (HITL) approval gate with a full audit trail.

**Action:**
Designed and built a 6-agent LangGraph pipeline with BM25 RAG grounded in real Stellantis procurement documents, a tiered HITL gate (AUTO / SOFT / HARD) with role-based approval authority, an independent LLM judge for blind validation, procedural memory for supplier and approval patterns, and a two-page Streamlit dashboard — all within a SOC2-compliant JSONL audit trail.

**Result:**
System runs end-to-end in under 60 seconds, generates RAG-grounded PO recommendations with confidence scores, triggers the correct HITL tier based on order value, enforces role-based approval authority, and produces a downloadable audit log per run.

**Data:**
Simulated using a Kaggle supply chain dataset (DataCo); RAG grounded in real Stellantis public procurement documents. Production deployment would swap the CSV for live SAP/ERP feeds.

---

## What It Does

The system continuously monitors SKU inventory levels across warehouses. When a stockout risk is detected, a 6-agent AI pipeline activates to:

1. **Sense** current inventory, disruptions, and supplier status
2. **Analyse** historical demand and supplier performance trends
3. **Simulate** three what-if scenarios (order today vs. wait 7 vs. 14 days)
4. **Recommend** a purchase order grounded in Stellantis procurement policy documents
5. **Validate** the recommendation with an independent LLM judge (blind evaluation) — low scores escalate the HITL tier before the human sees it
6. **Gate** the decision through a tiered HITL approval — only authorised roles can act on each tier
7. **Summarise** the full decision trail in plain English for procurement teams

Every decision routes through a tiered approval gate — agents prepare recommendations only and never execute purchases autonomously. Humans always confirm.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SUPPLY CHAIN AI PIPELINE                        │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ Agent 1  │─▶│ Agent 2  │─▶│ Agent 3  │─▶│     Agent 4        │  │
│  │ Sensing  │  │Historical│  │Simulation│  │  Optimization      │  │
│  │          │  │  Trend   │  │          │  │  + RAG + Memory    │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┬───────────┘  │
│                                                      │              │
│                                              ┌───────▼───────────┐  │
│                                              │   Cost Cap Check  │  │
│                                              │  Hard stop if     │  │
│                                              │  cost > $0.50/run │  │
│                                              └───────┬───────────┘  │
│                                                      │              │
│  ┌───────────────────────────────────────────────────▼───────────┐  │
│  │              Agent 5 — Validation Judge                       │  │
│  │  GPT-4o (Claude Sonnet 4.5 fallback) — blind evaluation       │  │
│  │  quantity_justified / timeline_realistic / reasoning_grounded │  │
│  │  Score out of 10 → PASS / FAIL                                │  │
│  │  Judge score < 7.0 on AUTO order → escalates tier to SOFT     │  │
│  └───────────────────────────────────────────┬───────────────────┘  │
│                                              │                      │
│  ┌───────────────────────────────────────────▼───────────────────┐  │
│  │                    TIERED HITL GATE                           │  │
│  │  ALL tiers pause — agents NEVER execute purchases autonomously│  │
│  │                                                               │  │
│  │  AUTO  (< $10k)    fast-track · escalates to SOFT after 24hr  │  │
│  │  SOFT  ($10k–$50k) 12hr window · notes mandatory · → HARD     │  │
│  │  HARD  (> $50k)    full block · no timeout · notes mandatory  │  │
│  │                                                               │  │
│  │  Role-based authority enforced at gate and API level:         │  │
│  │  Coordinator → AUTO only                                      │  │
│  │  Sr. Manager (L5) → AUTO + SOFT                               │  │
│  │  Director (L6) → AUTO + SOFT + HARD                           │  │
│  └───────────────────────────────────────────┬───────────────────┘  │
│                                              │  (approved only)     │
│  ┌───────────────────────────────────────────▼───────────────────┐  │
│  │        Agent 6 — Executive Summarizer (Claude Haiku 4.5)      │  │
│  │        5–6 sentence plain-English summary for procurement     │  │
│  └───────────────────────────────────────────┬───────────────────┘  │
│                                              │                      │
│  ┌───────────────────────────────────────────▼───────────────────┐  │
│  │              SOC2 AUDIT TRAIL  (append-only JSONL)            │  │
│  │  Every tool call, verification check, human decision,         │  │
│  │  approver role, rejection reason, judge score, token count,   │  │
│  │  cost, and integrity hash                                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

RAG:            BM25Retriever — local-first (data/docs/*.pdf → URL → stub)
                Stellantis Code of Conduct, Purchasing Guidelines,
                Supplier Management Principles, FAR Part 12
Memory:         supplier_memory.json — on-time rates, lead times per supplier
                hitl_memory.json — approval patterns, dynamic tier thresholds
Observability:  LangSmith tracing (opt-in via LANGCHAIN_TRACING_V2=true)
Self-healing:   Per-agent retry limits with LLM debug reasoning on failure
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
- Single-row PO summary strip: order qty, value, supplier, required-by date, confidence, HITL tier
- Pipeline panel with per-agent status and verification pill counts
- Four tabs: Verification Checks, Scenario Simulation chart, RAG Policy Sources, SOC2 Audit log
- HITL banner with role authentication — enter your approver key to unlock Approve/Reject; buttons stay disabled if your role lacks authority for the current tier
- Pipeline halted banner when cost cap or RAG failure stops the run early
- Fixed bottom strip: cumulative cost, tokens, latency, and judge verdict

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 0.2.74 — StateGraph, MemorySaver, interrupt/resume |
| LLMs | Claude Haiku 4.5 (Agents 1–3, 6) · Claude Sonnet 4.6 (Agent 4) · GPT-4o / Claude Sonnet 4.5 (Judge) |
| RAG | BM25Retriever — local-first PDF loading, query-level cache |
| Structured output | Pydantic v2 models for PO and judge output |
| UI | Streamlit 1.45 — live streaming, role-auth widget, JS button styling |
| API | FastAPI 0.115 — run, resume (role-gated), status, audit endpoints |
| Audit | Append-only SOC2 JSONL with SHA-256 integrity hash per event |
| Memory | JSON-backed procedural memory for supplier and HITL patterns |
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
│   ├── policy.py             # Role permissions, HITL tier logic, per-agent retry limits
│   ├── state.py              # Shared TypedDict + Pydantic models
│   ├── sensing.py            # Agent 1 — inventory level, disruption detection
│   ├── historical_trend.py   # Agent 2 — demand and supplier trend analysis
│   ├── simulation.py         # Agent 3 — three what-if scenario model
│   ├── optimization.py       # Agent 4 — RAG-grounded PO recommendation + prompt caching
│   ├── validation.py         # Agent 5 — independent LLM judge, AUTO→SOFT escalation
│   └── summarizer.py         # Agent 6 — plain-English executive summary
├── memory/
│   ├── supplier_memory.py    # Per-supplier on-time rate, lead time, disruption history
│   └── hitl_memory.py        # Approval patterns, dynamic AUTO/SOFT threshold adjustment
├── tools/
│   └── supply_chain_tools.py # Simulated SAP tools + per-run session cache
├── rag/
│   ├── document_loader.py    # Local-first PDF loading (LOCAL → URL → stub), startup table
│   └── retriever.py          # BM25Retriever, query-level cache, startup summary
├── audit/
│   └── soc2_logger.py        # Append-only SOC2 JSONL logger
├── api/
│   └── main.py               # FastAPI — run / resume (role-gated) / status / audit
├── ui/
│   └── dashboard.py          # Streamlit two-page dashboard, role-auth HITL widget
├── data/
│   ├── docs/                 # Place Stellantis PDFs here for local RAG loading
│   ├── data_mapper.py        # Kaggle CSV loader and SKU catalog builder
│   └── kaggle/               # CSV files (gitignored)
├── evals/
│   └── ragas_eval.py         # RAGAS + E2E evaluations
├── graph.py                  # LangGraph StateGraph, HITL gate, retry wrapper
├── config.py                 # Central config — env vars, approver keys, role resolution
├── demo.py                   # CLI demo script
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
# Add ANTHROPIC_API_KEY (and optionally OPENAI_API_KEY) to .env
```

### 2. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Add real policy documents

Place Stellantis PDFs in `data/docs/` for full RAG grounding:

```
data/docs/stellantis_code_of_conduct.pdf
data/docs/global_responsible_purchasing.pdf
data/docs/supplier_management_principles.pdf
```

If absent, the system falls back to curated stub text with a console warning. FAR Part 12 is fetched from the web automatically. A startup summary table shows which source each document loaded from.

### 4. Run the dashboard

```bash
streamlit run ui/dashboard.py --server.headless true
```

Open `http://localhost:8501`. Select a scenario preset and click **Run procurement analysis**. When the HITL banner appears, enter an approver key to authenticate before acting.

### 5. Run the full stack (API + UI)

```bash
# Terminal 1 — FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit
streamlit run ui/dashboard.py --server.headless true
```

API docs at `http://localhost:8000/docs`.

### 6. Docker

```bash
docker-compose up --build
```

---

## Scenario Presets

| Preset | SKU | Warehouse | Inventory | Expected Tier |
|---|---|---|---|---|
| Normal | SKU-0000 | WH-Detroit | 350 units | AUTO |
| Port Strike | SKU-0026 | WH-Chicago | 5 units | HARD (disruption) |
| High Value | SKU-0047 | WH-Dallas | 8 units | HARD |
| Critical Stockout | SKU-0000 | WH-Detroit | 10 units | SOFT / HARD |

---

## HITL Tiers and Role Authority

Agents **never** execute purchases autonomously. All tiers interrupt the pipeline and wait for explicit human confirmation. The tier controls urgency, blocking behaviour, and which roles can act.

| Tier | PO Value | Escalation | Notes |
|---|---|---|---|
| AUTO | < $10,000 | → SOFT after 24hr inaction | Optional |
| SOFT | $10k–$50k | → HARD after 12hr inaction | **Mandatory** |
| HARD | > $50,000 | No timeout | **Mandatory** |

A judge score below 7.0 on an AUTO order automatically escalates it to SOFT before the human sees the banner.

### Role permissions

| Role | AUTO | SOFT | HARD | Demo key |
|---|---|---|---|---|
| Analyst | — | — | — | `ANALYST-001` |
| Coordinator | ✓ | — | — | `COORD-001` |
| Sr. Manager (L5) | ✓ | ✓ | — | `SRMGR-001` |
| Director (L6) | ✓ | ✓ | ✓ | `DIR-001` |

Demo keys are active when `APPROVER_KEYS` is not set in `.env`. For production, set:

```
APPROVER_KEYS=realkey1:coordinator,realkey2:sr_manager_l5,realkey3:director_l6
```

Role and rejection reason are written to the SOC2 audit trail on every decision.

---

## Procedural Memory

The system builds up knowledge across runs:

**Supplier memory** (`memory/supplier_memory.json`) — tracks on-time delivery rate, average lead time, disruption frequency, and average PO value per supplier. This context is injected into the Optimization agent's prompt to improve recommendations over time.

**HITL memory** (`memory/hitl_memory.json`) — records approval and rejection patterns per tier. After enough runs, the AUTO and SOFT value thresholds adjust dynamically: if AUTO rejections exceed 30% of recent decisions, the AUTO cap lowers by 10%; if SOFT approvals are consistently high, the SOFT cap rises. Both memory files are excluded from version control (runtime-generated).

---

## Pipeline Safety

| Guard | Behaviour |
|---|---|
| Cost cap | Hard stop if cumulative LLM cost exceeds `MAX_COST_PER_RUN_USD` (default $0.50) — sets `pipeline_halted`, skips HITL gate |
| RAG failure | Hard stop if BM25 retrieval fails — prevents ungrounded recommendations |
| Validation failure | Zero retries — immediately escalates HITL tier to HARD |
| Per-agent retries | Sensing/Hist/Sim: 2 · Optimization: 1 · Summarizer: 3 |
| Self-healing | On retry, a mini LLM call diagnoses the failure and appends guidance to the verification log |

---

## RAG — Local-First Loading

On startup the retriever prints a summary of how each document was loaded:

```
──────────────────────────────────────────────────
  RAG Document Sources
──────────────────────────────────────────────────
  Document                        Source      Chunks
  --------------------------------------------------
  stellantis_code_of_conduct      📁 LOCAL  119
  global_responsible_purchasing   📁 LOCAL   49
  supplier_management_principles  📁 LOCAL   14
  far_part_12                     🌐 URL    154
──────────────────────────────────────────────────
```

Priority order: **LOCAL** (file in `data/docs/`) → **URL** (download) → **STUB** (curated fallback, prints a console warning). The BM25 index is pickled after the first build and reloaded on subsequent starts.

---

## SOC2 Audit Trail

Every run appends structured events to `audit/audit_trail.jsonl`:

```json
{
  "run_id": "uuid",
  "timestamp": "2026-05-10T09:14:02.341Z",
  "agent": "HITL_GATE",
  "action": "human_decision",
  "output_summary": {
    "human_confirmed": true,
    "notes": "Reviewed and approved — supplier on-time rate acceptable",
    "rejection_reason": null
  },
  "verification_passed": true,
  "cost_usd": 0.0,
  "tokens_used": 0,
  "integrity_hash": "a3f92b1c...",
  "human_confirmed": true,
  "escalation_tier": null
}
```

Fields logged per decision: `approver_role`, `hitl_tier`, `escalation_tier`, `rejection_reason`, `judge_score`, `cost_usd`, `tokens_used`. The audit log is excluded from version control. It can be downloaded as CSV from the dashboard SOC2 Audit tab.

---

## Confidence Scoring

The Validation agent scores the PO recommendation across three dimensions:

| Dimension | What it checks |
|---|---|
| `quantity_justified` | Order qty supported by inventory and forecast data |
| `timeline_realistic` | Required-by date achievable given lead time |
| `reasoning_grounded` | Reasoning free of hallucination, cites real sources |

| Result | Threshold |
|---|---|
| PASS | Overall score > 7.0 |
| FAIL | Overall score ≤ 7.0 |

A FAIL verdict on an AUTO tier order triggers escalation to SOFT before the HITL banner renders.

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
