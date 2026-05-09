# Stellantis Supply Chain AI

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.74-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)
![Claude](https://img.shields.io/badge/Claude-Sonnet_4.5-purple)
![GPT4o](https://img.shields.io/badge/GPT-4o-teal)
![SOC2](https://img.shields.io/badge/SOC2-Audit_Trail-gold)

Production-grade multi-agent AI system for Stellantis procurement decisions. Demonstrated to TCS Business Lead as a working prototype.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  STELLANTIS SUPPLY CHAIN AI PIPELINE                 │
│                                                                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ Agent 1  │──▶│ Agent 2  │──▶│ Agent 3  │──▶│    Agent 4       │ │
│  │ Sensing  │   │Historical│   │Simulation│   │  Optimization    │ │
│  │GPT-4o-m  │   │Trend     │   │GPT-4o-m  │   │  GPT-4o + RAG   │ │
│  │          │   │GPT-4o-m  │   │          │   │  FAISS+BM25     │ │
│  └──────────┘   └──────────┘   └──────────┘   └────────┬─────────┘ │
│                                                          │            │
│  ┌───────────────────────────────────────────────────────▼──────────┐ │
│  │                    TIERED HITL GATE                               │ │
│  │  🟢 AUTO (<$10k): log + proceed immediately                      │ │
│  │  🟡 SOFT ($10k-$50k): proceed after 12hr timeout                │ │
│  │  🔴 HARD (>$50k): full block — explicit approval required        │ │
│  └───────────────────────────────────┬───────────────────────────────┘ │
│                                       │                                  │
│  ┌────────────────────────────────────▼──────────────────────────────┐ │
│  │ Agent 5: Validation (Claude Sonnet 4.5 — independent 2nd opinion) │ │
│  │  Scores: quantity_justified / timeline_realistic / rag_grounded   │ │
│  └────────────────────────────────────┬──────────────────────────────┘ │
│                                        │                                 │
│  ┌─────────────────────────────────────▼─────────────────────────────┐ │
│  │ Agent 6: Executive Summarizer (GPT-4o-mini)                       │ │
│  └─────────────────────────────────────┬──────────────────────────────┘│
│                                         │                               │
│  ┌──────────────────────────────────────▼─────────────────────────────┐│
│  │              SOC2 AUDIT TRAIL (append-only JSONL)                  ││
│  │     Every tool call, verification, human decision, judge score     ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘

Observability: LangSmith tracing (LANGCHAIN_TRACING_V2=true)
RAG: FAISS + BM25 EnsembleRetriever (weights=[0.4, 0.6])
     Stellantis Code of Conduct + Purchasing Guidelines + FAR Part 12
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 0.2 (StateGraph + MemorySaver HITL) |
| LLMs | GPT-4o-mini (Agents 1,2,3,6) · GPT-4o (Agent 4) · Claude Sonnet 4.5 (Judge) |
| RAG | FAISS + BM25 EnsembleRetriever (BM25 weight=0.6) |
| Structured Output | Pydantic v2 + `.with_structured_output()` |
| Observability | LangSmith tracing |
| API | FastAPI (async, auto OpenAPI) |
| UI | Streamlit (live trace + metrics + approval) |
| Audit | Append-only SOC2 JSONL trail |
| Deploy | Docker + docker-compose |
| Data | Synthetic supply chain CSV (Kaggle-schema compatible) |
| Evals | RAGAS faithfulness + context_precision + E2E objective checks |

---

## Project Structure

```
supply_chain_agent/
├── agents/
│   ├── state.py              # Shared TypedDict + Pydantic models
│   ├── sensing.py            # Agent 1: inventory + disruption detection
│   ├── historical_trend.py   # Agent 2: Kaggle CSV trend analysis
│   ├── simulation.py         # Agent 3: 3 what-if scenario model
│   ├── optimization.py       # Agent 4: RAG-grounded PO recommendation
│   ├── validation.py         # Agent 5: Claude LLM judge
│   └── summarizer.py         # Agent 6: executive summary
├── tools/
│   └── supply_chain_tools.py # Simulated SAP tools + session cache
├── rag/
│   ├── document_loader.py    # PDF download + chunking + stubs
│   └── retriever.py          # FAISS + BM25 EnsembleRetriever
├── audit/
│   └── soc2_logger.py        # Append-only SOC2 JSONL
├── api/
│   └── main.py               # FastAPI (run/resume/status/audit)
├── ui/
│   └── dashboard.py          # Streamlit dashboard
├── data/
│   ├── generate_synthetic_data.py
│   ├── kaggle/               # supply_chain.csv (generated or real)
│   └── docs/                 # Downloaded Stellantis PDFs
├── evals/
│   └── ragas_eval.py         # RAGAS + E2E evaluations
├── graph.py                  # LangGraph StateGraph + HITL + retry
├── config.py                 # Central config + env vars
├── demo.py                   # 3-scenario demo script
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Setup

### Prerequisites
- Python 3.11+
- Docker + docker-compose (for containerized deployment)

### 1. Clone and configure

```bash
git clone <repo-url>
cd supply_chain_agent
cp .env.example .env
# Edit .env with your API keys
```

### 2. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python data/generate_synthetic_data.py
```

> The system also auto-generates data if the CSV is missing at runtime.
> To use the real Kaggle dataset, download from
> `kaggle.com/datasets/prashantk93/supply-chain-management-for-car`
> and save as `data/kaggle/supply_chain.csv`.

### 4. Run the demo

```bash
python demo.py --all
```

### 5. Start the full stack

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run ui/dashboard.py
```

Open `http://localhost:8501` for the dashboard, `http://localhost:8000/docs` for OpenAPI.

### 6. Docker deployment

```bash
docker-compose up --build
```

---

## Demo Scenarios

| Scenario | SKU | Supplier | Trigger | Expected HITL |
|---|---|---|---|---|
| Normal Reorder | SKU-4821 | SUP-001 | Inventory below safety stock | SOFT ($9k) |
| High-Value HARD Block | SKU-2024 | SUP-004 | ADAS sensor, unit_cost=$320 | **HARD** (>$50k) |
| Port Strike | SKU-4821 | SUP-002 | Active disruption + low inventory | HARD |

```bash
python demo.py --scenario normal
python demo.py --scenario high_value
python demo.py --scenario port_strike
```

---

## Data Schema (Synthetic CSV)

Column | Type | Maps To
---|---|---
`sku_id` | str | Part identifier
`supplier_id` | str | Supplier
`monthly_demand` | int | Demand signal
`inventory_level` | int | Current stock
`safety_stock` | int | Safety threshold
`lead_time_days` | int | Supplier lead time
`on_time_delivery` | 0/1 | On-time flag
`unit_cost` | float | Unit price
`seasonal_factor` | float | Seasonal multiplier
`year`, `month` | int | Time dimension

---

## HITL Tiers

| Tier | Threshold | Behavior | UI |
|---|---|---|---|
| 🟢 AUTO | < $10,000 | Proceeds immediately | Green badge |
| 🟡 SOFT | $10k–$50k | Auto-approves after 12hr countdown | Yellow badge + timer |
| 🔴 HARD | > $50,000 | Full block — approval button required | Red badge + button |

---

## SOC2 Audit Trail

Every run appends to `audit/audit_trail.jsonl`:
```json
{
  "run_id": "uuid",
  "timestamp": "2026-05-08T...",
  "agent": "OPTIMIZATION",
  "action": "agent_complete",
  "input_summary": null,
  "output_summary": {"quantity": 800, "hitl_tier": "HARD"},
  "verification_passed": true,
  "cost_usd": 0.0012,
  "tokens_used": 420,
  "integrity_hash": "a3f92b1c..."
}
```

---

## Evaluations

```bash
# Run E2E evals (requires a completed run)
python -m evals.ragas_eval
```

Metrics checked:
- `qty_coverage`: recommended_qty ≥ forecast + safety_stock
- `timeline_valid`: required_by > today + lead_time_days
- `cost_per_run`: < $0.50
- `judge_score`: > 7.0 (PASS verdict)
- RAGAS `faithfulness` + `context_precision`

---

## <!-- DEMO GIF PLACEHOLDER -->

![Demo GIF](docs/demo.gif)

*Architecture diagram and demo recording coming soon.*

---

## License

Proprietary — Stellantis / TCS prototype. Not for distribution.
