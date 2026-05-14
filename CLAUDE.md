# CLAUDE.md — Supply Chain Multi-Agent AI System

A multi-agent supply chain AI procurement recommendation system with tiered HITL oversight,
role-based approval authority, an independent LLM judge, BM25 RAG grounded in real procurement
policy documents, procedural memory, and a full SOC2 audit trail.

---

## Pipeline Order

```
Sensing → Historical Trend → Simulation → Optimization
       → Cost Check → Validation → HITL Gate → Summarizer → SOC2 Finalize
```

**Critical:** Validation runs BEFORE the HITL gate. The judge score can escalate the tier
(AUTO → SOFT when score < 7.0) before the human sees the banner. Do not move validation
after the HITL gate — it breaks the escalation logic in `agents/validation.py`.

LangGraph node names: `sensing`, `hist_trend`, `simulation`, `optimization`, `cost_check`,
`validation`, `hitl_gate`, `summarizer`, `soc2_finalize`.

**Note:** `hist_trend` is intentionally different from the state key `historical_trend`.
LangGraph raises an error if a node name matches a state key — do not rename it back.

---

## Agents

| # | Node | Model | Role |
|---|---|---|---|
| 1 | sensing | Claude Haiku 4.5 | Inventory level, disruption detection via tool calls |
| 2 | hist_trend | Claude Haiku 4.5 | Demand delta, supplier on-time rate from Kaggle CSV |
| 3 | simulation | Claude Haiku 4.5 | 3 what-if scenarios (order today / +7d / +14d) |
| 4 | optimization | Claude Sonnet 4.6 | BM25 RAG → Pydantic PurchaseOrderRecommendation |
| 5 | validation | GPT-4o (Claude Sonnet 4.5 fallback) | Blind judge: quantity, timeline, grounding (1–10 each) |
| 6 | summarizer | Claude Haiku 4.5 | Plain-English 5–6 sentence executive summary |

Model routing: `_USE_CLAUDE_ONLY = True` when `OPENAI_API_KEY` is absent or set to `none`.
All model constants are in `config.py` → `MODEL_MINI`, `MODEL_LARGE`, `MODEL_JUDGE`.
Use `get_llm("mini" | "large" | "judge")` from `agents/_llm.py` — never instantiate models directly.

---

## HITL Gate — All Tiers Interrupt

**ALL three tiers pause the pipeline and require explicit human confirmation.**
No tier auto-proceeds. `human_approved` is only ever set `True` by an explicit human click.

| Tier | PO Value | Escalation window | Notes |
|---|---|---|---|
| AUTO | < $10,000 | 24hr → SOFT | Optional |
| SOFT | $10k–$50k | 12hr → HARD | Mandatory |
| HARD | > $50,000 | None — hard block | Mandatory |

Tier thresholds are dynamic: `memory/hitl_memory.py` adjusts caps based on recent approval
patterns (loaded via `agents/policy.py → determine_hitl_tier()`). Static defaults are in
`config.py → HITL_AUTO_MAX_USD / HITL_SOFT_MAX_USD`.

### Role-based authority

| Role | AUTO | SOFT | HARD | Demo key |
|---|---|---|---|---|
| analyst | — | — | — | `ANALYST-001` |
| coordinator | ✓ | — | — | `COORD-001` |
| sr_manager_l5 | ✓ | ✓ | — | `SRMGR-001` |
| director_l6 | ✓ | ✓ | ✓ | `DIR-001` |

- Role permissions defined in `agents/policy.py → ROLE_TIER_PERMISSIONS`
- Key resolution in `config.py → resolve_approver_role(key)`
- Dashboard: key input in HITL banner → role badge → buttons unlock if authorised
- API: `X-Approver-Key` header on `POST /resume/{thread_id}` — 403 if wrong role for tier
- Demo keys active when `APPROVER_KEYS` env var is not set; override with
  `APPROVER_KEYS=key1:role,key2:role` in `.env` for production

---

## Shared State (`SupplyChainState` TypedDict)

Key fields — full definition in `agents/state.py`:

```
sku_id, warehouse, run_id, supplier_id
inventory_level, safety_stock_threshold, unit_cost
disruption_detected, disruption_details
historical_trend, demand_forecast, lead_time_days
simulation_scenarios, decision_window_days
po_recommendation          # serialised PurchaseOrderRecommendation
hitl_tier                  # AUTO | SOFT | HARD
human_approved             # True ONLY from explicit human click
hitl_deadline              # ISO-8601; AUTO=24hr, SOFT=12hr, HARD=None
escalation_tier            # e.g. "AUTO→SOFT" if deadline exceeded
hitl_notes                 # mandatory for SOFT/HARD
rejection_reason           # populated when human_approved=False; logged to SOC2
approver_role              # role used to approve/reject; stored for SOC2
pipeline_halted            # True = stop all downstream processing
judge_score, judge_verdict, judge_flags, judge_scores_detail
executive_summary
verification_logs          # "[AGENT] check: result ✓/✗"
decision_trail             # plain English per step
eval_metrics               # {agent: {latency_ms, tokens, cost_usd}}
tokens_used, cost_usd
error, retry_count
_tool_cache
```

---

## Pipeline Safety

| Guard | Behaviour |
|---|---|
| Cost cap | `cost_check_node` → sets `pipeline_halted=True` if `cost_usd > MAX_COST_PER_RUN_USD` (default $0.50). Hard stop — does not reach HITL gate. |
| RAG failure | `optimization_agent` returns `pipeline_halted=True` — never proceeds ungrounded |
| Validation failure | 0 retries — immediately escalates `hitl_tier` to HARD |
| Judge escalation | Score < 7.0 on AUTO → `hitl_tier` set to SOFT before HITL banner renders |
| Per-agent retries | Sensing/Hist/Sim: 2 · Optimization: 1 · Summarizer: 3 · Validation: 0 |

Retry limits defined in `agents/policy.py → AGENT_MAX_RETRIES`.
`_with_retry()` in `graph.py` wraps every agent node and skips immediately if `pipeline_halted`.

---

## Key Design Decisions

- **BM25-only RAG** — FAISS removed. Supplier codes and contract IDs need exact string
  matching; semantic similarity adds noise. Do not re-add FAISS.
- **Validation before HITL** — judge score influences the tier the human sees. This is
  intentional — a low-confidence AUTO recommendation gets a SOFT review, not a rubber stamp.
- **Blind judge** — `validation_agent` receives only raw inputs + PO recommendation, never
  `decision_trail`. This prevents earlier agents from anchoring the verdict.
- **All tiers interrupt** — there is no auto-proceed path anywhere in the graph. Any code
  that sets `human_approved=True` without an explicit human input is a bug.
- **Rejection reason mandatory on reject** — dashboard enforces it via notes; API enforces
  it via 422 validation. Always logged to SOC2.
- **Anthropic prompt caching** — optimization agent sends system prompt + RAG context as a
  single cached block when the model is Claude. Requires ≥1024 tokens to activate.

---

## RAG — Local-First Loading

Priority: LOCAL (`data/docs/{name}.pdf`) → URL download → STUB (prints WARNING to stdout).

Place real PDFs in `data/docs/` to avoid stubs:
```
stellantis_code_of_conduct.pdf
global_responsible_purchasing.pdf
supplier_management_principles.pdf
```
FAR Part 12 downloads automatically from the web. BM25 index pickled to
`data/bm25_retriever.pkl`; summary table saved to `data/bm25_summary.json` (gitignored).

Force rebuild: `get_retriever(force_rebuild=True)` in `rag/retriever.py`.
`force_rebuild=True` rebuilds the index but still loads LOCAL files first — it never skips
manually placed PDFs in favour of URL downloads.

---

## Procedural Memory

- `memory/supplier_memory.py` — on-time rate, lead time, disruption frequency per supplier.
  Updated in `soc2_finalize_node` after each completed run. Injected into Optimization prompt.
- `memory/hitl_memory.py` — approval/rejection patterns per tier. Adjusts AUTO/SOFT caps
  dynamically after ≥5 decisions in a tier. Consumed by `determine_hitl_tier()`.
- Both JSON files are gitignored (runtime-generated).

---

## Project Structure

```
Supply_Chain_AI_Agent/
├── agents/
│   ├── _llm.py               # LLM factory — always use get_llm(), never instantiate directly
│   ├── policy.py             # Role permissions, tier logic, per-agent retry limits
│   ├── state.py              # SupplyChainState TypedDict + Pydantic output models
│   ├── sensing.py            # Agent 1
│   ├── historical_trend.py   # Agent 2
│   ├── simulation.py         # Agent 3
│   ├── optimization.py       # Agent 4 — RAG + prompt caching + supplier memory
│   ├── validation.py         # Agent 5 — blind judge + AUTO→SOFT escalation
│   └── summarizer.py         # Agent 6
├── memory/
│   ├── supplier_memory.py
│   └── hitl_memory.py
├── tools/
│   └── supply_chain_tools.py # Simulated SAP tools + per-run cache
├── rag/
│   ├── document_loader.py    # Local-first PDF loading, startup summary table
│   └── retriever.py          # BM25Retriever, query-level cache, summary persistence
├── audit/
│   └── soc2_logger.py        # Append-only JSONL, SHA-256 integrity hash per event
├── api/
│   └── main.py               # FastAPI — /run, /resume (role-gated), /status, /audit
├── ui/
│   └── dashboard.py          # Streamlit — role-auth HITL widget, pipeline_halted banner
├── data/
│   └── docs/                 # Drop PDFs here for LOCAL RAG loading
├── evals/
│   └── ragas_eval.py
├── graph.py                  # StateGraph, retry wrapper, HITL gate, routing logic
├── config.py                 # All env vars + resolve_approver_role()
└── requirements.txt
```

---

## Running Locally

```bash
cp .env.example .env
# Required: ANTHROPIC_API_KEY
# Optional: OPENAI_API_KEY (enables GPT-4o judge), LANGCHAIN_API_KEY

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Dashboard only
streamlit run ui/dashboard.py --server.headless true

# Full stack
uvicorn api.main:app --reload --port 8000
streamlit run ui/dashboard.py --server.headless true
```

Streamlit: `http://localhost:8501` | FastAPI docs: `http://localhost:8000/docs`

---

## Evals

```bash
python -m evals.ragas_eval
```

- `qty_coverage` — recommended qty ≥ forecast + safety stock
- `timeline_valid` — required-by ≥ today + lead_time_days
- `cost_per_run` — < $0.50
- `judge_score` — > 7.0 for PASS
- RAGAS `faithfulness` + `context_precision` on RAG-grounded reasoning

---

## Commit Convention

Messages follow: `type: description` (feat / fix / docs / refactor / chore).
Trailer: `Made with help from Claude Code` (no co-author email tags).
