# Federal Reserve Macro Insights MVP

A notebook-first, local/open-source Retrieval-Augmented Generation (RAG) system that ingests Federal Reserve communications (PDFs), retrieves high-signal evidence, and generates a structured investor-focused U.S. macro view.

## 1) What This Project Is
This section defines the project in one sentence and sets expectations.

This repository is built as a practical MVP for interview-level walkthroughs and real local usage:
- It scrapes and probes Federal Reserve PDF sources.
- It extracts/chunks text and builds a local retrieval index.
- It retrieves topic-specific evidence for macro themes (inflation, unemployment, growth, policy rates, financial conditions, credit).
- It uses a local Ollama model to produce strict JSON analysis with citations and quote snippets.
- It includes a technical notebook and a user-facing investor UI notebook.

## 2) Problem Statement and Product Goal
This section explains why the architecture exists and what outcome it optimizes for.

Goal:
1. Build a local, reproducible pipeline to analyze recent Fed communications.
2. Produce investor-usable macro outputs quickly (recency- and relevance-biased).
3. Keep architecture simple and robust enough to run repeatedly on a laptop.

Non-goals for this MVP:
- No external hosted LLM APIs.
- No production web backend yet.
- No heavy evaluation framework integration (RAGAS/TruLens deferred).

## 3) Current Status (as of March 18, 2026)
This section gives the current implementation state at a glance.

Implemented:
- End-to-end ingestion -> indexing -> retrieval -> generation -> validation -> persistence.
- Runtime profiles (`fast_default`, `full_default`).
- Hybrid retrieval (dense FAISS + sparse TF-IDF + RRF fusion).
- Optional reranker stage.
- Robust JSON handling (balanced extraction, json-repair, retry/repair flow).
- Evidence hygiene (topic evidence enforcement, citation normalization, quote snippets).
- Investor UI notebook backed by modular Python code.
- Tests for profile behavior, catalog filtering, and evidence coercion/validation.

Deferred:
- Full incremental indexing workflow.
- Page-level PDF citations.
- Evaluation suite (RAGAS/TruLens) and historical backtesting dashboard.

## 4) Project Layout
This section is your file-system map for quick orientation.

```text
fed_macro_mvp/
├── README.md
├── requirements.txt
├── fed_macro_v1.ipynb                # legacy notebook (kept for history)
├── fed_macro_v2.ipynb                # technical workflow notebook
├── fed_macro_v3_investor_ui.ipynb    # user-facing interactive notebook UI
├── core/
│   ├── __init__.py
│   ├── config.py                     # central config + profile logic + paths
│   ├── ingest.py                     # discovery, filtering, download
│   ├── indexing.py                   # parse PDFs, chunk text, embeddings, FAISS
│   ├── retrieval.py                  # dense+sparse retrieval, RRF, recency, context build
│   ├── generation.py                 # Ollama generation + retries + JSON parsing/repair
│   ├── validation.py                 # schema coercion, evidence/citation checks, normalization
│   ├── analysis.py                   # orchestration of retrieval + generation + metrics
│   ├── artifacts.py                  # output save utilities
│   ├── observability.py              # structured diagnostics events + run artifacts
│   ├── pipeline.py                   # notebook-facing pipeline wrappers
│   └── investor_ui.py                # ipywidgets UI components and callbacks
├── docker-compose.yaml               # local observability stack (QuestDB/Telegraf/Grafana)
├── telegraf.conf                     # diagnostics tail/parse/forward config
├── Makefile                          # stack operation shortcuts
├── grafana/
│   └── provisioning/                 # datasource + dashboard provisioning
├── tests/
│   └── test_pipeline_enhancements.py
├── data/
│   ├── raw_pdfs/
│   └── processed/
├── index/
└── outputs/
```

Generated artifacts:
- `data/processed/download_manifest.csv`
- `data/processed/chunks.parquet` (or CSV fallback)
- `index/fed_chunks.index`
- `index/fed_chunks_meta.parquet` (or CSV fallback)
- `index/index_config.json`
- `outputs/hits_<timestamp>.csv`
- `outputs/macro_answer_<timestamp>.txt`
- `outputs/macro_answer_<timestamp>.json` (when parse/coercion succeeds)

## 5) Tech Stack and Why
This section maps each dependency to its job in the system.

Core libraries:
- `requests`, `beautifulsoup4`: scraping and URL discovery.
- `pypdf`: PDF text extraction.
- `sentence-transformers`: embeddings for dense retrieval.
- `faiss-cpu`: local vector index/search.
- `scikit-learn`: TF-IDF sparse retrieval.
- `ollama`: local LLM inference client.
- `json-repair`: malformed JSON recovery.
- `pyarrow`: parquet support (with CSV fallback).
- `ipywidgets`: interactive notebook UI.
- `pandas`, `numpy`, `tqdm`: data/metrics/iteration utilities.
- `telegraf`, `questdb`, `grafana` (Dockerized): observability stack for structured pipeline telemetry.

Design tradeoff:
- Chosen for local runnability and robustness over maximal model sophistication.

## 6) Runtime Profiles and Config Model
This section explains how scope and speed are controlled.

`core/config.py` defines `PipelineConfig` and defaults.

### 6.1 `fast_default` (recommended day-to-day)
- `days_back=180`
- `max_pdfs=24`
- `allowed_doc_types=["fomc_minutes", "mpr"]`

Why: faster turnaround and fresher macro signal for investor workflows.

### 6.2 `full_default`
- `days_back=540`
- `max_pdfs=40`
- `allowed_doc_types=["all"]`

Why: broader history and coverage when speed is less important.

### 6.3 Important knobs for section-4 latency/quality
- `top_k_topic`
- `context_chunks_per_topic`
- `max_chars_per_chunk`
- `max_context_chars`
- `ollama_num_predict`
- `enable_reranker`
- retry controls: `ollama_max_retries`, `retry_context_shrink`, `retry_predict_shrink`

## 7) End-to-End Architecture (Internal Flow)
This section is the deep implementation walkthrough interviewers usually ask for.

The detailed architecture has been split into a dedicated living document:

- [ARCHITECTURE.md](ARCHITECTURE.md)

## 8) Notebooks and Intended Usage
This section tells you which notebook to open for what objective.

1. `fed_macro_v3_investor_ui.ipynb`
- Primary demo notebook for users/interview walkthrough.
- Launches interactive widgets from `core/investor_ui.py`.
- Best for showing investor-centric usage and controls.

2. `fed_macro_v2.ipynb`
- Technical notebook for pipeline execution with transparent dataframes and diagnostics.

3. `fed_macro_v1.ipynb`
- Legacy retained for historical context; not preferred for current flow.

## 9) Investor UI Behavior (`core/investor_ui.py`)
This section details what the UI does internally.

Controls:
- Profile selector (`fast_default` / `full_default`)
- Run mode (`refresh_all` / `analysis_only`)
- Topic focus (all or one macro topic)
- Speed-quality sliders (`top_k_topic`, `max_context_chars`, `ollama_num_predict`)
- Optional reranker toggle

Tabs:
1. Dashboard:
- investor brief
- regime cards (growth, inflation, policy, recession risk, confidence badge)
- run metrics and analysis metrics
- warning list

2. Topic Drilldown:
- topic signal table
- evidence quote table by selected topic
- investor takeaway cards

3. Normalized JSON:
- exact post-processed JSON object used for persistence

## 10) Running the Project
This section is the reproducible runbook.

### 10.1 Setup
```bash
cd /Users/atheeshkrishnan/AK/DEV/hawkdove/fed_macro_mvp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 10.2 Ollama readiness
```bash
ollama serve
ollama list
ollama pull llama3:8b
curl http://127.0.0.1:11434/api/tags
```

If model tag differs, update `cfg.ollama_model` in notebook config/override cell.

### 10.3 Notebook execution order
1. Open `fed_macro_v3_investor_ui.ipynb` for investor demo or `fed_macro_v2.ipynb` for technical run.
2. Run dependency/import/config cells.
3. Pick profile (`fast_default` recommended for iteration speed).
4. Run ingestion/indexing (or `analysis_only` if index already exists).
5. Run analysis and inspect attempt log + checks.
6. Save outputs.

### 10.4 Optional observability stack
This project now includes a local telemetry stack inspired by the `systematic-trade-monitor` pattern:

1. the macro pipeline emits structured JSON diagnostics to `outputs/diagnostics/<run_id>/events.jsonl`
2. `telegraf` tails those files
3. `questdb` stores time-series telemetry
4. `grafana` visualizes run/stage health and latency

Start the stack:
```bash
cd /Users/atheeshkrishnan/AK/DEV/hawkdove/fed_macro_mvp
make up
make ps
```

Endpoints:
- Grafana: `http://localhost:3000`
- QuestDB UI / SQL API: `http://localhost:9000`
- QuestDB ILP: `localhost:9009`
- QuestDB Postgres wire: `localhost:8812`

Verification flow:
1. Run a macro analysis locally from notebook/UI.
2. Confirm `outputs/diagnostics/<run_id>/events.jsonl` exists.
3. Run `make count` and confirm rows are present in `rag_events`.
4. Open Grafana and confirm panels populate.

## 11) Data Contracts (Output Schema)
This section is useful for interview Q&A on reliability and integration readiness.

Top-level JSON keys:
- `generated_at_utc`
- `executive_summary`
- `regime_call`
- `topic_signals`
- `investor_takeaways`
- `citations`

Constraints:
- `topic_signals` must represent exactly:
  - `inflation`, `unemployment`, `growth`, `policy_rates`, `financial_conditions`, `credit`
- Evidence IDs must map to retrieved valid `chunk_id`s.
- Citations are deduplicated and capped.
- Citation quote snippets are populated from retrieved chunk text when available.

## 12) Testing
This section shows what is currently protected by automated tests.

Run tests:
```bash
cd /Users/atheeshkrishnan/AK/DEV/hawkdove/fed_macro_mvp
python -m unittest -q tests/test_pipeline_enhancements.py
```

Current coverage:
- Profile defaults and profile switching behavior.
- Catalog filtering by doc type and date window.
- Evidence minimum enforcement and citation quote generation.
- Observability schema and diagnostics artifact persistence.

Operational smoke tests:
- `docker compose up -d`
- `make count`
- inspect `outputs/diagnostics/<run_id>/events.jsonl`
- open Grafana dashboard `Fed Macro RAG Observability`

## 13) Major Issues Encountered and How They Were Resolved
This section documents real implementation failures and practical fixes.

1. Parquet engine ImportError (`pyarrow`/`fastparquet` missing)
- Symptom: `DataFrame.to_parquet` failed during chunk persistence.
- Fix:
  - added `pyarrow` to requirements
  - implemented parquet->CSV fallback in indexing save path
- Result: pipeline no longer blocks on parquet availability.

2. Ollama connection/model mismatch
- Symptom: `ConnectionError: Failed to connect to Ollama` and model not found errors.
- Fix:
  - added explicit Ollama readiness check (`/api/tags`)
  - clearer error messages for missing model tag
  - standardized default model to `llama3:8b`
- Result: fail-fast diagnostics before long pipeline stages.

3. Long and fragile generation stage (JSON parse failures)
- Symptom: frequent `parse_failed`, truncated JSON, malformed shapes.
- Fix:
  - retry plan with shrinking context/predict budgets
  - balanced JSON extraction
  - optional `json-repair`
  - final LLM repair pass
  - deterministic coercion/normalization after parse
- Result: significantly higher success rate and fewer hard failures.

4. Invalid evidence/citation IDs from model output
- Symptom: IDs not matching retrieved chunk format; validation failures.
- Fix:
  - ID resolver for common formatting drift
  - topic-based fallback evidence injection
  - strict validation report and quality gate
- Result: evidence/citation integrity became stable.

5. Non-ISO or stale timestamps in generated JSON
- Symptom: model emitted invalid or outdated `generated_at_utc`.
- Fix:
  - force timestamp server-side during post-processing (`ISO UTC`).
- Result: consistent run-time timestamp semantics.

6. Notebook install-cell error with `ipywidgets>=8.1.0`
- Symptom: `zsh: 8.1.0 not found` due to shell interpretation.
- Fix:
  - moved widget dependency into `requirements.txt`
  - simplified notebook install cell to `-r requirements.txt`.
- Result: stable dependency install cell behavior.

7. Notebook bloat and maintainability issues
- Symptom: overly long cells, hard debugging, low readability.
- Fix:
  - extracted logic to `core/*.py` modules
  - created UI module `core/investor_ui.py`
  - reduced notebook to orchestration/presentation calls
- Result: cleaner notebooks and testable modules.

8. Nested project-path confusion (`fed_macro_mvp/fed_macro_mvp`)
- Symptom: path resolution inconsistencies in some runs.
- Fix:
  - improved notebook marker detection in `PipelineConfig.__post_init__`.
- Result: more resilient path initialization.

## 14) Performance Notes and Practical Defaults
This section gives realistic operating guidance.

Observed practical guidance:
- Keep `fast_default` for iterative runs.
- If LLM stage is slow, reduce:
  - `max_context_chars`
  - `ollama_num_predict`
  - `top_k_topic`
- Keep reranker disabled for CPU-only quick runs.
- Use `analysis_only` mode when index already exists.

## 15) Known Limitations and Next Steps
This section is candid about MVP boundaries and realistic improvements.

Current limitations:
- PDF text extraction lacks page-coordinate citation granularity.
- Retrieval quality is good for MVP but not benchmarked with formal RAG metrics yet.
- Index rebuild can be expensive on repeated full-refresh runs.
- Observability dashboards are local/dev-oriented and do not yet include alerting or hardened auth.

Near-term enhancements (practical):
1. Incremental ingestion/indexing cache (skip unchanged docs/chunks by content hash).
2. Page-level citation support and richer evidence rendering in UI.
3. Lightweight evaluation suite (topic coverage, citation precision proxy, latency tracking over runs).

## 16) Living Document Note
This section states maintenance expectations.

This README is intended to be updated whenever:
- architecture changes,
- config defaults change,
- schema contracts change,
- major failure modes/fixes are discovered.

If this project is resumed after a gap, start with:
1. Section 6 (profiles),
2. Section 7 (architecture flow),
3. Section 13 (issues/fixes),
4. Section 15 (interview walkthrough).
