# Federal Reserve Macro Insights MVP

Notebook-first, local/open-source pipeline for gathering Federal Reserve communications (PDFs) and generating a structured macroeconomic view (inflation, unemployment, growth, policy rates, risk framing) with retrieval-backed citations.

## 1. Purpose

This restart focuses on a **minimum viable product** that is:
- US/Federal Reserve focused
- Free/open-source stack only
- Reproducible from a single Jupyter notebook
- Easy to extend into a web app later

Primary goal:
1. Retrieve publicly available Federal Reserve communication PDFs.
2. Parse content and build a local retrieval index.
3. Use a local LLM end-to-end for structured macro insight generation.

## 2. Scope of v1

Included:
- Discovery of Federal Reserve PDFs via:
  - Seed-page scraping
  - Known filename/date pattern probing
- PDF download + local storage
- Text extraction and chunking
- Semantic embeddings (SentenceTransformers)
- FAISS vector retrieval
- Local LLM synthesis through Ollama
- Basic output traceability checks (JSON parse + citation ID validation)
- Artifact persistence for reproducibility

Not included yet:
- UI/web app
- Scheduling/orchestration
- Advanced evaluation framework (RAGAS/TruLens)
- Quantitative time-series extraction and charting

## 3. Technology choices (open source)

- `requests` + `beautifulsoup4` for web scraping
- `pypdf` for PDF text extraction
- `sentence-transformers` for embeddings
- `faiss-cpu` for vector similarity search
- `scikit-learn` for sparse lexical retrieval (TF-IDF)
- `json-repair` for robust recovery of malformed JSON generations
- `ollama` Python client for local LLM inference
- `pyarrow` for Parquet support (with CSV fallback in notebook if unavailable)
- `pandas`/`numpy` for data handling

Why this stack:
- No paid API dependency required
- Runs locally and supports iteration speed
- Keeps architecture simple enough for MVP but extensible

## 4. Folder structure

Everything for this restart is inside this folder.

```text
fed_macro_mvp/
├── README.md
├── fed_macro_mvp.ipynb
├── fed_macro_mvp_streamlined.ipynb
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── validation.py
│   ├── analysis.py
│   ├── artifacts.py
│   └── pipeline.py
├── data/
│   ├── raw_pdfs/
│   └── processed/
├── index/
└── outputs/
```

Generated during notebook runs:
- `data/processed/download_manifest.csv`
- `data/processed/chunks.parquet`
- `index/fed_chunks.index`
- `index/fed_chunks_meta.parquet`
- `index/index_config.json`
- `outputs/hits_<timestamp>.csv`
- `outputs/macro_answer_<timestamp>.txt`
- `outputs/macro_answer_<timestamp>.json` (if JSON parse succeeds)

## 5. Environment prerequisites

1. Python 3.10+ recommended.
2. Jupyter Notebook or JupyterLab.
3. Ollama installed and running locally.
4. At least one local model pulled in Ollama.

Example:
```bash
ollama serve
ollama pull mistral:7b-instruct
```

If your model tag differs, update `OLLAMA_MODEL` in the notebook config cell.

## 6. Step-by-step runbook

Open:
- `fed_macro_mvp/fed_macro_mvp_streamlined.ipynb` (recommended)
- `fed_macro_mvp/fed_macro_mvp.ipynb` (legacy/full notebook retained)

Run cells top-to-bottom:

1. **Install dependencies**
   - Executes `%pip install ...`
   - Safe to skip if already installed.

2. **Imports + configuration**
   - Creates required subfolders:
     - `data/raw_pdfs`
     - `data/processed`
     - `index`
     - `outputs`
   - Set operational values:
     - `DAYS_BACK`
     - `MAX_PDFS`
     - `TOP_K`
     - `OLLAMA_MODEL`
   - Tune latency/quality controls:
     - `TOP_K_TOPIC`
     - `CONTEXT_CHUNKS_PER_TOPIC`
     - `MAX_CHARS_PER_CHUNK`
     - `MAX_CONTEXT_CHARS`
     - `OLLAMA_NUM_PREDICT`
     - `OLLAMA_NUM_CTX`
     - `OLLAMA_TEMPERATURE`
     - `ENABLE_HYBRID_RETRIEVAL`
     - `ENABLE_QUERY_FUSION`
     - `ENABLE_RERANKER`
     - `RERANK_MODEL_NAME`
     - `OLLAMA_MAX_RETRIES`
     - `RETRY_CONTEXT_SHRINK`
     - `RETRY_PREDICT_SHRINK`
     - `USE_JSON_REPAIR`

3. **Discover + download Federal Reserve PDFs**
   - Scrapes PDF links from configured Fed seed pages.
   - Probes common Fed filenames by date:
     - `fomcminutesYYYYMMDD.pdf`
     - `monetaryYYYYMMDDa1.pdf`
   - Combines + deduplicates + date-sorts.
   - Downloads up to `MAX_PDFS`.
   - Writes `download_manifest.csv`.

4. **Parse + chunk + index**
   - Extracts text from each downloaded PDF.
   - Chunks long text (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
   - Flags topic keyword presence.
   - Embeds chunks with SentenceTransformers.
   - Builds FAISS index.
   - Saves index + metadata + index config.

5. **Retrieval + local LLM macro synthesis**
   - Runs targeted retrieval per investor-relevant macro topic:
     - inflation
     - unemployment
     - growth
     - policy rates
     - financial conditions
     - credit
   - Uses robust hybrid retrieval:
     - dense semantic retrieval (FAISS)
     - sparse lexical retrieval (TF-IDF)
     - reciprocal-rank fusion (RRF)
   - Optional cross-encoder reranking on top candidate pool (safe fallback when disabled or unavailable).
   - Uses query-fusion per topic with deterministic query variants.
   - Applies recency-aware scoring so newer Fed communication receives moderate preference.
   - Extracts topic-focused snippets to reduce context length and latency.
   - Reserves context budget across topics so one compact evidence chunk per topic is favored before adding more detail.
   - Sends compact, structured context to Ollama with speed-oriented options (`num_predict`, `num_ctx`, low temperature).
   - Requests investor-grade strict JSON output containing:
     - regime call
     - topic signals
     - investor takeaways by horizon
     - evidence links (`chunk_id`-based citations)
     - citation quotes (short verbatim snippets from retrieved chunk text)
   - Prints retrieval and LLM latency to make runtime bottlenecks explicit.
   - Uses retry fallback for generation: if parsing/quality fails, retries with smaller context and lower `num_predict`.
   - Uses robust JSON parsing with balanced-object extraction and optional `json-repair`.
   - Includes final JSON-repair fallback via a short local LLM “repair pass” when direct parsing fails.
   - Applies deterministic post-processing to coerce near-valid outputs into schema-compliant JSON:
     - topic normalization (`policy_rate` -> `policy_rates`, etc.)
     - chunk ID normalization/fixing for common format drift
     - evidence/citation backfill from retrieved topic hits when sparse
   - Notebook display now prints normalized/coerced JSON (`analysis_result['normalized_json_text']`) when available.
   - `generated_at_utc` is normalized to current-run ISO UTC format.
   - Section 4 displays a citation preview table (`doc_id`, `chunk_id`, `quote`) for investor readability.

6. **Basic MVP quality checks**
   - Parse JSON from model output.
   - Verify schema shape and section presence.
   - Verify citation and evidence chunk IDs exist in retrieved context.
   - Verify expected macro topics are covered.

7. **Save artifacts**
   - Writes retrieval table and model outputs under `outputs/`.

## 7. Current implementation status

As of **March 15, 2026**:

- [x] New clean folder created for restart.
- [x] End-to-end notebook implemented in `fed_macro_mvp.ipynb`.
- [x] Local/open-source components only in current workflow.
- [x] Retrieval-backed generation with citation validation hooks.
- [x] Output persistence for reproducibility.
- [x] Hybrid retrieval (dense + sparse + RRF) with recency-aware ranking.
- [x] Topic query-fusion for investor-focused evidence gathering.
- [x] Optional cross-encoder reranking stage with safe fallback.
- [x] Core analysis split into `retrieval.py`, `generation.py`, and `validation.py` for cleaner notebooks and maintainability.
- [ ] Real run validation against live Federal Reserve endpoints in this environment.
- [x] Prompt/schema hardening for stricter JSON reliability.
- [ ] Better source filtering (document type/date relevance ranking).

## 8. Known limitations in v1

- Discovery currently emphasizes:
  - FOMC minutes
  - Monetary policy report patterns
  - PDF links found on seed pages
- Some Federal Reserve communications are HTML/non-PDF and are intentionally out-of-scope for this MVP.
- PDF text extraction quality depends on source formatting (scanned docs can degrade extraction).
- Citation quality is retrieval-dependent and should be reviewed before production use.

## 9. Troubleshooting

1. **No PDFs downloaded**
   - Check internet connectivity from notebook environment.
   - Confirm Federal Reserve pages are reachable.
   - Increase `DAYS_BACK` or `MAX_PDFS`.

2. **FAISS build fails**
   - Ensure `faiss-cpu` installed successfully.
   - Restart kernel after install cell.

3. **Ollama connection error**
   - Confirm `ollama serve` is running.
   - Confirm model is pulled locally.
   - Check `OLLAMA_HOST` and `OLLAMA_MODEL`.

4. **Model output not valid JSON**
   - Reduce temperature.
   - Tighten prompt instructions.
   - Keep retry logic enabled (`OLLAMA_MAX_RETRIES`).
   - Reduce `MAX_CONTEXT_CHARS` and `OLLAMA_NUM_PREDICT`.
   - Keep `USE_JSON_REPAIR = True`.
   - Keep `CONTEXT_CHUNKS_PER_TOPIC = 1` for faster, more stable structure completion.

5. **Section 4 is too slow**
   - Reduce `TOP_K_TOPIC` (for example `3 -> 2`).
   - Reduce `MAX_CHARS_PER_CHUNK` and `MAX_CONTEXT_CHARS`.
   - Reduce `OLLAMA_NUM_PREDICT`.
   - Keep `ENABLE_RERANKER = False` for fastest baseline.
   - Use a faster local model if needed.

## 10. Enhancement Priority (Robustness First)

Ordered by implementation realism and low bug risk.

1. **Hybrid retrieval + query-fusion + recency scoring** (`Implemented`)
2. **Schema-constrained investor output + strict validation** (`Implemented`)
3. **Cross-encoder reranking on top-N candidates** (`Implemented, optional toggle`)
4. **Evaluation harness (RAGAS + TruLens) integrated into run path** (`Next`)
5. **Broader source coverage (Fed speeches/testimony) + source quality filters** (`Next`)
6. **Scheduled incremental ingestion + index refresh** (`Next`)
7. **Web interface over stable notebook workflow** (`Later`)

## 11. Living change log

### 2026-03-15
- Initialized clean restart folder `fed_macro_mvp/`.
- Built notebook `fed_macro_mvp.ipynb` with end-to-end MVP pipeline.
- Added this comprehensive README as the implementation source of truth.
- Added parquet resilience: default `pyarrow` dependency plus automatic CSV fallback for table persistence.
- Tightened generation schema and validation checks so section-level `evidence` is validated as retrieved `chunk_id` references, not free-text claims.
- Refactored Section 4 for speed and investor use-case:
  - topic-targeted retrieval
  - context compaction per topic
  - explicit investor-grade JSON schema (`regime_call`, `topic_signals`, `investor_takeaways`)
  - stronger validation for evidence IDs and citations
- Added Ollama schema-constrained generation (`format=<JSON schema>`) for more reliable structured output.
- Added topic coverage validation so outputs are checked for all expected macro topics.
- Added robustness-first retrieval stack:
  - dense + sparse hybrid retrieval
  - reciprocal-rank fusion (RRF)
  - deterministic per-topic query fusion
  - recency-aware ranking boost
- Added optional reranker tier (`ENABLE_RERANKER`) using a cross-encoder over top candidates with safe fallback.
- Added explicit `scikit-learn` dependency for sparse retrieval.
- Added generation retry/fallback logic to reduce JSON truncation failures under slow local inference.
- Added robust parse recovery path (`extract_balanced_json` + `json-repair`) for malformed/partially emitted JSON.
- Added a compact generation contract and conservative retry schedule to reduce truncation and malformed JSON rates on CPU-bound local models.
- Refactored notebook code into modular Python files under `core/` and added a streamlined orchestrator notebook (`fed_macro_mvp_streamlined.ipynb`).
