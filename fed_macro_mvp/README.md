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
- `fed_macro_mvp/fed_macro_mvp.ipynb`

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
   - Embeds user query.
   - Retrieves top-k chunks from FAISS.
   - Sends context + instructions to Ollama model.
   - Requests strict JSON output containing:
     - executive summary
     - inflation/unemployment/growth/policy-rates/risk views
     - evidence links (`chunk_id`-based citations)

6. **Basic MVP quality checks**
   - Parse JSON from model output.
   - Verify section presence.
   - Verify citation chunk IDs exist in retrieved context.

7. **Save artifacts**
   - Writes retrieval table and model outputs under `outputs/`.

## 7. Current implementation status

As of **March 15, 2026**:

- [x] New clean folder created for restart.
- [x] End-to-end notebook implemented in `fed_macro_mvp.ipynb`.
- [x] Local/open-source components only in current workflow.
- [x] Retrieval-backed generation with citation validation hooks.
- [x] Output persistence for reproducibility.
- [ ] Real run validation against live Federal Reserve endpoints in this environment.
- [ ] Prompt/schema hardening for stricter JSON reliability.
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
   - Add a response-repair pass in a future iteration.

## 10. Next recommended enhancements

1. Add Federal Reserve speech/testimony PDF sources with metadata filtering.
2. Add re-ranking to improve retrieval precision.
3. Add periodic ingestion (daily/weekly) and incremental index update.
4. Add dedicated evaluation dataset for macro topics.
5. Build a thin web interface once notebook behavior is stable.

## 11. Living change log

### 2026-03-15
- Initialized clean restart folder `fed_macro_mvp/`.
- Built notebook `fed_macro_mvp.ipynb` with end-to-end MVP pipeline.
- Added this comprehensive README as the implementation source of truth.
- Added parquet resilience: default `pyarrow` dependency plus automatic CSV fallback for table persistence.
