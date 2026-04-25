# Federal Reserve Macro Insights MVP: End-to-End Architecture

This document expands the original README Section 7 into a deeper technical walkthrough of the implemented system. It is intended to be the living architecture reference for explaining the project in a technical data-science interview: what problem the pipeline solves, how evidence moves from Federal Reserve PDFs into model-ready context, how retrieval and generation are controlled, and where the current MVP is intentionally incomplete.

The source of truth is the project notebook flow, especially `fed_macro_v2.ipynb` for the technical run and `fed_macro_v3_investor_ui.ipynb` for the interactive investor-facing interface. The notebooks are thin by design: they expose the execution path and diagnostics while delegating reusable logic to `core/*.py`.

## 1) System Objective and Current Shape

The project builds a local Retrieval-Augmented Generation pipeline over recent Federal Reserve communications. The practical objective is not to forecast markets directly; it is to convert a small, changing corpus of Fed PDFs into a structured, evidence-cited macro read that can support investor interpretation across inflation, labor, growth, policy rates, financial conditions, and credit.

The implemented flow is:

1. Discover and download recent Fed PDFs.
2. Extract text, chunk it, embed chunks, and build a FAISS vector index.
3. Retrieve evidence separately for each macro topic using dense and sparse signals.
4. Build a compact, topic-balanced context window.
5. Ask a local Ollama model for strict JSON.
6. Parse, repair, coerce, validate, and persist the output.
7. Display results in either a technical notebook or an investor UI notebook.

Key notebook anchor from `fed_macro_v2.ipynb`:

```python
from pathlib import Path
import pandas as pd

from core.pipeline import create_config, run_ingest_and_index, run_full_analysis, persist_results

cfg = create_config(Path.cwd())
print('Project dir:', cfg.project_dir)
print('Model:', cfg.ollama_model)
print('Profile:', cfg.profile_name)
print('Days back / Max PDFs:', cfg.days_back, '/', cfg.max_pdfs)
print('Allowed doc types:', cfg.allowed_doc_types)
```

The main design choice is to keep the notebooks readable and interview-friendly while moving implementation details into modules that can be tested and reused. `core/pipeline.py` is the notebook-facing adapter, while `core/ingest.py`, `core/indexing.py`, `core/retrieval.py`, `core/generation.py`, `core/validation.py`, `core/analysis.py`, and `core/artifacts.py` own the actual stages.

## 2) Configuration as the Experiment Contract

`PipelineConfig` in `core/config.py` is the central experiment contract. It controls data scope, document types, chunking, retrieval depth, context size, local LLM settings, retry behavior, and validation strictness.

The two profiles encode the speed-quality tradeoff:

- `fast_default`: 180-day lookback, up to 24 PDFs, restricted to FOMC minutes and Monetary Policy Report documents.
- `full_default`: 540-day lookback, up to 40 PDFs, all discovered document types.

Notebook anchor from `fed_macro_v2.ipynb`:

```python
cfg.set_profile('fast_default')  # change to 'full_default' for broader, slower runs

cfg.top_k_topic = 2
cfg.max_context_chars = 2400
cfg.ollama_num_predict = 380
cfg.enable_reranker = False

cfg.ollama_max_retries = 3
cfg.retry_context_shrink = [1.0, 0.85, 0.7]
cfg.retry_predict_shrink = [1.0, 1.0, 0.9]
cfg.use_json_repair = True
```

Conceptually, the profile defines the sampling frame for the corpus. In a macro setting, this matters because the same question can produce different signals depending on whether the system emphasizes recent communications or a wider historical window. The current default intentionally favors recency and faster iteration because the output is positioned as a 6 to 12 month investor macro view.

Concrete implementation map:

- `core/config.py`: `PipelineConfig`, topic query definitions, profile switching, runtime defaults.
- `core/pipeline.py`: `create_config`, `run_ingest_and_index`, `run_full_analysis`, `persist_results`.
- `fed_macro_v2.ipynb`: technical execution and diagnostics.
- `fed_macro_v3_investor_ui.ipynb`: UI launch path over the same config object.

## 3) Data Acquisition: Building the Fed Document Corpus

The ingestion stage starts from Federal Reserve web sources, not from a static file list. This gives the MVP a repeatable way to refresh the local evidence base as new communications are published.

The implemented acquisition strategy combines two methods:

1. Seed-page scraping: parse Fed monetary policy pages and collect PDF links.
2. Known-pattern probing: issue `HEAD` requests for date-patterned filenames such as `fomcminutesYYYYMMDD.pdf` and `monetaryYYYYMMDDa1.pdf`.

The pattern probing is a pragmatic complement to scraping. Federal Reserve files follow predictable names for important document families, so probing helps recover recent documents even when page structure or link text changes.

Core code anchor from `core/ingest.py`:

```python
def probe_known_patterns(cfg: PipelineConfig) -> pd.DataFrame:
    base = "https://www.federalreserve.gov/monetarypolicy/files/"
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; hawkdove-mvp/1.0)"})

    rows = []
    today = datetime.now(timezone.utc).date()

    for offset in tqdm(range(cfg.days_back), desc="Probing patterns"):
        dt = today - timedelta(days=offset)
        ymd = dt.strftime("%Y%m%d")

        for fname in [f"fomcminutes{ymd}.pdf", f"monetary{ymd}a1.pdf"]:
            url = base + fname
            try:
                r = session.head(url, allow_redirects=True, timeout=8)
            except Exception:
                continue

            ctype = (r.headers.get("Content-Type") or "").lower()
            if r.status_code == 200 and ("pdf" in ctype or url.endswith(".pdf")):
                rows.append(
                    {
                        "source": "known_pattern",
                        "source_page": "monetarypolicy/files",
                        "pdf_url": url,
                        "title": fname,
                        "date_hint": dt.isoformat(),
                        "doc_type": classify_doc_type(fname),
                    }
                )
```

After discovery, `apply_catalog_filters` enforces the active profile. It parses dates, drops rows with invalid dates, applies the lookback window, filters document types, computes age in days, and sorts by recency.

Light mathematical framing:

Let each candidate document be \(d_i\), with parsed date \(t_i\). Given run date \(T\) and profile lookback \(L\), the eligible corpus is:

$$
\mathcal{D}_{eligible} = \{d_i : T - L \le t_i \le T,\ \text{doc_type}(d_i) \in A\}
$$

where \(A\) is the allowed document-type set for the selected profile.

Current implementation status:

- Implemented: seed scraping, known-pattern probing, date filtering, document-type filtering, recency sorting, download manifest persistence.
- Partial: no content-hash based incremental refresh yet.
- Caveat: date hints are inferred from filenames or URLs. This is adequate for the currently targeted Fed PDFs but is not equivalent to authoritative publication metadata extraction from PDF content.

## 4) Text Extraction and Chunk Construction

Indexing converts PDFs into model-addressable evidence units. `core/indexing.py` uses `pypdf` to extract page text, normalizes whitespace, and slices each document into overlapping character windows.

Core code anchor from `core/indexing.py`:

```python
def read_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""

    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")

    return re.sub(r"\s+", " ", "\n".join(pages)).strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []

    out = []
    step = max(1, size - overlap)
    i = 0
    while i < len(text):
        j = min(len(text), i + size)
        out.append(text[i:j])
        if j == len(text):
            break
        i += step
    return out
```

The chunking strategy is deliberately simple. With chunk size \(S\), overlap \(O\), and step \(S - O\), chunk \(c_j\) covers:

$$
c_j = x_{j(S-O):\ j(S-O)+S}
$$

The overlap reduces the chance that important economic statements are split across adjacent chunks. This matters for Fed documents because a single sentence or paragraph often carries both the observed condition and the policy interpretation, such as labor-market cooling plus implications for inflation pressure.

Each chunk receives metadata:

- `chunk_id`: stable local evidence identifier, for example `fomcminutes20260301.pdf::chunk0001`.
- `doc_id`: PDF filename.
- `pdf_url`: source URL.
- `date_hint`: parsed date used for recency scoring.
- `doc_type`: classified Fed document family.
- `chunk_index`: position within the document.
- `topic_flags`: simple keyword-based flags for quick inspection.
- `text` and `text_len`: raw evidence payload and length.

Current implementation status:

- Implemented: PDF text extraction, whitespace normalization, overlapping chunks, metadata rows, parquet persistence with CSV fallback.
- Partial: no page-level chunk coordinates or page-number citation mapping.
- Caveat: `pypdf` extraction can lose tables, columns, and layout. This is especially relevant for projection tables and Monetary Policy Report exhibits.

## 5) Embeddings and Dense Indexing

The dense index represents each text chunk as a vector using `sentence-transformers/all-MiniLM-L6-v2`. Vectors are L2-normalized and stored in a FAISS `IndexFlatIP`. Because vectors are normalized, inner product search is equivalent to cosine similarity:

$$
\cos(q, c_i) = \frac{q^\top c_i}{\lVert q \rVert_2 \lVert c_i \rVert_2}
$$

After normalization, \(\lVert q \rVert_2 = \lVert c_i \rVert_2 = 1\), so:

$$
\cos(q, c_i) = q^\top c_i
$$

Core code anchor from `core/indexing.py`:

```python
def build_faiss_index(chunks_df: pd.DataFrame, cfg: PipelineConfig) -> dict[str, Any]:
    if chunks_df.empty:
        raise ValueError("No chunks available; run ingestion/chunking first.")

    model = SentenceTransformer(cfg.embed_model_name)
    vectors = model.encode(chunks_df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True).astype("float32")
    vectors = normalize_rows(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = cfg.index_dir / "fed_chunks.index"
    faiss.write_index(index, str(index_path))
```

Method rationale:

- Dense embeddings capture semantic similarity, which is useful when Fed language uses phrasing that does not exactly match the query terms.
- `IndexFlatIP` is exact search, not approximate search. For this MVP corpus size, exact search is simple, deterministic, and fast enough.
- L2 normalization keeps scoring interpretable and aligns FAISS inner product with cosine similarity.

Current implementation status:

- Implemented: sentence-transformer embeddings, L2 normalization, FAISS index persistence, metadata persistence, index config JSON.
- Caveat: the embedding model is general-purpose, not macro-domain fine-tuned.
- Caveat: the index is rebuilt from the current download manifest instead of incrementally merged.

## 6) Topic-Aware Hybrid Retrieval

The retrieval stage is the core data-science component of the system. Rather than asking one broad question over the entire corpus, the pipeline decomposes the macro view into fixed topics:

- `inflation`
- `unemployment`
- `growth`
- `policy_rates`
- `financial_conditions`
- `credit`

Each topic has a base query and deterministic query variants in `core/config.py`. Retrieval is run independently per topic, which creates balanced evidence coverage instead of allowing one high-signal theme, such as inflation, to dominate the entire context window.

Core code anchor from `core/retrieval.py`:

```python
def retrieve_topic_hybrid(topic: str, index, meta_df: pd.DataFrame, emb_model, sparse_bundle, reranker, cfg: PipelineConfig) -> pd.DataFrame:
    query_fused = []

    for query in build_topic_queries(topic, cfg):
        dense_df = dense_retrieve(query, index, meta_df, emb_model, top_k=cfg.candidates_per_query)
        if cfg.enable_hybrid_retrieval:
            sparse_df = sparse_retrieve(query, sparse_bundle, meta_df, top_k=cfg.candidates_per_query)
            fused = rrf_fuse([dense_df, sparse_df], score_col="score", k=cfg.rrf_k)
        else:
            fused = rrf_fuse([dense_df], score_col="score", k=cfg.rrf_k)

        if not fused.empty:
            query_fused.append(fused.rename(columns={"fusion_score": "score"}))

    merged = rrf_fuse(query_fused, score_col="score", k=cfg.rrf_k)
```

### 6.1 Dense Retrieval

Dense retrieval is implemented in `dense_retrieve`. For each topic query, the code:

1. encodes the query with the same sentence-transformer used for chunk embeddings,
2. L2-normalizes the query vector,
3. searches the FAISS `IndexFlatIP`,
4. maps returned row IDs back to `chunk_id` values in `meta_df`.

Core code anchor from `core/retrieval.py`:

```python
def dense_retrieve(query: str, index, meta_df: pd.DataFrame, emb_model, top_k: int) -> pd.DataFrame:
    q = emb_model.encode([query], convert_to_numpy=True).astype("float32")
    q = normalize_rows(q)
    scores, ids = index.search(q, top_k)
    out = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0 or i >= len(meta_df):
            continue
        out.append({"chunk_id": meta_df.iloc[int(i)]["chunk_id"], "score": float(score)})
```

At a conceptual level, dense retrieval embeds the topic query and each chunk into the same semantic vector space. The score is:

$$
s_i^{(\mathrm{dense})} = q^\top c_i
$$

where \(q\) is the normalized query embedding and \(c_i\) is the normalized embedding of chunk \(i\).

In this project, dense retrieval is especially useful when the macro concept is expressed indirectly. A chunk may talk about "moderating wage pressures" or "softening hiring demand" without explicitly saying "unemployment" or "labor market slack." Because the embedding model captures semantic proximity, these chunks can still surface even when the lexical overlap with the query is weak.

What dense retrieval does well here:

- captures paraphrases and concept-level similarity,
- recovers chunks that are relevant but do not share exact query terms,
- works well with topic queries phrased as investor-style questions rather than keyword bags.

What it does less well:

- it can miss exact policy phrasing if the semantic similarity is noisy,
- raw dense scores are not directly comparable to TF-IDF scores,
- it may retrieve semantically nearby but less term-specific chunks.

### 6.2 Sparse Retrieval

Sparse retrieval is implemented through `build_sparse_index` and `sparse_retrieve`. The code fits a `TfidfVectorizer` over all chunk texts using English stop-word removal and unigram/bigram features, then transforms each query into that same sparse vocabulary space.

Core code anchor from `core/retrieval.py`:

```python
def build_sparse_index(meta_df: pd.DataFrame):
    if not HAVE_SKLEARN:
        return None
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.95)
        mat = vec.fit_transform(meta_df["text"].fillna("").astype(str).tolist())
        return vec, mat
```

```python
def sparse_retrieve(query: str, sparse_bundle, meta_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if sparse_bundle is None:
        return pd.DataFrame(columns=["chunk_id", "score"])

    vec, mat = sparse_bundle
    qv = vec.transform([query])
    sims = (qv @ mat.T).toarray().ravel()
```

The weighting intuition is standard TF-IDF:

$$
\operatorname{tfidf}(t, d) = \operatorname{tf}(t, d) \cdot \operatorname{idf}(t)
$$

The query-document score is a dot product between query and document vectors:

$$
s_i^{(\mathrm{sparse})} = v_q^\top v_i
$$

where \(v_q\) is the TF-IDF query vector and \(v_i\) is the TF-IDF vector for chunk \(i\).

In this project, sparse retrieval is valuable because Fed communications often rely on repeated institutional phrases and domain terms. Exact expressions like "federal funds rate," "PCE inflation," "credit availability," "lending standards," or "Treasury yields" can be highly informative, and TF-IDF preserves those lexical anchors more faithfully than dense retrieval alone.

What sparse retrieval does well here:

- rewards exact macro and policy terminology,
- picks up bigrams that matter operationally in Fed language,
- is often sharper when the user query contains the same language as the source.

What it does less well:

- it is weaker on paraphrases and concept drift,
- it depends on vocabulary overlap,
- it can under-rank relevant chunks that describe the same concept with different wording.

### 6.2.1 Dense vs. Sparse in This Project

The two retrieval modes are complementary rather than competing.

- Dense retrieval answers: "Which chunks are semantically about this topic?"
- Sparse retrieval answers: "Which chunks literally use the words and phrases associated with this topic?"

For this corpus, dense retrieval is better at capturing concept-level macro discussion, while sparse retrieval is better at preserving institutional phrase matching. That matters because Fed text mixes both styles: some passages are formulaic and term-heavy, while others describe similar conditions in broader prose.

In practice:

- for `policy_rates`, sparse retrieval is often strong because phrases like "federal funds rate" and "restrictive stance" matter directly;
- for `growth` or `unemployment`, dense retrieval can be stronger because the relevant evidence may be phrased more diffusely;
- for `financial_conditions` and `credit`, both methods help because domain terms matter, but semantic drift across related wording is also common.

### 6.3 Reciprocal Rank Fusion

The project uses Reciprocal Rank Fusion because dense and sparse retrieval produce different score scales. FAISS similarity scores and TF-IDF dot products are not directly calibrated, so averaging raw scores would be unstable. RRF avoids that problem by operating on ranks instead of raw score magnitudes.

For chunk \(c\), across rankers \(R\), the fusion score is:

$$
\operatorname{RRF}(c) = \sum_{r \in R} \frac{1}{k + \operatorname{rank}_r(c)}
$$

Core code anchor from `core/retrieval.py`:

```python
def rrf_fuse(rank_frames: list[pd.DataFrame], score_col: str, k: int) -> pd.DataFrame:
    agg: dict[str, float] = {}
    for df in rank_frames:
        if df is None or df.empty:
            continue
        ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        for rank, cid in enumerate(ordered["chunk_id"].tolist(), start=1):
            agg[cid] = agg.get(cid, 0.0) + 1.0 / (k + rank)
```

How it applies in this project:

1. For each query variant, the pipeline retrieves a dense ranking and, when enabled, a sparse ranking.
2. Those rankings are fused with RRF into a per-query candidate list.
3. If query fusion is enabled, the pipeline then applies RRF again across the different query-variant result lists.

So retrieval is effectively a two-stage fusion design:

- stage 1: fuse retrieval modalities for one query,
- stage 2: fuse multiple phrasing variants of the same topic query.

This is important technically because it reduces sensitivity to any single query wording and any single retrieval mechanism.

Why RRF is a good fit here:

- RRF avoids over-trusting absolute score calibration across dense and sparse methods.
- It rewards chunks that rank well across multiple retrieval views.
- It also supports query fusion, where topic variants are retrieved separately and fused again.

What RRF is doing conceptually is rewarding consensus. A chunk does not need to be rank 1 everywhere; it only needs to appear consistently near the top across multiple rankers or query variants. That is a good inductive bias for this problem because robust Fed evidence should often survive several retrieval views.

### 6.4 Optional Reranking

If enabled, a cross-encoder reranker scores query-chunk pairs over the top candidate pool. The current default leaves this disabled for CPU-friendly runs.

Core code anchor from `core/retrieval.py`:

```python
def apply_reranker(query: str, candidates: pd.DataFrame, reranker, cfg: PipelineConfig) -> pd.DataFrame:
    if reranker is None or candidates.empty:
        return candidates

    work = candidates.head(cfg.rerank_candidate_pool).copy()
    pairs = [[query, str(t)[:1200]] for t in work["text"].fillna("").tolist()]

    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        print(f"[warn] rerank predict failed: {e}")
        return candidates
```

### 6.5 Recency Weighting

Fed communications are time-sensitive. A statement from the most recent meeting should usually matter more than an older statement with similar semantic relevance. The retrieval layer applies exponential decay:

$$
r_i = \exp\left(-\ln(2) \cdot \frac{a_i}{h}\right)
$$

where \(a_i\) is the age in days of document \(i\), and \(h\) is `recency_half_life_days`.

The final score combines normalized retrieval/rerank score with recency:

$$
s_i^{(\mathrm{final})} = s_i^{(\mathrm{base})} \cdot \left((1-b) + b \cdot r_i\right)
$$

where \(s_i^{(\mathrm{base})}\) is the normalized retrieval score after fusion or reranking, and \(b\) is `recency_boost`.

Core code anchor from `core/retrieval.py`:

```python
out["base_score_norm"] = minmax_scale(out[base_col])
out["recency"] = out["date_hint"].apply(lambda x: recency_score(x, cfg.recency_half_life_days))
out["final_score"] = out["base_score_norm"] * ((1.0 - cfg.recency_boost) + cfg.recency_boost * out["recency"])
return out.sort_values("final_score", ascending=False).reset_index(drop=True).head(cfg.top_k_topic)
```

This matters in the current project because the corpus is explicitly recent-Fed oriented. Without recency weighting, an older but lexically strong chunk could dominate a newer and more relevant statement. The decay does not replace relevance; it modulates it. A weakly relevant new chunk should still not outrank a highly relevant one, but among similarly relevant candidates, the newer one receives a systematic advantage.

Current implementation status:

- Implemented: dense retrieval, sparse retrieval, query fusion, RRF, optional reranker, recency decay, final topic-level top-k selection.
- Partial: no formal retrieval benchmark yet.
- Caveat: relevance is evaluated through proxy diagnostics and evidence inspection, not through a labeled relevance dataset.

## 7) Context Construction for Controlled Generation

After retrieval, the pipeline builds a compact context for the LLM. The context builder does two important things:

1. It extracts topic-focused snippets from each chunk rather than passing entire chunks blindly.
2. It budgets the context across topics so all required macro themes receive coverage.

Core code anchor from `core/retrieval.py`:

```python
def build_investor_context(topic_hits: dict[str, pd.DataFrame], cfg: PipelineConfig, max_context_chars: int | None = None) -> tuple[str, list[str]]:
    if max_context_chars is None:
        max_context_chars = cfg.max_context_chars

    sections, all_ids = [], []
    total_chars = 0

    ordered_topics = list(cfg.topic_queries.keys())
    for idx, topic in enumerate(ordered_topics):
        remaining_topics = max(1, len(ordered_topics) - idx)
        remaining_chars = max_context_chars - total_chars
        if remaining_chars <= 140:
            break

        per_topic_budget = max(140, int(remaining_chars / remaining_topics) - 60)
        section, ids = build_topic_context(
            topic,
            topic_hits.get(topic, pd.DataFrame()),
            cfg,
            max_chunks=max(1, cfg.context_chunks_per_topic),
            max_chars_per_chunk=min(cfg.max_chars_per_chunk, per_topic_budget),
        )
```

This is a practical anti-failure mechanism. Local models with limited context can degrade when overloaded, especially when asked to produce schema-valid JSON. The project therefore treats context as a scarce resource and allocates it deliberately.

Each context block includes machine-readable anchors:

```text
[topic=inflation; chunk_id=fomcminutes20260301.pdf::chunk0001; doc_id=fomcminutes20260301.pdf; score=0.8123]
...
```

These anchors become the valid evidence ID universe for later validation.

Current implementation status:

- Implemented: topic snippet extraction, context budget control, per-topic coverage, valid context ID tracking.
- Caveat: snippet extraction uses keyword sentence selection, not a learned summarizer.
- Caveat: a relevant chunk can be excluded if the context budget is too small.

## 8) Local LLM Generation and JSON Discipline

Generation is performed through a local Ollama model. Before generation, the system checks server reachability and verifies that the configured model exists.

Core code anchor from `core/generation.py`:

```python
def check_ollama_ready(cfg: PipelineConfig) -> None:
    try:
        r = requests.get(f"{cfg.ollama_host}/api/tags", timeout=3)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise RuntimeError(
            "Ollama is not reachable. Start Ollama first (e.g., `ollama serve` or open the Ollama app). "
            f"Host tried: {cfg.ollama_host}. Original error: {e}"
        )
```

The LLM prompt is intentionally narrow. It requires JSON only, fixed top-level keys, exactly six topic signals, limited investor takeaways, and citations whose `chunk_id` values must come from the provided context.

Core code anchor from `core/generation.py`:

```python
system = " ".join(
    [
        "Return valid JSON only.",
        "No markdown, no code fences, no extra text.",
        "Top keys: generated_at_utc, executive_summary, regime_call, topic_signals, investor_takeaways, citations.",
        "regime_call keys: growth_momentum, inflation_trend, policy_bias, recession_risk, confidence.",
        "topic_signals must contain exactly six topics: inflation, unemployment, growth, policy_rates, financial_conditions, credit.",
        "Each topic_signals item keys: topic, view, confidence, evidence.",
        "investor_takeaways: 1 or 2 items, each keys: horizon, thesis, evidence.",
        "citations: up to 8 items, each keys: chunk_id, doc_id.",
        "Every evidence and citation chunk_id must come from provided context chunk_id values.",
        "Use concise strings.",
    ]
)
```

The generation objective is not free-form essay quality. It is controlled synthesis under an output contract:

- `executive_summary`: concise investor-facing macro read.
- `regime_call`: compact labels for growth, inflation, policy bias, recession risk, and confidence.
- `topic_signals`: one row per required macro topic.
- `investor_takeaways`: one or two thesis statements with evidence.
- `citations`: chunk-level citations, later enriched with quote snippets.

Current implementation status:

- Implemented: local Ollama chat call, JSON mode, zero temperature, strict system prompt, token and context controls.
- Caveat: the LLM can still produce malformed JSON or weak evidence references, so downstream validation is required.

## 9) Retry, Parse, Repair, and Validation

The system treats generation as probabilistic and validates it like an external dependency. This is one of the most important engineering choices in the project.

`run_generation_with_retries` executes a retry plan. If parsing or quality checks fail, it retries with a smaller context and adjusted output budget. If all attempts fail, it performs one final LLM-based JSON repair pass.

Core code anchor from `core/generation.py`:

```python
def generation_retry_plan(cfg: PipelineConfig) -> list[dict[str, Any]]:
    plan = []
    for i in range(cfg.ollama_max_retries):
        ctx_factor = cfg.retry_context_shrink[min(i, len(cfg.retry_context_shrink) - 1)]
        pred_factor = cfg.retry_predict_shrink[min(i, len(cfg.retry_predict_shrink) - 1)]
        plan.append(
            {
                "attempt": i + 1,
                "context_chars": max(1400, int(cfg.max_context_chars * ctx_factor)),
                "num_predict": max(260, int(cfg.ollama_num_predict * pred_factor)),
            }
        )
```

Parsing uses multiple fallbacks:

1. Direct `json.loads`.
2. Balanced JSON object extraction.
3. Optional `json-repair`.
4. Final short LLM repair prompt.

Validation then checks:

- required top-level keys,
- correct shapes for lists and dictionaries,
- exactly the required topic set,
- evidence IDs against valid retrieved/context IDs,
- citation IDs against valid IDs.

Core code anchor from `core/validation.py`:

```python
def quality_ok(report: dict[str, Any]) -> bool:
    return (
        len(report["missing_top_keys"]) == 0
        and len(report["bad_shape"]) == 0
        and len(report["bad_evidence_ids"]) == 0
        and len(report["unknown_citation_ids"]) == 0
        and len(report["missing_topics"]) == 0
    )
```

The coercion layer is intentionally deterministic. It stamps the current run timestamp, normalizes topic names, resolves common chunk ID drift, backfills missing evidence from retrieval hits when configured, deduplicates citations, caps citations, and adds quote snippets from retrieved text.

Core code anchor from `core/validation.py`:

```python
normalized_signals = []
for topic in REQUIRED_TOPICS:
    src = existing_map.get(topic, {})
    ev = normalize_evidence_list(src.get("evidence", []), valid_ids, fallback_ids=fbt.get(topic, []), max_items=3)
    if enforce_topic_min_evidence and not ev:
        ev = normalize_evidence_list([], valid_ids, fallback_ids=global_fallback, max_items=1)
    normalized_signals.append(
        {
            "topic": topic,
            "view": str(src.get("view") or "No clear signal"),
            "confidence": str(src.get("confidence") or "medium"),
            "evidence": ev,
        }
    )
```

Conceptual rationale:

- The model is allowed to synthesize, but it is not allowed to define the schema.
- Evidence IDs must be tied to retrieved chunks, not invented.
- Quote snippets are produced by deterministic post-processing, not trusted to the model.

Current implementation status:

- Implemented: robust parse path, retry log, repair flow, schema coercion, evidence normalization, citation quote enrichment, validation report.
- Caveat: evidence backfill improves schema completeness but can hide a weak model response. Reviewers should inspect `attempt_log`, `quality`, and `citation_preview_df`.

## 10) Orchestration, Diagnostics, and Outputs

`core/analysis.py` is the main orchestration layer for analysis. It loads the index, builds the sparse index, optionally loads the reranker, retrieves topic evidence, calls generation, creates diagnostics, and returns a single result dictionary.

Notebook anchor from `fed_macro_v2.ipynb`:

```python
analysis_result = run_full_analysis(cfg)

print('Sparse enabled:', analysis_result['sparse_enabled'])
print('Reranker enabled:', analysis_result['reranker_enabled'])
print('Timings:', analysis_result['timings'])
if 'analysis_counts' in analysis_result:
    print('Analysis counts:', analysis_result['analysis_counts'])

display(analysis_result['topic_summary_df'])
if not analysis_result['hits_df'].empty:
    display(analysis_result['hits_df'][['topic', 'chunk_id', 'doc_id', 'final_score', 'recency']].head(12))
if not analysis_result['attempt_log'].empty:
    display(analysis_result['attempt_log'])

if 'citation_preview_df' in analysis_result and not analysis_result['citation_preview_df'].empty:
    display(analysis_result['citation_preview_df'][['doc_id', 'chunk_id', 'quote']].head(8))
```

The important diagnostics are:

- `topic_summary_df`: confirms topic coverage and top chunk per topic.
- `hits_df`: merged evidence table across topics.
- `attempt_log`: shows parse/quality status per generation attempt.
- `quality`: validation report for schema and evidence integrity.
- `citation_preview_df`: deterministic quote snippets for cited chunks.
- `timings`: sparse index build, reranker load, retrieval, and LLM latency.
- `analysis_counts`: document count, chunk count, and retrieved unique chunks.

Persistence is handled by `core/artifacts.py`:

```python
saved_paths = persist_results(cfg, analysis_result)
print(saved_paths)
```

Outputs are timestamped:

- `hits_<timestamp>.csv`
- `macro_answer_<timestamp>.txt`
- `macro_answer_<timestamp>.json` when parsing and coercion produce a structured object

Current implementation status:

- Implemented: end-to-end orchestration, diagnostics, metrics, timestamped output persistence.
- Caveat: output comparison over time is manual; there is no formal run registry or dashboard yet.

## 10.1 Observability Layer

The project now includes an explicit observability layer modeled on the same principle used in `systematic-trade-monitor`: the application emits stable structured telemetry to local files first, and external ingestion/query/visualization happens outside the core pipeline.

For this project, observability is not generic host monitoring. It is pipeline monitoring for:

- ingestion and catalog building,
- chunking and index construction,
- sparse index creation,
- topic retrieval,
- reranker behavior,
- context construction,
- generation attempts,
- validation and citation grounding,
- artifact persistence.

The canonical local output is a run-scoped diagnostics bundle:

- `outputs/diagnostics/<run_id>/events.jsonl`
- `outputs/diagnostics/<run_id>/summary.json`
- `outputs/diagnostics/<run_id>/topic_retrieval.csv`
- `outputs/diagnostics/<run_id>/generation_attempts.csv`
- `outputs/diagnostics/<run_id>/validation_summary.json`
- `outputs/diagnostics/<run_id>/artifacts_manifest.json`

This file-first design has the same advantages as in the trade-monitor stack:

1. the macro pipeline stays decoupled from QuestDB-specific writes,
2. runs remain debuggable even without Docker services running,
3. telemetry can be replayed or tailed after the fact,
4. ingestion policy stays configurable in Telegraf rather than embedded in the app.

### 10.2 External Telemetry Stack

The external stack is local and Dockerized:

```text
fed_macro_mvp pipeline
  -> diagnostics JSONL/artifacts
  -> Telegraf tail input
  -> QuestDB time-series storage
  -> Grafana dashboards
```

Telegraf tails `outputs/diagnostics/*/events.jsonl`, parses the structured events, and forwards them to QuestDB. Grafana is provisioned against QuestDB and visualizes run count, stage latency, failure rates, generation-attempt behavior, and recent pipeline issues.

The core pipeline remains notebook-first and locally runnable without Docker. The Docker services are an observability extension, not a runtime dependency.

## 11) Investor UI Layer

`fed_macro_v3_investor_ui.ipynb` launches an interactive notebook UI over the same pipeline. It does not define a separate analysis path; it calls `create_config` and `launch_investor_dashboard`.

Notebook anchor from `fed_macro_v3_investor_ui.ipynb`:

```python
from pathlib import Path

from core.pipeline import create_config
from core.investor_ui import launch_investor_dashboard

cfg = create_config(Path.cwd())
print('Project dir:', cfg.project_dir)
print('Default profile:', cfg.profile_name)
```

```python
ui = launch_investor_dashboard(cfg)
```

The UI exposes:

- profile selection,
- refresh-all vs. analysis-only mode,
- topic focus,
- top-k per topic,
- maximum context characters,
- LLM output budget,
- optional reranker toggle,
- run and save actions.

The display layer maps the normalized JSON into:

- a dashboard summary,
- regime cards,
- run metrics,
- analysis metrics,
- topic signal tables,
- evidence quote tables,
- normalized JSON view.

Current implementation status:

- Implemented: notebook-native investor UI with controls, logs, tabs, metrics, and save action.
- Caveat: it is not a production web app and does not implement authentication, multi-user state, or scheduled refresh.

## 12) Alignment With Standard RAG Patterns

The project follows the standard RAG pattern: retrieve relevant external evidence, condition generation on that evidence, and return an answer grounded in retrieved context. It aligns with the conceptual design of retrieval-augmented generation in three main ways:

- Retrieval is explicit and separate from generation.
- The generator receives retrieved evidence at inference time.
- The final answer includes citations back to retrieved evidence units.

The project intentionally diverges from larger canonical RAG systems in several ways:

- It does not train or fine-tune a retriever or generator.
- It uses a local, small-corpus FAISS index instead of a hosted vector database.
- It uses exact local vector search because the corpus is small enough for an MVP.
- It adds macro-topic decomposition before retrieval to enforce analytical coverage.
- It validates and repairs JSON after generation because the output is intended to be machine-readable.
- It uses chunk IDs rather than page-level citations because page mapping is not implemented yet.

The controlled divergence is important. For this use case, the success criterion is not benchmark-leading RAG performance. It is a reproducible local pipeline that can produce a structured, inspectable macro brief with enough evidence discipline to support a technical conversation.

## 13) Implemented vs. Partial vs. Deferred

Implemented now:

- End-to-end ingestion, indexing, retrieval, generation, validation, and persistence.
- Runtime profiles for speed vs. breadth.
- Dense FAISS retrieval with normalized embeddings.
- Sparse TF-IDF retrieval and Reciprocal Rank Fusion.
- Deterministic query variants per macro topic.
- Optional cross-encoder reranker.
- Exponential recency scoring.
- Topic-balanced context construction.
- Local Ollama generation with strict JSON prompt.
- Retry, parse, JSON repair, coercion, and validation flow.
- Evidence ID normalization and citation quote enrichment.
- Technical notebook and investor UI notebook.
- Unit tests for profile behavior, catalog filtering, and evidence coercion.

Partial or incomplete:

- Incremental indexing and content-hash cache.
- Page-level citation granularity.
- Formal retrieval evaluation with labeled relevance judgments.
- Formal generation evaluation with RAGAS, TruLens, or similar frameworks.
- Historical backtesting of macro conclusions.
- Production web backend.

Material caveats:

- The generated macro read is evidence-grounded but not a quantitative macro forecast.
- The pipeline reflects the selected corpus window and document types.
- Fed PDF parsing can lose table structure and layout.
- Evidence backfill can make schema validation pass even when the raw model output was incomplete; inspect `attempt_log` and `quality`.
- The LLM is local and configurable, so output behavior can vary by installed Ollama model.

## 14) Interview-Ready Narrative

### Situation

Federal Reserve communications are dense, frequent, and market-relevant, but the signal is spread across long PDFs, repeated policy language, and evolving macro context. A human analyst can read the documents directly, but doing that repeatedly makes it difficult to maintain consistent coverage across inflation, labor, growth, policy, financial conditions, and credit.

The project frames that problem as an evidence retrieval and structured synthesis task. The goal is not to produce a standalone economic forecast. The goal is to turn recent Fed communications into a reproducible, citation-backed macro read that can be inspected, challenged, and rerun as new documents arrive.

### Task

The implementation target was a local MVP that could support both a technical walkthrough and practical analyst usage:

- build a fresh local corpus from recent Fed PDFs,
- retrieve high-signal evidence by macro topic,
- generate a compact 6 to 12 month investor view,
- preserve citations and quote snippets,
- expose diagnostics so failures are visible rather than hidden.

The key constraint was local runnability. That shaped the architecture toward FAISS, sentence-transformer embeddings, TF-IDF, Ollama, and notebook-native interfaces rather than hosted APIs or a production web stack.

### Action

The system decomposes the problem into a data pipeline followed by an evidence-grounded generation pipeline.

First, ingestion creates the document universe by scraping Fed seed pages and probing known filename patterns. The active profile then filters by lookback window and document type, which makes corpus selection an explicit modeling assumption rather than an accidental file-system state.

Second, indexing converts PDFs into chunk-level observations. Text is extracted, normalized, split into overlapping chunks, embedded, L2-normalized, and stored in FAISS. The chunk is the project's core evidence unit because it is small enough for retrieval and citation but large enough to preserve local macro context.

Third, retrieval is topic-aware and hybrid. Dense embeddings capture semantic similarity, TF-IDF preserves exact macro terms, Reciprocal Rank Fusion combines rank lists without relying on score calibration, and recency weighting reflects the time-sensitive nature of Fed communication. This is the most important data-science design choice because it balances semantic relevance, lexical precision, topic coverage, and freshness.

Fourth, generation is constrained rather than open-ended. The local LLM receives a topic-balanced context and must return strict JSON. Post-processing then validates schema shape, required topics, evidence IDs, citations, and quote snippets. This treats the LLM as a synthesis component inside a controlled data product, not as the source of truth.

### Result

The current result is a working notebook-first RAG system that can refresh a local Fed corpus, build a retrieval index, retrieve topic-specific evidence, generate a structured investor macro view, and persist both raw and normalized outputs. The technical notebook exposes intermediate tables and diagnostics, while the investor UI presents the same pipeline through controls, dashboard cards, topic drilldowns, citations, and normalized JSON.

The strongest technical points to emphasize in discussion are:

- configuration acts as the experiment contract,
- retrieval uses both semantic and lexical evidence signals,
- RRF avoids fragile score calibration between retrieval methods,
- recency decay encodes macro time relevance,
- topic decomposition prevents one theme from dominating the answer,
- validation converts probabilistic LLM output into a more reliable structured artifact,
- diagnostics make the system auditable through `topic_summary_df`, `hits_df`, `attempt_log`, `quality`, and `citation_preview_df`.

The main limitations are also clear: page-level citations, formal retrieval evaluation, incremental indexing, and historical backtesting are not complete yet. Those are natural next steps, but the current MVP already demonstrates the core architecture and the reasoning behind the method choices.
