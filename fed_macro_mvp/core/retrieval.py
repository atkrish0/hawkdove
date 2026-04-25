from __future__ import annotations

from typing import Any
import re
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import PipelineConfig
from .indexing import normalize_rows

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


def load_bundle(cfg: PipelineConfig):
    index = faiss.read_index(str(cfg.index_dir / "fed_chunks.index"))
    meta_parquet = cfg.index_dir / "fed_chunks_meta.parquet"
    meta_csv = cfg.index_dir / "fed_chunks_meta.csv"

    if meta_parquet.exists():
        meta_df = pd.read_parquet(meta_parquet)
    elif meta_csv.exists():
        meta_df = pd.read_csv(meta_csv)
    else:
        raise FileNotFoundError("No index metadata found (expected fed_chunks_meta.parquet or fed_chunks_meta.csv)")

    emb_model = SentenceTransformer(cfg.embed_model_name)
    return index, meta_df, emb_model


def load_reranker(cfg: PipelineConfig, observer=None):
    if not cfg.enable_reranker:
        return None
    try:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(cfg.rerank_model_name, max_length=512)
        if observer is not None:
            observer.emit(
                "stage_completed",
                stage="reranker_load",
                status="ok",
                payload={"model_name": cfg.rerank_model_name},
            )
        return reranker
    except Exception as e:
        print(f"[warn] Reranker disabled ({e})")
        if observer is not None:
            observer.emit(
                "stage_completed",
                stage="reranker_load",
                status="failed",
                payload={"model_name": cfg.rerank_model_name, "error": str(e)},
                level="warning",
            )
        return None


def recency_score(date_hint: Any, half_life_days: int) -> float:
    if not date_hint or pd.isna(date_hint):
        return 0.5
    ts = pd.to_datetime(date_hint, errors="coerce", utc=True)
    if pd.isna(ts):
        return 0.5
    age_days = max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 86400.0)
    return float(np.exp(-np.log(2) * age_days / max(1, half_life_days)))


def minmax_scale(values: pd.Series) -> pd.Series:
    v = values.astype(float)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return pd.Series([1.0] * len(v), index=v.index)
    return (v - lo) / (hi - lo)


def dense_retrieve(query: str, index, meta_df: pd.DataFrame, emb_model, top_k: int) -> pd.DataFrame:
    q = emb_model.encode([query], convert_to_numpy=True).astype("float32")
    q = normalize_rows(q)
    scores, ids = index.search(q, top_k)
    out = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0 or i >= len(meta_df):
            continue
        out.append({"chunk_id": meta_df.iloc[int(i)]["chunk_id"], "score": float(score)})
    if not out:
        return pd.DataFrame(columns=["chunk_id", "score"])
    return pd.DataFrame(out).drop_duplicates(subset=["chunk_id"]).sort_values("score", ascending=False).reset_index(drop=True)


def build_sparse_index(meta_df: pd.DataFrame, observer=None):
    t0 = time.time()
    if not HAVE_SKLEARN:
        if observer is not None:
            observer.emit(
                "stage_completed",
                stage="sparse_index_build",
                status="disabled",
                payload={"reason": "sklearn_unavailable"},
                level="warning",
            )
        return None
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.95)
        mat = vec.fit_transform(meta_df["text"].fillna("").astype(str).tolist())
        if observer is not None:
            observer.emit(
                "stage_completed",
                stage="sparse_index_build",
                status="ok",
                duration_ms=(time.time() - t0) * 1000.0,
                payload={"rows": int(len(meta_df)), "vocab_size": int(len(vec.vocabulary_))},
            )
        return vec, mat
    except Exception as e:
        print(f"[warn] Sparse index disabled: {e}")
        if observer is not None:
            observer.emit(
                "stage_completed",
                stage="sparse_index_build",
                status="failed",
                duration_ms=(time.time() - t0) * 1000.0,
                payload={"error": str(e)},
                level="warning",
            )
        return None


def sparse_retrieve(query: str, sparse_bundle, meta_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if sparse_bundle is None:
        return pd.DataFrame(columns=["chunk_id", "score"])

    vec, mat = sparse_bundle
    try:
        qv = vec.transform([query])
        sims = (qv @ mat.T).toarray().ravel()
    except Exception:
        return pd.DataFrame(columns=["chunk_id", "score"])

    if sims.size == 0:
        return pd.DataFrame(columns=["chunk_id", "score"])

    k = min(top_k, sims.size)
    idxs = np.argpartition(sims, -k)[-k:]
    ranked = idxs[np.argsort(sims[idxs])[::-1]]
    out = [{"chunk_id": meta_df.iloc[int(i)]["chunk_id"], "score": float(sims[int(i)])} for i in ranked]

    if not out:
        return pd.DataFrame(columns=["chunk_id", "score"])
    return pd.DataFrame(out).drop_duplicates(subset=["chunk_id"]).sort_values("score", ascending=False).reset_index(drop=True)


def rrf_fuse(rank_frames: list[pd.DataFrame], score_col: str, k: int) -> pd.DataFrame:
    agg: dict[str, float] = {}
    for df in rank_frames:
        if df is None or df.empty:
            continue
        ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        for rank, cid in enumerate(ordered["chunk_id"].tolist(), start=1):
            agg[cid] = agg.get(cid, 0.0) + 1.0 / (k + rank)

    if not agg:
        return pd.DataFrame(columns=["chunk_id", "fusion_score"])
    return pd.DataFrame({"chunk_id": list(agg.keys()), "fusion_score": list(agg.values())}).sort_values(
        "fusion_score", ascending=False
    ).reset_index(drop=True)


def build_topic_queries(topic: str, cfg: PipelineConfig) -> list[str]:
    queries = [cfg.topic_queries[topic]["query"]]
    if cfg.enable_query_fusion:
        for qv in cfg.topic_query_variants.get(topic, []):
            queries.append(f"Federal Reserve {qv}")
    return list(dict.fromkeys(queries))


def apply_reranker(query: str, candidates: pd.DataFrame, reranker, cfg: PipelineConfig, observer=None, topic: str | None = None) -> pd.DataFrame:
    if reranker is None or candidates.empty:
        return candidates

    work = candidates.head(cfg.rerank_candidate_pool).copy()
    pairs = [[query, str(t)[:1200]] for t in work["text"].fillna("").tolist()]

    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        print(f"[warn] rerank predict failed: {e}")
        if observer is not None:
            observer.emit(
                "stage_completed",
                stage="reranking",
                status="failed",
                payload={"topic": topic, "error": str(e)},
                level="warning",
            )
        return candidates

    work["rerank_score"] = [float(x) for x in scores]
    out = candidates.merge(work[["chunk_id", "rerank_score"]], on="chunk_id", how="left")
    out["rerank_score"] = out["rerank_score"].fillna(out["score"])
    return out


def retrieve_topic_hybrid(topic: str, index, meta_df: pd.DataFrame, emb_model, sparse_bundle, reranker, cfg: PipelineConfig, observer=None) -> pd.DataFrame:
    t0 = time.time()
    query_fused = []
    topic_queries = build_topic_queries(topic, cfg)

    for query in topic_queries:
        dense_df = dense_retrieve(query, index, meta_df, emb_model, top_k=cfg.candidates_per_query)
        if cfg.enable_hybrid_retrieval:
            sparse_df = sparse_retrieve(query, sparse_bundle, meta_df, top_k=cfg.candidates_per_query)
            fused = rrf_fuse([dense_df, sparse_df], score_col="score", k=cfg.rrf_k)
        else:
            fused = rrf_fuse([dense_df], score_col="score", k=cfg.rrf_k)

        if not fused.empty:
            query_fused.append(fused.rename(columns={"fusion_score": "score"}))

    merged = rrf_fuse(query_fused, score_col="score", k=cfg.rrf_k)
    if merged.empty:
        if observer is not None:
            observer.emit(
                "topic_retrieval_summary",
                stage="retrieval",
                status="empty",
                duration_ms=(time.time() - t0) * 1000.0,
                payload={
                    "topic": topic,
                    "queries": topic_queries,
                    "dense_enabled": True,
                    "sparse_enabled": cfg.enable_hybrid_retrieval and sparse_bundle is not None,
                    "reranker_enabled": reranker is not None,
                    "hit_count": 0,
                },
                level="warning",
            )
        return pd.DataFrame(columns=["chunk_id", "doc_id", "date_hint", "topic_flags", "text", "score", "final_score", "recency"])

    mlookup = meta_df.drop_duplicates(subset=["chunk_id"]).set_index("chunk_id")
    rows = []
    for _, r in merged.head(cfg.candidates_per_query).iterrows():
        cid = r["chunk_id"]
        if cid not in mlookup.index:
            continue
        meta = mlookup.loc[cid]
        rows.append(
            {
                "chunk_id": cid,
                "doc_id": meta.get("doc_id", ""),
                "date_hint": meta.get("date_hint", ""),
                "topic_flags": meta.get("topic_flags", []),
                "text": meta.get("text", ""),
                "score": float(r["fusion_score"]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["chunk_id", "doc_id", "date_hint", "topic_flags", "text", "score", "final_score", "recency"])

    out = pd.DataFrame(rows)
    prerank_ids = out["chunk_id"].head(cfg.top_k_topic).astype(str).tolist()
    out = apply_reranker(cfg.topic_queries[topic]["query"], out, reranker, cfg, observer=observer, topic=topic)

    base_col = "rerank_score" if "rerank_score" in out.columns else "score"
    out["base_score_norm"] = minmax_scale(out[base_col])
    out["recency"] = out["date_hint"].apply(lambda x: recency_score(x, cfg.recency_half_life_days))
    out["final_score"] = out["base_score_norm"] * ((1.0 - cfg.recency_boost) + cfg.recency_boost * out["recency"])
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True).head(cfg.top_k_topic)

    if observer is not None:
        logged_hits = []
        max_hits = max(1, int(getattr(cfg, "max_logged_hits_per_topic", 3)))
        for _, row in out.head(max_hits).iterrows():
            hit = {
                "chunk_id": str(row.get("chunk_id", "")),
                "doc_id": str(row.get("doc_id", "")),
                "date_hint": str(row.get("date_hint", "")),
                "score": float(row.get("score", 0.0)),
                "final_score": float(row.get("final_score", 0.0)),
                "recency": float(row.get("recency", 0.0)),
            }
            if "rerank_score" in row:
                hit["rerank_score"] = float(row.get("rerank_score", 0.0))
            if getattr(cfg, "emit_chunk_previews", False) and getattr(cfg, "chunk_preview_chars", 0) > 0:
                hit["preview"] = str(row.get("text", ""))[: int(cfg.chunk_preview_chars)]
            logged_hits.append(hit)

        observer.emit(
            "topic_retrieval_summary",
            stage="retrieval",
            status="ok",
            duration_ms=(time.time() - t0) * 1000.0,
            payload={
                "topic": topic,
                "queries": topic_queries,
                "dense_enabled": True,
                "sparse_enabled": cfg.enable_hybrid_retrieval and sparse_bundle is not None,
                "reranker_enabled": reranker is not None,
                "hit_count": int(len(out)),
                "rerank_changed": prerank_ids != out["chunk_id"].astype(str).tolist(),
                "top_hits": logged_hits,
            },
        )
    return out


def retrieve_multi_topic(index, meta_df: pd.DataFrame, emb_model, sparse_bundle, reranker, cfg: PipelineConfig, observer=None) -> dict[str, pd.DataFrame]:
    return {
        topic: retrieve_topic_hybrid(topic, index, meta_df, emb_model, sparse_bundle, reranker, cfg, observer=observer)
        for topic in cfg.topic_queries
    }


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text)) if s.strip()]


def topic_snippet(text: str, keywords: list[str], max_chars: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return str(text)[:max_chars]

    chosen = []
    lower_keys = [k.lower() for k in keywords]
    for s in sentences:
        if any(k in s.lower() for k in lower_keys):
            chosen.append(s)
        if len(" ".join(chosen)) >= max_chars:
            break

    if not chosen:
        chosen = sentences[:3]
    return " ".join(chosen)[:max_chars]


def build_topic_context(
    topic: str,
    hits_df: pd.DataFrame,
    cfg: PipelineConfig,
    max_chunks: int | None = None,
    max_chars_per_chunk: int | None = None,
) -> tuple[str, list[str]]:
    blocks, used_ids = [], []
    if hits_df.empty:
        return "", used_ids

    if max_chunks is None:
        max_chunks = cfg.top_k_topic
    if max_chars_per_chunk is None:
        max_chars_per_chunk = cfg.max_chars_per_chunk

    for _, r in hits_df.head(max_chunks).iterrows():
        snippet = topic_snippet(r["text"], cfg.topic_queries[topic]["keywords"], max_chars_per_chunk)
        blocks.append(
            f"[topic={topic}; chunk_id={r['chunk_id']}; doc_id={r['doc_id']}; score={r['final_score']:.4f}]\n{snippet}"
        )
        used_ids.append(r["chunk_id"])

    return "\n\n".join(blocks), used_ids


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
        if not section:
            continue

        block = f"## {topic}\n{section}"
        if total_chars + len(block) > max_context_chars:
            break

        sections.append(block)
        total_chars += len(block)
        all_ids.extend(ids)

    return "\n\n".join(sections), list(dict.fromkeys(all_ids))


def collect_valid_ids(topic_hits: dict[str, pd.DataFrame], per_topic: int = 3) -> set[str]:
    valid: set[str] = set()
    for _, df in topic_hits.items():
        if df is None or df.empty:
            continue
        for cid in df["chunk_id"].head(max(1, per_topic)).tolist():
            valid.add(str(cid))
    return valid
