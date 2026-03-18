from __future__ import annotations

import json
import time

import pandas as pd

from .config import PipelineConfig
from .generation import check_ollama_ready, run_generation_with_retries
from .retrieval import build_sparse_index, load_bundle, load_reranker, retrieve_multi_topic


def run_analysis(cfg: PipelineConfig) -> dict[str, object]:
    index, meta_df, emb_model = load_bundle(cfg)
    analysis_warnings: list[str] = []

    sparse_t0 = time.time()
    sparse_bundle = build_sparse_index(meta_df)
    sparse_latency = time.time() - sparse_t0

    rerank_t0 = time.time()
    reranker = load_reranker(cfg)
    rerank_latency = time.time() - rerank_t0

    retrieval_t0 = time.time()
    topic_hits = retrieve_multi_topic(index, meta_df, emb_model, sparse_bundle, reranker, cfg)
    retrieval_latency = time.time() - retrieval_t0

    topic_summary_df = pd.DataFrame(
        [
            {
                "topic": topic,
                "hits": int(len(h)),
                "top_final_score": float(h["final_score"].iloc[0]) if len(h) else None,
                "top_chunk_id": h["chunk_id"].iloc[0] if len(h) else None,
            }
            for topic, h in topic_hits.items()
        ]
    )

    flat_hits = [h.assign(topic=topic) for topic, h in topic_hits.items() if not h.empty]
    if flat_hits:
        hits_df = pd.concat(flat_hits, ignore_index=True)
        hits_df = hits_df.sort_values("final_score", ascending=False).drop_duplicates(subset=["chunk_id"]).reset_index(drop=True)
    else:
        hits_df = pd.DataFrame(columns=["topic", "chunk_id", "doc_id", "date_hint", "score", "recency", "final_score", "topic_flags", "text"])

    check_ollama_ready(cfg)
    question = "Build an investor-focused 6-12 month U.S. macro view from recent Federal Reserve communications."

    llm_t0 = time.time()
    llm_text, parsed, quality, attempt_log, final_context, context_chunk_ids = run_generation_with_retries(question, topic_hits, cfg)
    llm_latency = time.time() - llm_t0

    index_doc_count = int(meta_df["doc_id"].nunique()) if not meta_df.empty and "doc_id" in meta_df.columns else 0
    index_chunk_count = int(len(meta_df))
    if cfg.profile_name == "fast_default" and index_chunk_count > cfg.fast_profile_chunk_warning:
        analysis_warnings.append(
            f"fast_default: indexed chunks {index_chunk_count} exceed recommended cap {cfg.fast_profile_chunk_warning}"
        )

    normalized_json_text = llm_text
    citation_preview_df = pd.DataFrame(columns=["chunk_id", "doc_id", "quote"])
    if parsed is not None:
        try:
            normalized_json_text = json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            normalized_json_text = llm_text
        citations = parsed.get("citations", []) if isinstance(parsed, dict) else []
        if isinstance(citations, list):
            rows = []
            for c in citations:
                if not isinstance(c, dict):
                    continue
                rows.append(
                    {
                        "chunk_id": str(c.get("chunk_id", "")),
                        "doc_id": str(c.get("doc_id", "")),
                        "quote": str(c.get("quote", "")),
                    }
                )
            if rows:
                citation_preview_df = pd.DataFrame(rows)

    return {
        "question": question,
        "topic_hits": topic_hits,
        "topic_summary_df": topic_summary_df,
        "hits_df": hits_df,
        "llm_text": llm_text,
        "normalized_json_text": normalized_json_text,
        "parsed": parsed,
        "quality": quality,
        "attempt_log": attempt_log,
        "citation_preview_df": citation_preview_df,
        "final_context": final_context,
        "context_chunk_ids": context_chunk_ids,
        "sparse_enabled": sparse_bundle is not None,
        "reranker_enabled": reranker is not None,
        "timings": {
            "sparse_latency_s": sparse_latency,
            "rerank_load_s": rerank_latency,
            "retrieval_latency_s": retrieval_latency,
            "llm_stage_s": llm_latency,
        },
        "analysis_counts": {
            "index_docs": index_doc_count,
            "index_chunks": index_chunk_count,
            "retrieved_unique_chunks": int(len(hits_df)),
        },
        "analysis_warnings": analysis_warnings,
    }
