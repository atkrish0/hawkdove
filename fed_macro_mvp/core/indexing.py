from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re
import time

import faiss
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from .config import PipelineConfig


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


def topic_flags(text: str, focus_topics: list[str]) -> list[str]:
    t = text.lower()
    return [x for x in focus_topics if x in t]


def build_chunks(download_manifest: pd.DataFrame, cfg: PipelineConfig, observer=None) -> pd.DataFrame:
    t0 = time.time()
    rows = []
    if download_manifest.empty:
        return pd.DataFrame()

    ok = download_manifest[download_manifest["status"].isin(["downloaded", "exists"])].copy()
    skipped_missing = 0
    skipped_short = 0
    for _, row in tqdm(ok.iterrows(), total=len(ok), desc="Parsing PDFs"):
        fpath = Path(row["local_path"])
        if not fpath.exists():
            skipped_missing += 1
            continue

        text = read_pdf_text(fpath)
        if len(text) < 200:
            skipped_short += 1
            continue

        for idx, part in enumerate(chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)):
            if len(part.strip()) < 80:
                continue
            rows.append(
                {
                    "chunk_id": f"{fpath.name}::chunk{idx:04d}",
                    "doc_id": fpath.name,
                    "pdf_url": row.get("pdf_url", ""),
                    "date_hint": row.get("date_hint", ""),
                    "doc_type": row.get("doc_type", ""),
                    "chunk_index": idx,
                    "text": part,
                    "topic_flags": topic_flags(part, cfg.focus_topics),
                    "text_len": len(part),
                }
            )

    if not rows:
        out = pd.DataFrame()
    else:
        out = pd.DataFrame(rows).drop_duplicates(subset=["chunk_id"]).reset_index(drop=True)
    if observer is not None:
        observer.emit(
            "stage_completed",
            stage="chunk_build",
            status="ok",
            duration_ms=(time.time() - t0) * 1000.0,
            payload={
                "input_docs": int(len(ok)),
                "chunk_count": int(len(out)),
                "skipped_missing_files": skipped_missing,
                "skipped_short_docs": skipped_short,
            },
        )
    return out


def save_table(df: pd.DataFrame, preferred_path: Path, observer=None, stage: str = "table_save") -> Path:
    try:
        df.to_parquet(preferred_path, index=False)
        if observer is not None:
            observer.emit(
                "stage_completed",
                stage=stage,
                status="ok",
                payload={"path": str(preferred_path), "format": "parquet", "rows": int(len(df))},
            )
        return preferred_path
    except Exception as e:
        fallback = preferred_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        print(f"[warn] Parquet unavailable ({e}); saved CSV fallback: {fallback}")
        if observer is not None:
            observer.emit(
                "stage_completed",
                stage=stage,
                status="fallback_csv",
                payload={"path": str(fallback), "format": "csv", "rows": int(len(df)), "error": str(e)},
                level="warning",
            )
        return fallback


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def build_faiss_index(chunks_df: pd.DataFrame, cfg: PipelineConfig, observer=None) -> dict[str, Any]:
    if chunks_df.empty:
        raise ValueError("No chunks available; run ingestion/chunking first.")

    t0 = time.time()
    model = SentenceTransformer(cfg.embed_model_name)
    vectors = model.encode(chunks_df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True).astype("float32")
    vectors = normalize_rows(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = cfg.index_dir / "fed_chunks.index"
    faiss.write_index(index, str(index_path))

    meta_path = save_table(chunks_df, cfg.index_dir / "fed_chunks_meta.parquet", observer=observer, stage="index_meta_save")

    config = {
        "embed_model": cfg.embed_model_name,
        "dimension": dim,
        "rows": int(len(chunks_df)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "index_path": str(index_path),
        "meta_path": str(meta_path),
    }
    with open(cfg.index_dir / "index_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if observer is not None:
        observer.emit(
            "stage_completed",
            stage="faiss_index_build",
            status="ok",
            duration_ms=(time.time() - t0) * 1000.0,
            payload={
                "embed_model": cfg.embed_model_name,
                "dimension": int(dim),
                "rows": int(len(chunks_df)),
                "index_path": str(index_path),
                "meta_path": str(meta_path),
            },
        )

    return config


def run_indexing(download_df: pd.DataFrame, cfg: PipelineConfig, observer=None) -> tuple[pd.DataFrame, Path | None, dict[str, Any] | None]:
    t0 = time.time()
    chunks_df = build_chunks(download_df, cfg, observer=observer)
    chunks_path = None
    index_cfg = None

    if not chunks_df.empty:
        chunks_path = save_table(chunks_df, cfg.processed_dir / "chunks.parquet", observer=observer, stage="chunks_persist")
        index_cfg = build_faiss_index(chunks_df, cfg, observer=observer)

    if observer is not None:
        observer.emit(
            "stage_completed",
            stage="indexing",
            status="ok",
            duration_ms=(time.time() - t0) * 1000.0,
            payload={
                "chunk_count": int(len(chunks_df)),
                "chunks_path": str(chunks_path) if chunks_path else None,
                "index_created": index_cfg is not None,
            },
        )

    return chunks_df, chunks_path, index_cfg
