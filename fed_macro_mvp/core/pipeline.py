from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import pandas as pd

from .artifacts import save_outputs
from .analysis import run_analysis
from .config import PipelineConfig, default_config
from .indexing import run_indexing
from .ingest import run_ingestion


def create_config(project_dir: Path | None = None) -> PipelineConfig:
    return default_config(project_dir)


def run_ingest_and_index(cfg: PipelineConfig) -> dict[str, Any]:
    ingest_t0 = time.time()
    catalog_df, download_df, manifest_path, ingest_stats = run_ingestion(cfg)
    ingest_latency = time.time() - ingest_t0

    index_t0 = time.time()
    chunks_df, chunks_path, index_cfg = run_indexing(download_df, cfg)
    index_latency = time.time() - index_t0

    warnings = list(ingest_stats.get("warnings", []))
    if cfg.profile_name == "fast_default" and len(chunks_df) > cfg.fast_profile_chunk_warning:
        warnings.append(
            f"fast_default: chunks {len(chunks_df)} exceed recommended cap {cfg.fast_profile_chunk_warning}"
        )

    stage_metrics = {
        "profile_name": cfg.profile_name,
        "ingest_latency_s": ingest_latency,
        "index_latency_s": index_latency,
        "catalog_candidates": int(len(catalog_df)),
        "downloaded_or_exists": int(download_df["status"].isin(["downloaded", "exists"]).sum()) if not download_df.empty else 0,
        "chunks": int(len(chunks_df)),
        "warnings": warnings,
    }

    return {
        "catalog_df": catalog_df,
        "download_df": download_df,
        "manifest_path": manifest_path,
        "chunks_df": chunks_df,
        "chunks_path": chunks_path,
        "index_cfg": index_cfg,
        "ingest_stats": ingest_stats,
        "stage_metrics": stage_metrics,
    }


def run_full_analysis(cfg: PipelineConfig) -> dict[str, Any]:
    return run_analysis(cfg)


def persist_results(cfg: PipelineConfig, analysis_result: dict[str, Any]) -> dict[str, Path | None]:
    hits_df: pd.DataFrame = analysis_result["hits_df"]
    llm_text: str = analysis_result["llm_text"]
    parsed = analysis_result["parsed"]
    return save_outputs(cfg, hits_df, llm_text, parsed)
