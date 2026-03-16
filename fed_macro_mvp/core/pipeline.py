from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import save_outputs
from .analysis import run_analysis
from .config import PipelineConfig, default_config
from .indexing import run_indexing
from .ingest import run_ingestion


def create_config(project_dir: Path | None = None) -> PipelineConfig:
    return default_config(project_dir)


def run_ingest_and_index(cfg: PipelineConfig) -> dict[str, Any]:
    catalog_df, download_df, manifest_path = run_ingestion(cfg)
    chunks_df, chunks_path, index_cfg = run_indexing(download_df, cfg)

    return {
        "catalog_df": catalog_df,
        "download_df": download_df,
        "manifest_path": manifest_path,
        "chunks_df": chunks_df,
        "chunks_path": chunks_path,
        "index_cfg": index_cfg,
    }


def run_full_analysis(cfg: PipelineConfig) -> dict[str, Any]:
    return run_analysis(cfg)


def persist_results(cfg: PipelineConfig, analysis_result: dict[str, Any]) -> dict[str, Path | None]:
    hits_df: pd.DataFrame = analysis_result["hits_df"]
    llm_text: str = analysis_result["llm_text"]
    parsed = analysis_result["parsed"]
    return save_outputs(cfg, hits_df, llm_text, parsed)
