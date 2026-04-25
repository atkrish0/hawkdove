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
from .observability import create_recorder


def create_config(project_dir: Path | None = None) -> PipelineConfig:
    return default_config(project_dir)


def run_ingest_and_index(cfg: PipelineConfig, observer=None, mode: str = "refresh_all") -> dict[str, Any]:
    if observer is None:
        observer = create_recorder(cfg, mode=mode)
    try:
        ingest_t0 = time.time()
        catalog_df, download_df, manifest_path, ingest_stats = run_ingestion(cfg, observer=observer)
        ingest_latency = time.time() - ingest_t0

        index_t0 = time.time()
        chunks_df, chunks_path, index_cfg = run_indexing(download_df, cfg, observer=observer)
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

        diagnostics_summary = observer.diagnostics_summary()
        diagnostics_summary.update(
            {
                "catalog_candidates": int(len(catalog_df)),
                "downloaded_or_exists": int(download_df["status"].isin(["downloaded", "exists"]).sum()) if not download_df.empty else 0,
                "chunks": int(len(chunks_df)),
                "ingest_latency_s": round(ingest_latency, 4),
                "index_latency_s": round(index_latency, 4),
            }
        )
        observer.write_summary(diagnostics_summary)
        if manifest_path:
            observer.add_artifact("download_manifest", manifest_path)
        if chunks_path:
            observer.add_artifact("chunks_table", chunks_path)
        if index_cfg is not None:
            observer.add_artifact("index_config", cfg.index_dir / "index_config.json")

        return {
            "run_id": observer.run_id,
            "catalog_df": catalog_df,
            "download_df": download_df,
            "manifest_path": manifest_path,
            "chunks_df": chunks_df,
            "chunks_path": chunks_path,
            "index_cfg": index_cfg,
            "ingest_stats": ingest_stats,
            "stage_metrics": stage_metrics,
            "diagnostics_summary": diagnostics_summary,
            "diagnostics_paths": observer.diagnostics_paths(),
            "_observer": observer,
        }
    except Exception as e:
        observer.emit(
            "run_failed",
            stage="run",
            status="failed",
            payload={"error": str(e), "mode": mode},
            level="warning",
        )
        raise


def run_full_analysis(cfg: PipelineConfig, observer=None, mode: str = "analysis_only") -> dict[str, Any]:
    return run_analysis(cfg, observer=observer, mode=mode)


def persist_results(cfg: PipelineConfig, analysis_result: dict[str, Any]) -> dict[str, Path | None]:
    hits_df: pd.DataFrame = analysis_result["hits_df"]
    llm_text: str = analysis_result["llm_text"]
    parsed = analysis_result["parsed"]
    observer = analysis_result.get("_observer")
    saved = save_outputs(cfg, hits_df, llm_text, parsed, observer=observer)
    if observer is not None:
        for name, path in saved.items():
            if path is not None:
                observer.add_artifact(name, path)
        analysis_result["diagnostics_paths"] = observer.diagnostics_paths()
        analysis_result["diagnostics_summary"] = observer.diagnostics_summary()
    return saved
