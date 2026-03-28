from __future__ import annotations

from pathlib import Path
from typing import Any
import os

os.environ.setdefault("PANDAS_NO_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_BOTTLENECK", "1")
import pandas as pd

from .config import LabConfig
from .dataset_builder import discover_pairs
from .io_utils import load_hits, load_json, write_json
from .validation import is_quality_ok, validate_target_json


def evaluate_existing_outputs(cfg: LabConfig) -> dict[str, Any]:
    pairs = discover_pairs(cfg.source_outputs_dir)
    rows = []

    for pair in pairs:
        status = "ok"
        reason = ""
        valid_ids_count = 0
        report = {}

        try:
            obj = load_json(pair.json_path)
            hits = load_hits(pair.hits_path)
            valid_ids = set(str(x) for x in hits.get("chunk_id", pd.Series(dtype=str)).dropna().tolist())
            valid_ids_count = len(valid_ids)
            report = validate_target_json(obj, valid_ids)
            if not is_quality_ok(report):
                status = "quality_failed"
                reason = str(report)
        except Exception as e:
            status = "load_error"
            reason = str(e)
            report = {"exception": str(e)}

        rows.append(
            {
                "timestamp": pair.timestamp,
                "status": status,
                "valid_ids": valid_ids_count,
                "json_path": str(pair.json_path),
                "hits_path": str(pair.hits_path),
                "report": report,
                "reason": reason,
            }
        )

    df = pd.DataFrame(rows)

    total = len(df)
    ok = int((df["status"] == "ok").sum()) if total else 0
    quality_failed = int((df["status"] == "quality_failed").sum()) if total else 0
    load_error = int((df["status"] == "load_error").sum()) if total else 0

    metrics = {
        "total_pairs": total,
        "ok": ok,
        "quality_failed": quality_failed,
        "load_error": load_error,
        "quality_pass_rate": (ok / total) if total else 0.0,
    }

    eval_dir = cfg.artifacts_dir / "eval"
    df.to_csv(eval_dir / "baseline_eval_rows.csv", index=False)
    write_json(eval_dir / "baseline_eval_summary.json", metrics)

    return metrics
