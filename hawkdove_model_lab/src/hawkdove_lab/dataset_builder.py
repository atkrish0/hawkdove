from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random
import os

os.environ.setdefault("PANDAS_NO_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_BOTTLENECK", "1")
import pandas as pd

from .config import LabConfig
from .constants import DEFAULT_INSTRUCTION
from .io_utils import extract_timestamp, load_hits, load_json, write_json, write_jsonl
from .validation import is_quality_ok, validate_target_json


@dataclass
class PairRecord:
    timestamp: str
    json_path: Path
    hits_path: Path


def discover_pairs(outputs_dir: Path) -> list[PairRecord]:
    json_by_ts: dict[str, Path] = {}
    hits_by_ts: dict[str, Path] = {}

    for p in outputs_dir.glob("macro_answer_*.json"):
        ts = extract_timestamp(p.name)
        if ts:
            json_by_ts[ts] = p

    for p in outputs_dir.glob("hits_*.csv"):
        ts = extract_timestamp(p.name)
        if ts:
            hits_by_ts[ts] = p

    out = []
    for ts in sorted(set(json_by_ts).intersection(hits_by_ts)):
        out.append(PairRecord(timestamp=ts, json_path=json_by_ts[ts], hits_path=hits_by_ts[ts]))
    return out


def _safe_score(row: pd.Series) -> float:
    try:
        return float(row.get("final_score", 0.0))
    except Exception:
        return 0.0


def build_context(hits_df: pd.DataFrame, context_rows: int = 10) -> str:
    if hits_df.empty:
        return ""

    work = hits_df.copy()
    if "final_score" in work.columns:
        work = work.sort_values("final_score", ascending=False)

    blocks = []
    for _, row in work.head(context_rows).iterrows():
        chunk_id = str(row.get("chunk_id", "")).strip()
        doc_id = str(row.get("doc_id", "")).strip()
        topic = str(row.get("topic", "")).strip()
        score = _safe_score(row)
        text = str(row.get("text", "")).strip().replace("\n", " ")
        if len(text) > 900:
            text = text[:900]
        blocks.append(
            f"[chunk_id={chunk_id}; doc_id={doc_id}; topic={topic}; score={score:.4f}]\\n{text}"
        )

    return "\\n\\n".join(blocks)


def _split_rows(rows: list[dict[str, Any]], seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {"train": [], "val": [], "test": []}

    ordered = sorted(rows, key=lambda x: x["timestamp"])
    random.Random(seed).shuffle(ordered)

    n = len(ordered)
    if n == 1:
        return {"train": ordered, "val": [], "test": []}
    if n == 2:
        return {"train": [ordered[0]], "val": [ordered[1]], "test": []}

    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    if n_train + n_val >= n:
        n_val = 1
        n_train = max(1, n - 2)

    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]

    if not test and len(val) > 1:
        test = [val.pop()]

    return {"train": train, "val": val, "test": test}


def build_sft_dataset(cfg: LabConfig) -> dict[str, Any]:
    pairs = discover_pairs(cfg.source_outputs_dir)

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for pair in pairs:
        try:
            obj = load_json(pair.json_path)
            hits = load_hits(pair.hits_path)
        except Exception as e:
            rejected_rows.append({
                "timestamp": pair.timestamp,
                "status": "load_error",
                "reason": str(e),
                "json_path": str(pair.json_path),
                "hits_path": str(pair.hits_path),
            })
            continue

        valid_ids = set(str(x) for x in hits.get("chunk_id", pd.Series(dtype=str)).dropna().tolist())
        report = validate_target_json(obj, valid_ids)
        ok = is_quality_ok(report)

        if not ok:
            rejected_rows.append({
                "timestamp": pair.timestamp,
                "status": "quality_reject",
                "reason": str(report),
                "json_path": str(pair.json_path),
                "hits_path": str(pair.hits_path),
            })
            continue

        context = build_context(hits, context_rows=cfg.context_rows)

        accepted_rows.append(
            {
                "timestamp": pair.timestamp,
                "instruction": DEFAULT_INSTRUCTION,
                "context": context,
                "target_json": obj,
                "meta": {
                    "source_json": str(pair.json_path),
                    "source_hits": str(pair.hits_path),
                    "num_hits_rows": int(len(hits)),
                    "num_valid_ids": int(len(valid_ids)),
                },
            }
        )

    split = _split_rows(accepted_rows, seed=cfg.seed)

    datasets_dir = cfg.artifacts_dir / "datasets"
    all_path = datasets_dir / "sft_dataset_all.jsonl"
    train_path = datasets_dir / "sft_train.jsonl"
    val_path = datasets_dir / "sft_val.jsonl"
    test_path = datasets_dir / "sft_test.jsonl"

    write_jsonl(all_path, accepted_rows)
    write_jsonl(train_path, split["train"])
    write_jsonl(val_path, split["val"])
    write_jsonl(test_path, split["test"])

    summary = {
        "pairs_found": len(pairs),
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),
        "train": len(split["train"]),
        "val": len(split["val"]),
        "test": len(split["test"]),
        "paths": {
            "all": str(all_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }
    write_json(datasets_dir / "dataset_summary.json", summary)

    if rejected_rows:
        pd.DataFrame(rejected_rows).to_csv(datasets_dir / "dataset_rejections.csv", index=False)
    else:
        pd.DataFrame(columns=["timestamp", "status", "reason", "json_path", "hits_path"]).to_csv(
            datasets_dir / "dataset_rejections.csv", index=False
        )

    return summary
