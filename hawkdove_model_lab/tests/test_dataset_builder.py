from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import os

os.environ.setdefault("PANDAS_NO_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_BOTTLENECK", "1")
import pandas as pd

from hawkdove_lab.config import LabConfig
from hawkdove_lab.dataset_builder import build_sft_dataset, discover_pairs


class TestDatasetBuilder(unittest.TestCase):
    def test_discover_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "macro_answer_20260318_120000.json").write_text("{}")
            (p / "hits_20260318_120000.csv").write_text("chunk_id\nabc\n")
            (p / "macro_answer_20260318_120500.json").write_text("{}")
            pairs = discover_pairs(p)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].timestamp, "20260318_120000")

    def test_build_dataset_accepts_valid_row(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            outputs = root / "fed_macro_mvp" / "outputs"
            artifacts = root / "hawkdove_model_lab" / "artifacts"
            outputs.mkdir(parents=True)
            (artifacts / "datasets").mkdir(parents=True)
            (artifacts / "eval").mkdir(parents=True)

            ts = "20260318_121212"
            chunk_id = "fomcminutes20260301.pdf::chunk0001"

            obj = {
                "generated_at_utc": "2026-03-18T12:12:12Z",
                "executive_summary": "x",
                "regime_call": {
                    "growth_momentum": "Moderate",
                    "inflation_trend": "Stable",
                    "policy_bias": "Neutral",
                    "recession_risk": "Low",
                    "confidence": "0.7",
                },
                "topic_signals": [
                    {"topic": "inflation", "view": "x", "confidence": "0.7", "evidence": [chunk_id]},
                    {"topic": "unemployment", "view": "x", "confidence": "0.7", "evidence": [chunk_id]},
                    {"topic": "growth", "view": "x", "confidence": "0.7", "evidence": [chunk_id]},
                    {"topic": "policy_rates", "view": "x", "confidence": "0.7", "evidence": [chunk_id]},
                    {"topic": "financial_conditions", "view": "x", "confidence": "0.7", "evidence": [chunk_id]},
                    {"topic": "credit", "view": "x", "confidence": "0.7", "evidence": [chunk_id]},
                ],
                "investor_takeaways": [{"horizon": "6-12 months", "thesis": "x", "evidence": [chunk_id]}],
                "citations": [{"chunk_id": chunk_id, "doc_id": "fomcminutes20260301.pdf"}],
            }

            with open(outputs / f"macro_answer_{ts}.json", "w") as f:
                json.dump(obj, f)

            hits_df = pd.DataFrame(
                [
                    {
                        "chunk_id": chunk_id,
                        "doc_id": "fomcminutes20260301.pdf",
                        "topic": "inflation",
                        "final_score": 0.9,
                        "text": "Inflation sentence.",
                    }
                ]
            )
            hits_df.to_csv(outputs / f"hits_{ts}.csv", index=False)

            cfg = LabConfig(project_root=root, source_outputs_dir=outputs, artifacts_dir=artifacts)
            cfg.ensure_dirs()
            out = build_sft_dataset(cfg)

            self.assertEqual(out["accepted"], 1)
            self.assertEqual(out["rejected"], 0)
            self.assertTrue((artifacts / "datasets" / "sft_dataset_all.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
