# Hawkdove Model Lab (Parallel Track)

This directory is a **new, isolated implementation track** for building a domain-specialized Hawkdove model.
It does **not modify** existing notebooks, code, data, or logs under `fed_macro_mvp/`.

## Purpose

Build a realistic path from current RAG outputs to a trainable Hawkdove model workflow:
1. Reuse validated historical outputs as **silver-label supervision**.
2. Build strict JSON SFT datasets for domain alignment.
3. Evaluate baseline output quality quantitatively.
4. Provide a LoRA training entrypoint (dependency-gated) for local-first iteration.

## What This Lab Contains

- `src/hawkdove_lab/dataset_builder.py`
  - Discovers paired artifacts (`macro_answer_<ts>.json` + `hits_<ts>.csv`).
  - Validates schema + evidence/citation IDs.
  - Builds SFT JSONL dataset and train/val/test splits.
- `src/hawkdove_lab/eval_baseline.py`
  - Computes baseline pass/fail quality metrics over existing artifacts.
- `src/hawkdove_lab/train_lora.py`
  - LoRA SFT training pipeline.
  - If train dependencies are missing, writes `train_status.json` with actionable reason.
- `src/hawkdove_lab/validation.py`
  - Contract checks for strict JSON output and evidence integrity.
- `src/hawkdove_lab/cli.py`
  - Commands: `build-dataset`, `eval-baseline`, `train-lora`.
- `tests/test_dataset_builder.py`
  - Unit tests for pair-discovery and dataset construction.
- `IMPLEMENTATION_LOG.md`
  - Living record of actions, bugs, fixes, and outcomes.

## Data Flow

1. Source artifacts are read from: `../fed_macro_mvp/outputs/`
2. Pairing key is timestamp (`YYYYMMDD_HHMMSS`).
3. Only rows passing validation enter the dataset.
4. Outputs are written to:
   - `artifacts/datasets/`
   - `artifacts/eval/`
   - `artifacts/lora/` (training artifacts)

## Output Contracts

### SFT dataset row shape

```json
{
  "timestamp": "20260318_145936",
  "instruction": "Build a 6-12 month US macro investor view...",
  "context": "[chunk_id=...; doc_id=...; topic=...; score=...]\\n...",
  "target_json": {"generated_at_utc": "...", "topic_signals": [...], "citations": [...]},
  "meta": {
    "source_json": ".../macro_answer_*.json",
    "source_hits": ".../hits_*.csv",
    "num_hits_rows": 10,
    "num_valid_ids": 10
  }
}
```

### Baseline eval summary

```json
{
  "total_pairs": 7,
  "ok": 6,
  "quality_failed": 1,
  "load_error": 0,
  "quality_pass_rate": 0.8571
}
```

## Quick Start

From repo root:

```bash
cd /Users/atheeshkrishnan/AK/DEV/hawkdove
python3 -m venv .venv_hawkdove_lab
source .venv_hawkdove_lab/bin/activate
pip install -r hawkdove_model_lab/requirements.txt
```

Run commands:

```bash
PYTHONPATH=hawkdove_model_lab/src python3 -m hawkdove_lab.cli build-dataset --project-root /Users/atheeshkrishnan/AK/DEV/hawkdove
PYTHONPATH=hawkdove_model_lab/src python3 -m hawkdove_lab.cli eval-baseline --project-root /Users/atheeshkrishnan/AK/DEV/hawkdove
PYTHONPATH=hawkdove_model_lab/src python3 -m hawkdove_lab.cli train-lora --project-root /Users/atheeshkrishnan/AK/DEV/hawkdove
```

To actually run training (opt-in safety gate):

```bash
HAWKDOVE_ENABLE_TRAIN=1 PYTHONPATH=hawkdove_model_lab/src python3 -m hawkdove_lab.cli train-lora --project-root /Users/atheeshkrishnan/AK/DEV/hawkdove
```

## Design Notes

- Reliability-first: dataset rows are rejected unless strict quality checks pass.
- Fed-only by construction: source artifacts currently come from the Fed-focused MVP pipeline.
- Non-destructive: this lab only reads from existing `fed_macro_mvp/outputs` and writes to its own `artifacts/`.
- Local-first: LoRA training has graceful dependency gating instead of hard failures.

## Known Limitations

- Current source dataset is small; robust fine-tuning requires more accepted runs.
- Validation focuses on structure/evidence integrity, not macro correctness scoring.
- Training script uses a lightweight baseline model (`TinyLlama`) for portability; model choice can be upgraded later.

## Next Enhancements

1. Add manual-QA reviewed labels on top of silver labels.
2. Add held-out time-based split policies (chronological validation).
3. Add comparison harness: baseline vs tuned model on the same prompt/context set.
4. Add richer metrics (topic-level precision/recall proxies, citation coverage depth).
