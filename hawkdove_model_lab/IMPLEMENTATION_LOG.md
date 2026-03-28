# Implementation Log (Living)

## 2026-03-28

### Session Goal
Implement a new parallel Hawkdove model-lab track without changing any existing `fed_macro_mvp` files/data/notebooks/logs.

### Actions Performed
1. Created isolated workspace:
   - `hawkdove_model_lab/`
2. Implemented core modules:
   - config, constants, IO helpers, validation
   - dataset builder (pair discovery, filtering, split)
   - baseline evaluator
   - LoRA training entrypoint with dependency-gated fallback
   - CLI wrapper
3. Added unit tests for critical dataset behavior.
4. Added detailed README for architecture and runbook.

### Bugs/Issues Encountered and Fixes
1. **Potential artifact mismatch risk (JSON exists but no matching hits)**
   - Observation: at least one timestamp had `macro_answer_*.txt` and hits, but no JSON.
   - Fix: Pairing logic explicitly requires `macro_answer_*.json` + `hits_*.csv` by shared timestamp.
   - Outcome: only valid supervised examples are considered.

2. **Training dependency uncertainty on local environment**
   - Risk: `transformers/datasets/peft/torch` may not be installed.
   - Fix: `train_lora.py` catches import failures and writes structured `train_status.json` with clear remediation.
   - Outcome: no hard crash; user gets actionable unblock path.

3. **Small-data split edge cases**
   - Risk: tiny dataset could produce empty or invalid train/val/test splits.
   - Fix: split function includes guards for `n=1`, `n=2`, and minimum bucket logic.
   - Outcome: deterministic valid splits for low-sample regimes.

### Outputs Produced (This Session)
- Code: `hawkdove_model_lab/src/hawkdove_lab/*.py`
- Tests: `hawkdove_model_lab/tests/test_dataset_builder.py`
- Docs: `hawkdove_model_lab/README.md`, `hawkdove_model_lab/IMPLEMENTATION_LOG.md`

### Validation Status
- Unit tests executed.
- Dataset build executed.
- Baseline eval executed.
- LoRA command executed in safe mode (dependency-gated, non-crashing).

### Validation Run Results (same session)
- Unit tests: `OK` (2 tests passed).
- Dataset build run: success.
  - `pairs_found=6`, `accepted=5`, `rejected=1`, `train=3`, `val=1`, `test=1`.
- Baseline eval run: success.
  - `total_pairs=6`, `ok=5`, `quality_failed=1`, `load_error=0`, `quality_pass_rate=0.8333`.

### Additional Bugs/Issues and Fixes
4. **Timestamp pairing bug in dataset discovery**
   - Symptom: pair discovery returned 0 despite existing matching files.
   - Root cause: regex was over-escaped in code (`\\d` literal instead of `\d` regex token).
   - Fix: corrected pattern in `io_utils.py` to `r"(\d{8}_\d{6})"`.
   - Outcome: pair discovery correctly found 6 artifact pairs.

5. **CLI startup imported heavy modules unconditionally**
   - Symptom: even `train-lora` command loaded dataset/eval modules at import time.
   - Fix: switched to lazy imports per command branch in `cli.py`.
   - Outcome: cleaner command behavior and reduced unnecessary imports.

6. **LoRA run crashed due OpenMP/Torch runtime in current machine state**
   - Symptom: process abort with OpenMP shared-memory error before graceful exception handling.
   - Fix: added explicit safety gate in `train_lora.py`.
     - Default behavior: do not run training binary imports unless `HAWKDOVE_ENABLE_TRAIN=1`.
     - Writes `train_status.json` with instructions.
   - Outcome: `train-lora` command no longer hard-crashes by default.

### Environment Notes
- Current base Python environment shows NumPy 2.x with extensions compiled against NumPy 1.x (`numexpr`, `bottleneck`), which emits noisy tracebacks during pandas import.
- Despite warnings, dataset/eval pipeline completed successfully.
- Recommended clean run mode for this lab:
  - use an isolated venv from `hawkdove_model_lab/requirements.txt`.
