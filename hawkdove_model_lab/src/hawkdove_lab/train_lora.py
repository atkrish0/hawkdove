from __future__ import annotations

from pathlib import Path
import json
import os

from .config import LabConfig


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _format_row(row: dict) -> str:
    target = json.dumps(row.get("target_json", {}), ensure_ascii=False)
    return (
        "### Instruction\n"
        f"{row.get('instruction', '')}\n\n"
        "### Context\n"
        f"{row.get('context', '')}\n\n"
        "### Response\n"
        f"{target}"
    )


def run_lora_training(
    cfg: LabConfig,
    train_path: Path | None = None,
    val_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    if train_path is None:
        train_path = cfg.artifacts_dir / "datasets" / "sft_train.jsonl"
    if val_path is None:
        val_path = cfg.artifacts_dir / "datasets" / "sft_val.jsonl"
    if output_dir is None:
        output_dir = cfg.artifacts_dir / "lora" / "hawkdove_v1"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Safety gate: training is opt-in because some local environments crash on torch/OpenMP init.
    if os.environ.get("HAWKDOVE_ENABLE_TRAIN", "0") != "1":
        msg = {
            "status": "skipped_training_disabled_by_default",
            "hint": (
                "Set HAWKDOVE_ENABLE_TRAIN=1 and install requirements-train.txt "
                "to run LoRA training."
            ),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "output_dir": str(output_dir),
        }
        with open(output_dir / "train_status.json", "w") as f:
            json.dump(msg, f, indent=2)
        print(msg)
        return

    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as e:
        msg = {
            "status": "blocked_missing_train_dependencies",
            "message": str(e),
            "hint": "Install requirements-train.txt to enable LoRA training.",
            "train_path": str(train_path),
            "val_path": str(val_path),
            "output_dir": str(output_dir),
        }
        with open(output_dir / "train_status.json", "w") as f:
            json.dump(msg, f, indent=2)
        print(msg)
        return

    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")

    train_rows = _load_jsonl(train_path)
    val_rows = _load_jsonl(val_path) if val_path.exists() else []

    if len(train_rows) == 0:
        raise ValueError("Training dataset is empty")

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    def tokenize_batch(batch):
        texts = [_format_row(x) for x in batch["raw"]]
        tok = tokenizer(texts, truncation=True, max_length=2048, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return tok

    train_ds = Dataset.from_dict({"raw": train_rows}).map(tokenize_batch, batched=True, remove_columns=["raw"])
    eval_ds = Dataset.from_dict({"raw": val_rows}).map(tokenize_batch, batched=True, remove_columns=["raw"]) if val_rows else None

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_ds is not None else "no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()

    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    with open(output_dir / "train_status.json", "w") as f:
        json.dump(
            {
                "status": "ok",
                "model_id": model_id,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "output_dir": str(output_dir),
            },
            f,
            indent=2,
        )

    print({"status": "ok", "output_dir": str(output_dir)})
