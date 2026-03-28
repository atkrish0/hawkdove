from __future__ import annotations

import argparse
from pathlib import Path

from .config import default_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hawkdove model lab CLI")
    p.add_argument("command", choices=["build-dataset", "eval-baseline", "train-lora"])
    p.add_argument("--project-root", default=None, help="Path to hawkdove repo root")
    p.add_argument("--train-path", default=None, help="Optional training JSONL path")
    p.add_argument("--val-path", default=None, help="Optional validation JSONL path")
    p.add_argument("--output-dir", default=None, help="Output directory for LoRA artifacts")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve() if args.project_root else None
    cfg = default_config(project_root=root)

    if args.command == "build-dataset":
        from .dataset_builder import build_sft_dataset

        summary = build_sft_dataset(cfg)
        print(summary)
    elif args.command == "eval-baseline":
        from .eval_baseline import evaluate_existing_outputs

        metrics = evaluate_existing_outputs(cfg)
        print(metrics)
    else:
        from .train_lora import run_lora_training

        run_lora_training(
            cfg=cfg,
            train_path=Path(args.train_path).resolve() if args.train_path else None,
            val_path=Path(args.val_path).resolve() if args.val_path else None,
            output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        )


if __name__ == "__main__":
    main()
