from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig


def save_outputs(cfg: PipelineConfig, hits_df: pd.DataFrame, llm_text: str, parsed: dict[str, Any] | None) -> dict[str, Path | None]:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    hits_path = cfg.output_dir / f"hits_{run_ts}.csv"
    hits_df.to_csv(hits_path, index=False)

    txt_path = cfg.output_dir / f"macro_answer_{run_ts}.txt"
    txt_path.write_text(llm_text)

    json_path: Path | None = None
    if parsed is not None:
        json_path = cfg.output_dir / f"macro_answer_{run_ts}.json"
        with open(json_path, "w") as f:
            json.dump(parsed, f, indent=2)

    return {"hits_path": hits_path, "text_path": txt_path, "json_path": json_path}
