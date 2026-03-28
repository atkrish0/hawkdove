from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

import pandas as pd


TS_PATTERN = re.compile(r"(\d{8}_\d{6})")


def extract_timestamp(name: str) -> str | None:
    m = TS_PATTERN.search(name)
    return m.group(1) if m else None


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_hits(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")
