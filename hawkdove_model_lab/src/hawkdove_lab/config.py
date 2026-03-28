from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabConfig:
    project_root: Path
    source_outputs_dir: Path
    artifacts_dir: Path

    context_rows: int = 10
    quote_chars: int = 220
    seed: int = 42

    def ensure_dirs(self) -> None:
        (self.artifacts_dir / "datasets").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "eval").mkdir(parents=True, exist_ok=True)



def default_config(project_root: Path | None = None) -> LabConfig:
    if project_root is None:
        project_root = Path.cwd()

    source_outputs = project_root / "fed_macro_mvp" / "outputs"
    artifacts = project_root / "hawkdove_model_lab" / "artifacts"

    cfg = LabConfig(
        project_root=project_root,
        source_outputs_dir=source_outputs,
        artifacts_dir=artifacts,
    )
    cfg.ensure_dirs()
    return cfg
