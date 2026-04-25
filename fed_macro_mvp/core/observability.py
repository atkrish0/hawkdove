from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid
from typing import Any

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{ts}_{uuid.uuid4().hex[:8]}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, set):
        return [_json_safe(x) for x in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))


def _flatten_payload(prefix: str, value: Any) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    safe = _json_safe(value)
    if isinstance(safe, Mapping):
        for key, item in safe.items():
            flat.update(_flatten_payload(f"{prefix}_{key}", item))
        return flat
    if isinstance(safe, list):
        flat[prefix] = json.dumps(safe, ensure_ascii=False)
        return flat
    flat[prefix] = safe
    return flat


class DiagnosticsRecorder:
    def __init__(
        self,
        cfg,
        *,
        run_id: str | None = None,
        query_id: str | None = None,
        mode: str = "analysis_only",
    ) -> None:
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "enable_observability", True))
        self.run_id = str(run_id or new_run_id())
        self.query_id = str(query_id or self.run_id)
        self.mode = str(mode or "analysis_only")
        self.profile_name = str(getattr(cfg, "profile_name", "unknown"))
        self.events: list[dict[str, Any]] = []
        self.artifacts: dict[str, str] = {}
        self.summary: dict[str, Any] = {}
        self.run_dir: Path | None = None
        self.events_path: Path | None = None

        if self.enabled:
            try:
                self.run_dir = cfg.diagnostics_dir / self.run_id
                self.run_dir.mkdir(parents=True, exist_ok=True)
                self.events_path = self.run_dir / "events.jsonl"
            except Exception:
                self.enabled = False
                self.run_dir = None
                self.events_path = None

    def diagnostics_paths(self) -> dict[str, str | None]:
        base = self.run_dir
        return {
            "run_dir": str(base) if base else None,
            "events_path": str(base / "events.jsonl") if base else None,
            "summary_path": str(base / "summary.json") if base else None,
            "topic_retrieval_path": str(base / "topic_retrieval.csv") if base else None,
            "generation_attempts_path": str(base / "generation_attempts.csv") if base else None,
            "validation_summary_path": str(base / "validation_summary.json") if base else None,
            "artifacts_manifest_path": str(base / "artifacts_manifest.json") if base else None,
        }

    def emit(
        self,
        event_type: str,
        *,
        stage: str,
        status: str,
        payload: Mapping[str, Any] | None = None,
        duration_ms: float | None = None,
        level: str = "info",
    ) -> dict[str, Any]:
        safe_payload = _json_safe(dict(payload or {}))
        event = {
            "event_type": str(event_type),
            "run_id": self.run_id,
            "query_id": self.query_id,
            "ts_utc": utc_now_iso(),
            "stage": str(stage),
            "status": str(status),
            "level": str(level),
            "profile_name": self.profile_name,
            "mode": self.mode,
            "payload": safe_payload,
        }
        if isinstance(safe_payload, Mapping):
            for key, value in safe_payload.items():
                event.update(_flatten_payload(f"payload_{key}", value))
        if duration_ms is not None:
            event["duration_ms"] = round(float(duration_ms), 2)

        self.events.append(event)
        if not self.enabled:
            return event

        try:
            assert self.events_path is not None
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            if getattr(self.cfg, "emit_diagnostics_stdout", False):
                sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except Exception:
            pass
        return event

    def add_artifact(self, name: str, path: Path | str | None) -> None:
        if not path:
            return
        self.artifacts[str(name)] = str(path)
        self.emit(
            "artifact_saved",
            stage="artifacts",
            status="ok",
            payload={"artifact_name": str(name), "path": str(path)},
        )
        self.write_manifest()

    def write_manifest(self) -> None:
        if not self.enabled or not self.run_dir:
            return
        try:
            _write_json(self.run_dir / "artifacts_manifest.json", {"run_id": self.run_id, "artifacts": self.artifacts})
        except Exception:
            pass

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        self.summary = dict(summary)
        if not self.enabled or not self.run_dir:
            return
        try:
            _write_json(self.run_dir / "summary.json", self.summary)
        except Exception:
            pass

    def write_topic_retrieval(self, topic_rows: list[dict[str, Any]] | pd.DataFrame | None) -> None:
        if not self.enabled or not self.run_dir:
            return
        try:
            df = topic_rows if isinstance(topic_rows, pd.DataFrame) else pd.DataFrame(topic_rows or [])
            df.to_csv(self.run_dir / "topic_retrieval.csv", index=False)
        except Exception:
            pass

    def write_generation_attempts(self, attempt_df: pd.DataFrame | None) -> None:
        if not self.enabled or not self.run_dir or attempt_df is None:
            return
        try:
            attempt_df.to_csv(self.run_dir / "generation_attempts.csv", index=False)
        except Exception:
            pass

    def write_validation_summary(self, validation: Mapping[str, Any] | None) -> None:
        if not self.enabled or not self.run_dir or validation is None:
            return
        try:
            _write_json(self.run_dir / "validation_summary.json", validation)
        except Exception:
            pass

    def diagnostics_summary(self) -> dict[str, Any]:
        stage_status_counts: dict[str, int] = {}
        warning_count = 0
        for event in self.events:
            key = f"{event.get('stage', 'unknown')}:{event.get('status', 'unknown')}"
            stage_status_counts[key] = stage_status_counts.get(key, 0) + 1
            if event.get("level") == "warning":
                warning_count += 1
        return {
            "run_id": self.run_id,
            "query_id": self.query_id,
            "event_count": len(self.events),
            "warning_count": warning_count,
            "artifacts_count": len(self.artifacts),
            "stage_status_counts": stage_status_counts,
        }

    def finalize_run(self, status: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        summary = self.diagnostics_summary()
        merged = dict(summary)
        if payload:
            merged.update(_json_safe(dict(payload)))
        self.write_summary(merged)
        self.write_manifest()
        self.emit("run_completed", stage="run", status=status, payload=merged)
        return merged


def create_recorder(cfg, *, run_id: str | None = None, query_id: str | None = None, mode: str = "analysis_only") -> DiagnosticsRecorder:
    recorder = DiagnosticsRecorder(cfg, run_id=run_id, query_id=query_id, mode=mode)
    recorder.emit(
        "run_started",
        stage="run",
        status="started",
        payload={
            "ollama_model": getattr(cfg, "ollama_model", None),
            "enable_hybrid_retrieval": getattr(cfg, "enable_hybrid_retrieval", None),
            "enable_query_fusion": getattr(cfg, "enable_query_fusion", None),
            "enable_reranker": getattr(cfg, "enable_reranker", None),
        },
    )
    return recorder


def summarize_generation_attempts(attempt_df: pd.DataFrame | None) -> dict[str, Any]:
    if attempt_df is None or attempt_df.empty:
        return {"attempts": 0, "successful_attempts": 0, "statuses": {}}
    statuses = attempt_df["status"].fillna("unknown").astype(str).value_counts().to_dict()
    return {
        "attempts": int(len(attempt_df)),
        "successful_attempts": int(sum(1 for x in attempt_df["status"].astype(str).tolist() if x in {"ok", "repaired_ok"})),
        "statuses": statuses,
    }


def quality_summary(quality: Mapping[str, Any] | None) -> dict[str, Any]:
    if quality is None:
        return {"available": False}
    return {
        "available": True,
        "missing_top_keys": len(quality.get("missing_top_keys", [])),
        "bad_shape": len(quality.get("bad_shape", [])),
        "bad_evidence_ids": len(quality.get("bad_evidence_ids", [])),
        "unknown_citation_ids": len(quality.get("unknown_citation_ids", [])),
        "missing_topics": len(quality.get("missing_topics", [])),
        "extra_topics": len(quality.get("extra_topics", [])),
    }
