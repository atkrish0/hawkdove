from __future__ import annotations

from typing import Any

from .constants import REQUIRED_TOPICS


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_target_json(obj: dict[str, Any], valid_ids: set[str]) -> dict[str, Any]:
    report = {
        "missing_top_keys": [],
        "bad_shape": [],
        "missing_topics": [],
        "extra_topics": [],
        "invalid_evidence_ids": [],
        "invalid_citation_ids": [],
    }

    required_top = [
        "generated_at_utc",
        "executive_summary",
        "regime_call",
        "topic_signals",
        "investor_takeaways",
        "citations",
    ]
    report["missing_top_keys"] = [k for k in required_top if k not in obj]

    topic_signals = obj.get("topic_signals")
    if not isinstance(topic_signals, list):
        report["bad_shape"].append("topic_signals")
        topic_signals = []

    found_topics = []
    for i, item in enumerate(topic_signals):
        if not isinstance(item, dict):
            report["bad_shape"].append(f"topic_signals[{i}]")
            continue
        t = item.get("topic")
        if isinstance(t, str):
            found_topics.append(t)
        evidence = _as_list(item.get("evidence"))
        if not evidence:
            report["bad_shape"].append(f"topic_signals[{i}].evidence_empty")
        for e in evidence:
            if str(e) not in valid_ids:
                report["invalid_evidence_ids"].append(str(e))

    found = set(found_topics)
    expected = set(REQUIRED_TOPICS)
    report["missing_topics"] = sorted(expected - found)
    report["extra_topics"] = sorted(found - expected)

    citations = obj.get("citations")
    if not isinstance(citations, list):
        report["bad_shape"].append("citations")
        citations = []
    for i, c in enumerate(citations):
        if not isinstance(c, dict):
            report["bad_shape"].append(f"citations[{i}]")
            continue
        cid = str(c.get("chunk_id", ""))
        if cid and cid not in valid_ids:
            report["invalid_citation_ids"].append(cid)

    return report


def is_quality_ok(report: dict[str, Any]) -> bool:
    return (
        len(report["missing_top_keys"]) == 0
        and len(report["bad_shape"]) == 0
        and len(report["missing_topics"]) == 0
        and len(report["extra_topics"]) == 0
        and len(report["invalid_evidence_ids"]) == 0
        and len(report["invalid_citation_ids"]) == 0
    )
