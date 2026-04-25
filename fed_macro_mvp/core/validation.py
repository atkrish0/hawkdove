from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import re

import pandas as pd


REQUIRED_TOPICS = ["inflation", "unemployment", "growth", "policy_rates", "financial_conditions", "credit"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _quote_snippet(text: str, max_chars: int = 220) -> str:
    clean = _clean_text(text)
    if not clean:
        return ""

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean) if p.strip()]
    if not parts:
        return clean[:max_chars]

    out = parts[0]
    i = 1
    while i < len(parts) and len(out) < max_chars:
        out = f"{out} {parts[i]}"
        i += 1
    return out[:max_chars]


def _chunk_text_lookup(topic_hits: dict[str, pd.DataFrame]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for _, df in topic_hits.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            cid = str(row.get("chunk_id", "")).strip()
            if not cid:
                continue
            txt = str(row.get("text", "") or "")
            if cid not in lookup and txt:
                lookup[cid] = txt
    return lookup


def canonical_topic_name(topic: str) -> str:
    t = str(topic or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "policy_rate": "policy_rates",
        "policyrates": "policy_rates",
        "policy": "policy_rates",
        "financial_condition": "financial_conditions",
    }
    return aliases.get(t, t)


def resolve_chunk_id(candidate: Any, valid_ids: set[str]) -> str | None:
    c = str(candidate or "").strip()
    if not c:
        return None

    if c in valid_ids:
        return c

    c = c.replace(" ", "")
    if c in valid_ids:
        return c

    m = re.match(r"^(20\d{6})_([A-Za-z0-9_-]+)\.pdf(::chunk\d+)$", c)
    if m:
        ymd, stem, suffix = m.groups()
        alt = f"{stem}{ymd}.pdf{suffix}"
        if alt in valid_ids:
            return alt

    chunk_m = re.search(r"::chunk\d{4}$", c)
    date_m = re.search(r"(20\d{6})", c)
    if chunk_m:
        suffix = chunk_m.group(0)
        cands = [vid for vid in valid_ids if vid.endswith(suffix)]
        if date_m:
            dated = [vid for vid in cands if date_m.group(1) in vid]
            if dated:
                cands = dated
        if cands:
            return sorted(cands)[0]

    return None


def normalize_evidence_list(evidence: Any, valid_ids: set[str], fallback_ids: list[str] | None = None, max_items: int = 3) -> list[str]:
    out: list[str] = []
    if isinstance(evidence, list):
        for e in evidence:
            rid = resolve_chunk_id(e, valid_ids)
            if rid and rid not in out:
                out.append(rid)
            if len(out) >= max_items:
                break

    if not out and fallback_ids:
        for fb in fallback_ids:
            rid = resolve_chunk_id(fb, valid_ids) or (fb if fb in valid_ids else None)
            if rid and rid not in out:
                out.append(rid)
            if len(out) >= max_items:
                break

    return out


def fallback_ids_by_topic(topic_hits: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for topic, df in topic_hits.items():
        if df is None or df.empty:
            out[topic] = []
            continue
        out[topic] = [str(x) for x in df["chunk_id"].head(3).tolist()]
    return out


def coerce_investor_json(
    parsed: dict[str, Any],
    topic_hits: dict[str, pd.DataFrame],
    valid_ids: set[str],
    enforce_topic_min_evidence: bool = True,
    return_meta: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    obj = parsed if isinstance(parsed, dict) else {}
    meta = {
        "topic_fallback_injections": 0,
        "global_fallback_injections": 0,
        "takeaway_fallback_used": False,
        "citation_fallback_used": False,
    }

    # Always stamp generation time to the current run in normalized ISO UTC format.
    obj["generated_at_utc"] = _utc_now_iso()

    obj["executive_summary"] = str(obj.get("executive_summary") or "Macro view generated from retrieved Federal Reserve communications.")

    regime = obj.get("regime_call") if isinstance(obj.get("regime_call"), dict) else {}
    regime_defaults = {
        "growth_momentum": "uncertain",
        "inflation_trend": "uncertain",
        "policy_bias": "uncertain",
        "recession_risk": "uncertain",
        "confidence": "medium",
    }
    for k, v in regime_defaults.items():
        if not str(regime.get(k, "")).strip():
            regime[k] = v
        else:
            regime[k] = str(regime[k])
    obj["regime_call"] = regime

    fbt = fallback_ids_by_topic(topic_hits)
    text_lookup = _chunk_text_lookup(topic_hits)
    global_fallback = []
    for topic in REQUIRED_TOPICS:
        global_fallback.extend(fbt.get(topic, []))
    global_fallback = [x for x in dict.fromkeys(global_fallback) if x]
    if not global_fallback:
        global_fallback = sorted(valid_ids)

    existing_map: dict[str, dict[str, Any]] = {}
    topic_signals = obj.get("topic_signals")
    if isinstance(topic_signals, list):
        for item in topic_signals:
            if not isinstance(item, dict):
                continue
            t = canonical_topic_name(item.get("topic", ""))
            if t in REQUIRED_TOPICS and t not in existing_map:
                existing_map[t] = item

    normalized_signals = []
    for topic in REQUIRED_TOPICS:
        src = existing_map.get(topic, {})
        ev = normalize_evidence_list(src.get("evidence", []), valid_ids, fallback_ids=fbt.get(topic, []), max_items=3)
        if not src.get("evidence") and ev:
            meta["topic_fallback_injections"] += 1
        if enforce_topic_min_evidence and not ev:
            ev = normalize_evidence_list([], valid_ids, fallback_ids=global_fallback, max_items=1)
            if ev:
                meta["global_fallback_injections"] += 1
        normalized_signals.append(
            {
                "topic": topic,
                "view": str(src.get("view") or "No clear signal"),
                "confidence": str(src.get("confidence") or "medium"),
                "evidence": ev,
            }
        )
    obj["topic_signals"] = normalized_signals

    all_signal_evidence: list[str] = []
    for sig in normalized_signals:
        for e in sig["evidence"]:
            if e not in all_signal_evidence:
                all_signal_evidence.append(e)

    takeaways = obj.get("investor_takeaways")
    norm_takeaways = []
    if isinstance(takeaways, list):
        for t in takeaways[:2]:
            if not isinstance(t, dict):
                continue
            ev = normalize_evidence_list(t.get("evidence", []), valid_ids, fallback_ids=all_signal_evidence, max_items=4)
            norm_takeaways.append(
                {
                    "horizon": str(t.get("horizon") or "6-12 months"),
                    "thesis": str(t.get("thesis") or obj["executive_summary"][:200]),
                    "evidence": ev,
                }
            )

    if not norm_takeaways:
        norm_takeaways = [
            {
                "horizon": "6-12 months",
                "thesis": obj["executive_summary"][:200],
                "evidence": all_signal_evidence[:4],
            }
        ]
        meta["takeaway_fallback_used"] = True
    obj["investor_takeaways"] = norm_takeaways

    citations = obj.get("citations")
    norm_citations = []
    seen = set()
    if isinstance(citations, list):
        for c in citations:
            if not isinstance(c, dict):
                continue
            cid = resolve_chunk_id(c.get("chunk_id"), valid_ids)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            norm_citations.append(
                {
                    "chunk_id": cid,
                    "doc_id": cid.split("::")[0],
                    "quote": _quote_snippet(text_lookup.get(cid, "")),
                }
            )
            if len(norm_citations) >= 8:
                break

    if not norm_citations:
        for cid in all_signal_evidence:
            if cid in seen:
                continue
            seen.add(cid)
            norm_citations.append(
                {
                    "chunk_id": cid,
                    "doc_id": cid.split("::")[0],
                    "quote": _quote_snippet(text_lookup.get(cid, "")),
                }
            )
            if len(norm_citations) >= 8:
                break
        if norm_citations:
            meta["citation_fallback_used"] = True

    obj["citations"] = norm_citations
    if return_meta:
        return obj, meta
    return obj


def postprocess_obj(obj: dict[str, Any]) -> dict[str, Any]:
    obj["generated_at_utc"] = _utc_now_iso()
    return obj


def validate_investor_json(parsed: dict[str, Any], valid_ids: set[str], enforce_topic_min_evidence: bool = True) -> dict[str, Any]:
    required_top = ["generated_at_utc", "executive_summary", "regime_call", "topic_signals", "investor_takeaways", "citations"]
    report = {
        "missing_top_keys": [k for k in required_top if k not in parsed],
        "bad_shape": [],
        "bad_evidence_ids": [],
        "unknown_citation_ids": [],
        "missing_topics": [],
        "extra_topics": [],
    }

    topic_signals = parsed.get("topic_signals")
    if not isinstance(topic_signals, list):
        report["bad_shape"].append("topic_signals")
        topic_signals = []

    found_topics = []
    for i, x in enumerate(topic_signals):
        if not isinstance(x, dict):
            report["bad_shape"].append(f"topic_signals[{i}]")
            continue
        topic = x.get("topic")
        if isinstance(topic, str):
            found_topics.append(topic)

        ev = x.get("evidence")
        if not isinstance(ev, list):
            report["bad_shape"].append(f"topic_signals[{i}].evidence")
        else:
            if enforce_topic_min_evidence and len(ev) == 0:
                report["bad_shape"].append(f"topic_signals[{i}].evidence_empty")
            for e in ev:
                if e not in valid_ids:
                    report["bad_evidence_ids"].append({"section": f"topic_signals[{i}]", "value": e})

    found_set = set(found_topics)
    expected_topics = set(REQUIRED_TOPICS)
    report["missing_topics"] = sorted(expected_topics - found_set)
    report["extra_topics"] = sorted(found_set - expected_topics)

    takeaways = parsed.get("investor_takeaways")
    if not isinstance(takeaways, list):
        report["bad_shape"].append("investor_takeaways")
    else:
        for i, x in enumerate(takeaways):
            if not isinstance(x, dict):
                report["bad_shape"].append(f"investor_takeaways[{i}]")
                continue
            ev = x.get("evidence")
            if not isinstance(ev, list):
                report["bad_shape"].append(f"investor_takeaways[{i}].evidence")
            else:
                for e in ev:
                    if e not in valid_ids:
                        report["bad_evidence_ids"].append({"section": f"investor_takeaways[{i}]", "value": e})

    citations = parsed.get("citations")
    if not isinstance(citations, list):
        report["bad_shape"].append("citations")
    else:
        for i, c in enumerate(citations):
            if not isinstance(c, dict) or "chunk_id" not in c:
                report["bad_shape"].append(f"citations[{i}]")
                continue
            if c["chunk_id"] not in valid_ids:
                report["unknown_citation_ids"].append(c["chunk_id"])

    return report


def quality_ok(report: dict[str, Any]) -> bool:
    return (
        len(report["missing_top_keys"]) == 0
        and len(report["bad_shape"]) == 0
        and len(report["bad_evidence_ids"]) == 0
        and len(report["unknown_citation_ids"]) == 0
        and len(report["missing_topics"]) == 0
    )
