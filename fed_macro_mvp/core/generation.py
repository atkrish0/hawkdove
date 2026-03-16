from __future__ import annotations

from typing import Any
import json
import time

import pandas as pd
import requests
from ollama import Client

from .config import PipelineConfig
from .retrieval import build_investor_context, collect_valid_ids
from .validation import coerce_investor_json, postprocess_obj, quality_ok, validate_investor_json

try:
    from json_repair import repair_json
    HAVE_JSON_REPAIR = True
except Exception:
    HAVE_JSON_REPAIR = False


def check_ollama_ready(cfg: PipelineConfig) -> None:
    try:
        r = requests.get(f"{cfg.ollama_host}/api/tags", timeout=3)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise RuntimeError(
            "Ollama is not reachable. Start Ollama first (e.g., `ollama serve` or open the Ollama app). "
            f"Host tried: {cfg.ollama_host}. Original error: {e}"
        )

    names = {m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)}
    if cfg.ollama_model not in names:
        hint = ", ".join(sorted(names)) if names else "<no models found>"
        raise RuntimeError(
            f"Model '{cfg.ollama_model}' not found in Ollama. Available: {hint}. "
            f"Run: ollama pull {cfg.ollama_model} (or update config)."
        )


def generate_investor_view(question: str, context: str, cfg: PipelineConfig, num_predict: int | None = None) -> str:
    if num_predict is None:
        num_predict = cfg.ollama_num_predict

    system = " ".join(
        [
            "Return valid JSON only.",
            "No markdown, no code fences, no extra text.",
            "Top keys: generated_at_utc, executive_summary, regime_call, topic_signals, investor_takeaways, citations.",
            "regime_call keys: growth_momentum, inflation_trend, policy_bias, recession_risk, confidence.",
            "topic_signals must contain exactly six topics: inflation, unemployment, growth, policy_rates, financial_conditions, credit.",
            "Each topic_signals item keys: topic, view, confidence, evidence.",
            "investor_takeaways: 1 or 2 items, each keys: horizon, thesis, evidence.",
            "citations: up to 8 items, each keys: chunk_id, doc_id.",
            "Every evidence and citation chunk_id must come from provided context chunk_id values.",
            "Use concise strings.",
        ]
    )

    client = Client(host=cfg.ollama_host)
    out = client.chat(
        model=cfg.ollama_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}],
        options={"temperature": cfg.ollama_temperature, "num_predict": int(num_predict), "num_ctx": cfg.ollama_num_ctx},
        format="json",
    )
    return out["message"]["content"]


def extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    end = None

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is not None:
        return text[start:end]
    if depth > 0:
        return text[start:] + ("}" * depth)
    return text[start:]


def parse_json_robust(text: str, use_json_repair: bool = True) -> dict[str, Any] | None:
    candidates = []
    raw = text.strip()
    if raw:
        candidates.append(raw)

    extracted = extract_balanced_json(raw)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            pass

    if HAVE_JSON_REPAIR and use_json_repair:
        for c in candidates or [raw]:
            try:
                repaired = repair_json(c)
                return json.loads(repaired)
            except Exception:
                continue

    return None


def repair_json_with_llm(raw_text: str, cfg: PipelineConfig, num_predict: int = 240) -> dict[str, Any] | None:
    if not raw_text.strip():
        return None

    client = Client(host=cfg.ollama_host)
    system = " ".join([
        "You repair malformed JSON.",
        "Return a valid JSON object only.",
        "Do not add explanations or markdown.",
        "Preserve keys and values when possible.",
    ])

    try:
        out = client.chat(
            model=cfg.ollama_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Repair this malformed JSON into valid JSON:\n\n" + raw_text[:9000]},
            ],
            options={"temperature": 0.0, "num_predict": int(num_predict), "num_ctx": cfg.ollama_num_ctx},
            format="json",
        )
        return parse_json_robust(out["message"]["content"], use_json_repair=cfg.use_json_repair)
    except Exception:
        return None


def generation_retry_plan(cfg: PipelineConfig) -> list[dict[str, Any]]:
    plan = []
    for i in range(cfg.ollama_max_retries):
        ctx_factor = cfg.retry_context_shrink[min(i, len(cfg.retry_context_shrink) - 1)]
        pred_factor = cfg.retry_predict_shrink[min(i, len(cfg.retry_predict_shrink) - 1)]
        plan.append(
            {
                "attempt": i + 1,
                "context_chars": max(1400, int(cfg.max_context_chars * ctx_factor)),
                "num_predict": max(260, int(cfg.ollama_num_predict * pred_factor)),
            }
        )

    unique, seen = [], set()
    for x in plan:
        key = (x["context_chars"], x["num_predict"])
        if key not in seen:
            unique.append(x)
            seen.add(key)
    return unique


def run_generation_with_retries(question: str, topic_hits: dict[str, pd.DataFrame], cfg: PipelineConfig):
    attempts = generation_retry_plan(cfg)
    last_text = ""
    last_context = ""
    last_valid_ids: set[str] = set()
    last_parsed = None
    last_quality = None
    logs = []
    retrieval_valid_ids = collect_valid_ids(topic_hits, per_topic=max(2, cfg.top_k_topic))

    for a in attempts:
        context, context_ids = build_investor_context(topic_hits, cfg, max_context_chars=a["context_chars"])
        valid_ids = set(context_ids) | retrieval_valid_ids

        t0 = time.time()
        text = generate_investor_view(question, context, cfg, num_predict=a["num_predict"])
        lat = time.time() - t0

        parsed_obj = parse_json_robust(text, use_json_repair=cfg.use_json_repair)
        if parsed_obj is None:
            logs.append(
                {
                    "attempt": a["attempt"],
                    "status": "parse_failed",
                    "latency_s": round(lat, 2),
                    "context_chars": a["context_chars"],
                    "num_predict": a["num_predict"],
                }
            )
            last_text, last_context, last_valid_ids = text, context, valid_ids
            continue

        parsed_obj = postprocess_obj(parsed_obj)
        parsed_obj = coerce_investor_json(parsed_obj, topic_hits, valid_ids)
        q = validate_investor_json(parsed_obj, valid_ids)
        ok = quality_ok(q)

        logs.append(
            {
                "attempt": a["attempt"],
                "status": "ok" if ok else "quality_failed",
                "latency_s": round(lat, 2),
                "context_chars": a["context_chars"],
                "num_predict": a["num_predict"],
            }
        )

        last_text, last_context, last_valid_ids = text, context, valid_ids
        last_parsed, last_quality = parsed_obj, q

        if ok:
            return text, parsed_obj, q, pd.DataFrame(logs), context, list(valid_ids)

    repaired = repair_json_with_llm(last_text, cfg, num_predict=max(220, int(cfg.ollama_num_predict * 0.75)))
    if repaired is not None:
        repaired = postprocess_obj(repaired)
        repaired = coerce_investor_json(repaired, topic_hits, last_valid_ids)
        rq = validate_investor_json(repaired, last_valid_ids)
        logs.append(
            {
                "attempt": len(attempts) + 1,
                "status": "repaired_ok" if quality_ok(rq) else "repaired_quality_failed",
                "latency_s": None,
                "context_chars": len(last_context),
                "num_predict": None,
            }
        )
        return last_text, repaired, rq, pd.DataFrame(logs), last_context, list(last_valid_ids)

    return last_text, last_parsed, last_quality, pd.DataFrame(logs), last_context, list(last_valid_ids)
