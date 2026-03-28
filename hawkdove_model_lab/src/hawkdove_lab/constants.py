from __future__ import annotations

REQUIRED_TOPICS = [
    "inflation",
    "unemployment",
    "growth",
    "policy_rates",
    "financial_conditions",
    "credit",
]

DEFAULT_INSTRUCTION = (
    "Build a 6-12 month US macro investor view from Federal Reserve communications. "
    "Return strict JSON only with keys: generated_at_utc, executive_summary, regime_call, "
    "topic_signals, investor_takeaways, citations. Evidence and citations must use provided chunk_ids."
)
