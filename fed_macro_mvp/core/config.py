from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_TOPIC_QUERIES = {
    "inflation": {
        "query": "What do recent Federal Reserve communications imply about inflation trend, stickiness, and upside/downside risks?",
        "keywords": ["inflation", "prices", "PCE", "CPI", "expectations"],
    },
    "unemployment": {
        "query": "What do recent communications imply about unemployment, labor demand, and job market slack?",
        "keywords": ["unemployment", "labor", "employment", "wages", "payroll"],
    },
    "growth": {
        "query": "What do recent Federal Reserve communications imply about GDP growth, demand, and activity momentum?",
        "keywords": ["growth", "GDP", "activity", "demand", "consumption", "investment"],
    },
    "policy_rates": {
        "query": "What do recent Federal Reserve communications imply about policy rate path, cuts/hikes, and reaction function?",
        "keywords": ["federal funds", "policy", "rate", "hike", "cut", "restrictive"],
    },
    "financial_conditions": {
        "query": "What do recent communications imply about financial conditions, yields, equities, dollar, and credit spreads?",
        "keywords": ["financial conditions", "yields", "equity", "dollar", "spreads", "treasury"],
    },
    "credit": {
        "query": "What do recent communications imply about bank lending standards, credit availability, and funding conditions?",
        "keywords": ["credit", "lending", "bank", "loan", "funding", "delinquency"],
    },
}


DEFAULT_TOPIC_QUERY_VARIANTS = {
    "inflation": ["inflation trend and persistence", "inflation expectations and upside risks"],
    "unemployment": ["labor market slack and unemployment trajectory", "employment, wages, and labor demand"],
    "growth": ["real activity and GDP momentum", "consumer demand and business investment outlook"],
    "policy_rates": ["federal funds rate path and policy stance", "conditions for rate cuts or hikes"],
    "financial_conditions": [
        "financial conditions, yields, and risk appetite",
        "treasury yields, dollar strength, and market pricing",
    ],
    "credit": ["bank lending standards and credit supply", "credit availability and funding stress"],
}


@dataclass
class PipelineConfig:
    project_dir: Path

    # Paths
    data_dir: Path = field(init=False)
    raw_pdf_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    index_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    # Collection
    days_back: int = 540
    max_pdfs: int = 40
    request_timeout: int = 20

    # Chunking / indexing
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 8
    chunk_size: int = 1200
    chunk_overlap: int = 180

    # LLM
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3:8b"

    # Focus
    focus_topics: list[str] = field(
        default_factory=lambda: [
            "inflation",
            "unemployment",
            "labor market",
            "growth",
            "interest rates",
            "financial conditions",
            "credit",
        ]
    )
    seed_pages: list[str] = field(
        default_factory=lambda: [
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            "https://www.federalreserve.gov/monetarypolicy/fomcminutes.htm",
            "https://www.federalreserve.gov/monetarypolicy/mpr_default.htm",
            "https://www.federalreserve.gov/monetarypolicy/beigebookdefault.htm",
        ]
    )

    # Investor analysis
    top_k_topic: int = 2
    context_chunks_per_topic: int = 1
    max_chars_per_chunk: int = 700
    max_context_chars: int = 2800
    ollama_num_predict: int = 320
    ollama_num_ctx: int = 4096
    ollama_temperature: float = 0.0

    enable_hybrid_retrieval: bool = True
    enable_query_fusion: bool = True
    rrf_k: int = 60
    recency_half_life_days: int = 180
    recency_boost: float = 0.35
    candidates_per_query: int = 8

    enable_reranker: bool = False
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_pool: int = 10

    ollama_max_retries: int = 3
    retry_context_shrink: list[float] = field(default_factory=lambda: [1.0, 0.85, 0.7])
    retry_predict_shrink: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.9])
    use_json_repair: bool = True

    topic_queries: dict[str, Any] = field(default_factory=lambda: DEFAULT_TOPIC_QUERIES.copy())
    topic_query_variants: dict[str, list[str]] = field(default_factory=lambda: DEFAULT_TOPIC_QUERY_VARIANTS.copy())

    def __post_init__(self) -> None:
        if not (self.project_dir / "fed_macro_mvp.ipynb").exists() and (self.project_dir / "fed_macro_mvp").exists():
            self.project_dir = self.project_dir / "fed_macro_mvp"

        self.data_dir = self.project_dir / "data"
        self.raw_pdf_dir = self.data_dir / "raw_pdfs"
        self.processed_dir = self.data_dir / "processed"
        self.index_dir = self.project_dir / "index"
        self.output_dir = self.project_dir / "outputs"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for p in [self.raw_pdf_dir, self.processed_dir, self.index_dir, self.output_dir]:
            p.mkdir(parents=True, exist_ok=True)


def default_config(project_dir: Path | None = None) -> PipelineConfig:
    if project_dir is None:
        project_dir = Path.cwd()
    return PipelineConfig(project_dir=project_dir)
