from .config import PipelineConfig, default_config


def create_config(*args, **kwargs):
    from .pipeline import create_config as _create_config

    return _create_config(*args, **kwargs)


def run_ingest_and_index(*args, **kwargs):
    from .pipeline import run_ingest_and_index as _run_ingest_and_index

    return _run_ingest_and_index(*args, **kwargs)


def run_full_analysis(*args, **kwargs):
    from .pipeline import run_full_analysis as _run_full_analysis

    return _run_full_analysis(*args, **kwargs)


def persist_results(*args, **kwargs):
    from .pipeline import persist_results as _persist_results

    return _persist_results(*args, **kwargs)


__all__ = [
    "PipelineConfig",
    "default_config",
    "create_config",
    "run_ingest_and_index",
    "run_full_analysis",
    "persist_results",
]
