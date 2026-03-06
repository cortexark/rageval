"""Evaluation pipeline and storage.

Orchestrates end-to-end RAG evaluation: run retrieval metrics,
generation metrics, store results in DuckDB, compute aggregates.
"""

from rageval.pipeline.runner import EvalRunner
from rageval.pipeline.storage import ResultStore

__all__ = ["EvalRunner", "ResultStore"]
