"""Retriever assessment harness.

Wraps any LlamaIndex retriever and assesses it against a golden
dataset of queries with known relevant documents.
"""

from __future__ import annotations

from typing import Any, Protocol

import structlog

from rageval.core.models import EvalResult, EvalSample, RetrievalMetrics
from rageval.metrics.retrieval import RetrievalEvaluator

logger = structlog.get_logger(__name__)


class Retriever(Protocol):
    """Protocol for retriever implementations."""

    def retrieve(self, query: str) -> list[dict[str, Any]]: ...


class RetrieverHarness:
    """Assess a retriever against golden query-document pairs.

    Supports any retriever that follows the Retriever protocol,
    including LlamaIndex retrievers wrapped with an adapter.

    Example::

        harness = RetrieverHarness(retriever=my_retriever)
        results = harness.run_samples(samples)
        for r in results:
            print(f"Query: {r.sample.query}")
            print(f"  Precision: {r.retrieval_metrics.precision:.2f}")
            print(f"  Recall: {r.retrieval_metrics.recall:.2f}")
    """

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._top_k = top_k
        self._metric_scorer = RetrievalEvaluator()

    def run_samples(
        self,
        samples: list[EvalSample],
        *,
        retrieved_results: list[list[str]] | None = None,
    ) -> list[EvalResult]:
        """Run retrieval assessment on a batch of samples.

        If retrieved_results is provided, uses those instead of calling
        the retriever. This enables offline scoring of pre-computed results.

        Args:
            samples: Assessment samples with queries and relevant doc IDs.
            retrieved_results: Pre-computed retrieved doc IDs per sample.

        Returns:
            List of EvalResult with retrieval metrics populated.
        """
        results: list[EvalResult] = []

        for i, sample in enumerate(samples):
            if retrieved_results is not None:
                retrieved_ids = retrieved_results[i] if i < len(retrieved_results) else []
            elif self._retriever is not None:
                raw_results = self._retriever.retrieve(sample.query)
                retrieved_ids = [r.get("id", str(j)) for j, r in enumerate(raw_results)]
            else:
                logger.warning("retriever.none", msg="No retriever and no pre-computed results")
                retrieved_ids = []

            metrics = self._metric_scorer.compute(
                retrieved_ids=retrieved_ids,
                relevant_ids=sample.reference_contexts,
            )

            results.append(
                EvalResult(
                    sample=sample,
                    retrieved_doc_ids=retrieved_ids,
                    retrieval_metrics=metrics,
                )
            )

        return results

    def run_with_metrics(
        self,
        samples: list[EvalSample],
        *,
        retrieved_results: list[list[str]],
    ) -> tuple[list[EvalResult], RetrievalMetrics]:
        """Run assessment and return both individual results and aggregate metrics.

        Args:
            samples: Assessment samples.
            retrieved_results: Pre-computed retrieved doc IDs.

        Returns:
            Tuple of (individual results, aggregate metrics).
        """
        results = self.run_samples(samples, retrieved_results=retrieved_results)
        aggregate = self._aggregate_metrics([r.retrieval_metrics for r in results])
        return results, aggregate

    @staticmethod
    def _aggregate_metrics(metrics_list: list[RetrievalMetrics]) -> RetrievalMetrics:
        """Compute average metrics across all samples."""
        if not metrics_list:
            return RetrievalMetrics()

        n = len(metrics_list)
        return RetrievalMetrics(
            precision=sum(m.precision for m in metrics_list) / n,
            recall=sum(m.recall for m in metrics_list) / n,
            f1_score=sum(m.f1_score for m in metrics_list) / n,
            mrr=sum(m.mrr for m in metrics_list) / n,
            ndcg=sum(m.ndcg for m in metrics_list) / n,
            hit_rate=sum(m.hit_rate for m in metrics_list) / n,
            retrieved_count=sum(m.retrieved_count for m in metrics_list),
            relevant_count=sum(m.relevant_count for m in metrics_list),
            relevant_retrieved_count=sum(m.relevant_retrieved_count for m in metrics_list),
        )
