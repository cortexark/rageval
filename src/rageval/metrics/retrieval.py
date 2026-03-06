"""Retrieval quality metrics.

Computes precision, recall, F1, MRR, NDCG, and hit rate by comparing
retrieved document IDs against ground-truth relevant document IDs.

These metrics are deterministic and do not require LLM calls.
"""

from __future__ import annotations

import math

import structlog

from rageval.core.models import RetrievalMetrics

logger = structlog.get_logger(__name__)


class RetrievalEvaluator:
    """Compute retrieval metrics from retrieved vs. relevant document sets.

    Example::

        evaluator = RetrievalEvaluator()
        metrics = evaluator.compute(
            retrieved_ids=["doc1", "doc3", "doc5"],
            relevant_ids=["doc1", "doc2", "doc3"],
        )
        print(f"Precision: {metrics.precision:.2f}")
        print(f"Recall: {metrics.recall:.2f}")
        print(f"F1: {metrics.f1_score:.2f}")
    """

    def compute(
        self,
        *,
        retrieved_ids: list[str],
        relevant_ids: list[str],
    ) -> RetrievalMetrics:
        """Compute all retrieval metrics for a single query.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids: Set of ground-truth relevant document IDs.

        Returns:
            RetrievalMetrics with all computed values.
        """
        if not relevant_ids:
            logger.warning("retrieval.no_relevant_ids", msg="No relevant IDs provided")
            return RetrievalMetrics(
                retrieved_count=len(retrieved_ids),
                relevant_count=0,
            )

        relevant_set = set(relevant_ids)
        retrieved_set = set(retrieved_ids)
        relevant_retrieved = relevant_set & retrieved_set

        precision = self._precision(retrieved_ids, relevant_set)
        recall = self._recall(retrieved_ids, relevant_set)
        f1 = self._f1(precision, recall)
        mrr = self._mrr(retrieved_ids, relevant_set)
        ndcg = self._ndcg(retrieved_ids, relevant_set)
        hit_rate = 1.0 if relevant_retrieved else 0.0

        return RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            mrr=mrr,
            ndcg=ndcg,
            hit_rate=hit_rate,
            retrieved_count=len(retrieved_ids),
            relevant_count=len(relevant_ids),
            relevant_retrieved_count=len(relevant_retrieved),
        )

    def compute_batch(
        self,
        *,
        batch: list[tuple[list[str], list[str]]],
    ) -> list[RetrievalMetrics]:
        """Compute retrieval metrics for a batch of queries.

        Args:
            batch: List of (retrieved_ids, relevant_ids) tuples.

        Returns:
            List of RetrievalMetrics, one per query.
        """
        return [
            self.compute(retrieved_ids=retrieved, relevant_ids=relevant)
            for retrieved, relevant in batch
        ]

    @staticmethod
    def _precision(retrieved: list[str], relevant: set[str]) -> float:
        """Fraction of retrieved documents that are relevant."""
        if not retrieved:
            return 0.0
        hits = sum(1 for doc_id in retrieved if doc_id in relevant)
        return hits / len(retrieved)

    @staticmethod
    def _recall(retrieved: list[str], relevant: set[str]) -> float:
        """Fraction of relevant documents that were retrieved."""
        if not relevant:
            return 0.0
        hits = sum(1 for doc_id in retrieved if doc_id in relevant)
        return hits / len(relevant)

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        """Harmonic mean of precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def _mrr(retrieved: list[str], relevant: set[str]) -> float:
        """Mean Reciprocal Rank — 1/rank of the first relevant result."""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _ndcg(retrieved: list[str], relevant: set[str]) -> float:
        """Normalized Discounted Cumulative Gain.

        Binary relevance: 1 if relevant, 0 otherwise.
        """
        if not relevant or not retrieved:
            return 0.0

        # DCG: sum of 1/log2(i+2) for relevant docs at position i
        dcg = 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(i + 2)

        # Ideal DCG: relevant docs at top positions
        ideal_count = min(len(relevant), len(retrieved))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

        if idcg == 0:
            return 0.0
        return dcg / idcg
