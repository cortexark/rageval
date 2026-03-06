"""Tests for retrieval metrics (precision, recall, F1, MRR, NDCG)."""

from __future__ import annotations

import math

import pytest

from rageval.metrics.retrieval import RetrievalEvaluator


@pytest.fixture
def evaluator() -> RetrievalEvaluator:
    return RetrievalEvaluator()


class TestPrecision:
    """Precision = relevant_retrieved / total_retrieved."""

    def test_perfect_precision(self, evaluator: RetrievalEvaluator) -> None:
        """All retrieved docs are relevant."""
        result = evaluator.compute(
            retrieved_ids=["d1", "d2", "d3"],
            relevant_ids=["d1", "d2", "d3", "d4"],
        )
        assert result.precision == 1.0

    def test_half_precision(self, evaluator: RetrievalEvaluator) -> None:
        """Half of retrieved docs are relevant."""
        result = evaluator.compute(
            retrieved_ids=["d1", "d2", "d3", "d4"],
            relevant_ids=["d1", "d3"],
        )
        assert result.precision == 0.5

    def test_zero_precision(self, evaluator: RetrievalEvaluator) -> None:
        """No retrieved docs are relevant."""
        result = evaluator.compute(
            retrieved_ids=["d1", "d2"],
            relevant_ids=["d3", "d4"],
        )
        assert result.precision == 0.0

    def test_empty_retrieved(self, evaluator: RetrievalEvaluator) -> None:
        """No docs retrieved."""
        result = evaluator.compute(
            retrieved_ids=[],
            relevant_ids=["d1", "d2"],
        )
        assert result.precision == 0.0


class TestRecall:
    """Recall = relevant_retrieved / total_relevant."""

    def test_perfect_recall(self, evaluator: RetrievalEvaluator) -> None:
        """All relevant docs are retrieved."""
        result = evaluator.compute(
            retrieved_ids=["d1", "d2", "d3", "d5"],
            relevant_ids=["d1", "d2", "d3"],
        )
        assert result.recall == 1.0

    def test_partial_recall(self, evaluator: RetrievalEvaluator) -> None:
        """Some relevant docs are missing."""
        result = evaluator.compute(
            retrieved_ids=["d1", "d5"],
            relevant_ids=["d1", "d2", "d3"],
        )
        assert abs(result.recall - 1 / 3) < 1e-9

    def test_zero_recall(self, evaluator: RetrievalEvaluator) -> None:
        """No relevant docs retrieved."""
        result = evaluator.compute(
            retrieved_ids=["d5", "d6"],
            relevant_ids=["d1", "d2"],
        )
        assert result.recall == 0.0

    def test_empty_relevant(self, evaluator: RetrievalEvaluator) -> None:
        """No ground truth relevant docs."""
        result = evaluator.compute(
            retrieved_ids=["d1", "d2"],
            relevant_ids=[],
        )
        assert result.recall == 0.0


class TestF1Score:
    """F1 = 2 * (precision * recall) / (precision + recall)."""

    def test_perfect_f1(self, evaluator: RetrievalEvaluator) -> None:
        """Perfect precision and recall yield F1 = 1.0."""
        result = evaluator.compute(
            retrieved_ids=["d1", "d2", "d3"],
            relevant_ids=["d1", "d2", "d3"],
        )
        assert result.f1_score == 1.0

    def test_f1_with_half_precision_full_recall(self, evaluator: RetrievalEvaluator) -> None:
        """P=0.5, R=1.0 -> F1=2/3."""
        result = evaluator.compute(
            retrieved_ids=["d1", "d2", "d3", "d4"],
            relevant_ids=["d1", "d2"],
        )
        # P = 2/4 = 0.5, R = 2/2 = 1.0, F1 = 2*(0.5*1.0)/(0.5+1.0) = 2/3
        assert abs(result.f1_score - 2 / 3) < 1e-9

    def test_zero_f1(self, evaluator: RetrievalEvaluator) -> None:
        """No overlap yields F1 = 0."""
        result = evaluator.compute(
            retrieved_ids=["d5", "d6"],
            relevant_ids=["d1", "d2"],
        )
        assert result.f1_score == 0.0


class TestMRR:
    """Mean Reciprocal Rank = 1 / rank of first relevant result."""

    def test_mrr_first_position(self, evaluator: RetrievalEvaluator) -> None:
        """First result is relevant -> MRR = 1.0."""
        result = evaluator.compute(
            retrieved_ids=["d1", "d5", "d6"],
            relevant_ids=["d1"],
        )
        assert result.mrr == 1.0

    def test_mrr_second_position(self, evaluator: RetrievalEvaluator) -> None:
        """Second result is relevant -> MRR = 0.5."""
        result = evaluator.compute(
            retrieved_ids=["d5", "d1", "d6"],
            relevant_ids=["d1"],
        )
        assert result.mrr == 0.5

    def test_mrr_third_position(self, evaluator: RetrievalEvaluator) -> None:
        """Third result is relevant -> MRR = 1/3."""
        result = evaluator.compute(
            retrieved_ids=["d5", "d6", "d1"],
            relevant_ids=["d1"],
        )
        assert abs(result.mrr - 1 / 3) < 1e-9

    def test_mrr_no_relevant(self, evaluator: RetrievalEvaluator) -> None:
        """No relevant result -> MRR = 0."""
        result = evaluator.compute(
            retrieved_ids=["d5", "d6", "d7"],
            relevant_ids=["d1"],
        )
        assert result.mrr == 0.0


class TestNDCG:
    """Normalized Discounted Cumulative Gain."""

    def test_ndcg_perfect_ranking(self, evaluator: RetrievalEvaluator) -> None:
        """All relevant at top -> NDCG = 1.0."""
        result = evaluator.compute(
            retrieved_ids=["d1", "d2", "d5"],
            relevant_ids=["d1", "d2"],
        )
        assert result.ndcg == 1.0

    def test_ndcg_worst_ranking(self, evaluator: RetrievalEvaluator) -> None:
        """Relevant at bottom -> NDCG < 1.0."""
        result = evaluator.compute(
            retrieved_ids=["d5", "d6", "d1"],
            relevant_ids=["d1"],
        )
        # DCG = 1/log2(3+1) = 1/log2(4) = 0.5
        # IDCG = 1/log2(0+2) = 1.0
        expected = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
        assert abs(result.ndcg - expected) < 1e-9

    def test_ndcg_no_relevant(self, evaluator: RetrievalEvaluator) -> None:
        """No relevant docs -> NDCG = 0."""
        result = evaluator.compute(
            retrieved_ids=["d5", "d6"],
            relevant_ids=["d1"],
        )
        assert result.ndcg == 0.0


class TestHitRate:
    """Hit rate: 1 if any relevant doc retrieved, 0 otherwise."""

    def test_hit(self, evaluator: RetrievalEvaluator) -> None:
        result = evaluator.compute(
            retrieved_ids=["d5", "d1", "d6"],
            relevant_ids=["d1", "d2"],
        )
        assert result.hit_rate == 1.0

    def test_miss(self, evaluator: RetrievalEvaluator) -> None:
        result = evaluator.compute(
            retrieved_ids=["d5", "d6"],
            relevant_ids=["d1", "d2"],
        )
        assert result.hit_rate == 0.0


class TestBatchComputation:
    """Test batch metric computation."""

    def test_batch_returns_correct_count(self, evaluator: RetrievalEvaluator) -> None:
        batch = [
            (["d1", "d2"], ["d1", "d3"]),
            (["d1"], ["d1"]),
            (["d5"], ["d1", "d2"]),
        ]
        results = evaluator.compute_batch(batch=batch)
        assert len(results) == 3

    def test_batch_metrics_valid(self, evaluator: RetrievalEvaluator) -> None:
        batch = [
            (["d1", "d2", "d3"], ["d1", "d2"]),
        ]
        results = evaluator.compute_batch(batch=batch)
        assert results[0].precision == pytest.approx(2 / 3)
        assert results[0].recall == 1.0


class TestCounts:
    """Test count tracking."""

    def test_counts(self, evaluator: RetrievalEvaluator) -> None:
        result = evaluator.compute(
            retrieved_ids=["d1", "d2", "d3", "d4", "d5"],
            relevant_ids=["d1", "d3", "d6"],
        )
        assert result.retrieved_count == 5
        assert result.relevant_count == 3
        assert result.relevant_retrieved_count == 2
