"""Tests for DuckDB-backed result storage."""

from __future__ import annotations

import pytest

from rageval.core.models import (
    EvalResult,
    EvalSample,
    EvalStatus,
    GenerationMetrics,
    RetrievalMetrics,
)
from rageval.pipeline.storage import ResultStore


@pytest.fixture
def store() -> ResultStore:
    s = ResultStore(db_path=":memory:")
    yield s
    s.close()


def _make_result(
    sample_id: str,
    query: str = "test query",
    *,
    precision: float = 0.5,
    recall: float = 0.5,
    f1: float = 0.5,
    faithfulness: float = 0.5,
) -> EvalResult:
    """Create a test EvalResult."""
    return EvalResult(
        sample=EvalSample(
            id=sample_id,
            query=query,
            reference_answer="ref answer",
        ),
        retrieval_metrics=RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
        ),
        generation_metrics=GenerationMetrics(faithfulness=faithfulness),
        generated_answer="generated answer",
    )


class TestStoreResults:
    """Test storing and retrieving results."""

    def test_store_single_result(self, store: ResultStore) -> None:
        result = _make_result("s1")
        count = store.store_results("run-1", [result])
        assert count == 1

    def test_store_multiple_results(self, store: ResultStore) -> None:
        results = [_make_result(f"s{i}") for i in range(10)]
        count = store.store_results("run-1", results)
        assert count == 10

    def test_store_empty_list(self, store: ResultStore) -> None:
        count = store.store_results("run-1", [])
        assert count == 0

    def test_upsert_on_duplicate_id(self, store: ResultStore) -> None:
        """Same result ID should be updated, not duplicated."""
        result = _make_result("s1", precision=0.5)
        store.store_results("run-1", [result])

        updated = EvalResult(
            id=result.id,
            sample=result.sample,
            retrieval_metrics=RetrievalMetrics(precision=0.9, recall=0.9, f1_score=0.9),
            generation_metrics=GenerationMetrics(faithfulness=0.9),
        )
        store.store_results("run-1", [updated])

        summary = store.get_run_summary("run-1")
        assert summary.avg_precision == pytest.approx(0.9)


class TestRunSummary:
    """Test aggregate summary computation."""

    def test_summary_averages(self, store: ResultStore) -> None:
        results = [
            _make_result("s1", precision=0.8, recall=0.6, f1=0.685),
            _make_result("s2", precision=0.6, recall=0.8, f1=0.685),
        ]
        store.store_results("run-1", results)
        summary = store.get_run_summary("run-1")

        assert summary.sample_count == 2
        assert summary.avg_precision == pytest.approx(0.7)
        assert summary.avg_recall == pytest.approx(0.7)
        assert summary.status == EvalStatus.COMPLETED

    def test_summary_nonexistent_run(self, store: ResultStore) -> None:
        summary = store.get_run_summary("nonexistent")
        assert summary.status == EvalStatus.FAILED
        assert summary.sample_count == 0


class TestCompareRuns:
    """Test run comparison."""

    def test_compare_two_runs(self, store: ResultStore) -> None:
        baseline = [
            _make_result("s1", precision=0.7, recall=0.6, f1=0.646),
        ]
        candidate = [
            _make_result("s1c", precision=0.8, recall=0.8, f1=0.8),
        ]
        store.store_results("baseline", baseline)
        store.store_results("candidate", candidate)

        comparison = store.compare_runs("baseline", "candidate")
        assert "avg_precision" in comparison
        assert comparison["avg_precision"]["delta"] == pytest.approx(0.1)
        assert comparison["avg_recall"]["delta"] == pytest.approx(0.2)

    def test_compare_detects_regression(self, store: ResultStore) -> None:
        baseline = [_make_result("s1", f1=0.8)]
        candidate = [_make_result("s2", f1=0.5)]
        store.store_results("baseline", baseline)
        store.store_results("candidate", candidate)

        comparison = store.compare_runs("baseline", "candidate")
        assert comparison["avg_f1"]["delta"] < 0  # Regression


class TestQueryResults:
    """Test filtered result queries."""

    def test_query_by_run_id(self, store: ResultStore) -> None:
        store.store_results("run-1", [_make_result("s1")])
        store.store_results("run-2", [_make_result("s2")])

        results = store.query_results("run-1")
        assert len(results) == 1

    def test_query_with_f1_filter(self, store: ResultStore) -> None:
        results = [
            _make_result("s1", f1=0.3),
            _make_result("s2", f1=0.7),
            _make_result("s3", f1=0.9),
        ]
        store.store_results("run-1", results)

        filtered = store.query_results("run-1", min_f1=0.5)
        assert len(filtered) == 2

    def test_query_with_limit(self, store: ResultStore) -> None:
        results = [_make_result(f"s{i}") for i in range(20)]
        store.store_results("run-1", results)

        limited = store.query_results("run-1", limit=5)
        assert len(limited) == 5

    def test_query_empty_run(self, store: ResultStore) -> None:
        results = store.query_results("nonexistent")
        assert len(results) == 0
