"""Tests for the evaluation pipeline runner."""

from __future__ import annotations

import pytest

from rageval.core.config import EvalConfig
from rageval.core.models import EvalSample, EvalStatus
from rageval.pipeline.runner import EvalRunner, RegressionChecker
from rageval.pipeline.storage import ResultStore


@pytest.fixture
def config() -> EvalConfig:
    return EvalConfig(name="test-run", db_path=":memory:")


@pytest.fixture
def runner(config: EvalConfig) -> EvalRunner:
    r = EvalRunner(config)
    yield r
    r.close()


def _make_samples(count: int) -> list[EvalSample]:
    return [
        EvalSample(
            id=f"sample-{i}",
            query=f"What is concept {i}?",
            reference_answer=f"Concept {i} is a key idea in the field",
            reference_contexts=[f"doc-{i}-a", f"doc-{i}-b"],
        )
        for i in range(count)
    ]


class TestEvalRunner:
    """Test full pipeline evaluation."""

    def test_evaluate_single_sample(self, runner: EvalRunner) -> None:
        samples = _make_samples(1)
        results = runner.run_assessment(
            samples=samples,
            retrieved_results=[["doc-0-a", "doc-0-c", "doc-0-b"]],
            generated_answers=["Concept 0 is a key idea in the field of study"],
            retrieved_contexts=[["Concept 0 is a fundamental idea."]],
        )
        assert len(results) == 1
        assert results[0].retrieval_metrics.precision > 0
        assert results[0].retrieval_metrics.recall > 0
        assert results[0].retrieval_metrics.f1_score > 0

    def test_evaluate_batch(self, runner: EvalRunner) -> None:
        samples = _make_samples(5)
        retrieved = [[f"doc-{i}-a", f"doc-{i}-c"] for i in range(5)]
        answers = [f"Answer for concept {i}" for i in range(5)]
        contexts = [[f"Context about concept {i}"] for i in range(5)]

        results = runner.run_assessment(
            samples=samples,
            retrieved_results=retrieved,
            generated_answers=answers,
            retrieved_contexts=contexts,
        )
        assert len(results) == 5

    def test_evaluate_stores_results(self, runner: EvalRunner) -> None:
        samples = _make_samples(3)
        retrieved = [[f"doc-{i}-a"] for i in range(3)]
        answers = [f"Answer {i}" for i in range(3)]

        runner.run_assessment(
            samples=samples,
            retrieved_results=retrieved,
            generated_answers=answers,
        )

        summary = runner.get_summary()
        assert summary.sample_count == 3
        assert summary.status == EvalStatus.COMPLETED

    def test_retrieval_only_evaluation(self, runner: EvalRunner) -> None:
        samples = _make_samples(2)
        retrieved = [["doc-0-a", "doc-0-b"], ["doc-1-a"]]

        results = runner.run_retrieval_only(
            samples=samples,
            retrieved_results=retrieved,
        )
        assert len(results) == 2
        # First sample: perfect retrieval
        assert results[0].retrieval_metrics.recall == 1.0
        # Generation metrics should be zeros
        assert results[0].generation_metrics.faithfulness == 0.0

    def test_get_summary_aggregates(self, runner: EvalRunner) -> None:
        samples = _make_samples(4)
        retrieved = [
            ["doc-0-a", "doc-0-b"],  # Perfect
            ["doc-1-a"],  # 50% recall
            ["doc-2-a", "doc-2-b"],  # Perfect
            ["doc-3-c"],  # 0% recall
        ]
        answers = ["A0", "A1", "A2", "A3"]

        runner.run_assessment(
            samples=samples,
            retrieved_results=retrieved,
            generated_answers=answers,
        )
        summary = runner.get_summary()
        assert summary.sample_count == 4
        assert 0 < summary.avg_recall < 1

    def test_empty_samples(self, runner: EvalRunner) -> None:
        results = runner.run_assessment(
            samples=[],
            retrieved_results=[],
            generated_answers=[],
        )
        assert len(results) == 0


class TestRegressionChecker:
    """Test regression detection between runs."""

    def test_no_regression(self) -> None:
        store = ResultStore(db_path=":memory:")
        samples = _make_samples(3)

        config_a = EvalConfig(name="baseline", db_path=":memory:")
        runner_a = EvalRunner(config_a, store=store)
        runner_a.run_assessment(
            samples=samples,
            retrieved_results=[["doc-0-a", "doc-0-b"] for _ in range(3)],
            generated_answers=["Answer" for _ in range(3)],
        )

        # Candidate with same or better metrics
        config_b = EvalConfig(name="candidate", db_path=":memory:")
        runner_b = EvalRunner(config_b, store=store)
        runner_b.run_assessment(
            samples=samples,
            retrieved_results=[["doc-0-a", "doc-0-b"] for _ in range(3)],
            generated_answers=["Answer" for _ in range(3)],
        )

        checker = RegressionChecker(threshold=0.05)
        has_regression, _ = checker.check(store, "baseline", "candidate")
        assert has_regression is False

        store.close()

    def test_detects_regression(self) -> None:
        store = ResultStore(db_path=":memory:")
        samples = _make_samples(3)

        # Baseline with good retrieval
        config_a = EvalConfig(name="baseline", db_path=":memory:")
        runner_a = EvalRunner(config_a, store=store)
        runner_a.run_assessment(
            samples=samples,
            retrieved_results=[["doc-0-a", "doc-0-b"] for _ in range(3)],
            generated_answers=["Good answer" for _ in range(3)],
        )

        # Candidate with bad retrieval (wrong docs)
        config_b = EvalConfig(name="candidate", db_path=":memory:")
        runner_b = EvalRunner(config_b, store=store)
        runner_b.run_assessment(
            samples=samples,
            retrieved_results=[["wrong-doc"] for _ in range(3)],
            generated_answers=["Bad answer" for _ in range(3)],
        )

        checker = RegressionChecker(threshold=0.05)
        has_regression, comparison = checker.check(store, "baseline", "candidate")
        assert has_regression is True
        assert comparison["avg_recall"]["delta"] < 0

        store.close()


class TestRunComparison:
    """Test comparing evaluation runs."""

    def test_compare_with_method(self) -> None:
        store = ResultStore(db_path=":memory:")
        samples = _make_samples(2)

        config_v1 = EvalConfig(name="run-v1", db_path=":memory:")
        runner_v1 = EvalRunner(config_v1, store=store)
        runner_v1.run_assessment(
            samples=samples,
            retrieved_results=[["doc-0-a"], ["doc-1-a"]],
            generated_answers=["A1", "A2"],
        )

        config_v2 = EvalConfig(name="run-v2", db_path=":memory:")
        runner_v2 = EvalRunner(config_v2, store=store)
        runner_v2.run_assessment(
            samples=samples,
            retrieved_results=[["doc-0-a", "doc-0-b"], ["doc-1-a", "doc-1-b"]],
            generated_answers=["Better A1", "Better A2"],
        )

        comparison = runner_v2.compare_with("run-v1")
        # v2 should have better recall (retrieved both docs)
        assert comparison["avg_recall"]["delta"] > 0

        store.close()
