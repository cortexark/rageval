"""End-to-end evaluation pipeline runner.

Orchestrates retrieval evaluation, generation evaluation, result
storage, and summary reporting in a single pipeline call.
"""

from __future__ import annotations

import time

import structlog

from rageval.core.config import EvalConfig
from rageval.core.models import (
    EvalResult,
    EvalRunSummary,
    EvalSample,
    EvalStatus,
    GenerationMetrics,
)
from rageval.metrics.generation import GenerationEvaluator
from rageval.metrics.retrieval import RetrievalEvaluator
from rageval.pipeline.storage import ResultStore

logger = structlog.get_logger(__name__)


class EvalRunner:
    """Run end-to-end RAG evaluation pipelines.

    Coordinates retrieval metrics, generation metrics, storage, and
    comparison across evaluation runs.

    Example::

        config = EvalConfig(name="v1-eval")
        runner = EvalRunner(config)

        results = runner.run_assessment(
            samples=samples,
            retrieved_results=retrieved_per_sample,
            generated_answers=answers_per_sample,
        )
        summary = runner.get_summary()
        print(f"Avg F1: {summary.avg_f1:.3f}")
        print(f"Avg Faithfulness: {summary.avg_faithfulness:.3f}")
    """

    def __init__(self, config: EvalConfig, *, store: ResultStore | None = None) -> None:
        self._config = config
        self._run_id = config.name
        self._retrieval_eval = RetrievalEvaluator()
        self._generation_eval = GenerationEvaluator(
            judge_model=config.judge_model,
            api_key=config.llm.api_key,
        )
        self._store = store or ResultStore(db_path=config.db_path)

    def run_assessment(
        self,
        *,
        samples: list[EvalSample],
        retrieved_results: list[list[str]],
        generated_answers: list[str],
        retrieved_contexts: list[list[str]] | None = None,
    ) -> list[EvalResult]:
        """Run full assessment pipeline.

        Args:
            samples: Evaluation samples with queries and ground truth.
            retrieved_results: Retrieved doc IDs per sample.
            generated_answers: Generated answers per sample.
            retrieved_contexts: Actual text contexts per sample (for generation eval).

        Returns:
            List of EvalResult with all metrics computed.
        """
        start = time.monotonic()
        results: list[EvalResult] = []

        for i, sample in enumerate(samples):
            sample_start = time.monotonic()

            # Retrieval metrics
            retrieved_ids = retrieved_results[i] if i < len(retrieved_results) else []
            retrieval_metrics = self._retrieval_eval.compute(
                retrieved_ids=retrieved_ids,
                relevant_ids=sample.reference_contexts,
            )

            # Generation metrics
            answer = generated_answers[i] if i < len(generated_answers) else ""
            contexts = (
                retrieved_contexts[i] if retrieved_contexts and i < len(retrieved_contexts) else []
            )
            generation_metrics = self._generation_eval.evaluate(
                query=sample.query,
                generated_answer=answer,
                retrieved_contexts=contexts,
                reference_answer=sample.reference_answer,
            )

            latency = (time.monotonic() - sample_start) * 1000

            result = EvalResult(
                sample=sample,
                retrieved_doc_ids=retrieved_ids,
                retrieved_contexts=contexts,
                generated_answer=answer,
                retrieval_metrics=retrieval_metrics,
                generation_metrics=generation_metrics,
                latency_ms=latency,
            )
            results.append(result)

        # Store results
        self._store.store_results(self._run_id, results)

        duration = time.monotonic() - start
        logger.info(
            "runner.evaluation_complete",
            run_id=self._run_id,
            samples=len(results),
            duration_s=f"{duration:.2f}",
        )

        return results

    def run_retrieval_only(
        self,
        *,
        samples: list[EvalSample],
        retrieved_results: list[list[str]],
    ) -> list[EvalResult]:
        """Run retrieval-only assessment (no generation metrics).

        Useful for assessing retriever changes independently.
        """
        results: list[EvalResult] = []

        for i, sample in enumerate(samples):
            retrieved_ids = retrieved_results[i] if i < len(retrieved_results) else []
            retrieval_metrics = self._retrieval_eval.compute(
                retrieved_ids=retrieved_ids,
                relevant_ids=sample.reference_contexts,
            )

            result = EvalResult(
                sample=sample,
                retrieved_doc_ids=retrieved_ids,
                retrieval_metrics=retrieval_metrics,
                generation_metrics=GenerationMetrics(),
            )
            results.append(result)

        self._store.store_results(self._run_id, results)
        return results

    def get_summary(self) -> EvalRunSummary:
        """Get aggregate metrics for the current run.

        Returns:
            EvalRunSummary with averages.
        """
        return self._store.get_run_summary(self._run_id)

    def compare_with(self, other_run_id: str) -> dict[str, dict[str, float]]:
        """Compare current run with another run.

        Args:
            other_run_id: Run ID to compare against (baseline).

        Returns:
            Comparison dict with delta for each metric.
        """
        return self._store.compare_runs(other_run_id, self._run_id)

    @property
    def store(self) -> ResultStore:
        """Access the underlying result store."""
        return self._store

    def close(self) -> None:
        """Clean up resources."""
        self._store.close()


class RegressionChecker:
    """Check for regression between two evaluation runs.

    Example::

        checker = RegressionChecker(threshold=0.05)
        has_regression = checker.check(store, "baseline-run", "candidate-run")
    """

    def __init__(self, *, threshold: float = 0.05) -> None:
        """Initialize regression checker.

        Args:
            threshold: Maximum allowed drop in any metric.
        """
        self._threshold = threshold

    def check(
        self,
        store: ResultStore,
        baseline_run: str,
        candidate_run: str,
    ) -> tuple[bool, dict[str, dict[str, float]]]:
        """Check if candidate run regresses relative to baseline.

        Args:
            store: ResultStore with both runs.
            baseline_run: Baseline run ID.
            candidate_run: Candidate run ID.

        Returns:
            Tuple of (has_regression, comparison_dict).
        """
        comparison = store.compare_runs(baseline_run, candidate_run)

        has_regression = False
        for metric, values in comparison.items():
            if values["delta"] < -self._threshold:
                has_regression = True
                logger.warning(
                    "regression.detected",
                    metric=metric,
                    delta=f"{values['delta']:+.4f}",
                    baseline=f"{values['baseline']:.4f}",
                    candidate=f"{values['candidate']:.4f}",
                )

        status = EvalStatus.FAILED if has_regression else EvalStatus.COMPLETED
        logger.info("regression.check_complete", status=status, has_regression=has_regression)

        return has_regression, comparison
