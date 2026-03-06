"""End-to-end tests simulating real customer use cases.

Each test class represents a complete RAG assessment workflow from start
to finish, exercising the full rageval stack.

Customer Scenarios Covered:
1. Customer Support Bot Assessment -- measure Q&A accuracy and retrieval
2. Legal Document RAG -- domain-specific retrieval precision
3. CI/CD Regression Guard -- block deploys when retrieval quality drops
4. Multi-Retriever Comparison -- compare BM25 vs. dense vs. hybrid
5. Version Tracking -- track quality over multiple index versions
6. Batch Pipeline -- nightly assessment of 500+ samples
"""

from __future__ import annotations

import pytest

from rageval.core.config import EvalConfig
from rageval.core.models import EvalResult, EvalSample, EvalStatus, GenerationMetrics
from rageval.metrics.retrieval import RetrievalEvaluator
from rageval.pipeline.runner import EvalRunner, RegressionChecker
from rageval.pipeline.storage import ResultStore
from rageval.retrieval.evaluator import RetrieverHarness

# ===========================================================================
# Helpers
# ===========================================================================


def _make_samples(
    count: int,
    *,
    prefix: str = "sample",
    relevant_per_sample: int = 3,
) -> list[EvalSample]:
    """Generate assessment samples with ground truth."""
    return [
        EvalSample(
            id=f"{prefix}-{i}",
            query=f"What is topic {i} about?",
            reference_answer=f"Topic {i} covers key concepts in the field",
            reference_contexts=[f"{prefix}-doc-{i}-{j}" for j in range(relevant_per_sample)],
        )
        for i in range(count)
    ]


# ===========================================================================
# Scenario 1: Customer Support Bot Assessment
# ===========================================================================


class TestCustomerSupportBotAssessment:
    """Measure a customer support RAG bot's Q&A quality.

    Simulates: A team deploying a support bot that answers product questions.
    They want to measure retrieval accuracy and answer quality before launch.
    """

    def test_full_support_bot_assessment(self) -> None:
        """Complete assessment: index docs, query, measure metrics, report."""
        # Step 1: Define assessment dataset
        samples = [
            EvalSample(
                id="support-1",
                query="How do I reset my password?",
                reference_answer="Go to Settings > Security > Reset Password",
                reference_contexts=["doc-password-reset", "doc-security-settings"],
            ),
            EvalSample(
                id="support-2",
                query="What are the pricing tiers?",
                reference_answer="We offer Free, Pro ($10/mo), and Enterprise plans",
                reference_contexts=["doc-pricing", "doc-plans-comparison"],
            ),
            EvalSample(
                id="support-3",
                query="How do I export my data?",
                reference_answer="Use Settings > Data > Export to download a CSV",
                reference_contexts=["doc-data-export", "doc-csv-format"],
            ),
        ]

        # Step 2: Simulate retrieval results (what the bot retrieved)
        retrieved = [
            ["doc-password-reset", "doc-account-help", "doc-security-settings"],
            ["doc-pricing", "doc-billing-faq"],
            ["doc-data-export", "doc-api-docs", "doc-csv-format"],
        ]

        # Step 3: Simulate generated answers
        answers = [
            "To reset your password, go to Settings then Security and click Reset",
            "We have Free and Pro plans at $10 per month",
            "Go to Settings, then Data, and click Export for a CSV download",
        ]

        contexts = [
            ["To reset your password, navigate to Settings > Security."],
            ["Pricing: Free tier, Pro at $10/month, Enterprise custom."],
            ["Export data via Settings > Data > Export button."],
        ]

        # Step 4: Run assessment
        config = EvalConfig(name="support-bot-v1", db_path=":memory:")
        runner = EvalRunner(config)

        results = runner.run_assessment(
            samples=samples,
            retrieved_results=retrieved,
            generated_answers=answers,
            retrieved_contexts=contexts,
        )

        # Step 5: Verify results
        assert len(results) == 3

        # Sample 1: retrieved 2/2 relevant (perfect recall)
        assert results[0].retrieval_metrics.recall == 1.0
        assert results[0].retrieval_metrics.precision == pytest.approx(2 / 3)

        # All samples should have F1 > 0
        for r in results:
            assert r.retrieval_metrics.f1_score > 0

        # Step 6: Check summary
        summary = runner.get_summary()
        assert summary.sample_count == 3
        assert summary.avg_recall > 0.5
        assert summary.avg_precision > 0.4
        assert summary.status == EvalStatus.COMPLETED

        runner.close()


# ===========================================================================
# Scenario 2: Legal Document RAG Precision
# ===========================================================================


class TestLegalDocumentRAG:
    """Measure retrieval for a legal document search system.

    Simulates: A law firm using RAG to search case law. Precision is critical
    because citing irrelevant cases is worse than missing some.
    """

    def test_high_precision_retrieval(self) -> None:
        """Legal RAG needs high precision -- no irrelevant citations."""
        samples = [
            EvalSample(
                id="legal-1",
                query="Cases involving breach of contract in California",
                reference_contexts=["case-001", "case-015", "case-042"],
            ),
            EvalSample(
                id="legal-2",
                query="Precedent for intellectual property disputes",
                reference_contexts=["case-100", "case-105"],
            ),
        ]

        # High precision retrieval (all retrieved are relevant)
        retrieved = [
            ["case-001", "case-042"],  # 2/2 precise, recall=2/3
            ["case-100"],  # 1/1 precise, recall=1/2
        ]

        retrieval_scorer = RetrievalEvaluator()
        for i, sample in enumerate(samples):
            metrics = retrieval_scorer.compute(
                retrieved_ids=retrieved[i],
                relevant_ids=sample.reference_contexts,
            )
            # All retrieved docs should be relevant (precision = 1.0)
            assert metrics.precision == 1.0

    def test_precision_recall_tradeoff(self) -> None:
        """Increasing top_k hurts precision but helps recall."""
        sample = EvalSample(
            id="legal-tradeoff",
            query="Environmental regulation cases",
            reference_contexts=["case-200", "case-205", "case-210"],
        )

        retrieval_scorer = RetrievalEvaluator()

        # top_k=2: high precision, low recall
        top2 = retrieval_scorer.compute(
            retrieved_ids=["case-200", "case-205"],
            relevant_ids=sample.reference_contexts,
        )

        # top_k=10: lower precision, higher recall
        top10 = retrieval_scorer.compute(
            retrieved_ids=[
                "case-200",
                "case-205",
                "case-210",
                "case-300",
                "case-301",
                "case-302",
                "case-303",
                "case-304",
                "case-305",
                "case-306",
            ],
            relevant_ids=sample.reference_contexts,
        )

        assert top2.precision > top10.precision
        assert top2.recall < top10.recall
        assert top10.recall == 1.0  # All 3 relevant found


# ===========================================================================
# Scenario 3: CI/CD Regression Guard
# ===========================================================================


class TestCICDRegressionGuard:
    """Block deployment when RAG quality drops below threshold.

    Simulates: A CI pipeline that assesses a new retriever version
    and blocks deployment if retrieval F1 regresses.
    """

    def test_ci_gate_pass(self) -> None:
        """Candidate retriever is as good or better -- deploy allowed."""
        # Use a shared store for both runs (in-memory DBs are per-connection)
        store = ResultStore(db_path=":memory:")
        samples = _make_samples(10)

        # Baseline: moderate retrieval (2 of 3 relevant docs)
        retrieval_scorer = RetrievalEvaluator()
        baseline_results = []
        for i, sample in enumerate(samples):
            retrieved = [f"sample-doc-{i}-0", f"sample-doc-{i}-1"]
            metrics = retrieval_scorer.compute(
                retrieved_ids=retrieved, relevant_ids=sample.reference_contexts
            )
            baseline_results.append(
                EvalResult(
                    sample=sample,
                    retrieved_doc_ids=retrieved,
                    retrieval_metrics=metrics,
                    generation_metrics=GenerationMetrics(),
                )
            )
        store.store_results("baseline-v3", baseline_results)

        # Candidate: better (all 3 relevant docs)
        candidate_results = []
        for i, sample in enumerate(samples):
            retrieved = [f"sample-doc-{i}-0", f"sample-doc-{i}-1", f"sample-doc-{i}-2"]
            metrics = retrieval_scorer.compute(
                retrieved_ids=retrieved, relevant_ids=sample.reference_contexts
            )
            candidate_results.append(
                EvalResult(
                    sample=sample,
                    retrieved_doc_ids=retrieved,
                    retrieval_metrics=metrics,
                    generation_metrics=GenerationMetrics(),
                )
            )
        store.store_results("candidate-v4", candidate_results)

        checker = RegressionChecker(threshold=0.05)
        has_regression, _ = checker.check(store, "baseline-v3", "candidate-v4")
        assert has_regression is False  # CI gate passes

        store.close()

    def test_ci_gate_block(self) -> None:
        """Candidate retriever is worse -- deployment blocked."""
        store = ResultStore(db_path=":memory:")
        samples = _make_samples(10)

        # Good baseline
        config = EvalConfig(name="good-baseline", db_path=":memory:")
        runner = EvalRunner(config, store=store)
        runner.run_assessment(
            samples=samples,
            retrieved_results=[
                [f"sample-doc-{i}-0", f"sample-doc-{i}-1", f"sample-doc-{i}-2"] for i in range(10)
            ],
            generated_answers=[f"A{i}" for i in range(10)],
        )

        # Bad candidate (shared store)
        config_b = EvalConfig(name="bad-candidate", db_path=":memory:")
        runner_b = EvalRunner(config_b, store=store)
        runner_b.run_assessment(
            samples=samples,
            retrieved_results=[["wrong-doc"] for _ in range(10)],
            generated_answers=[f"A{i}" for i in range(10)],
        )

        checker = RegressionChecker(threshold=0.05)
        has_regression, comparison = checker.check(store, "good-baseline", "bad-candidate")
        assert has_regression is True
        assert comparison["avg_recall"]["delta"] < -0.5

        store.close()


# ===========================================================================
# Scenario 4: Multi-Retriever Comparison
# ===========================================================================


class TestMultiRetrieverComparison:
    """Compare BM25 vs. dense vs. hybrid retrieval strategies.

    Simulates: An ML engineer assessing three retriever configurations
    to pick the best one for their use case.
    """

    def test_three_way_retriever_comparison(self) -> None:
        """Compare BM25, dense, and hybrid retrievers."""
        samples = _make_samples(5, relevant_per_sample=2)

        # BM25: good for keyword matches
        bm25_retrieved = [
            ["sample-doc-0-0", "irrelevant-1", "sample-doc-0-1"],
            ["sample-doc-1-0"],
            ["sample-doc-2-0", "sample-doc-2-1"],
            ["irrelevant-2", "irrelevant-3"],
            ["sample-doc-4-0"],
        ]

        # Dense: good for semantic matches
        dense_retrieved = [
            ["sample-doc-0-0", "sample-doc-0-1"],
            ["sample-doc-1-0", "sample-doc-1-1"],
            ["sample-doc-2-0"],
            ["sample-doc-3-0", "sample-doc-3-1"],
            ["irrelevant-4"],
        ]

        # Hybrid: best of both
        hybrid_retrieved = [
            ["sample-doc-0-0", "sample-doc-0-1"],
            ["sample-doc-1-0", "sample-doc-1-1"],
            ["sample-doc-2-0", "sample-doc-2-1"],
            ["sample-doc-3-0", "sample-doc-3-1"],
            ["sample-doc-4-0", "sample-doc-4-1"],
        ]

        harness = RetrieverHarness()

        _, bm25_agg = harness.run_with_metrics(samples, retrieved_results=bm25_retrieved)
        _, dense_agg = harness.run_with_metrics(samples, retrieved_results=dense_retrieved)
        _, hybrid_agg = harness.run_with_metrics(samples, retrieved_results=hybrid_retrieved)

        # Hybrid should have best overall metrics
        assert hybrid_agg.f1_score >= bm25_agg.f1_score
        assert hybrid_agg.f1_score >= dense_agg.f1_score
        assert hybrid_agg.recall == 1.0  # Perfect recall


# ===========================================================================
# Scenario 5: Version History Tracking
# ===========================================================================


class TestVersionHistoryTracking:
    """Track retrieval quality across multiple index versions.

    Simulates: An engineering team monitoring RAG quality over weekly
    index rebuilds, detecting gradual degradation.
    """

    def test_track_five_versions(self) -> None:
        """Track metrics across 5 index versions."""
        store = ResultStore(db_path=":memory:")
        retrieval_scorer = RetrievalEvaluator()
        samples = _make_samples(10, relevant_per_sample=2)

        versions = ["v1.0", "v1.1", "v1.2", "v1.3", "v1.4"]
        # Simulate improving then degrading quality
        recall_targets = [0.5, 0.7, 0.8, 0.6, 0.4]

        for version, target_recall in zip(versions, recall_targets):
            results = []
            for i, sample in enumerate(samples):
                relevant = sample.reference_contexts
                # Simulate retrieval quality
                if target_recall >= 0.8:
                    retrieved = [*relevant, f"extra-{i}"]
                elif target_recall >= 0.6:
                    retrieved = [relevant[0], f"extra-{i}"]
                else:
                    retrieved = [f"extra-{i}", f"extra-{i}-2"]

                metrics = retrieval_scorer.compute(
                    retrieved_ids=retrieved,
                    relevant_ids=relevant,
                )

                results.append(
                    EvalResult(
                        sample=sample,
                        retrieved_doc_ids=retrieved,
                        retrieval_metrics=metrics,
                        generation_metrics=GenerationMetrics(),
                    )
                )

            store.store_results(version, results)

        # Verify we can query each version
        for version in versions:
            summary = store.get_run_summary(version)
            assert summary.sample_count == 10
            assert summary.status == EvalStatus.COMPLETED

        # Compare first vs last version
        comparison = store.compare_runs("v1.0", "v1.4")
        # v1.4 should be worse than v1.0 (recall dropped from 0.5 to 0.4)
        assert comparison["avg_recall"]["delta"] <= 0

        store.close()


# ===========================================================================
# Scenario 6: Batch Pipeline (500 samples)
# ===========================================================================


class TestBatchAssessmentPipeline:
    """Nightly batch assessment of 500+ samples.

    Simulates: A production team running nightly assessments on their
    RAG pipeline to catch quality regressions early.
    """

    def test_batch_500_samples(self) -> None:
        """Assess 500 samples in a single pipeline run."""
        config = EvalConfig(name="nightly-2024-03-05", db_path=":memory:")
        runner = EvalRunner(config)

        samples = _make_samples(500, relevant_per_sample=3)

        # Simulate retrieval: mix of good and bad results
        retrieved = []
        answers = []
        for i in range(500):
            if i % 5 == 0:
                # Every 5th sample: bad retrieval
                retrieved.append([f"wrong-doc-{i}"])
                answers.append("I don't know")
            else:
                # Good retrieval
                retrieved.append(
                    [
                        f"sample-doc-{i}-0",
                        f"sample-doc-{i}-1",
                        f"sample-doc-{i}-2",
                    ]
                )
                answers.append(f"Topic {i} covers key concepts")

        results = runner.run_assessment(
            samples=samples,
            retrieved_results=retrieved,
            generated_answers=answers,
        )

        assert len(results) == 500

        summary = runner.get_summary()
        assert summary.sample_count == 500
        assert summary.status == EvalStatus.COMPLETED
        # 80% of samples have perfect retrieval
        assert summary.avg_recall > 0.7
        assert summary.avg_f1 > 0.6

        # Query low-performing samples
        low_f1 = runner.store.query_results("nightly-2024-03-05", max_f1=0.1)
        # Should be ~100 samples (every 5th)
        assert len(low_f1) == 100

        runner.close()


# ===========================================================================
# Scenario 7: Retriever Harness with Pre-computed Results
# ===========================================================================


class TestRetrieverHarnessScenarios:
    """Test the retriever harness with pre-computed results."""

    def test_offline_assessment(self) -> None:
        """Assess pre-computed retrieval results without a live retriever."""
        harness = RetrieverHarness()
        samples = _make_samples(3, relevant_per_sample=2)

        retrieved = [
            ["sample-doc-0-0", "sample-doc-0-1"],
            ["sample-doc-1-0"],
            ["wrong-doc"],
        ]

        results = harness.run_samples(samples, retrieved_results=retrieved)

        assert len(results) == 3
        assert results[0].retrieval_metrics.recall == 1.0
        assert results[1].retrieval_metrics.recall == 0.5
        assert results[2].retrieval_metrics.recall == 0.0

    def test_aggregate_metrics(self) -> None:
        """Verify aggregate metrics across all samples."""
        harness = RetrieverHarness()
        samples = _make_samples(4, relevant_per_sample=2)

        retrieved = [
            ["sample-doc-0-0", "sample-doc-0-1"],  # Perfect
            ["sample-doc-1-0", "sample-doc-1-1"],  # Perfect
            ["sample-doc-2-0"],  # 50% recall
            ["wrong"],  # 0% recall
        ]

        _, aggregate = harness.run_with_metrics(samples, retrieved_results=retrieved)

        # Average recall: (1 + 1 + 0.5 + 0) / 4 = 0.625
        assert aggregate.recall == pytest.approx(0.625)
        assert aggregate.hit_rate == pytest.approx(0.75)
