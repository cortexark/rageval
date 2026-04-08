"""Tests for core data models and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rageval.core.config import EvalConfig, LLMProviderConfig, RetrieverConfig
from rageval.core.models import (
    EvalMode,
    EvalResult,
    EvalRunSummary,
    EvalSample,
    EvalStatus,
    GenerationMetrics,
    RetrievalMetrics,
)


class TestRetrievalMetrics:
    """Test RetrievalMetrics model validation."""

    def test_default_values(self) -> None:
        metrics = RetrievalMetrics()
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.mrr == 0.0
        assert metrics.ndcg == 0.0
        assert metrics.hit_rate == 0.0

    def test_valid_ranges(self) -> None:
        metrics = RetrievalMetrics(
            precision=0.8,
            recall=0.6,
            f1_score=0.685,
            mrr=1.0,
            ndcg=0.9,
            hit_rate=1.0,
        )
        assert metrics.precision == 0.8
        assert metrics.recall == 0.6

    def test_rejects_negative_precision(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalMetrics(precision=-0.1)

    def test_rejects_precision_above_one(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalMetrics(precision=1.1)

    def test_rejects_negative_recall(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalMetrics(recall=-0.5)

    def test_counts_non_negative(self) -> None:
        metrics = RetrievalMetrics(
            retrieved_count=10,
            relevant_count=5,
            relevant_retrieved_count=3,
        )
        assert metrics.retrieved_count == 10

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalMetrics(retrieved_count=-1)


class TestGenerationMetrics:
    """Test GenerationMetrics model validation."""

    def test_default_values(self) -> None:
        metrics = GenerationMetrics()
        assert metrics.faithfulness == 0.0
        assert metrics.relevance == 0.0
        assert metrics.correctness == 0.0
        assert metrics.rouge_l == 0.0
        assert metrics.context_utilization == 0.0
        assert metrics.eval_mode == EvalMode.NONE

    def test_valid_scores(self) -> None:
        metrics = GenerationMetrics(
            faithfulness=0.9,
            relevance=0.85,
            correctness=0.7,
            rouge_l=0.8,
            context_utilization=0.6,
            eval_mode=EvalMode.HEURISTIC,
        )
        assert metrics.faithfulness == 0.9
        assert metrics.rouge_l == 0.8
        assert metrics.eval_mode == EvalMode.HEURISTIC

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            GenerationMetrics(faithfulness=1.5)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            GenerationMetrics(relevance=-0.1)

    def test_rejects_rouge_l_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            GenerationMetrics(rouge_l=1.5)


class TestEvalMode:
    """Test EvalMode enum."""

    def test_values(self) -> None:
        assert EvalMode.LLM_JUDGE == "llm_judge"
        assert EvalMode.HEURISTIC == "heuristic"
        assert EvalMode.NONE == "none"


class TestEvalSample:
    """Test EvalSample model."""

    def test_minimal_sample(self) -> None:
        sample = EvalSample(query="What is RAG?")
        assert sample.query == "What is RAG?"
        assert sample.reference_answer == ""
        assert sample.reference_contexts == []
        assert sample.id  # Auto-generated UUID

    def test_full_sample(self) -> None:
        sample = EvalSample(
            id="test-1",
            query="What is RAG?",
            reference_answer="Retrieval-Augmented Generation",
            reference_contexts=["doc1", "doc2"],
            metadata={"topic": "AI"},
        )
        assert sample.id == "test-1"
        assert len(sample.reference_contexts) == 2
        assert sample.metadata["topic"] == "AI"


class TestEvalResult:
    """Test EvalResult model."""

    def test_default_result(self) -> None:
        sample = EvalSample(query="test")
        result = EvalResult(sample=sample)
        assert result.generated_answer == ""
        assert result.retrieval_metrics.precision == 0.0
        assert result.generation_metrics.faithfulness == 0.0

    def test_full_result(self) -> None:
        sample = EvalSample(query="test", reference_contexts=["d1"])
        result = EvalResult(
            sample=sample,
            retrieved_doc_ids=["d1", "d2"],
            generated_answer="Test answer",
            retrieval_metrics=RetrievalMetrics(precision=0.5, recall=1.0, f1_score=0.667),
            generation_metrics=GenerationMetrics(faithfulness=0.8),
            latency_ms=42.5,
        )
        assert result.retrieval_metrics.f1_score == 0.667
        assert result.latency_ms == 42.5


class TestEvalRunSummary:
    """Test EvalRunSummary model."""

    def test_default_summary(self) -> None:
        summary = EvalRunSummary()
        assert summary.status == EvalStatus.PENDING
        assert summary.sample_count == 0

    def test_completed_summary(self) -> None:
        summary = EvalRunSummary(
            status=EvalStatus.COMPLETED,
            sample_count=100,
            avg_precision=0.75,
            avg_recall=0.80,
            avg_f1=0.774,
        )
        assert summary.status == EvalStatus.COMPLETED
        assert summary.avg_f1 == 0.774


class TestEvalStatus:
    """Test EvalStatus enum."""

    def test_values(self) -> None:
        assert EvalStatus.PENDING == "pending"
        assert EvalStatus.RUNNING == "running"
        assert EvalStatus.COMPLETED == "completed"
        assert EvalStatus.FAILED == "failed"


class TestEvalConfig:
    """Test EvalConfig model."""

    def test_defaults(self) -> None:
        config = EvalConfig()
        assert config.name == "rag-eval"
        assert config.db_path == ":memory:"
        assert config.batch_size == 10

    def test_custom_config(self) -> None:
        config = EvalConfig(
            name="my-eval",
            llm=LLMProviderConfig(model="gpt-4o-mini"),
            db_path="/tmp/test.db",
        )
        assert config.name == "my-eval"
        assert config.llm.model == "gpt-4o-mini"


class TestRetrieverConfig:
    """Test RetrieverConfig validation."""

    def test_defaults(self) -> None:
        config = RetrieverConfig()
        assert config.top_k == 5

    def test_rejects_zero_top_k(self) -> None:
        with pytest.raises(ValidationError):
            RetrieverConfig(top_k=0)

    def test_rejects_excessive_top_k(self) -> None:
        with pytest.raises(ValidationError):
            RetrieverConfig(top_k=200)
