"""Tests for generation quality metrics (faithfulness, relevance, correctness, ROUGE-L)."""

from __future__ import annotations

import pytest

from rageval.core.models import EvalMode
from rageval.metrics.generation import GenerationEvaluator


@pytest.fixture
def evaluator() -> GenerationEvaluator:
    """Evaluator without LLM (uses heuristic mode)."""
    return GenerationEvaluator()


class TestHeuristicFaithfulness:
    """Faithfulness via token overlap between answer and context."""

    def test_high_faithfulness(self, evaluator: GenerationEvaluator) -> None:
        """Answer closely matches context."""
        result = evaluator.evaluate(
            query="What is machine learning?",
            generated_answer=(
                "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn from data"
            ),
            retrieved_contexts=[
                "Machine learning is a subset of artificial intelligence. "
                "It enables systems to learn from data and improve over time."
            ],
        )
        assert result.faithfulness > 0.5

    def test_low_faithfulness(self, evaluator: GenerationEvaluator) -> None:
        """Answer has little overlap with context."""
        result = evaluator.evaluate(
            query="What is machine learning?",
            generated_answer="The weather is sunny today with clear skies",
            retrieved_contexts=["Machine learning is a branch of artificial intelligence."],
        )
        assert result.faithfulness < 0.3

    def test_empty_context(self, evaluator: GenerationEvaluator) -> None:
        """No context yields zero faithfulness."""
        result = evaluator.evaluate(
            query="What is ML?",
            generated_answer="ML is machine learning",
            retrieved_contexts=[],
        )
        assert result.faithfulness == 0.0


class TestHeuristicRelevance:
    """Relevance via token overlap between answer and query."""

    def test_relevant_answer(self, evaluator: GenerationEvaluator) -> None:
        """Answer directly addresses the query."""
        result = evaluator.evaluate(
            query="What is the capital of France?",
            generated_answer="The capital of France is Paris",
            retrieved_contexts=["Paris is the capital city of France."],
        )
        assert result.relevance > 0.3

    def test_irrelevant_answer(self, evaluator: GenerationEvaluator) -> None:
        """Answer doesn't address the query."""
        result = evaluator.evaluate(
            query="What is the capital of France?",
            generated_answer="Python is a programming language used for data science",
            retrieved_contexts=["Paris is the capital of France."],
        )
        assert result.relevance < 0.2


class TestHeuristicCorrectness:
    """Correctness via token overlap with reference answer."""

    def test_correct_answer(self, evaluator: GenerationEvaluator) -> None:
        """Answer matches reference."""
        result = evaluator.evaluate(
            query="What is 2+2?",
            generated_answer="The answer is four",
            retrieved_contexts=["Basic arithmetic."],
            reference_answer="The answer is four",
        )
        assert result.correctness == 1.0

    def test_partially_correct(self, evaluator: GenerationEvaluator) -> None:
        """Answer partially matches reference."""
        result = evaluator.evaluate(
            query="What are the primary colors?",
            generated_answer="The primary colors are red, blue, and yellow",
            retrieved_contexts=["Color theory basics."],
            reference_answer="Red, blue, and yellow are the three primary colors",
        )
        assert result.correctness > 0.4

    def test_no_reference(self, evaluator: GenerationEvaluator) -> None:
        """No reference answer yields zero correctness."""
        result = evaluator.evaluate(
            query="What is ML?",
            generated_answer="ML is machine learning",
            retrieved_contexts=["ML stands for machine learning."],
            reference_answer="",
        )
        assert result.correctness == 0.0


class TestContextUtilization:
    """How well the answer uses retrieved context."""

    def test_high_utilization(self, evaluator: GenerationEvaluator) -> None:
        """Answer uses content from retrieved contexts."""
        result = evaluator.evaluate(
            query="What is RAG?",
            generated_answer=(
                "RAG combines retrieval with generation to produce "
                "answers grounded in external knowledge sources"
            ),
            retrieved_contexts=[
                "RAG combines retrieval with generation",
                "It produces answers grounded in external knowledge sources",
            ],
        )
        assert result.context_utilization > 0.5

    def test_low_utilization(self, evaluator: GenerationEvaluator) -> None:
        """Answer doesn't use the retrieved context."""
        result = evaluator.evaluate(
            query="What is RAG?",
            generated_answer="The sky is blue and water is wet",
            retrieved_contexts=[
                "RAG is a technique for grounding LLM outputs.",
                "Retrieval augmented generation improves factuality.",
            ],
        )
        assert result.context_utilization < 0.3


class TestBatchEvaluation:
    """Test batch evaluation."""

    def test_batch_returns_correct_count(self, evaluator: GenerationEvaluator) -> None:
        samples = [
            {
                "query": "Q1",
                "generated_answer": "A1",
                "retrieved_contexts": ["C1"],
                "reference_answer": "R1",
            },
            {
                "query": "Q2",
                "generated_answer": "A2",
                "retrieved_contexts": ["C2"],
                "reference_answer": "R2",
            },
        ]
        results = evaluator.evaluate_batch(samples=samples)
        assert len(results) == 2

    def test_batch_all_metrics_present(self, evaluator: GenerationEvaluator) -> None:
        samples = [
            {
                "query": "What is AI?",
                "generated_answer": "AI is artificial intelligence",
                "retrieved_contexts": ["Artificial intelligence overview."],
                "reference_answer": "AI stands for artificial intelligence",
            },
        ]
        results = evaluator.evaluate_batch(samples=samples)
        assert results[0].faithfulness >= 0.0
        assert results[0].relevance >= 0.0
        assert results[0].correctness >= 0.0
        assert results[0].context_utilization >= 0.0


class TestRougeL:
    """ROUGE-L integration in generation metrics."""

    def test_rouge_l_with_reference(self, evaluator: GenerationEvaluator) -> None:
        """ROUGE-L should be computed when reference is provided."""
        result = evaluator.evaluate(
            query="What is RAG?",
            generated_answer="RAG combines retrieval with generation for better answers",
            retrieved_contexts=["RAG is a technique."],
            reference_answer="RAG combines retrieval with generation",
        )
        assert result.rouge_l > 0.5

    def test_rouge_l_without_reference(self, evaluator: GenerationEvaluator) -> None:
        """ROUGE-L should be 0 when no reference is provided."""
        result = evaluator.evaluate(
            query="What is RAG?",
            generated_answer="RAG combines retrieval with generation",
            retrieved_contexts=["RAG is a technique."],
        )
        assert result.rouge_l == 0.0

    def test_rouge_l_identical(self, evaluator: GenerationEvaluator) -> None:
        """Identical answer and reference should give ROUGE-L = 1.0."""
        result = evaluator.evaluate(
            query="What is 2+2?",
            generated_answer="The answer is four",
            retrieved_contexts=["Basic arithmetic."],
            reference_answer="The answer is four",
        )
        assert result.rouge_l == 1.0


class TestEvalMode:
    """Test evaluation mode tracking."""

    def test_heuristic_mode_tracked(self, evaluator: GenerationEvaluator) -> None:
        """Heuristic evaluator should report its mode."""
        result = evaluator.evaluate(
            query="What is RAG?",
            generated_answer="RAG is a technique",
            retrieved_contexts=["RAG is a technique."],
        )
        assert result.eval_mode == EvalMode.HEURISTIC

    def test_mode_property(self, evaluator: GenerationEvaluator) -> None:
        """Mode property should reflect configuration."""
        assert evaluator.mode == EvalMode.HEURISTIC

    def test_llm_mode_when_configured(self) -> None:
        """Evaluator with API key should report LLM mode."""
        llm_eval = GenerationEvaluator(api_key="test-key-123")
        assert llm_eval.mode == EvalMode.LLM_JUDGE


class TestEdgeCases:
    """Edge cases for generation evaluation."""

    def test_empty_answer(self, evaluator: GenerationEvaluator) -> None:
        result = evaluator.evaluate(
            query="Test query",
            generated_answer="",
            retrieved_contexts=["Some context."],
        )
        assert result.faithfulness == 0.0
        assert result.relevance == 0.0
        assert result.rouge_l == 0.0

    def test_empty_query(self, evaluator: GenerationEvaluator) -> None:
        result = evaluator.evaluate(
            query="",
            generated_answer="Some answer",
            retrieved_contexts=["Context."],
        )
        assert result.relevance == 0.0
