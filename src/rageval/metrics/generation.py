"""Generation quality metrics using LLM-as-Judge or heuristic fallback.

Evaluates the quality of RAG-generated answers using either:
1. LLM-as-Judge (LlamaIndex evaluators) — accurate, ~2s/sample
2. Heuristic fallback (ROUGE-L + Jaccard + context overlap) — fast, free

When no LLM is configured or LlamaIndex is not installed, falls back
to heuristic mode automatically. The evaluation mode is tracked in
every GenerationMetrics result via the eval_mode field.
"""

from __future__ import annotations

import structlog

from rageval.core.models import EvalMode, GenerationMetrics
from rageval.metrics.rouge import rouge_l_score

logger = structlog.get_logger(__name__)


class GenerationEvaluator:
    """Evaluate generation quality with LLM judge or heuristic fallback.

    The evaluator automatically selects the best available mode:
    - If an API key is provided and LlamaIndex is installed: LLM-as-Judge
    - Otherwise: heuristic mode (ROUGE-L + Jaccard + context utilization)

    Every result includes an ``eval_mode`` field so you always know
    which mode produced the scores.

    Example::

        evaluator = GenerationEvaluator()
        metrics = evaluator.evaluate(
            query="What is RAG?",
            generated_answer="RAG combines retrieval with generation.",
            retrieved_contexts=["RAG is a technique..."],
            reference_answer="Retrieval-Augmented Generation...",
        )
        print(f"Faithfulness: {metrics.faithfulness:.2f}")
        print(f"ROUGE-L: {metrics.rouge_l:.2f}")
        print(f"Mode: {metrics.eval_mode}")
    """

    def __init__(self, *, judge_model: str = "gpt-4o", api_key: str = "") -> None:
        """Initialize the generation evaluator.

        Args:
            judge_model: LLM model to use as judge (when LLM mode is active).
            api_key: API key for the LLM provider. If empty, uses heuristic mode.
        """
        self._judge_model = judge_model
        self._api_key = api_key
        self._llm_available = bool(api_key)

    @property
    def mode(self) -> EvalMode:
        """Return the current evaluation mode."""
        return EvalMode.LLM_JUDGE if self._llm_available else EvalMode.HEURISTIC

    def evaluate(
        self,
        *,
        query: str,
        generated_answer: str,
        retrieved_contexts: list[str],
        reference_answer: str = "",
    ) -> GenerationMetrics:
        """Evaluate a single generated answer.

        Automatically selects LLM-as-Judge or heuristic mode based on
        configuration. Falls back to heuristic on any LLM error.

        Args:
            query: The original user query.
            generated_answer: The RAG pipeline's answer.
            retrieved_contexts: Context chunks used for generation.
            reference_answer: Ground truth answer for correctness scoring.

        Returns:
            GenerationMetrics with all quality scores and eval_mode.
        """
        if self._llm_available:
            return self._evaluate_with_llm(
                query=query,
                generated_answer=generated_answer,
                retrieved_contexts=retrieved_contexts,
                reference_answer=reference_answer,
            )
        return self._evaluate_with_heuristics(
            query=query,
            generated_answer=generated_answer,
            retrieved_contexts=retrieved_contexts,
            reference_answer=reference_answer,
        )

    def evaluate_batch(
        self,
        *,
        samples: list[dict[str, object]],
    ) -> list[GenerationMetrics]:
        """Evaluate a batch of samples.

        Args:
            samples: List of dicts with keys: query, generated_answer,
                     retrieved_contexts, reference_answer.

        Returns:
            List of GenerationMetrics.
        """
        results: list[GenerationMetrics] = []
        for sample in samples:
            raw_contexts = sample.get("retrieved_contexts")
            contexts: list[str] = (
                [str(c) for c in raw_contexts] if isinstance(raw_contexts, list) else []
            )
            metrics = self.evaluate(
                query=str(sample.get("query", "")),
                generated_answer=str(sample.get("generated_answer", "")),
                retrieved_contexts=contexts,
                reference_answer=str(sample.get("reference_answer", "")),
            )
            results.append(metrics)
        return results

    # ------------------------------------------------------------------
    # LLM-as-Judge mode
    # ------------------------------------------------------------------

    def _evaluate_with_llm(
        self,
        *,
        query: str,
        generated_answer: str,
        retrieved_contexts: list[str],
        reference_answer: str,
    ) -> GenerationMetrics:
        """Use LlamaIndex LLM evaluators for quality assessment.

        Each evaluator is called independently so a failure in one
        metric doesn't lose the others. Falls back to heuristic
        for the entire sample only if LlamaIndex is not installed.
        """
        try:
            from llama_index.core.evaluation import (
                CorrectnessEvaluator,
                FaithfulnessEvaluator,
                RelevancyEvaluator,
            )
            from llama_index.llms.openai import OpenAI
        except ImportError:
            logger.warning("generation.llm_unavailable", msg="LlamaIndex not installed")
            return self._evaluate_with_heuristics(
                query=query,
                generated_answer=generated_answer,
                retrieved_contexts=retrieved_contexts,
                reference_answer=reference_answer,
            )

        llm = OpenAI(model=self._judge_model, api_key=self._api_key)

        # Faithfulness: is the answer grounded in context?
        faithfulness = self._llm_faithfulness(
            llm, FaithfulnessEvaluator, query, generated_answer, retrieved_contexts
        )

        # Relevance: does the answer address the query?
        relevance = self._llm_relevance(
            llm, RelevancyEvaluator, query, generated_answer, retrieved_contexts
        )

        # Correctness: semantic match to reference answer
        correctness = self._llm_correctness(
            llm, CorrectnessEvaluator, query, generated_answer, reference_answer
        )

        # ROUGE-L: always computed (deterministic, free)
        rouge_l = rouge_l_score(generated_answer, reference_answer) if reference_answer else 0.0

        # Context utilization: always computed (deterministic)
        context_util = self._context_utilization(generated_answer, retrieved_contexts)

        return GenerationMetrics(
            faithfulness=faithfulness,
            relevance=relevance,
            correctness=correctness,
            rouge_l=rouge_l,
            context_utilization=context_util,
            eval_mode=EvalMode.LLM_JUDGE,
        )

    @staticmethod
    def _llm_faithfulness(
        llm: object,
        evaluator_cls: type,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> float:
        """Evaluate faithfulness with per-metric error handling."""
        try:
            evaluator = evaluator_cls(llm=llm)
            result = evaluator.evaluate(
                query=query,
                response=answer,
                contexts=contexts,
            )
            return 1.0 if result.passing else 0.0
        except Exception:
            logger.exception("generation.llm_faithfulness_failed")
            return 0.0

    @staticmethod
    def _llm_relevance(
        llm: object,
        evaluator_cls: type,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> float:
        """Evaluate relevance with per-metric error handling."""
        try:
            evaluator = evaluator_cls(llm=llm)
            result = evaluator.evaluate(
                query=query,
                response=answer,
                contexts=contexts,
            )
            return 1.0 if result.passing else 0.0
        except Exception:
            logger.exception("generation.llm_relevance_failed")
            return 0.0

    @staticmethod
    def _llm_correctness(
        llm: object,
        evaluator_cls: type,
        query: str,
        answer: str,
        reference: str,
    ) -> float:
        """Evaluate correctness with per-metric error handling."""
        if not reference:
            return 0.0
        try:
            evaluator = evaluator_cls(llm=llm)
            result = evaluator.evaluate(
                query=query,
                response=answer,
                reference=reference,
            )
            score = result.score or 0.0
            return min(score / 5.0, 1.0)  # Normalize 0-5 → 0-1
        except Exception:
            logger.exception("generation.llm_correctness_failed")
            return 0.0

    # ------------------------------------------------------------------
    # Heuristic mode
    # ------------------------------------------------------------------

    def _evaluate_with_heuristics(
        self,
        *,
        query: str,
        generated_answer: str,
        retrieved_contexts: list[str],
        reference_answer: str,
    ) -> GenerationMetrics:
        """Deterministic heuristic evaluation using ROUGE-L and token overlap.

        Used when no LLM judge is configured. Provides proxy metrics:
        - Faithfulness: Jaccard overlap between answer and context tokens
        - Relevance: Jaccard overlap between answer and query tokens
        - Correctness: Jaccard overlap between answer and reference tokens
        - ROUGE-L: Longest Common Subsequence F1 vs reference (order-aware)
        - Context utilization: fraction of contexts with >20% token overlap
        """
        answer_tokens = set(generated_answer.lower().split())

        # Faithfulness proxy: overlap between answer and context
        context_text = " ".join(retrieved_contexts)
        context_tokens = set(context_text.lower().split())
        faithfulness = self._token_overlap(answer_tokens, context_tokens)

        # Relevance proxy: overlap between answer and query
        query_tokens = set(query.lower().split())
        relevance = self._token_overlap(answer_tokens, query_tokens)

        # Correctness proxy: overlap between answer and reference
        correctness = 0.0
        if reference_answer:
            ref_tokens = set(reference_answer.lower().split())
            correctness = self._token_overlap(answer_tokens, ref_tokens)

        # ROUGE-L: order-aware similarity vs reference
        rouge_l = rouge_l_score(generated_answer, reference_answer) if reference_answer else 0.0

        # Context utilization
        context_util = self._context_utilization(generated_answer, retrieved_contexts)

        return GenerationMetrics(
            faithfulness=min(faithfulness, 1.0),
            relevance=min(relevance, 1.0),
            correctness=min(correctness, 1.0),
            rouge_l=min(rouge_l, 1.0),
            context_utilization=min(context_util, 1.0),
            eval_mode=EvalMode.HEURISTIC,
        )

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _token_overlap(set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two token sets.

        Jaccard = |A & B| / |A | B|
        """
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    @staticmethod
    def _context_utilization(answer: str, contexts: list[str]) -> float:
        """Measure what fraction of retrieved contexts contributed to the answer.

        A context is considered "utilized" if >20% of its tokens appear
        in the generated answer. This threshold is empirical — adjust
        per use case.
        """
        if not contexts or not answer:
            return 0.0

        answer_tokens = set(answer.lower().split())
        utilized = 0
        for ctx in contexts:
            ctx_tokens = set(ctx.lower().split())
            overlap = answer_tokens & ctx_tokens
            # Consider a context "utilized" if >20% of its tokens appear in answer
            if ctx_tokens and len(overlap) / len(ctx_tokens) > 0.2:
                utilized += 1

        return utilized / len(contexts)
