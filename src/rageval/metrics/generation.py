"""Generation quality metrics using LLM-as-Judge.

Evaluates the quality of RAG-generated answers using LlamaIndex's
evaluation modules. Measures faithfulness (grounding in context),
relevance (addressing the query), and correctness (vs. reference answer).

When no LLM is configured, falls back to deterministic heuristics
(token overlap, embedding similarity proxies).
"""

from __future__ import annotations

import structlog

from rageval.core.models import GenerationMetrics

logger = structlog.get_logger(__name__)


class GenerationEvaluator:
    """Evaluate generation quality with LLM judge or heuristic fallback.

    Example::

        evaluator = GenerationEvaluator()
        metrics = evaluator.evaluate(
            query="What is RAG?",
            generated_answer="RAG combines retrieval with generation.",
            retrieved_contexts=["RAG is a technique..."],
            reference_answer="Retrieval-Augmented Generation...",
        )
        print(f"Faithfulness: {metrics.faithfulness:.2f}")
    """

    def __init__(self, *, judge_model: str = "gpt-4o", api_key: str = "") -> None:
        """Initialize the generation evaluator.

        Args:
            judge_model: LLM model to use as judge.
            api_key: API key for the LLM provider.
        """
        self._judge_model = judge_model
        self._api_key = api_key
        self._llm_available = bool(api_key)

    def evaluate(
        self,
        *,
        query: str,
        generated_answer: str,
        retrieved_contexts: list[str],
        reference_answer: str = "",
    ) -> GenerationMetrics:
        """Evaluate a single generated answer.

        Args:
            query: The original user query.
            generated_answer: The RAG pipeline's answer.
            retrieved_contexts: Context chunks used for generation.
            reference_answer: Ground truth answer for correctness scoring.

        Returns:
            GenerationMetrics with all quality scores.
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

    def _evaluate_with_llm(
        self,
        *,
        query: str,
        generated_answer: str,
        retrieved_contexts: list[str],
        reference_answer: str,
    ) -> GenerationMetrics:
        """Use LlamaIndex LLM evaluators for quality assessment.

        This uses LlamaIndex's FaithfulnessEvaluator, RelevancyEvaluator,
        and CorrectnessEvaluator under the hood.
        """
        try:
            from llama_index.core.evaluation import (
                CorrectnessEvaluator,
                FaithfulnessEvaluator,
                RelevancyEvaluator,
            )
            from llama_index.llms.openai import OpenAI

            llm = OpenAI(model=self._judge_model, api_key=self._api_key)

            # Faithfulness: is the answer grounded in the context?
            faith_eval = FaithfulnessEvaluator(llm=llm)
            faith_result = faith_eval.evaluate(
                query=query,
                response=generated_answer,
                contexts=retrieved_contexts,
            )
            faithfulness = 1.0 if faith_result.passing else 0.0

            # Relevancy: does the answer address the query?
            rel_eval = RelevancyEvaluator(llm=llm)
            rel_result = rel_eval.evaluate(
                query=query,
                response=generated_answer,
                contexts=retrieved_contexts,
            )
            relevance = 1.0 if rel_result.passing else 0.0

            # Correctness: semantic match to reference answer
            correctness = 0.0
            if reference_answer:
                corr_eval = CorrectnessEvaluator(llm=llm)
                corr_result = corr_eval.evaluate(
                    query=query,
                    response=generated_answer,
                    reference=reference_answer,
                )
                correctness = (corr_result.score or 0.0) / 5.0  # Normalize to 0-1

            context_util = self._context_utilization(generated_answer, retrieved_contexts)

            return GenerationMetrics(
                faithfulness=faithfulness,
                relevance=relevance,
                correctness=correctness,
                context_utilization=context_util,
            )

        except ImportError:
            logger.warning("generation.llm_unavailable", msg="LlamaIndex not installed")
            return self._evaluate_with_heuristics(
                query=query,
                generated_answer=generated_answer,
                retrieved_contexts=retrieved_contexts,
                reference_answer=reference_answer,
            )
        except Exception:
            logger.exception("generation.llm_eval_failed")
            return self._evaluate_with_heuristics(
                query=query,
                generated_answer=generated_answer,
                retrieved_contexts=retrieved_contexts,
                reference_answer=reference_answer,
            )

    def _evaluate_with_heuristics(
        self,
        *,
        query: str,
        generated_answer: str,
        retrieved_contexts: list[str],
        reference_answer: str,
    ) -> GenerationMetrics:
        """Deterministic heuristic evaluation using token overlap.

        Used when no LLM judge is configured. Provides rough proxy
        metrics based on word overlap analysis.
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

        # Context utilization
        context_util = self._context_utilization(generated_answer, retrieved_contexts)

        return GenerationMetrics(
            faithfulness=min(faithfulness, 1.0),
            relevance=min(relevance, 1.0),
            correctness=min(correctness, 1.0),
            context_utilization=min(context_util, 1.0),
        )

    @staticmethod
    def _token_overlap(set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard-like overlap between two token sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    @staticmethod
    def _context_utilization(answer: str, contexts: list[str]) -> float:
        """Measure how much of the retrieved context is used in the answer."""
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
