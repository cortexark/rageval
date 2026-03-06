"""Generation evaluation components.

Provides quality assessment of RAG-generated answers using
LLM-as-Judge evaluation (faithfulness, relevance, correctness).
"""

from rageval.metrics.generation import GenerationEvaluator

__all__ = ["GenerationEvaluator"]
