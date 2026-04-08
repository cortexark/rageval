"""Metrics computation for RAG evaluation.

Provides retrieval metrics (precision, recall, F1, MRR, NDCG),
generation quality metrics (faithfulness, relevance, correctness),
and ROUGE-L (word-order-aware text similarity).
"""

from rageval.metrics.generation import GenerationEvaluator
from rageval.metrics.retrieval import RetrievalEvaluator
from rageval.metrics.rouge import rouge_l_score

__all__ = [
    "GenerationEvaluator",
    "RetrievalEvaluator",
    "rouge_l_score",
]
