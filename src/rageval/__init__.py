"""rageval - Production-grade RAG evaluation framework.

Evaluates Retrieval-Augmented Generation pipelines with:
- Retrieval metrics: precision, recall, F1, MRR, NDCG, hit rate
- Generation metrics: faithfulness, relevance, correctness, ROUGE-L
- Dual evaluation: LLM-as-Judge (LlamaIndex) + heuristic fallback
- DuckDB storage: persistent results, cross-run comparison, JSON export
- CI/CD gate: threshold-based regression detection
"""

__version__ = "0.2.0"
