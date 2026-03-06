# rageval

**Production-grade RAG evaluation framework with retrieval and generation metrics**

[![CI](https://github.com/cortexark/rageval/actions/workflows/ci.yml/badge.svg)](https://github.com/cortexark/rageval/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

rageval measures the quality of Retrieval-Augmented Generation pipelines end-to-end. It computes **retrieval metrics** (precision, recall, F1, MRR, NDCG) and **generation quality metrics** (faithfulness, relevance, correctness) using LlamaIndex evaluators and deterministic heuristics.

---

## Why rageval?

RAG pipelines have two failure modes that require separate measurement:

1. **Retrieval failure** -- the right documents aren't found
2. **Generation failure** -- the answer doesn't use retrieved context correctly

Most evaluation tools measure only generation quality. rageval evaluates both stages independently and together, giving engineering teams the metrics they need to diagnose and fix quality issues.

---

## Features

- **Retrieval metrics**: Precision, Recall, F1, MRR, NDCG, Hit Rate
- **Generation metrics**: Faithfulness, Relevance, Correctness, Context Utilization
- **LLM-as-Judge**: Uses LlamaIndex evaluators (FaithfulnessEvaluator, RelevancyEvaluator, CorrectnessEvaluator) with GPT-4o
- **Heuristic fallback**: Token-overlap scoring when no LLM is configured (zero-cost, deterministic)
- **DuckDB storage**: Persistent results with SQL-queryable history
- **Regression detection**: Compare runs and block deploys when metrics drop
- **Batch evaluation**: Process 500+ samples in a single pipeline run
- **Multi-retriever comparison**: Side-by-side assessment of BM25, dense, hybrid strategies

---

## Architecture

```
                    rageval Architecture
  +----------------------------------------------------+
  |                                                    |
  |   Retrieval Metrics        Generation Metrics      |
  |   +----------------+      +-------------------+   |
  |   | Precision       |      | Faithfulness      |   |
  |   | Recall          |      | Relevance         |   |
  |   | F1 Score        |      | Correctness       |   |
  |   | MRR             |      | Context Util.     |   |
  |   | NDCG            |      |                   |   |
  |   | Hit Rate        |      | LLM Judge /       |   |
  |   |                 |      | Heuristic Fallback|   |
  |   +----------------+      +-------------------+   |
  |           |                        |               |
  |           v                        v               |
  |   +--------------------------------------------+   |
  |   |           EvalRunner Pipeline              |   |
  |   |   Batch processing, timing, orchestration  |   |
  |   +--------------------------------------------+   |
  |                        |                           |
  |                        v                           |
  |   +--------------------------------------------+   |
  |   |          DuckDB ResultStore                |   |
  |   |   Storage, queries, run comparison,        |   |
  |   |   regression detection                     |   |
  |   +--------------------------------------------+   |
  |                                                    |
  +----------------------------------------------------+
```

### Data Flow

```
EvalSample (query + ground truth)
    |
    +---> RetrievalEvaluator.compute()
    |         retrieved_ids vs relevant_ids
    |         => RetrievalMetrics (P/R/F1/MRR/NDCG)
    |
    +---> GenerationEvaluator.score()
    |         answer vs context vs reference
    |         => GenerationMetrics (faithfulness/relevance/correctness)
    |
    +---> ResultStore.store_results()
    |         DuckDB persistence
    |
    +---> RegressionChecker.check()
              baseline vs candidate
              => pass/fail decision
```

---

## Quick Start

```python
from rageval.core.config import EvalConfig
from rageval.core.models import EvalSample
from rageval.pipeline.runner import EvalRunner

# Define evaluation dataset
samples = [
    EvalSample(
        query="What is RAG?",
        reference_answer="Retrieval-Augmented Generation combines retrieval with LLMs",
        reference_contexts=["doc-rag-overview", "doc-rag-tutorial"],
    ),
]

# What your RAG pipeline returned
retrieved_ids = [["doc-rag-overview", "doc-embeddings"]]
generated_answers = ["RAG is a technique that retrieves relevant documents..."]
retrieved_texts = [["RAG combines retrieval with generation..."]]

# Run evaluation
config = EvalConfig(name="my-rag-v1")
runner = EvalRunner(config)
results = runner.run_assessment(
    samples=samples,
    retrieved_results=retrieved_ids,
    generated_answers=generated_answers,
    retrieved_contexts=retrieved_texts,
)

# Check results
summary = runner.get_summary()
print(f"Avg Precision: {summary.avg_precision:.3f}")
print(f"Avg Recall:    {summary.avg_recall:.3f}")
print(f"Avg F1:        {summary.avg_f1:.3f}")
```

---

## Metrics Reference

### Retrieval Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Precision** | relevant_retrieved / total_retrieved | Are retrieved docs relevant? |
| **Recall** | relevant_retrieved / total_relevant | Were all relevant docs found? |
| **F1 Score** | 2 * (P * R) / (P + R) | Balance of precision and recall |
| **MRR** | 1 / rank_of_first_relevant | Is the first relevant doc ranked high? |
| **NDCG** | DCG / IDCG | Quality of the ranking order |
| **Hit Rate** | 1 if any relevant doc found | At least one hit? |

### Generation Metrics

| Metric | What It Measures | Method |
|--------|-----------------|--------|
| **Faithfulness** | Answer grounded in context? | LLM judge or token overlap |
| **Relevance** | Answer addresses the query? | LLM judge or token overlap |
| **Correctness** | Answer matches reference? | LLM judge or token overlap |
| **Context Utilization** | How much context was used? | Token overlap analysis |

---

## CI/CD Integration

```python
from rageval.pipeline.runner import EvalRunner, RegressionChecker

# Run baseline and candidate evaluations
# ...

# Check for regression
checker = RegressionChecker(threshold=0.05)
has_regression, comparison = checker.check(store, "baseline", "candidate")

if has_regression:
    print("REGRESSION DETECTED - blocking deployment")
    sys.exit(1)
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-7C3AED?style=for-the-badge&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

---

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/rageval/
```

---

## License

MIT
