"""DuckDB-backed result storage for evaluation runs.

Provides persistent storage, querying, comparison, and export of
evaluation results across runs, models, and dataset versions.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb
import structlog

from rageval.core.models import EvalResult, EvalRunSummary, EvalStatus

logger = structlog.get_logger(__name__)


class ResultStore:
    """Store and query evaluation results in DuckDB.

    Supports both in-memory (for tests) and file-backed (for production)
    databases. Provides SQL-powered aggregation, cross-run comparison,
    filtered queries, and JSON export.

    Example::

        store = ResultStore(db_path=":memory:")
        store.store_results("run-1", results)
        summary = store.get_run_summary("run-1")
        print(f"Avg F1: {summary.avg_f1:.3f}")
        store.export_json("run-1", "/tmp/run-1-results.json")
    """

    def __init__(self, *, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = duckdb.connect(db_path)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                sample_id VARCHAR NOT NULL,
                query VARCHAR,
                reference_answer VARCHAR,
                generated_answer VARCHAR,
                -- Retrieval metrics
                precision DOUBLE DEFAULT 0,
                recall DOUBLE DEFAULT 0,
                f1_score DOUBLE DEFAULT 0,
                mrr DOUBLE DEFAULT 0,
                ndcg DOUBLE DEFAULT 0,
                hit_rate DOUBLE DEFAULT 0,
                retrieved_count INTEGER DEFAULT 0,
                relevant_count INTEGER DEFAULT 0,
                relevant_retrieved_count INTEGER DEFAULT 0,
                -- Generation metrics
                faithfulness DOUBLE DEFAULT 0,
                relevance DOUBLE DEFAULT 0,
                correctness DOUBLE DEFAULT 0,
                rouge_l DOUBLE DEFAULT 0,
                context_utilization DOUBLE DEFAULT 0,
                eval_mode VARCHAR DEFAULT 'none',
                -- Timing
                latency_ms DOUBLE DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                status VARCHAR DEFAULT 'pending',
                sample_count INTEGER DEFAULT 0,
                avg_precision DOUBLE DEFAULT 0,
                avg_recall DOUBLE DEFAULT 0,
                avg_f1 DOUBLE DEFAULT 0,
                avg_mrr DOUBLE DEFAULT 0,
                avg_ndcg DOUBLE DEFAULT 0,
                avg_hit_rate DOUBLE DEFAULT 0,
                avg_faithfulness DOUBLE DEFAULT 0,
                avg_relevance DOUBLE DEFAULT 0,
                avg_correctness DOUBLE DEFAULT 0,
                avg_rouge_l DOUBLE DEFAULT 0,
                avg_context_utilization DOUBLE DEFAULT 0,
                avg_latency_ms DOUBLE DEFAULT 0,
                total_duration_seconds DOUBLE DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.debug("storage.tables_created")

    def store_results(self, run_id: str, results: list[EvalResult]) -> int:
        """Store evaluation results for a run.

        Uses INSERT OR REPLACE for idempotent upserts — re-running
        an evaluation on the same samples updates rather than duplicates.

        Args:
            run_id: Unique identifier for this evaluation run.
            results: List of evaluation results to store.

        Returns:
            Number of results stored.
        """
        for result in results:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO eval_results (
                    id, run_id, sample_id, query, reference_answer,
                    generated_answer, precision, recall, f1_score,
                    mrr, ndcg, hit_rate, retrieved_count, relevant_count,
                    relevant_retrieved_count, faithfulness, relevance,
                    correctness, rouge_l, context_utilization, eval_mode,
                    latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result.id,
                    run_id,
                    result.sample.id,
                    result.sample.query,
                    result.sample.reference_answer,
                    result.generated_answer,
                    result.retrieval_metrics.precision,
                    result.retrieval_metrics.recall,
                    result.retrieval_metrics.f1_score,
                    result.retrieval_metrics.mrr,
                    result.retrieval_metrics.ndcg,
                    result.retrieval_metrics.hit_rate,
                    result.retrieval_metrics.retrieved_count,
                    result.retrieval_metrics.relevant_count,
                    result.retrieval_metrics.relevant_retrieved_count,
                    result.generation_metrics.faithfulness,
                    result.generation_metrics.relevance,
                    result.generation_metrics.correctness,
                    result.generation_metrics.rouge_l,
                    result.generation_metrics.context_utilization,
                    result.generation_metrics.eval_mode.value,
                    result.latency_ms,
                    result.created_at.isoformat(),
                ],
            )

        logger.info("storage.results_stored", run_id=run_id, count=len(results))
        return len(results)

    def get_run_summary(self, run_id: str) -> EvalRunSummary:
        """Compute aggregate metrics for a run via SQL.

        Uses DuckDB's AVG() aggregation — no materializing rows into Python.

        Args:
            run_id: The run to summarize.

        Returns:
            EvalRunSummary with averages across all samples.
        """
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) as sample_count,
                AVG(precision) as avg_precision,
                AVG(recall) as avg_recall,
                AVG(f1_score) as avg_f1,
                AVG(mrr) as avg_mrr,
                AVG(ndcg) as avg_ndcg,
                AVG(hit_rate) as avg_hit_rate,
                AVG(faithfulness) as avg_faithfulness,
                AVG(relevance) as avg_relevance,
                AVG(correctness) as avg_correctness,
                AVG(rouge_l) as avg_rouge_l,
                AVG(context_utilization) as avg_context_utilization,
                AVG(latency_ms) as avg_latency_ms
            FROM eval_results
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()

        if not row or row[0] == 0:
            return EvalRunSummary(id=run_id, status=EvalStatus.FAILED)

        return EvalRunSummary(
            id=run_id,
            status=EvalStatus.COMPLETED,
            sample_count=row[0],
            avg_precision=row[1] or 0.0,
            avg_recall=row[2] or 0.0,
            avg_f1=row[3] or 0.0,
            avg_mrr=row[4] or 0.0,
            avg_ndcg=row[5] or 0.0,
            avg_hit_rate=row[6] or 0.0,
            avg_faithfulness=row[7] or 0.0,
            avg_relevance=row[8] or 0.0,
            avg_correctness=row[9] or 0.0,
            avg_rouge_l=row[10] or 0.0,
            avg_context_utilization=row[11] or 0.0,
            avg_latency_ms=row[12] or 0.0,
        )

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict[str, dict[str, float]]:
        """Compare two evaluation runs side by side.

        Computes delta (candidate - baseline) for every metric.

        Args:
            run_id_a: First run (baseline).
            run_id_b: Second run (candidate).

        Returns:
            Dict with metric names as keys, each containing
            'baseline', 'candidate', 'delta' values.
        """
        summary_a = self.get_run_summary(run_id_a)
        summary_b = self.get_run_summary(run_id_b)

        metrics = [
            "avg_precision",
            "avg_recall",
            "avg_f1",
            "avg_mrr",
            "avg_ndcg",
            "avg_hit_rate",
            "avg_faithfulness",
            "avg_relevance",
            "avg_correctness",
            "avg_rouge_l",
            "avg_context_utilization",
        ]

        comparison: dict[str, dict[str, float]] = {}
        for metric in metrics:
            val_a = getattr(summary_a, metric, 0.0)
            val_b = getattr(summary_b, metric, 0.0)
            comparison[metric] = {
                "baseline": val_a,
                "candidate": val_b,
                "delta": val_b - val_a,
            }

        return comparison

    def query_results(
        self,
        run_id: str,
        *,
        min_f1: float | None = None,
        max_f1: float | None = None,
        min_faithfulness: float | None = None,
        max_faithfulness: float | None = None,
        min_rouge_l: float | None = None,
        max_rouge_l: float | None = None,
        eval_mode: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query results with flexible filters.

        Supports filtering on any metric range, evaluation mode, and
        result count limit.

        Args:
            run_id: Run to query.
            min_f1: Minimum F1 score filter.
            max_f1: Maximum F1 score filter.
            min_faithfulness: Minimum faithfulness filter.
            max_faithfulness: Maximum faithfulness filter.
            min_rouge_l: Minimum ROUGE-L filter.
            max_rouge_l: Maximum ROUGE-L filter.
            eval_mode: Filter by evaluation mode ('llm_judge' or 'heuristic').
            limit: Max results to return.

        Returns:
            List of result dicts.
        """
        conditions = ["run_id = ?"]
        params: list[Any] = [run_id]

        filter_map: list[tuple[str, str, float | str | None]] = [
            ("f1_score >= ?", "f1_score >= ?", min_f1),
            ("f1_score <= ?", "f1_score <= ?", max_f1),
            ("faithfulness >= ?", "faithfulness >= ?", min_faithfulness),
            ("faithfulness <= ?", "faithfulness <= ?", max_faithfulness),
            ("rouge_l >= ?", "rouge_l >= ?", min_rouge_l),
            ("rouge_l <= ?", "rouge_l <= ?", max_rouge_l),
            ("eval_mode = ?", "eval_mode = ?", eval_mode),
        ]

        for condition, _, value in filter_map:
            if value is not None:
                conditions.append(condition)
                params.append(value)

        where = " AND ".join(conditions)
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM eval_results WHERE {where} LIMIT ?",
            params,
        ).fetchall()

        desc = self._conn.description
        columns = [d[0] for d in desc] if desc else []
        return [dict(zip(columns, row)) for row in rows]

    def export_json(self, run_id: str, file_path: str) -> int:
        """Export a run's results to a JSON file.

        Exports both individual results and the aggregate summary
        in a single JSON document suitable for dashboards, reports,
        or archiving.

        Args:
            run_id: Run to export.
            file_path: Output file path (.json).

        Returns:
            Number of results exported.
        """
        results = self.query_results(run_id, limit=10000)
        summary = self.get_run_summary(run_id)

        export_data = {
            "run_id": run_id,
            "summary": {
                "status": summary.status.value,
                "sample_count": summary.sample_count,
                "avg_precision": summary.avg_precision,
                "avg_recall": summary.avg_recall,
                "avg_f1": summary.avg_f1,
                "avg_mrr": summary.avg_mrr,
                "avg_ndcg": summary.avg_ndcg,
                "avg_hit_rate": summary.avg_hit_rate,
                "avg_faithfulness": summary.avg_faithfulness,
                "avg_relevance": summary.avg_relevance,
                "avg_correctness": summary.avg_correctness,
                "avg_rouge_l": summary.avg_rouge_l,
                "avg_context_utilization": summary.avg_context_utilization,
                "avg_latency_ms": summary.avg_latency_ms,
            },
            "results": _serialize_results(results),
        }

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info("storage.exported_json", run_id=run_id, count=len(results), path=file_path)
        return len(results)

    def list_runs(self) -> list[dict[str, Any]]:
        """List all evaluation runs with their sample counts.

        Returns:
            List of dicts with run_id and sample_count.
        """
        rows = self._conn.execute(
            """
            SELECT run_id, COUNT(*) as sample_count,
                   MIN(created_at) as first_result,
                   MAX(created_at) as last_result
            FROM eval_results
            GROUP BY run_id
            ORDER BY MIN(created_at) DESC
            """
        ).fetchall()

        return [
            {
                "run_id": row[0],
                "sample_count": row[1],
                "first_result": str(row[2]),
                "last_result": str(row[3]),
            }
            for row in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


def _serialize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert DuckDB result rows to JSON-serializable dicts."""
    serialized = []
    for row in results:
        clean: dict[str, Any] = {}
        for key, value in row.items():
            clean[key] = str(value) if hasattr(value, "isoformat") else value
        serialized.append(clean)
    return serialized
