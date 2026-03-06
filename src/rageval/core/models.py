"""Core data models for RAG evaluation.

Defines the evaluation sample structure, metric containers, and result
types used throughout the rageval framework.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class EvalStatus(StrEnum):
    """Status of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RetrievalMetrics(BaseModel):
    """Retrieval quality metrics for a single sample.

    Measures how well the retriever finds relevant documents.
    All metrics range from 0.0 to 1.0.
    """

    precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of retrieved docs that are relevant",
    )
    recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of relevant docs that were retrieved",
    )
    f1_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Harmonic mean of precision and recall",
    )
    mrr: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean Reciprocal Rank of first relevant result",
    )
    ndcg: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized Discounted Cumulative Gain",
    )
    hit_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Whether at least one relevant doc was retrieved",
    )
    retrieved_count: int = Field(default=0, ge=0)
    relevant_count: int = Field(default=0, ge=0)
    relevant_retrieved_count: int = Field(default=0, ge=0)


class GenerationMetrics(BaseModel):
    """Generation quality metrics for a single sample.

    Measures how well the generated answer uses retrieved context.
    All scores range from 0.0 to 1.0 unless noted.
    """

    faithfulness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Whether the answer is grounded in the retrieved context",
    )
    relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Whether the answer addresses the query",
    )
    correctness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Semantic similarity to the reference answer",
    )
    context_utilization: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How well the answer uses relevant retrieved context",
    )


class EvalSample(BaseModel):
    """A single evaluation sample with query, context, and expected answer.

    This is the input to the evaluation pipeline. Each sample represents
    one question-answer pair with optional ground truth context references.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = Field(description="The user query / question")
    reference_answer: str = Field(
        default="",
        description="Ground truth answer for correctness evaluation",
    )
    reference_contexts: list[str] = Field(
        default_factory=list,
        description="Ground truth relevant document IDs or text chunks",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary metadata (topic, difficulty, source, etc.)",
    )


class EvalResult(BaseModel):
    """Complete evaluation result for a single sample.

    Contains the input sample, retrieved documents, generated answer,
    and all computed metrics.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample: EvalSample
    retrieved_contexts: list[str] = Field(
        default_factory=list,
        description="Document chunks returned by the retriever",
    )
    retrieved_doc_ids: list[str] = Field(
        default_factory=list,
        description="IDs of retrieved documents for relevance matching",
    )
    generated_answer: str = Field(
        default="",
        description="The RAG pipeline's generated answer",
    )
    retrieval_metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    generation_metrics: GenerationMetrics = Field(default_factory=GenerationMetrics)
    latency_ms: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvalRunSummary(BaseModel):
    """Aggregate metrics for an evaluation run across all samples."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    status: EvalStatus = Field(default=EvalStatus.PENDING)
    sample_count: int = Field(default=0, ge=0)
    # Retrieval aggregates
    avg_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_mrr: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_ndcg: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    # Generation aggregates
    avg_faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_context_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    # Timing
    avg_latency_ms: float = Field(default=0.0, ge=0.0)
    total_duration_seconds: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
