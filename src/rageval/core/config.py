"""Configuration models for rageval.

Defines settings for LLM providers, embedding models, retrieval
parameters, and evaluation run configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMProviderConfig(BaseModel):
    """Configuration for LLM-based evaluation judges."""

    provider: str = Field(default="openai", description="LLM provider: openai, anthropic")
    model: str = Field(default="gpt-4o", description="Model name")
    api_key: str = Field(default="", description="API key (can also use env vars)")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    provider: str = Field(default="openai", description="Embedding provider")
    model: str = Field(default="text-embedding-3-small", description="Model name")
    api_key: str = Field(default="", description="API key")
    dimensions: int = Field(default=1536, ge=1)


class RetrieverConfig(BaseModel):
    """Configuration for the retrieval step."""

    top_k: int = Field(default=5, ge=1, le=100, description="Number of documents to retrieve")
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to include",
    )


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    name: str = Field(default="rag-eval", description="Name of this eval run")
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    judge_model: str = Field(
        default="gpt-4o",
        description="Model used as an LLM judge for generation quality",
    )
    batch_size: int = Field(default=10, ge=1, description="Samples per batch")
    max_concurrency: int = Field(default=5, ge=1, description="Max parallel evaluations")
    db_path: str = Field(default=":memory:", description="DuckDB database path")
