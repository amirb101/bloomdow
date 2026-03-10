"""Pipeline configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """All user-facing and internal configuration for a Bloomdow run."""

    target_model: str
    target_api_key: str | None = None
    target_api_base: str | None = None
    evaluator_model: str = "anthropic/claude-sonnet-4-20250514"
    evaluator_api_key: str | None = None
    evaluator_api_base: str | None = None
    concern: str
    num_rollouts: int = Field(
        default=20,
        ge=1,
        description="Downstream cap on scenarios per behavior for rollout (subsample if exceeded)",
    )
    num_diverse_understandings: int = Field(default=5, ge=1)
    seed_scenarios_per_understanding: int = Field(default=10, ge=1)
    target_scenarios_per_understanding: int = Field(default=100, ge=1)
    min_cosine_distance: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Minimum pairwise cosine distance between understanding embeddings",
    )
    genrm_threshold: float = Field(
        default=7.0,
        ge=1.0,
        le=10.0,
        description="Minimum quality score (1-10) for augmented scenarios to pass validation",
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="LiteLLM-compatible embedding model ID",
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="API key for the embedding model (defaults to evaluator key if not set)",
    )
    embedding_api_base: str | None = Field(
        default=None,
        description="Base URL for the embedding model API (e.g. http://localhost:8776/v1 for llama.cpp)",
    )
    max_diversity_retries: int = Field(default=3, ge=0)
    max_turns: int = Field(default=5, ge=1)
    max_concurrency: int = Field(default=3, ge=1)
    judge_samples: int = Field(
        default=1,
        ge=1,
        description="Number of independent judge samples per transcript. >1 enables variance estimation.",
    )
    output_dir: str = "bloomdow-results"
