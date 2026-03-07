"""Pipeline configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """All user-facing and internal configuration for a Bloomdow run."""

    target_model: str
    target_api_key: str | None = None
    target_api_base: str | None = None
    evaluator_model: str = "bedrock/anthropic.claude-sonnet-4-20250514-v1:0"
    evaluator_api_key: str | None = None
    evaluator_api_base: str | None = None
    concern: str
    num_rollouts: int = Field(default=20, ge=1)
    diversity: float = Field(default=0.5, ge=0.0, le=1.0)
    max_turns: int = Field(default=5, ge=1)
    max_concurrency: int = Field(default=10, ge=1)
    judge_samples: int = Field(default=1, ge=1)
    output_dir: str = "bloomdow-results"
