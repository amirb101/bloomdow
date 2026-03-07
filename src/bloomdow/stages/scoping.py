"""Stage 1: Scoping — decompose a user concern into measurable behavioral dimensions."""

from __future__ import annotations

import logging

from bloomdow.config import PipelineConfig
from bloomdow.llm import complete_structured
from bloomdow.models import ScopingResult
from bloomdow.prompts.scoping import SCOPING_SYSTEM, scoping_user_message

logger = logging.getLogger(__name__)


async def run_scoping(config: PipelineConfig) -> ScopingResult:
    """Decompose the user's concern prompt into concrete behavioral dimensions."""
    logger.info("Stage 1 — Scoping: decomposing concern into behavioral dimensions")

    messages = [
        {"role": "system", "content": SCOPING_SYSTEM},
        {"role": "user", "content": scoping_user_message(config.concern)},
    ]

    result = await complete_structured(
        model=config.evaluator_model,
        messages=messages,
        schema=ScopingResult,
        api_key=config.evaluator_api_key,
        api_base=config.evaluator_api_base,
        temperature=0.6,
        max_tokens=8192,
    )

    logger.info(
        "Scoping identified %d behavioral dimensions: %s",
        len(result.behaviors),
        [b.name for b in result.behaviors],
    )
    return result
