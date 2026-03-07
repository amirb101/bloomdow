"""Stage 2: Understanding — produce a detailed analysis of each behavioral dimension."""

from __future__ import annotations

import asyncio
import logging

from bloomdow.config import PipelineConfig
from bloomdow.llm import complete_structured
from bloomdow.models import BehaviorDimension, UnderstandingDocument
from bloomdow.prompts.understanding import (
    UNDERSTANDING_SYSTEM,
    understanding_user_message,
)

logger = logging.getLogger(__name__)


async def _understand_one(
    behavior: BehaviorDimension,
    config: PipelineConfig,
) -> UnderstandingDocument:
    rubric_text = "\n".join(
        f"  - Score {ex.score}: {ex.description}" for ex in behavior.scoring_rubric
    )
    messages = [
        {"role": "system", "content": UNDERSTANDING_SYSTEM},
        {
            "role": "user",
            "content": understanding_user_message(
                behavior.name,
                behavior.description,
                rubric_text,
            ),
        },
    ]

    return await complete_structured(
        model=config.evaluator_model,
        messages=messages,
        schema=UnderstandingDocument,
        api_key=config.evaluator_api_key,
        api_base=config.evaluator_api_base,
        temperature=0.5,
        max_tokens=4096,
    )


async def run_understanding(
    behaviors: list[BehaviorDimension],
    config: PipelineConfig,
) -> dict[str, UnderstandingDocument]:
    """Generate understanding documents for all behaviors concurrently."""
    logger.info(
        "Stage 2 — Understanding: analysing %d behaviors", len(behaviors)
    )

    tasks = [_understand_one(b, config) for b in behaviors]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    docs: dict[str, UnderstandingDocument] = {}
    for behavior, result in zip(behaviors, results):
        if isinstance(result, Exception):
            logger.error(
                "Understanding failed for %s: %s", behavior.name, result
            )
            continue
        docs[behavior.name] = result

    logger.info("Understanding complete for %d/%d behaviors", len(docs), len(behaviors))
    return docs
