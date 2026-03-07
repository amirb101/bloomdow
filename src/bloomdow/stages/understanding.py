"""Stage 2: Understanding — produce N diverse analyses per behavior with diversity enforcement."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from bloomdow.config import PipelineConfig
from bloomdow.llm import complete_json, embed
from bloomdow.models import BehaviorDimension, UnderstandingDocument
from bloomdow.prompts.understanding import (
    DIVERSE_UNDERSTANDING_SYSTEM,
    diverse_understanding_user_message,
)

logger = logging.getLogger(__name__)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two unit-normalized vectors: 1 - cos_sim."""
    return float(1.0 - np.dot(a, b))


async def _generate_diverse_understandings(
    behavior: BehaviorDimension,
    config: PipelineConfig,
) -> list[UnderstandingDocument]:
    """Generate N diverse understanding documents for one behavior."""
    rubric_text = "\n".join(
        f"  - Score {ex.score}: {ex.description}" for ex in behavior.scoring_rubric
    )
    messages = [
        {"role": "system", "content": DIVERSE_UNDERSTANDING_SYSTEM},
        {
            "role": "user",
            "content": diverse_understanding_user_message(
                behavior.name,
                behavior.description,
                rubric_text,
                config.num_diverse_understandings,
            ),
        },
    ]

    data = await complete_json(
        model=config.evaluator_model,
        messages=messages,
        api_key=config.evaluator_api_key,
        api_base=config.evaluator_api_base,
        temperature=0.8,
        max_tokens=8192,
    )

    understandings: list[UnderstandingDocument] = []
    raw_list = data if isinstance(data, list) else data.get("understandings", [])
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        doc = UnderstandingDocument(
            behavior_name=raw.get("behavior_name", behavior.name),
            mechanisms=raw.get("mechanisms", ""),
            existential_risk_relevance=raw.get("existential_risk_relevance", ""),
            subtle_expressions=raw.get("subtle_expressions", ""),
            overt_expressions=raw.get("overt_expressions", ""),
            full_text=raw.get("full_text", ""),
            angle=raw.get("angle", ""),
        )
        understandings.append(doc)

    return understandings[: config.num_diverse_understandings]


async def _enforce_understanding_diversity_with_behavior(
    behavior: BehaviorDimension,
    understandings: list[UnderstandingDocument],
    config: PipelineConfig,
) -> list[UnderstandingDocument]:
    """Enforce pairwise cosine distance; regenerate understanding at idx_j if needed."""
    if len(understandings) < 2:
        return understandings

    emb_api_key = config.embedding_api_key if config.embedding_api_key else config.evaluator_api_key
    emb_api_base = config.embedding_api_base if config.embedding_api_base else config.evaluator_api_base

    for attempt in range(config.max_diversity_retries + 1):
        texts = [u.full_text for u in understandings]
        vectors = await embed(
            model=config.embedding_model,
            texts=texts,
            api_key=emb_api_key,
            api_base=emb_api_base,
        )
        vectors_np = np.array(vectors, dtype=np.float64)
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vectors_np = vectors_np / norms

        for i, doc in enumerate(understandings):
            doc.embedding = vectors_np[i].tolist()

        min_dist = 2.0
        idx_i, idx_j = -1, -1
        for i in range(len(vectors_np)):
            for j in range(i + 1, len(vectors_np)):
                d = _cosine_distance(vectors_np[i], vectors_np[j])
                if d < min_dist:
                    min_dist = d
                    idx_i, idx_j = i, j

        if min_dist >= config.min_cosine_distance:
            logger.info(
                "Diversity enforced: min pairwise cosine distance = %.3f",
                min_dist,
            )
            return understandings

        logger.warning(
            "Pair %d,%d too close (dist=%.3f), regenerating understanding %d",
            idx_i, idx_j, min_dist, idx_j,
        )
        other_angles = [u.angle for k, u in enumerate(understandings) if k != idx_j]
        rubric_text = "\n".join(
            f"  - Score {ex.score}: {ex.description}" for ex in behavior.scoring_rubric
        )
        diverge_instruction = (
            f"\n\nCRITICAL: Your understanding MUST explore a completely different angle "
            f"from these existing ones: {other_angles}. Do NOT paraphrase them. "
            f"Choose a different mechanism, deployment context, or severity framing."
        )
        messages = [
            {"role": "system", "content": DIVERSE_UNDERSTANDING_SYSTEM},
            {
                "role": "user",
                "content": diverse_understanding_user_message(
                    behavior.name,
                    behavior.description,
                    rubric_text,
                    1,
                )
                + diverge_instruction
                + "\n\nReturn a JSON object with \"understandings\" containing a single object.",
            },
        ]
        data = await complete_json(
            model=config.evaluator_model,
            messages=messages,
            api_key=config.evaluator_api_key,
            api_base=config.evaluator_api_base,
            temperature=0.9,
            max_tokens=4096,
        )
        raws = data.get("understandings", [])
        if raws:
            raw = raws[0]
            understandings[idx_j] = UnderstandingDocument(
                behavior_name=raw.get("behavior_name", behavior.name),
                mechanisms=raw.get("mechanisms", ""),
                existential_risk_relevance=raw.get("existential_risk_relevance", ""),
                subtle_expressions=raw.get("subtle_expressions", ""),
                overt_expressions=raw.get("overt_expressions", ""),
                full_text=raw.get("full_text", ""),
                angle=raw.get("angle", ""),
            )

    return understandings


async def run_understanding(
    behaviors: list[BehaviorDimension],
    config: PipelineConfig,
) -> dict[str, list[UnderstandingDocument]]:
    """Generate N diverse understanding documents per behavior with diversity enforcement."""
    logger.info(
        "Stage 2 — Understanding: generating %d diverse understandings for %d behaviors",
        config.num_diverse_understandings,
        len(behaviors),
    )

    result: dict[str, list[UnderstandingDocument]] = {}

    for behavior in behaviors:
        understandings = await _generate_diverse_understandings(behavior, config)
        if not understandings:
            logger.error("No understandings generated for %s", behavior.name)
            continue
        understandings = await _enforce_understanding_diversity_with_behavior(
            behavior, understandings, config
        )
        result[behavior.name] = understandings
        logger.info(
            "Understanding for %s: %d diverse documents",
            behavior.name,
            len(understandings),
        )

    logger.info(
        "Understanding complete for %d/%d behaviors",
        len(result),
        len(behaviors),
    )
    return result
