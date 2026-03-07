"""Stage 3: Ideation — generate evaluation scenarios with variation/perturbation."""

from __future__ import annotations

import asyncio
import json
import logging
import math

from bloomdow.config import PipelineConfig
from bloomdow.llm import complete_json
from bloomdow.models import (
    BehaviorDimension,
    Scenario,
    UnderstandingDocument,
)
from bloomdow.prompts.ideation import (
    IDEATION_SYSTEM,
    VARIATION_SYSTEM,
    ideation_user_message,
    variation_user_message,
)

logger = logging.getLogger(__name__)

_MAX_SCENARIOS_PER_CALL = 3
_BEDROCK_CONCURRENCY = 3


async def _ideate_batch(
    behavior: BehaviorDimension,
    understanding: UnderstandingDocument,
    count: int,
    config: PipelineConfig,
    semaphore: asyncio.Semaphore,
) -> list[Scenario]:
    """Generate a small batch of base scenarios in a single LLM call."""
    messages = [
        {"role": "system", "content": IDEATION_SYSTEM},
        {
            "role": "user",
            "content": ideation_user_message(
                behavior.name,
                understanding.full_text,
                count,
                behavior.modality.value,
                behavior.suggested_turns,
            ),
        },
    ]

    async with semaphore:
        data = await complete_json(
            model=config.evaluator_model,
            messages=messages,
            api_key=config.evaluator_api_key,
            api_base=config.evaluator_api_base,
            temperature=0.8,
            max_tokens=6144,
        )

    raw_list = data if isinstance(data, list) else data.get("scenarios", [])
    scenarios: list[Scenario] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        scenarios.append(
            Scenario(
                behavior_name=behavior.name,
                situation=raw.get("situation", ""),
                user_persona=raw.get("user_persona", ""),
                target_system_prompt=raw.get("target_system_prompt", ""),
                environment=raw.get("environment", ""),
                example_manifestation=raw.get("example_manifestation", ""),
                modality=behavior.modality,
                max_turns=behavior.suggested_turns,
            )
        )
    return scenarios[:count]


async def _ideate_base_scenarios(
    behavior: BehaviorDimension,
    understanding: UnderstandingDocument,
    num_base: int,
    config: PipelineConfig,
    semaphore: asyncio.Semaphore,
) -> list[Scenario]:
    """Generate base scenarios, batching into smaller LLM calls to avoid timeouts."""
    remaining = num_base
    all_scenarios: list[Scenario] = []

    while remaining > 0:
        batch_size = min(remaining, _MAX_SCENARIOS_PER_CALL)
        logger.info(
            "Ideating %d scenarios for %s (%d remaining)",
            batch_size, behavior.name, remaining,
        )
        try:
            batch = await _ideate_batch(behavior, understanding, batch_size, config, semaphore)
            all_scenarios.extend(batch)
            remaining -= len(batch)
            if not batch:
                logger.warning("Empty batch for %s, breaking", behavior.name)
                break
        except Exception as exc:
            logger.error("Ideation batch failed for %s: %s", behavior.name, exc)
            break

    return all_scenarios


async def _generate_variation(
    base: Scenario,
    config: PipelineConfig,
    semaphore: asyncio.Semaphore,
) -> Scenario:
    """Produce a single surface-level variation of a base scenario."""
    base_dict = {
        "situation": base.situation,
        "user_persona": base.user_persona,
        "target_system_prompt": base.target_system_prompt,
        "environment": base.environment,
        "example_manifestation": base.example_manifestation,
    }
    messages = [
        {"role": "system", "content": VARIATION_SYSTEM},
        {"role": "user", "content": variation_user_message(json.dumps(base_dict, indent=2))},
    ]

    async with semaphore:
        data = await complete_json(
            model=config.evaluator_model,
            messages=messages,
            api_key=config.evaluator_api_key,
            api_base=config.evaluator_api_base,
            temperature=0.9,
            max_tokens=4096,
        )

    return Scenario(
        behavior_name=base.behavior_name,
        situation=data.get("situation", base.situation),
        user_persona=data.get("user_persona", base.user_persona),
        target_system_prompt=data.get("target_system_prompt", base.target_system_prompt),
        environment=data.get("environment", base.environment),
        example_manifestation=data.get("example_manifestation", base.example_manifestation),
        modality=base.modality,
        max_turns=base.max_turns,
        is_variation=True,
        parent_scenario_id=base.id,
    )


async def _expand_scenarios(
    base_scenarios: list[Scenario],
    total_needed: int,
    config: PipelineConfig,
    semaphore: asyncio.Semaphore,
) -> list[Scenario]:
    """Expand base scenarios via variation to reach total_needed."""
    if len(base_scenarios) >= total_needed:
        return base_scenarios[:total_needed]

    all_scenarios = list(base_scenarios)
    variations_needed = total_needed - len(base_scenarios)

    variation_tasks = []
    for i in range(variations_needed):
        base = base_scenarios[i % len(base_scenarios)]
        variation_tasks.append(_generate_variation(base, config, semaphore))

    results = await asyncio.gather(*variation_tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Variation generation failed: %s", result)
            continue
        all_scenarios.append(result)

    return all_scenarios[:total_needed]


async def _ideate_for_behavior(
    behavior: BehaviorDimension,
    understanding: UnderstandingDocument,
    config: PipelineConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[Scenario]]:
    """Generate all scenarios (base + variations) for a single behavior."""
    diversity = behavior.suggested_diversity
    n = config.num_rollouts
    num_base = max(1, math.ceil(n * diversity))

    logger.info(
        "Ideating for %s: %d base scenarios -> %d total (diversity=%.2f)",
        behavior.name, num_base, n, diversity,
    )

    base = await _ideate_base_scenarios(behavior, understanding, num_base, config, semaphore)
    if not base:
        logger.error("No base scenarios generated for %s", behavior.name)
        return behavior.name, []

    expanded = await _expand_scenarios(base, n, config, semaphore)
    logger.info("Ideation for %s: got %d scenarios", behavior.name, len(expanded))
    return behavior.name, expanded


async def run_ideation(
    behaviors: list[BehaviorDimension],
    understandings: dict[str, UnderstandingDocument],
    config: PipelineConfig,
) -> dict[str, list[Scenario]]:
    """Generate evaluation scenarios for all behaviors in parallel."""
    logger.info(
        "Stage 3 — Ideation: generating scenarios for %d behaviors", len(behaviors)
    )

    semaphore = asyncio.Semaphore(_BEDROCK_CONCURRENCY)

    tasks = []
    for behavior in behaviors:
        understanding = understandings.get(behavior.name)
        if not understanding:
            logger.warning("No understanding for %s, skipping ideation", behavior.name)
            continue
        tasks.append(_ideate_for_behavior(behavior, understanding, config, semaphore))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_scenarios: dict[str, list[Scenario]] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error("Ideation failed: %s", result)
            continue
        name, scenarios = result
        if scenarios:
            all_scenarios[name] = scenarios

    return all_scenarios
