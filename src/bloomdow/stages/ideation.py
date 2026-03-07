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


async def _ideate_base_scenarios(
    behavior: BehaviorDimension,
    understanding: UnderstandingDocument,
    num_base: int,
    config: PipelineConfig,
) -> list[Scenario]:
    """Generate the base (unique) scenarios for one behavior."""
    messages = [
        {"role": "system", "content": IDEATION_SYSTEM},
        {
            "role": "user",
            "content": ideation_user_message(
                behavior.name,
                understanding.full_text,
                num_base,
                behavior.modality.value,
                behavior.suggested_turns,
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

    scenarios: list[Scenario] = []
    for raw in data.get("scenarios", []):
        scenario = Scenario(
            behavior_name=behavior.name,
            situation=raw.get("situation", ""),
            user_persona=raw.get("user_persona", ""),
            target_system_prompt=raw.get("target_system_prompt", ""),
            environment=raw.get("environment", ""),
            example_manifestation=raw.get("example_manifestation", ""),
            modality=behavior.modality,
            max_turns=behavior.suggested_turns,
        )
        scenarios.append(scenario)

    return scenarios[:num_base]


async def _generate_variation(
    base: Scenario,
    config: PipelineConfig,
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
) -> list[Scenario]:
    """Expand base scenarios via variation to reach total_needed."""
    if len(base_scenarios) >= total_needed:
        return base_scenarios[:total_needed]

    all_scenarios = list(base_scenarios)
    variations_needed = total_needed - len(base_scenarios)

    variation_tasks = []
    for i in range(variations_needed):
        base = base_scenarios[i % len(base_scenarios)]
        variation_tasks.append(_generate_variation(base, config))

    results = await asyncio.gather(*variation_tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Variation generation failed: %s", result)
            continue
        all_scenarios.append(result)

    return all_scenarios[:total_needed]


async def run_ideation(
    behaviors: list[BehaviorDimension],
    understandings: dict[str, UnderstandingDocument],
    config: PipelineConfig,
) -> dict[str, list[Scenario]]:
    """Generate evaluation scenarios for all behaviors."""
    logger.info(
        "Stage 3 — Ideation: generating scenarios for %d behaviors", len(behaviors)
    )

    all_scenarios: dict[str, list[Scenario]] = {}

    for behavior in behaviors:
        understanding = understandings.get(behavior.name)
        if not understanding:
            logger.warning("No understanding for %s, skipping ideation", behavior.name)
            continue

        diversity = behavior.suggested_diversity
        n = config.num_rollouts
        num_base = max(1, math.ceil(n * diversity))

        logger.info(
            "Ideating for %s: %d base scenarios -> %d total (diversity=%.2f)",
            behavior.name, num_base, n, diversity,
        )

        base = await _ideate_base_scenarios(behavior, understanding, num_base, config)
        expanded = await _expand_scenarios(base, n, config)
        all_scenarios[behavior.name] = expanded

        logger.info(
            "Ideation for %s: got %d scenarios", behavior.name, len(expanded)
        )

    return all_scenarios
