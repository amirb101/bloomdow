"""Stage 3: Ideation (SDG) — seed generation, augmentation, genRM validation, combination."""

from __future__ import annotations

import asyncio
import json
import logging

from bloomdow.config import PipelineConfig
from bloomdow.llm import complete_json
from bloomdow.models import (
    BehaviorDimension,
    Scenario,
    UnderstandingDocument,
    ValidationResult,
    normalize_behavior_name,
)
from bloomdow.prompts.ideation import (
    AUGMENTATION_SYSTEM,
    GENRM_SYSTEM,
    SEED_GENERATION_SYSTEM,
    augmentation_user_message,
    genrm_user_message,
    seed_generation_user_message,
)

logger = logging.getLogger(__name__)

_BATCH_AUGMENT_SIZE = 5
_GENRM_BATCH_SIZE = 5


async def _generate_seed_scenarios(
    understanding: UnderstandingDocument,
    behavior: BehaviorDimension,
    config: PipelineConfig,
) -> list[Scenario]:
    """Generate K seed scenarios for one understanding."""
    understanding_text = understanding.angle and understanding.full_text.strip()
    if not understanding_text:
        understanding_text = understanding.full_text

    messages = [
        {"role": "system", "content": SEED_GENERATION_SYSTEM},
        {
            "role": "user",
            "content": seed_generation_user_message(
                behavior_name=behavior.name,
                understanding_angle=understanding.angle or "general",
                understanding_text=understanding_text,
                num_scenarios=config.seed_scenarios_per_understanding,
                modality=behavior.modality.value,
                max_turns=behavior.suggested_turns,
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

    raw_list = data if isinstance(data, list) else data.get("scenarios", [])
    scenarios: list[Scenario] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        s = Scenario(
            behavior_name=behavior.name,
            situation=raw.get("situation", ""),
            user_persona=raw.get("user_persona", ""),
            target_system_prompt=raw.get("target_system_prompt", ""),
            environment=raw.get("environment", ""),
            example_manifestation=raw.get("example_manifestation", ""),
            modality=behavior.modality,
            max_turns=behavior.suggested_turns,
        )
        scenarios.append(s)

    return scenarios[: config.seed_scenarios_per_understanding]


async def _augment_batch(
    seeds: list[Scenario],
    num_to_generate: int,
    behavior: BehaviorDimension,
    config: PipelineConfig,
) -> list[Scenario]:
    """Generate num_to_generate augmented scenarios from the given seeds (one LLM call)."""
    if num_to_generate <= 0:
        return []
    seed_dicts = [
        {
            "id": s.id,
            "situation": s.situation,
            "user_persona": s.user_persona,
            "target_system_prompt": s.target_system_prompt,
            "environment": s.environment,
            "example_manifestation": s.example_manifestation,
        }
        for s in seeds[:3]
    ]
    scenario_json_list = json.dumps(seed_dicts, indent=2)

    messages = [
        {"role": "system", "content": AUGMENTATION_SYSTEM},
        {
            "role": "user",
            "content": augmentation_user_message(
                scenario_json_list,
                behavior.name,
                num_to_generate,
            ),
        },
    ]

    data = await complete_json(
        model=config.evaluator_model,
        messages=messages,
        api_key=config.evaluator_api_key,
        api_base=config.evaluator_api_base,
        temperature=0.9,
        max_tokens=8192,
    )

    result: list[Scenario] = []
    raw_list = data if isinstance(data, list) else data.get("scenarios", [])
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        s = Scenario(
            behavior_name=behavior.name,
            situation=raw.get("situation", ""),
            user_persona=raw.get("user_persona", ""),
            target_system_prompt=raw.get("target_system_prompt", ""),
            environment=raw.get("environment", ""),
            example_manifestation=raw.get("example_manifestation", ""),
            modality=behavior.modality,
            max_turns=behavior.suggested_turns,
            is_variation=True,
            parent_scenario_id=seeds[0].id if seeds else None,
        )
        result.append(s)
    return result[:num_to_generate]


async def _augment_scenarios(
    seeds: list[Scenario],
    understanding: UnderstandingDocument,
    behavior: BehaviorDimension,
    config: PipelineConfig,
) -> list[Scenario]:
    """Expand seed scenarios to target_scenarios_per_understanding via batch augmentation."""
    target = config.target_scenarios_per_understanding
    if len(seeds) >= target:
        return seeds[:target]

    all_scenarios = list(seeds)
    seed_index = 0

    while len(all_scenarios) < target:
        to_gen = min(_BATCH_AUGMENT_SIZE, target - len(all_scenarios))
        base_seeds = [seeds[seed_index % len(seeds)]]
        if len(seeds) > 1:
            base_seeds.append(seeds[(seed_index + 1) % len(seeds)])
        batch = await _augment_batch(base_seeds, to_gen, behavior, config)
        all_scenarios.extend(batch)
        seed_index += 1

    return all_scenarios[:target]


async def _validate_batch(
    scenarios: list[Scenario],
    behavior: BehaviorDimension,
    config: PipelineConfig,
) -> list[ValidationResult]:
    """Score a batch of scenarios with genRM; return ValidationResult list."""
    if not scenarios:
        return []
    scenarios_payload = [
        {
            "scenario_id": s.id,
            "situation": s.situation,
            "user_persona": s.user_persona,
            "target_system_prompt": s.target_system_prompt[:500],
            "environment": s.environment[:300],
            "example_manifestation": s.example_manifestation[:300],
        }
        for s in scenarios
    ]
    scenarios_json = json.dumps(scenarios_payload, indent=2)

    messages = [
        {"role": "system", "content": GENRM_SYSTEM},
        {
            "role": "user",
            "content": genrm_user_message(
                behavior.name,
                behavior.description,
                config.genrm_threshold,
                scenarios_json,
            ),
        },
    ]

    data = await complete_json(
        model=config.evaluator_model,
        messages=messages,
        api_key=config.evaluator_api_key,
        api_base=config.evaluator_api_base,
        temperature=0.2,
        max_tokens=4096,
    )

    results: list[ValidationResult] = []
    raw_list = data if isinstance(data, list) else data.get("results", [])
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        q = float(raw.get("quality_score", 5.0))
        r = float(raw.get("relevance_score", 5.0))
        v = float(raw.get("validity_score", 5.0))
        overall = raw.get("overall_score")
        if overall is None:
            overall = (q + r + v) / 3.0
        else:
            overall = float(overall)
        passed = overall >= config.genrm_threshold
        if isinstance(raw.get("passed"), bool):
            passed = raw["passed"]
        results.append(
            ValidationResult(
                scenario_id=str(raw.get("scenario_id", "")),
                quality_score=q,
                relevance_score=r,
                validity_score=v,
                overall_score=overall,
                passed=passed,
                feedback=str(raw.get("feedback", "")),
            )
        )
    return results


async def _validate_scenarios(
    scenarios: list[Scenario],
    behavior: BehaviorDimension,
    config: PipelineConfig,
    semaphore: asyncio.Semaphore,
) -> list[Scenario]:
    """Run genRM on all scenarios in batches; return passing scenarios with scores attached."""
    if not scenarios:
        return []
    score_map: dict[str, ValidationResult] = {}
    for i in range(0, len(scenarios), _GENRM_BATCH_SIZE):
        batch = scenarios[i : i + _GENRM_BATCH_SIZE]
        async with semaphore:
            results = await _validate_batch(batch, behavior, config)
        for vr in results:
            score_map[vr.scenario_id] = vr

    passing: list[Scenario] = []
    for s in scenarios:
        vr = score_map.get(s.id)
        if vr is not None:
            # Attach genRM scores regardless of pass/fail for downstream analysis
            s = s.model_copy(update={
                "genrm_quality": vr.quality_score,
                "genrm_relevance": vr.relevance_score,
                "genrm_validity": vr.validity_score,
                "genrm_overall": vr.overall_score,
            })
        if vr is not None and vr.passed:
            passing.append(s)
    return passing


async def _process_one_understanding(
    understanding: UnderstandingDocument,
    behavior: BehaviorDimension,
    config: PipelineConfig,
    semaphore: asyncio.Semaphore,
) -> list[Scenario]:
    """Generate seeds, augment, validate for one understanding; return validated scenarios."""
    seeds = await _generate_seed_scenarios(understanding, behavior, config)
    if not seeds:
        logger.warning("No seed scenarios for behavior %s (angle: %s)", behavior.name, understanding.angle)
        return []
    augmented = await _augment_scenarios(seeds, understanding, behavior, config)
    validated = await _validate_scenarios(augmented, behavior, config, semaphore)
    logger.info(
        "Understanding angle %r: %d seeds -> %d augmented -> %d passed genRM",
        understanding.angle or "general",
        len(seeds),
        len(augmented),
        len(validated),
    )
    return validated


async def run_ideation(
    behaviors: list[BehaviorDimension],
    all_understandings: dict[str, list[UnderstandingDocument]],
    config: PipelineConfig,
) -> dict[str, list[Scenario]]:
    """Generate seed sets per understanding, augment, validate with genRM, combine per behavior."""
    logger.info(
        "Stage 3 — Ideation (SDG): seed -> augment -> validate for %d behaviors",
        len(behaviors),
    )

    semaphore = asyncio.Semaphore(config.max_concurrency)
    result: dict[str, list[Scenario]] = {}

    norm_understandings = {
        normalize_behavior_name(k): v for k, v in all_understandings.items()
    }

    async def _ideate_one_behavior(behavior: BehaviorDimension) -> tuple[str, list[Scenario]]:
        norm_key = normalize_behavior_name(behavior.name)
        understandings = norm_understandings.get(norm_key)
        if not understandings:
            logger.warning("No understandings for %s, skipping ideation", behavior.name)
            return behavior.name, []

        per_understanding_tasks = [
            _process_one_understanding(understanding, behavior, config, semaphore)
            for understanding in understandings
        ]
        lists_per_understanding = await asyncio.gather(*per_understanding_tasks, return_exceptions=True)

        combined: list[Scenario] = []
        for i, outcome in enumerate(lists_per_understanding):
            if isinstance(outcome, Exception):
                logger.warning("Ideation failed for %s (understanding %d): %s", behavior.name, i, outcome)
                continue
            combined.extend(outcome)

        if config.num_rollouts > 0 and len(combined) > config.num_rollouts:
            combined = combined[: config.num_rollouts]
            logger.info("Subsampled %s to %d scenarios (num_rollouts cap)", behavior.name, config.num_rollouts)

        logger.info("Ideation for %s: %d total validated scenarios", behavior.name, len(combined))
        return behavior.name, combined

    # All behaviors run concurrently; per-understanding tasks within each are also concurrent
    behavior_tasks = [_ideate_one_behavior(behavior) for behavior in behaviors]
    outcomes = await asyncio.gather(*behavior_tasks, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            logger.warning("Ideation behavior task failed: %s", outcome)
            continue
        behavior_name, combined = outcome
        result[behavior_name] = combined

    return result
