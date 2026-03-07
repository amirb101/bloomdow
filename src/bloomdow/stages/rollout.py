"""Stage 4: Rollout — execute evaluation scenarios against the target model."""

from __future__ import annotations

import asyncio
import json
import logging

from bloomdow.config import PipelineConfig
from bloomdow.llm import complete, complete_json
from bloomdow.models import (
    Message,
    Role,
    Scenario,
    Transcript,
    UnderstandingDocument,
    normalize_behavior_name,
)
from bloomdow.prompts.rollout import (
    ROLLOUT_SYSTEM,
    rollout_continue_message,
    rollout_initial_message,
)

logger = logging.getLogger(__name__)

END_MARKER = "<END_EVAL>"


async def _run_single_rollout(
    scenario: Scenario,
    understanding: UnderstandingDocument,
    config: PipelineConfig,
    semaphore: asyncio.Semaphore,
) -> Transcript:
    """Execute one scenario against the target model and return a transcript."""
    async with semaphore:
        transcript = Transcript(
            scenario_id=scenario.id,
            behavior_name=scenario.behavior_name,
            target_model=config.target_model,
            evaluator_model=config.evaluator_model,
        )

        # The target sees its system prompt from the scenario
        target_messages: list[dict[str, str]] = [
            {"role": "system", "content": scenario.target_system_prompt},
        ]

        # The evaluator has the full context
        evaluator_messages: list[dict[str, str]] = [
            {"role": "system", "content": ROLLOUT_SYSTEM},
            {
                "role": "user",
                "content": rollout_initial_message(
                    scenario.situation,
                    scenario.user_persona,
                    scenario.target_system_prompt,
                    scenario.environment,
                    scenario.behavior_name,
                    understanding.full_text,
                    scenario.modality.value,
                    scenario.max_turns,
                ),
            },
        ]

        # Get the evaluator's first user message
        try:
            first_msg_raw = await complete_json(
                model=config.evaluator_model,
                messages=evaluator_messages,
                api_key=config.evaluator_api_key,
                api_base=config.evaluator_api_base,
                temperature=0.7,
                max_tokens=2048,
            )
            first_user_content = first_msg_raw.get("content", "Hello")
        except Exception:
            first_user_content = "Hello, I need your help with something."
            first_msg_raw = {"role": "user", "content": first_user_content}

        # Track the evaluator's own response in its conversation history
        evaluator_messages.append(
            {"role": "assistant", "content": json.dumps(first_msg_raw)}
        )

        transcript.messages.append(
            Message(role=Role.USER, content=first_user_content)
        )
        target_messages.append({"role": "user", "content": first_user_content})

        for turn in range(1, scenario.max_turns + 1):
            # Target responds
            try:
                target_response = await complete(
                    model=config.target_model,
                    messages=target_messages,
                    api_key=config.target_api_key,
                    api_base=config.target_api_base,
                    temperature=0.7,
                    max_tokens=2048,
                )
            except Exception as exc:
                logger.warning(
                    "Target model failed on turn %d for scenario %s: %s",
                    turn, scenario.id, exc,
                )
                break

            transcript.messages.append(
                Message(role=Role.ASSISTANT, content=target_response)
            )
            target_messages.append({"role": "assistant", "content": target_response})
            transcript.turns_completed = turn

            if turn >= scenario.max_turns:
                break

            # Evaluator decides next move
            evaluator_messages.append(
                {
                    "role": "user",
                    "content": rollout_continue_message(
                        target_response, turn, scenario.max_turns, scenario.behavior_name,
                    ),
                }
            )

            try:
                next_msg_data = await complete_json(
                    model=config.evaluator_model,
                    messages=evaluator_messages,
                    api_key=config.evaluator_api_key,
                    api_base=config.evaluator_api_base,
                    temperature=0.7,
                    max_tokens=2048,
                )
                next_content = next_msg_data.get("content", "")
            except Exception as exc:
                logger.warning("Evaluator failed on turn %d: %s", turn, exc)
                break

            # Track evaluator's response in its history
            evaluator_messages.append(
                {"role": "assistant", "content": json.dumps(next_msg_data)}
            )

            if END_MARKER in next_content:
                next_content = next_content.replace(END_MARKER, "").strip()
                transcript.early_termination = True
                if next_content:
                    transcript.messages.append(
                        Message(role=Role.USER, content=next_content)
                    )
                    target_messages.append({"role": "user", "content": next_content})
                break

            if next_content:
                role = Role.TOOL if next_msg_data.get("role") == "tool" else Role.USER
                transcript.messages.append(
                    Message(
                        role=role,
                        content=next_content,
                        name=next_msg_data.get("name"),
                    )
                )
                target_messages.append({"role": role.value, "content": next_content})

        return transcript


def _get_understanding(
    understandings: dict[str, list[UnderstandingDocument]],
    behavior_name: str,
) -> UnderstandingDocument | None:
    """Resolve one understanding per behavior (first of list) for rollout context."""
    lst = understandings.get(behavior_name)
    if lst:
        return lst[0]
    norm_key = normalize_behavior_name(behavior_name)
    for key, val in understandings.items():
        if normalize_behavior_name(key) == norm_key and val:
            return val[0]
    return None


async def run_rollouts(
    scenarios: dict[str, list[Scenario]],
    understandings: dict[str, list[UnderstandingDocument]],
    config: PipelineConfig,
    progress_callback: object | None = None,
) -> dict[str, list[Transcript]]:
    """Run all evaluation scenarios in parallel with concurrency control."""
    logger.info("Stage 4 — Rollout: executing scenarios against target model")

    semaphore = asyncio.Semaphore(config.max_concurrency)
    all_transcripts: dict[str, list[Transcript]] = {}

    tasks: list[tuple[str, asyncio.Task]] = []
    for behavior_name, behavior_scenarios in scenarios.items():
        understanding = _get_understanding(understandings, behavior_name)
        if not understanding:
            continue

        for scenario in behavior_scenarios:
            task = asyncio.create_task(
                _run_single_rollout(scenario, understanding, config, semaphore)
            )
            tasks.append((behavior_name, task))

    results = await asyncio.gather(
        *[t for _, t in tasks], return_exceptions=True
    )

    for (behavior_name, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.warning("Rollout failed for %s: %s", behavior_name, result)
            continue
        all_transcripts.setdefault(behavior_name, []).append(result)

    for name, transcripts in all_transcripts.items():
        logger.info("Rollout complete: %s — %d transcripts", name, len(transcripts))

    return all_transcripts
