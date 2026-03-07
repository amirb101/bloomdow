"""Async LiteLLM wrapper with retry logic and structured JSON output parsing."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel

litellm.drop_params = True

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_TIMEOUT = 120


async def complete(
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> str:
    """Return the assistant content string from a chat completion."""
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        num_retries=max_retries,
    )
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = await litellm.acompletion(**kwargs)
    return response.choices[0].message.content or ""


async def complete_json(
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 4096,
    timeout: int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Return parsed JSON dict from a chat completion.

    Instructs the model to return JSON and strips markdown fences if present.
    """
    json_instruction = {
        "role": "system",
        "content": "You MUST respond with valid JSON only. No markdown fences, no commentary outside the JSON object.",
    }
    augmented = [json_instruction] + messages

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=augmented,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        num_retries=max_retries,
    )
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    try:
        kwargs["response_format"] = {"type": "json_object"}
    except Exception:
        pass

    raw = ""
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await litellm.acompletion(**kwargs)
            raw = response.choices[0].message.content or ""
            return _parse_json(raw)
        except json.JSONDecodeError as exc:
            last_err = exc
            logger.warning(
                "JSON parse failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )
        except Exception as exc:
            last_err = exc
            logger.warning(
                "LLM call failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )

    raise RuntimeError(
        f"Failed to get valid JSON after {max_retries + 1} attempts. "
        f"Last raw output: {raw[:500]!r}. Last error: {last_err}"
    )


async def complete_structured(
    model: str,
    messages: list[dict[str, str]],
    schema: type[T],
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 4096,
    timeout: int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> T:
    """Return a validated Pydantic model instance from a JSON completion."""
    schema_text = json.dumps(schema.model_json_schema(), indent=2)
    schema_msg = {
        "role": "system",
        "content": (
            "You MUST respond with valid JSON matching this schema exactly:\n"
            f"```\n{schema_text}\n```\n"
            "No markdown fences or commentary outside the JSON."
        ),
    }
    augmented = [schema_msg] + messages

    data = await complete_json(
        model,
        augmented,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    return schema.model_validate(data)


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse JSON from a string, stripping common markdown wrappers."""
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)
