"""Async LiteLLM wrapper with retry logic and structured JSON output parsing."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

litellm.drop_params = True

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_TIMEOUT = 300


def _is_bedrock(model: str) -> bool:
    return model.startswith("bedrock/")


def _build_kwargs(
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None,
    api_base: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        num_retries=0,
    )
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    return kwargs


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
    kwargs = _build_kwargs(
        model, messages,
        api_key=api_key, api_base=api_base,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout,
    )

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d, retrying in %ds): %s",
                    attempt + 1, max_retries + 1, wait, exc,
                )
                await asyncio.sleep(wait)
            else:
                logger.error("LLM call failed after %d attempts: %s", max_retries + 1, exc)

    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {last_err}")


async def _raw_call(kwargs: dict[str, Any]) -> str:
    """Make a single acompletion call and return raw content."""
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
) -> Any:
    """Return parsed JSON from a chat completion."""
    json_instruction = {
        "role": "system",
        "content": (
            "You MUST respond with valid JSON only. "
            "No markdown fences, no explanatory text, no commentary. "
            "Your entire response must be a single parseable JSON value."
        ),
    }
    augmented = [json_instruction] + messages

    kwargs = _build_kwargs(
        model, augmented,
        api_key=api_key, api_base=api_base,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout,
    )
    if not _is_bedrock(model):
        kwargs["response_format"] = {"type": "json_object"}

    raw = ""
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = await _raw_call(kwargs)
            return _parse_json(raw)
        except json.JSONDecodeError as exc:
            last_err = exc
            logger.warning(
                "JSON parse failed (attempt %d/%d): %s — raw: %.300s",
                attempt + 1, max_retries + 1, exc, raw,
            )
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt if attempt < max_retries else 0
            logger.warning(
                "LLM call failed (attempt %d/%d): %s%s",
                attempt + 1, max_retries + 1, exc,
                f" — retrying in {wait}s" if wait else "",
            )
            if wait:
                await asyncio.sleep(wait)

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
    """Return a validated Pydantic model instance from a JSON completion.

    Retries both LLM failures and validation failures.
    """
    schema_text = json.dumps(schema.model_json_schema(), indent=2)
    schema_msg = {
        "role": "system",
        "content": (
            "You MUST respond with a single JSON object (not an array) "
            "matching this schema exactly:\n"
            f"```json\n{schema_text}\n```\n"
            "No markdown fences or commentary outside the JSON object. "
            "Your entire response must be a single parseable JSON object."
        ),
    }
    augmented = [schema_msg] + messages

    kwargs = _build_kwargs(
        model, augmented,
        api_key=api_key, api_base=api_base,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout,
    )
    if not _is_bedrock(model):
        kwargs["response_format"] = {"type": "json_object"}

    raw = ""
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = await _raw_call(kwargs)
            parsed = _parse_json(raw)
            coerced = _coerce_to_schema(parsed, schema)
            return schema.model_validate(coerced)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_err = exc
            logger.warning(
                "Structured parse failed (attempt %d/%d): %s — raw: %.500s",
                attempt + 1, max_retries + 1, exc, raw,
            )
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt if attempt < max_retries else 0
            logger.warning(
                "LLM call failed (attempt %d/%d): %s%s",
                attempt + 1, max_retries + 1, exc,
                f" — retrying in {wait}s" if wait else "",
            )
            if wait:
                await asyncio.sleep(wait)

    raise RuntimeError(
        f"Structured completion failed after {max_retries + 1} attempts. "
        f"Last raw: {raw[:500]!r}. Last error: {last_err}"
    )


def _coerce_to_schema(data: Any, schema: type) -> dict[str, Any]:
    """Best-effort coercion when the LLM returns a slightly wrong shape."""
    json_schema = schema.model_json_schema()
    properties = json_schema.get("properties", {})

    array_fields = [
        name for name, info in properties.items()
        if info.get("type") == "array" or "items" in info
    ]

    if isinstance(data, list) and array_fields:
        field = array_fields[0]
        logger.info("Coercing bare array into {%r: [%d items]}", field, len(data))
        return {field: data}

    if not isinstance(data, dict):
        logger.warning("Unexpected JSON type %s, returning as-is", type(data).__name__)
        return data

    for field in array_fields:
        if field in data:
            continue
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0:
                logger.info("Remapping key %r -> %r (%d items)", key, field, len(val))
                data[field] = data.pop(key)
                break

    return data


def _parse_json(raw: str) -> Any:
    """Parse JSON from a string, stripping common markdown wrappers."""
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)
