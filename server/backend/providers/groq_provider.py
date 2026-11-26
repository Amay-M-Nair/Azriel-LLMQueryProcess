"""Groq - OpenAI's open-weight gpt-oss models, served fast.

Get a key at https://console.groq.com/keys and put it in .env as GROQ_API_KEY.

gpt-oss is a reasoning model: it thinks before it answers, and how hard is
set per request with `reasoning_effort`. The thinking is never streamed as
answer text - `include_reasoning=False` asks Groq not to send it at all - but
it is billed, so its token count is kept in `last_usage` for the cost log.

Each pipeline stage can run on its own model and effort (`for_stage`), which
is what lets the small structured jobs go to gpt-oss-20b at low effort while
answers go to gpt-oss-120b.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Iterator

from backend.providers.base import Provider, ProviderError, ProviderNotReady
from utils import config

ENV_VAR = "GROQ_API_KEY"

# Worth trying again, possibly on another model: rate limited, or overloaded.
RETRYABLE_CODES = {429, 498, 500, 502, 503}


class _ModelUnavailable(Exception):
    """Internal: this model is busy or rate limited; another one might work."""

    def __init__(self, model: str, cause: Exception):
        self.model = model
        self.cause = cause
        self.code = getattr(cause, "status_code", None)
        self.retry_after = retry_after_seconds(cause)


def retry_after_seconds(exc) -> float | None:
    """How long Groq asked us to wait, from the `retry-after` header."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class GroqProvider(Provider):
    name = "groq"
    note = "gpt-oss on Groq. Your question and excerpts are sent to Groq."
    key_variable = ENV_VAR

    def __init__(
        self,
        model: str = config.GROQ_MODEL,
        api_key: str | None = None,
        reasoning_effort: str = config.GROQ_REASONING_EFFORT,
        temperature: float = config.GROQ_TEMPERATURE,
        max_tokens: int = config.GROQ_MAX_TOKENS,
    ):
        self.model = model
        # A key passed in wins over the environment: someone typed it into
        # this session on purpose.
        self.api_key = (api_key or "").strip() or None
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage = None

    # --- configuration ---------------------------------------------------

    def for_stage(self, stage: str) -> "GroqProvider":
        model, effort = config.GROQ_STAGES.get(
            stage, (config.GROQ_MODEL, config.GROQ_REASONING_EFFORT)
        )
        staged = copy.copy(self)
        staged.model, staged.reasoning_effort, staged.last_usage = model, effort, None
        staged.stage = stage
        return staged

    def _api_key(self) -> str:
        key = self.api_key or os.environ.get(ENV_VAR)
        if key:
            return key
        raise ProviderNotReady(
            "No Groq API key found. Get one at https://console.groq.com/keys, "
            "then put it in your .env file as GROQ_API_KEY=... and restart."
        )

    def check_ready(self) -> None:
        self._api_key()

    def _client(self):
        from groq import Groq

        # Retries are done here rather than by the SDK, so a rate-limited
        # model can fall through to the next one instead of waiting.
        return Groq(api_key=self._api_key(), max_retries=0)

    def candidates(self) -> list[str]:
        """This model first, then the fallbacks, without duplicates."""
        return list(dict.fromkeys([self.model, *config.GROQ_FALLBACK_MODELS]))

    # --- calling ---------------------------------------------------------

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        busy: list[str] = []
        last: _ModelUnavailable | None = None

        for round_number in range(config.GROQ_RETRY_ROUNDS):
            if round_number:
                asked = last.retry_after if last else None
                time.sleep(min(asked or config.GROQ_RETRY_BACKOFF * round_number,
                               config.GROQ_MAX_WAIT))
            for model in self.candidates():
                try:
                    yield from self._stream_one(model, system, prompt)
                    return
                except _ModelUnavailable as exc:
                    # Nothing was emitted yet, so another model is safe to try.
                    busy.append(exc.model)
                    last = exc

        tried = ", ".join(dict.fromkeys(busy))
        if last is not None and last.code == 429:
            raise ProviderError(
                f"Groq rate limit reached on: {tried}. Wait a minute and try "
                "again, or switch the provider."
            )
        raise ProviderError(f"Groq is unavailable right now ({tried}). Try again shortly.")

    def _stream_one(self, model: str, system: str, prompt: str) -> Iterator[str]:
        import groq

        yielded = False
        try:
            chunks = self._client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                include_reasoning=False,
                max_completion_tokens=self.max_tokens,
            )
            for chunk in chunks:
                self._record_usage(model, chunk)
                for choice in chunk.choices or []:
                    # `delta.reasoning` is the model thinking; never answer text.
                    text = getattr(choice.delta, "content", None)
                    if text:
                        yielded = True
                        yield text
        except groq.APIStatusError as exc:
            if not yielded and exc.status_code in RETRYABLE_CODES:
                raise _ModelUnavailable(model, exc) from exc
            raise ProviderError(_explain(exc, model)) from exc
        except groq.APIConnectionError as exc:
            if not yielded:
                raise _ModelUnavailable(model, exc) from exc
            raise ProviderError(f"Lost the connection to Groq mid-answer: {exc}") from exc

        if not yielded:
            raise ProviderError(
                "Groq returned an empty answer. With a reasoning model this "
                "usually means GROQ_MAX_TOKENS ran out while it was still "
                "thinking - raise it in utils/config.py, or lower the effort."
            )

    def _record_usage(self, model: str, chunk) -> None:
        """Groq sends usage on the final chunk, under `usage` or `x_groq.usage`."""
        usage = getattr(chunk, "usage", None) or getattr(
            getattr(chunk, "x_groq", None), "usage", None
        )
        if usage is None:
            return
        details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        self._report_usage({
            "model": model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "cached_tokens": getattr(prompt_details, "cached_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
        })


def _explain(exc, model: str) -> str:
    """Turn the common API errors into something actionable."""
    code = getattr(exc, "status_code", None)
    message = getattr(exc, "message", None) or str(exc)
    if code == 404:
        return (
            f"Groq does not serve the model {model!r} ({message}). Check the "
            "model list at https://console.groq.com/docs/models and set "
            "GROQ_MODEL in utils/config.py."
        )
    if code in (401, 403):
        return f"Groq rejected the API key: {message}"
    if code == 413:
        return "The question and excerpts were too long for Groq's request limit."
    return f"Groq request failed: {message}"
