"""The contract every answer provider implements.

A provider takes a system prompt and a user prompt, and yields the answer as
text arrives. That is the whole interface - everything else (retrieval,
chunking, the UI) is provider-independent.
"""

import copy
from abc import ABC, abstractmethod
from typing import Callable, Iterator


class ProviderNotReady(RuntimeError):
    """Provider can't run yet - missing key, server not started, etc.

    The message is shown directly to the user, so it should say what to do.
    """


class ProviderError(RuntimeError):
    """The provider was reachable but the request failed."""


class Provider(ABC):
    name: str = "provider"
    #: Human-readable note shown in the UI (cost, privacy, etc.)
    note: str = ""
    #: Environment variable the key is normally read from. None means this
    #: provider needs no key at all, and the UI should not ask for one.
    key_variable: str | None = None

    @property
    def needs_key(self) -> bool:
        return self.key_variable is not None

    @abstractmethod
    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Yield the answer in pieces as it is generated."""

    @abstractmethod
    def check_ready(self) -> None:
        """Raise ProviderNotReady if this provider cannot run right now."""

    def complete(self, system: str, prompt: str) -> str:
        """The whole answer as one string, for callers that cannot stream.

        Classification and rewriting need the complete reply before they can
        act on it, so streaming buys them nothing.
        """
        return "".join(self.stream(system, prompt))

    def for_stage(self, stage: str) -> "Provider":
        """This provider, configured for one pipeline stage.

        Stages are "classify", "rewrite", "answer", "decompose", "claims".
        A provider with one model and no per-stage settings returns itself;
        Groq returns a copy on the stage's model and reasoning effort.
        """
        return self

    #: Token counts from the most recent call, when the provider reports them:
    #: {"model", "prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens"}.
    last_usage: dict | None = None
    #: The pipeline stage this copy is configured for (see `for_stage`).
    stage: str = "default"
    #: Called with each call's usage and stage, on a metered copy.
    usage_sink: Callable[[dict], None] | None = None

    def metered(self, sink: Callable[[dict], None]) -> "Provider":
        """A copy that reports every call's usage to `sink`, stages included.

        A copy, so a provider shared between questions never mixes their costs.
        """
        meter = copy.copy(self)
        meter.usage_sink = sink
        return meter

    def _report_usage(self, usage: dict) -> None:
        self.last_usage = usage
        if self.usage_sink is not None:
            self.usage_sink(usage | {"stage": self.stage})

    def is_ready(self) -> bool:
        try:
            self.check_ready()
            return True
        except ProviderNotReady:
            return False
