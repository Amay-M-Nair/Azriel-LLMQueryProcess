"""How a caller reaches the pipeline.

The UI depends on the `Client` interface rather than on the pipeline itself.
`LocalClient.ask` yields the newline-delimited event stream (stage, route,
token, sources) that the FastAPI service will serve in Phase 2, so the React
page and this one read the same events.
"""

from dataclasses import dataclass
from typing import Iterator, Protocol

from utils import config, logs

log = logs.get(__name__)


@dataclass
class IngestResult:
    indexed: list[str]
    reindexed: list[str]
    skipped: list[tuple[str, str]]
    chunks_added: int
    chunks: int

    @property
    def changed(self) -> bool:
        return bool(self.indexed or self.reindexed)


class Client(Protocol):
    """What the UI needs, and nothing else."""

    def describe(self, collection: str) -> dict: ...
    def add_documents(self, collection: str, files, on_progress=None) -> IngestResult: ...
    def clear(self, collection: str) -> None: ...
    def ask(self, collection, question, history, length,
            provider_name=None, api_key=None, on_stage=None) -> Iterator[dict]: ...


class LocalClient:
    """Calls the pipeline in this process. No server, no network."""

    def describe(self, collection: str) -> dict:
        from api import collections

        return collections.describe(collection)

    def add_documents(self, collection, files, on_progress=None) -> IngestResult:
        from api import collections

        paths = collections.save_uploads(collection, files)
        _, report = collections.ingest(collection, paths, on_progress=on_progress)
        return IngestResult(
            report.indexed, report.reindexed, report.skipped,
            report.chunks_added, report.total_chunks,
        )

    def clear(self, collection: str) -> None:
        from api import collections

        collections.clear(collection)

    def ask(self, collection, question, history, length,
            provider_name=None, api_key=None, on_stage=None) -> Iterator[dict]:
        from api import collections
        from backend.llm import get_provider
        from backend.query_processor import process

        store = collections.load(collection)
        # Both arguments matter and neither can be guessed from config:
        # the provider is whichever the sidebar has selected, and the key
        # may have been typed into the page rather than set in .env. A
        # deployed copy has no .env at all, so dropping the key here left
        # the sidebar reporting Ready and every question failing.
        provider = get_provider(provider_name or config.PROVIDER, api_key=api_key or None)

        stages: list[str] = []
        plan = process(
            question, store=store, history=history, provider=provider,
            on_stage=stages.append, length=length,
        )
        for stage in stages:
            yield {"type": "stage", "text": stage}

        trace = plan.trace
        yield {
            "type": "route",
            "intent": trace.intent.name, "method": trace.intent.method,
            "intent_reason": trace.intent.reason,
            "route": trace.route.name, "route_reason": trace.route.reason,
            "searched_for": (
                trace.rewrite.query if (trace.rewrite and trace.rewrite.changed) else None
            ),
            "api_calls": trace.api_calls,
            "timings": {k: round(v * 1000) for k, v in trace.timings.items()},
            "notes": trace.notes,
        }

        if plan.answer is not None or plan.message is not None:
            yield {"type": "token",
                   "text": plan.answer if plan.answer is not None else plan.message}
        else:
            for piece in plan.stream():
                yield {"type": "token", "text": piece}

        yield {
            "type": "sources",
            "sources": [
                {"label": c.label, "source": c.source, "page": c.page,
                 "score": round(s, 3), "text": c.text}
                for c, s in plan.sources
            ],
        }


def get_client() -> Client:
    """The pipeline in this process. The FastAPI service (Phase 2) wraps it."""
    return LocalClient()
