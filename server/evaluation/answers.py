"""The pipeline's answers to golden-set questions, generated once and kept.

An answer is what the evaluation grades, so it has to stay fixed between
runs: the answer model samples at temperature 1.0, and a regenerated answer
would be a different answer, graded afresh. Each one is cached under a
fingerprint of everything that produced it (snapshot version, index, models,
retrieval settings), so changing any of those starts a new set rather than
mixing two.

Each record carries what the question cost (ROADMAP 1.7): every model call's
tokens and stage, its dollars, and its latency. `summarize` turns a set of them
into the $/query and p50/p95 latency every eval result reports.

    data/answer_cache/{fingerprint}/{case_id}.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.llm import get_provider
from backend.query_processor import answer_now, process
from evaluation import golden
from evaluation import snapshot as snap
from ingestion import filings
from utils import config
from vectorstore.faiss_store import VectorStore


# Bumped when what an answer record holds, or how it is produced, changes.
# 2: records carry usage, cost and latency, and the eval passes the configured
#    provider, so the classifier and the rewriter run (before, they were skipped).
RECORD_VERSION = 2


def fingerprint(snapshot: dict | None = None) -> str:
    """A short hash of everything that decides what an answer will be."""
    snapshot = snapshot or snap.load()
    manifest = filings.load_manifest(filings.index_dir(snapshot)) or {}
    settings = {
        "record": RECORD_VERSION,
        "snapshot": snapshot["version"],
        "index": manifest,
        "provider": config.PROVIDER,
        "stages": config.GROQ_STAGES,
        "temperature": config.GROQ_TEMPERATURE,
        "top_k": config.TOP_K,
        "min_similarity": config.MIN_SIMILARITY,
        "keyword_weight": config.KEYWORD_WEIGHT,
        "full_context_words": config.FULL_CONTEXT_WORDS,
    }
    blob = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def history_of(case: golden.GoldenCase) -> list[dict]:
    """The case's earlier turns in the shape the pipeline takes."""
    return [{"question": t.question, "answer": t.answer} for t in case.history]


def answer(case: golden.GoldenCase, store: VectorStore, cache_dir: Path, provider=None) -> dict:
    """The pipeline's answer to one case: from the cache, or generated and cached.

    The configured provider unless one is given. Never none: with no provider
    the pipeline skips the classifier and the rewriter, so a follow-up would go
    unrewritten and the evaluation would not measure the pipeline as it runs.
    """
    path = Path(cache_dir) / f"{case.id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    plan = process(case.question, store=store, history=history_of(case), provider=provider or get_provider())
    response = answer_now(plan)
    trace = plan.trace
    record = {
        "case_id": case.id,
        "question": case.question,
        "response": response,
        "contexts": [
            {"chunk_id": c.chunk_id, "accession": c.accession, "start_char": c.start_char,
             "end_char": c.end_char, "text": c.text, "score": score}
            for c, score in plan.sources
        ],
        "intent": trace.intent.name if trace.intent else None,
        "intent_method": trace.intent.method if trace.intent else None,
        "intent_reason": trace.intent.reason if trace.intent else None,
        "rewrite_reason": trace.rewrite.reason if trace.rewrite else None,
        "route": trace.route.name if trace.route else None,
        "search_query": trace.search_query,
        "timings": trace.timings,
        "seconds": trace.total_seconds,
        "api_calls": trace.api_calls,
        "usage": trace.usage,
        "tokens": trace.tokens,
        "cost": trace.cost,
        "answered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def load_store(snapshot: dict | None = None) -> VectorStore:
    snapshot = snapshot or snap.load()
    store = VectorStore.load(filings.index_dir(snapshot))
    if store is None:
        raise FileNotFoundError("no snapshot index: run  python -m ingestion.filings  (from server/)")
    return store


def summarize(records: list[dict]) -> dict:
    """What a set of answers cost and how long they took: the 1.7 columns of every eval result.

    `cost_per_query` is None when any answer's cost is unknown (an unpriced
    model, or a call that reported no usage): a mean over the rest would look
    cheaper than the run was.
    """
    if not records:
        raise ValueError("no answers to summarize")
    seconds = np.array([r["seconds"] for r in records], dtype=float)
    costs = [r["cost"] for r in records]
    known = [c for c in costs if c is not None]
    unknown = len(costs) - len(known)
    tokens = {k: float(np.mean([r["tokens"][k] for r in records]))
              for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")}
    return {
        "questions": len(records),
        "cost_per_query": None if unknown else sum(known) / len(known),
        "cost_total": None if unknown else sum(known),
        "cost_unknown": unknown,
        "tokens_per_query": tokens,
        "p50_seconds": float(np.percentile(seconds, 50)),
        "p95_seconds": float(np.percentile(seconds, 95)),
        "prices_checked": config.PRICES_CHECKED,
    }
