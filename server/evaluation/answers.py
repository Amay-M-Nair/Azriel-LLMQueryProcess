"""The pipeline's answers to golden-set questions, generated once and kept.

An answer is what the evaluation grades, so it has to stay fixed between
runs: the answer model samples at temperature 1.0, and a regenerated answer
would be a different answer, graded afresh. Each one is cached under a
fingerprint of everything that produced it (snapshot version, index, models,
retrieval settings), so changing any of those starts a new set rather than
mixing two.

    data/answer_cache/{fingerprint}/{case_id}.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.query_processor import answer_now, process
from evaluation import golden
from evaluation import snapshot as snap
from ingestion import filings
from utils import config
from vectorstore.faiss_store import VectorStore


def fingerprint(snapshot: dict | None = None) -> str:
    """A short hash of everything that decides what an answer will be."""
    snapshot = snapshot or snap.load()
    manifest = filings.load_manifest(filings.index_dir(snapshot)) or {}
    settings = {
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
    """The pipeline's answer to one case: from the cache, or generated and cached."""
    path = Path(cache_dir) / f"{case.id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    plan = process(case.question, store=store, history=history_of(case), provider=provider)
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
        "route": trace.route.name if trace.route else None,
        "search_query": trace.search_query,
        "timings": trace.timings,
        "api_calls": trace.api_calls,
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
