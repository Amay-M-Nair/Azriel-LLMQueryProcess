"""Index the pinned snapshot's filings (ROADMAP 1.3): the baseline the rest is measured against.

Deliberately the pipeline as it stands: the minimal HTML loader, the same
chunker, embedder and FAISS store an uploaded PDF goes through. What is new
is only what the evaluation needs to know about each chunk - which filing it
came from (ticker, accession, form, fiscal year and period) and where in the
filing's text it sits (character offsets), so golden-set answers labelled by
text span (1.4) can be matched to whatever chunks retrieval returns.

    python -m ingestion.filings            # index snapshot.json into data/snapshot_index/v{version}/
    python -m ingestion.filings --check    # report what is indexed, change nothing

The index is saved after every company, and a filing already indexed with the
same hash is skipped, so an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from evaluation import snapshot as snap
from ingestion import embedder
from ingestion.chunker import chunk_pages
from ingestion.document_loader import html_text
from ingestion.edgar import _sha256
from utils import config, logs
from vectorstore.faiss_store import VectorStore
from vectorstore.metadata_store import DocumentRecord

log = logs.get(__name__)

MANIFEST = "manifest.json"


def source_name(pinned: dict) -> str:
    """How a filing is named in the index and in citations: "AAPL 10-K FY2026"."""
    return f"{pinned['ticker']} {pinned['form']} {pinned['fiscal_period']}{pinned['fiscal_year']}"


def index_dir(snapshot: dict) -> Path:
    return config.SNAPSHOT_INDEX_DIR / f"v{snapshot['version']}"


@dataclass
class IndexReport:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)      # already indexed, same hash
    chunks_added: int = 0
    seconds: float = 0.0


def load_filing_text(path: Path, sha256: str) -> str:
    """A pinned filing's visible text, refusing a file that is not the pinned one."""
    content = path.read_bytes()
    if _sha256(content) != sha256:
        raise ValueError(f"{path} does not match its pinned hash: run  python -m evaluation.snapshot")
    return html_text(content.decode("utf-8", errors="replace"))


def index_snapshot(
    snapshot: dict,
    directory: Path | None = None,
    cache_dir: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[VectorStore, IndexReport]:
    """Add every pinned filing to the index in `directory`, saving after each company."""
    directory = Path(directory or index_dir(snapshot))
    store = VectorStore.load(directory) or VectorStore()
    report = IndexReport()
    started = time.monotonic()
    paths = dict(zip((f["accession"] for f in snapshot["filings"]), snap.paths(snapshot, cache_dir)))

    for ticker in snapshot["companies"]:
        changed = False
        for pinned in (f for f in snapshot["filings"] if f["ticker"] == ticker):
            name = source_name(pinned)
            if store.is_current(name, pinned["sha256"]):
                report.skipped.append(name)
                continue
            if on_progress:
                on_progress(f"reading {name}")
            text = load_filing_text(paths[pinned["accession"]], pinned["sha256"])
            chunks = chunk_pages([(1, text)], source=name)
            for chunk in chunks:
                chunk.ticker = pinned["ticker"]
                chunk.accession = pinned["accession"]
                chunk.form = pinned["form"]
                chunk.fiscal_year = pinned["fiscal_year"]
                chunk.fiscal_period = pinned["fiscal_period"]
            if on_progress:
                on_progress(f"embedding {len(chunks)} chunks of {name}")
            vectors = embedder.embed([c.text for c in chunks])
            if store.has_name(name):
                store.remove_document(name)
            store.add(vectors, chunks, DocumentRecord(
                name=name, content_hash=pinned["sha256"], pages=1, chunk_count=len(chunks),
            ))
            report.indexed.append(name)
            report.chunks_added += len(chunks)
            changed = True
            log.info("indexed %s: %d words, %d chunks", name, len(text.split()), len(chunks))
        if changed:
            save(store, directory, snapshot)

    if not (directory / MANIFEST).exists():
        save(store, directory, snapshot)
    report.seconds = time.monotonic() - started
    return store, report


def save(store: VectorStore, directory: Path, snapshot: dict) -> None:
    """The index, plus a manifest saying what it was built from and how."""
    store.save(directory)
    manifest = {
        "snapshot_version": snapshot["version"],
        "filings": len(store.documents),
        "chunks": len(store),
        "embed_model": config.EMBED_MODEL,
        "chunk_words": config.CHUNK_WORDS,
        "chunk_overlap_words": config.CHUNK_OVERLAP_WORDS,
        "loader": "document_loader.html_text (minimal, ROADMAP 1.3 baseline)",
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifest(directory: Path) -> dict | None:
    path = Path(directory) / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report what is indexed; change nothing")
    args = parser.parse_args(argv)
    logs.setup()

    snapshot = snap.load()
    directory = index_dir(snapshot)
    if args.check:
        store = VectorStore.load(directory)
        pinned = {source_name(f): f["sha256"] for f in snapshot["filings"]}
        have = {} if store is None else {n: r.content_hash for n, r in store.documents.items()}
        missing = sorted(n for n, h in pinned.items() if have.get(n) != h)
        print(f"{directory}: {len(have)} filings, {len(store) if store else 0} chunks; "
              f"{len(pinned) - len(missing)}/{len(pinned)} pinned filings indexed")
        for name in missing:
            print(f"  missing {name}")
        return 1 if missing else 0

    store, report = index_snapshot(snapshot, on_progress=lambda m: print(f"  {m}", flush=True))
    print(f"snapshot v{snapshot['version']} -> {directory}")
    print(f"  {len(report.indexed)} indexed, {len(report.skipped)} already there, "
          f"{report.chunks_added} chunks added, {len(store)} in total, {report.seconds / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
