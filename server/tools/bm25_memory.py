"""ROADMAP 2.3: how much memory does keyword search need at full size?

    python -m tools.bm25_memory          # from server/, needs data/snapshot_index/v1/

Builds BM25 over every chunk of the pinned snapshot (31,119) two ways, each in
a fresh subprocess, and records what the process holds afterwards and at its
peak, against the ~75 MB of headroom 0.7 left on a 512 MB server:

- `BM25` (vectorstore/keyword.py), what the FAISS store uses: built from a
  list holding every chunk's text, as the FAISS metadata does.
- `CompactBM25` (db/keyword.py): fed one chunk at a time, as `KeywordIndex.load`
  feeds it from Postgres in batches, so the texts are never all in memory.

Writes evaluation/results/bm25_memory.json. MB here is 10^6 bytes, as in 0.7.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

from utils import config

OUT = config.PROJECT_ROOT / "evaluation" / "results" / "bm25_memory.json"
HEADROOM_MB = 75   # ROADMAP 0.7: 512 MB minus the 435 MB peak of embedder + reranker

CHILD = r"""
import gc, json, sys, time
import psutil
mode, path = sys.argv[1], sys.argv[2]
proc = psutil.Process()
import numpy
from vectorstore.keyword import BM25, tokenize
from db.keyword import CompactBM25
gc.collect()
base = proc.memory_info().rss
t0 = time.monotonic()
if mode == "bm25":
    with open(path, encoding="utf-8") as f:
        texts = [json.loads(line) for line in f]        # the FAISS metadata holds them all
    index = BM25(texts)
    del texts
    size = None
else:
    def stream():
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    index = CompactBM25(stream())
    size = index.nbytes()
seconds = time.monotonic() - t0
gc.collect()
info = proc.memory_info()
peak = getattr(info, "peak_wset", None)            # Windows: true peak working set
if peak is None:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024   # Linux: KB
t = time.monotonic()
for q in ["What were Apple's total net sales in fiscal 2025?",
          "How many employees did Chevron have?", "goodwill impairment testing"]:
    index.scores(q)
print(json.dumps({"chunks": index.count, "held_mb": (info.rss - base) / 1e6,
                  "peak_above_start_mb": (peak - base) / 1e6, "build_seconds": seconds,
                  "query_ms": (time.monotonic() - t) / 3 * 1000,
                  "index_nbytes_mb": None if size is None else size / 1e6}))
"""


def run(mode: str, path: Path) -> dict:
    out = subprocess.run([sys.executable, "-c", CHILD, mode, str(path)], cwd=config.PROJECT_ROOT,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    metadata = config.SNAPSHOT_INDEX_DIR / "v1" / "metadata.json"
    if not metadata.exists():
        print(f"no FAISS index at {metadata.parent}: run  python -m ingestion.filings", file=sys.stderr)
        return 1
    chunks = json.loads(metadata.read_text(encoding="utf-8"))["chunks"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "texts.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk["text"]) + "\n")
        del chunks
        results = {}
        for mode, label in [("bm25", "BM25 (vectorstore/keyword.py, every text in a list)"),
                            ("compact", "CompactBM25 (db/keyword.py, texts streamed)")]:
            results[mode] = {"label": label, **run(mode, path)}
            r = results[mode]
            print(f"{label:55} holds {r['held_mb']:6.1f} MB, peak +{r['peak_above_start_mb']:6.1f} MB, "
                  f"built in {r['build_seconds']:.1f}s, {r['query_ms']:.0f} ms a query")

    report = {
        "date": date.today().isoformat(),
        "machine": f"{platform.system()} {platform.release()}, Python {platform.python_version()}",
        "headroom_mb": HEADROOM_MB,
        "results": results,
        "fits": results["compact"]["peak_above_start_mb"] < HEADROOM_MB,
        "note": ("held = RSS after building minus RSS before; peak = peak working set (Windows) or "
                 "max RSS (Linux) minus RSS before. The scores of both are identical "
                 "(tests/test_db_keyword.py)."),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(config.PROJECT_ROOT)}; fits the {HEADROOM_MB} MB headroom: {report['fits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
