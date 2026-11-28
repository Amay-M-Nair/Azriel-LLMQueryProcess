"""ROADMAP 2.3: keyword search inside Postgres, checked as an alternative.

    python -m tools.postgres_keyword_check     # from server/, needs TEST_DATABASE_URL and the FAISS index

The roadmap names pg_search (ParadeDB). Neon has deprecated it: the extension
is still listed, but creating it is refused, and Neon's replacement for BM25
is its own `lakebase_text` (`lakebase_bm25` indexes over a tsvector). This
records the pg_search refusal, then checks lakebase_text instead.

It loads Apple's 1,808 real chunks into a throwaway schema on the Neon *test*
branch, indexes them, and runs the answerable golden questions about Apple
through it and through our BM25 over the same chunks: how far the two agree,
how each does on the golden spans (keyword search alone, top 10), how long
the query takes inside Postgres, and the index's size. The schema is dropped
afterwards, and any extension this run created.

A comparison for the record, not a switch: lakebase_text tokenizes with
Postgres's English text search (stemming, its own stopwords) and scores with
its own BM25, so adopting it would be its own Results row, never parity (2.6).

Writes evaluation/results/postgres_keyword_check.json.
"""

from __future__ import annotations

import json
import statistics
import sys
import uuid
from datetime import date

import numpy as np
import psycopg

from db import connection
from db.keyword import CompactBM25
from evaluation import golden as golden_set
from evaluation.metrics import Range, coverage_matrix, recall_at_k
from utils import config

OUT = config.PROJECT_ROOT / "evaluation" / "results" / "postgres_keyword_check.json"
TICKER = "AAPL"
K = 10
QUERY = ("SELECT id FROM (SELECT id, tsv <@> to_bm25query(to_tsvector('english', %s), 'kw_bm25') AS score"
         " FROM kw) AS scored WHERE score < 0 ORDER BY score, id LIMIT %s")


def try_create(conn, name: str) -> tuple[bool, str | None]:
    """(created by this run, refusal message)."""
    if conn.execute("SELECT 1 FROM pg_extension WHERE extname = %s", (name,)).fetchone():
        return False, None
    try:
        conn.execute(f"CREATE EXTENSION {name}")
        return True, None
    except psycopg.Error as error:
        return False, str(error).strip()


def main() -> int:
    metadata = config.SNAPSHOT_INDEX_DIR / "v1" / "metadata.json"
    chunks = [c for c in json.loads(metadata.read_text(encoding="utf-8"))["chunks"] if c["ticker"] == TICKER]
    cases = [c for c in golden_set.load() if c.spans and any(a.ticker == TICKER for a in c.about)]
    print(f"{len(chunks)} {TICKER} chunks, {len(cases)} answerable golden questions about {TICKER}")

    ours = CompactBM25(c["text"] for c in chunks)
    ranges = [Range(c["accession"], c["start_char"], c["end_char"]) for c in chunks]
    schema = f"kwc_{uuid.uuid4().hex[:8]}"
    report = {"date": date.today().isoformat(), "ticker": TICKER, "chunks": len(chunks),
              "questions": len(cases), "k": K}

    with connection.connect(connection.TEST, register_vector=False) as conn:
        conn.autocommit = True
        _, refusal = try_create(conn, "pg_search")
        report["pg_search"] = refusal or "created (unexpected: Neon lists it as deprecated)"
        print(f"pg_search: {report['pg_search']}")
        created, refusal = try_create(conn, "lakebase_text")
        if refusal:
            report["lakebase_text_error"] = refusal
            OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"lakebase_text: {refusal}", file=sys.stderr)
            return 1
        report["lakebase_text_version"] = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'lakebase_text'").fetchone()[0]
        conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            conn.execute(f'SET search_path = "{schema}", public')
            conn.execute("CREATE TABLE kw (id bigint PRIMARY KEY, text text NOT NULL,"
                         " tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED)")
            with conn.cursor() as cur, cur.copy("COPY kw (id, text) FROM STDIN") as copy:
                for n, chunk in enumerate(chunks):
                    copy.write_row((n, chunk["text"]))
            conn.execute("CREATE INDEX kw_bm25 ON kw USING lakebase_bm25 (tsv)")   # after loading: stats at build
            report["index_mb"] = conn.execute("SELECT pg_relation_size('kw_bm25')").fetchone()[0] / 1e6
            report["table_with_tsvector_mb"] = conn.execute(
                "SELECT pg_table_size('kw')").fetchone()[0] / 1e6

            per_case, times = [], []
            for case in cases:
                theirs = [r[0] for r in conn.execute(QUERY, (case.question, K)).fetchall()]
                plan = conn.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + QUERY, (case.question, K)).fetchone()[0]
                times.append(plan[0]["Execution Time"])
                scores = ours.scores(case.question)
                mine = [int(i) for i in np.argsort(-scores, kind="stable")[:K] if scores[i] > 0]
                spans = [Range(s.accession, s.start_char, s.end_char) for s in case.spans]
                per_case.append({
                    "id": case.id, "returned_ours": len(mine), "returned_lakebase": len(theirs),
                    "overlap": len(set(mine) & set(theirs)) / K,
                    "recall_ours": recall_at_k(coverage_matrix([ranges[i] for i in mine], spans), K),
                    "recall_lakebase": recall_at_k(coverage_matrix([ranges[i] for i in theirs], spans), K),
                })
        finally:
            conn.execute("SET search_path = public")
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
            if created:
                conn.execute("DROP EXTENSION IF EXISTS lakebase_text")

    report.update({
        "top10_overlap_mean": statistics.mean(c["overlap"] for c in per_case),
        "recall_at_10_ours": statistics.mean(c["recall_ours"] for c in per_case),
        "recall_at_10_lakebase": statistics.mean(c["recall_lakebase"] for c in per_case),
        "lakebase_ms_median": statistics.median(times),
        "lakebase_ms_max": max(times),
        "per_case": per_case,
        "note": ("Keyword search alone, both over the same Apple chunks (BM25 statistics from that subset). "
                 "Times are execution inside Postgres on Neon's free compute, without the network round trip. "
                 "lakebase_text uses its defaults (k1 1.2, b 0.75; ours is k1 1.5, b 0.75)."),
    })
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_case"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
