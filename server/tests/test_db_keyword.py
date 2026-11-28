"""Keyword search loaded from Postgres, small enough to fit (ROADMAP 2.3).

Done when: BM25 scores match the FAISS-era ones and its memory is recorded.
- `CompactBM25` gives `BM25`'s scores bit for bit: on made-up corpora with
  every awkward case, and on all 31,119 real chunks for every golden question.
- `KeywordIndex.load` builds the same index from Postgres, batch by batch.
- `tools.bm25_memory` recorded its memory at full size, inside 0.7's headroom.
"""

import json

import numpy as np
import pytest

from db.keyword import CompactBM25, KeywordIndex
from db.store import Filters
from utils import config
from vectorstore.keyword import BM25

pytestmark = pytest.mark.roadmap("2.3")

WORDS = ("revenue net sales services iphone the of and cloud margin 10-K Q3 $391.0 billion "
         "goodwill xenoberry fiscal 2025 2024 employees @contact a.b+c").split()


def made_up(seed, n=300):
    rng = np.random.default_rng(seed)
    docs = [" ".join(rng.choice(WORDS, size=int(rng.integers(0, 40)))) for _ in range(n)]
    docs[3] = ""                        # a chunk with no words
    docs[7] = "the of and a"            # only stopwords
    return docs


QUERIES = ["net sales revenue", "xenoberry xenoberry xenoberry",   # a repeated word counts each time
           "the of and", "", "nothing-here matches", "Q3 $391.0 billion", "@contact", "fiscal 2025 vs 2024"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_scores_equal_bm25_bit_for_bit(seed):
    docs = made_up(seed)
    old, new = BM25(docs), CompactBM25(iter(docs))   # the compact one takes a stream
    assert new.count == old.count and new.avg_length == old.avg_length
    for query in QUERIES:
        assert np.array_equal(old.scores(query), new.scores(query)), query
        assert new.scores(query).dtype == np.float32


def test_an_empty_corpus_scores_nothing():
    assert CompactBM25([]).scores("revenue").shape == (0,)


def test_scores_equal_bm25_on_every_real_chunk_for_every_golden_question():
    """The FAISS-era index is what the baseline ran on: same chunks, same order."""
    metadata = config.SNAPSHOT_INDEX_DIR / "v1" / "metadata.json"
    if not metadata.exists():
        pytest.skip(f"no FAISS index at {metadata.parent}: run  python -m ingestion.filings  (from server/)")
    texts = [c["text"] for c in json.loads(metadata.read_text(encoding="utf-8"))["chunks"]]
    old, new = BM25(texts), CompactBM25(texts)
    golden = config.PROJECT_ROOT / "evaluation" / "golden.jsonl"
    questions = [json.loads(line)["question"] for line in golden.read_text(encoding="utf-8").splitlines()]
    assert len(texts) == 31_119 and len(questions) == 101
    mismatched = [q for q in questions if not np.array_equal(old.scores(q), new.scores(q))]
    assert mismatched == []
    assert new.nbytes() < 30e6, "the index itself should be ~19 MB"


def test_memory_at_full_size_is_recorded_and_fits():
    path = config.PROJECT_ROOT / "evaluation" / "results" / "bm25_memory.json"
    assert path.exists(), "run  python -m tools.bm25_memory  (from server/)"
    report = json.loads(path.read_text(encoding="utf-8"))
    compact = report["results"]["compact"]
    assert compact["chunks"] == 31_119
    assert compact["peak_above_start_mb"] < report["headroom_mb"] == 75
    assert report["fits"] is True


def test_ids_can_come_in_any_order():
    index = KeywordIndex([30, 10, 20], ["cash cash", "cash and debt", "xenoberry pie"])
    assert index.positions([10, 20, 30]).tolist() == [1, 2, 0]
    assert index.scores("xenoberry", [20, 10, 30]).tolist()[1:] == [0, 0]
    with pytest.raises(LookupError, match="Rebuild it"):
        index.scores("cash", [10, 40])
    with pytest.raises(ValueError, match="2 ids but 3 texts"):
        KeywordIndex([1, 2], ["a", "b", "c"])


# --- Loading from Postgres -----------------------------------------------------

ACCESSIONS = ["0000320193-25-000079", "0000789019-25-000100"]


@pytest.fixture(scope="module")
def db(neon_migrated):
    conn = neon_migrated
    conn.execute("TRUNCATE filings, tables, eval_runs CASCADE")
    for i, accession in enumerate(ACCESSIONS):
        conn.execute(
            "INSERT INTO filings (accession, cik, ticker, company, form, fiscal_year, fiscal_period, filed,"
            " primary_document, sha256) VALUES (%s, %s, %s, 'Co', '10-K', 2025, 'FY', '2025-01-01', 'f.htm', %s)",
            (accession, f"000000000{i}", ["AAPL", "MSFT"][i], "c" * 64))
    docs = made_up(9, n=50)
    vector = np.ones(config.EMBED_DIM, dtype=np.float32) / np.sqrt(config.EMBED_DIM)
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (chunk_id, accession, ticker, cik, form, fiscal_year, fiscal_period, start_char,"
            " end_char, start_word, end_word, words, text, embedding)"
            " VALUES (%s, %s, %s, %s, '10-K', 2025, 'FY', 0, 1, 0, 1, %s, %s, %s)",
            [(f"c{n}", ACCESSIONS[n % 2], ["AAPL", "MSFT"][n % 2], f"000000000{n % 2}", len(t.split()), t, vector)
             for n, t in enumerate(docs)])
    return conn


def rows(conn, accessions=None):
    found = conn.execute("SELECT id, text, accession FROM chunks ORDER BY id").fetchall()
    return [(i, t) for i, t, a in found if accessions is None or a in accessions]


@pytest.mark.parametrize("batch", [7, 50, 5000])   # across batch edges, exactly one batch, one short batch
def test_load_from_postgres_equals_building_from_the_rows(db, batch):
    expected = rows(db)
    loaded = KeywordIndex.load(db, batch=batch)
    built = KeywordIndex([i for i, _ in expected], [t for _, t in expected])
    ids = [i for i, _ in expected]
    assert len(loaded) == len(expected) == 50
    for query in QUERIES:
        assert np.array_equal(loaded.scores(query, ids), built.scores(query, ids)), query


def test_load_respects_the_scope(db):
    scope = Filters(accessions=ACCESSIONS[:1])
    expected = rows(db, ACCESSIONS[:1])
    loaded = KeywordIndex.load(db, scope, batch=4)
    assert len(loaded) == len(expected) == 25
    ids = [i for i, _ in expected]
    assert np.array_equal(loaded.scores("net sales", ids),
                          KeywordIndex(ids, [t for _, t in expected]).scores("net sales", ids))
    outside = rows(db, ACCESSIONS[1:])[0][0]
    with pytest.raises(LookupError):
        loaded.scores("net sales", [outside])


def test_loading_nothing_gives_an_empty_index(db):
    empty = KeywordIndex.load(db, Filters(accessions=["0000000000-00-000000"]))
    assert len(empty) == 0
    assert empty.scores("net sales", []).shape == (0,)
