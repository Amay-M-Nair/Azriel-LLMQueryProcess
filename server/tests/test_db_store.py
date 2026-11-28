"""Search over Postgres gives what the FAISS store gives (ROADMAP 2.2).

Done when: store tests pass against a test database. The central check runs
the FAISS `VectorStore` and `PgVectorStore` over the same chunks and vectors
and asks for identical results, in every branch of the search: meaning only,
hybrid over every chunk, hybrid over a pool, the similarity floor with its
keyword rescue, and a store scoped to some filings. 2.6 repeats this at full
size on the golden set.
"""

import numpy as np
import pytest

from db.store import Filters, KeywordIndex, PgVectorStore, where
from ingestion.chunker import Chunk
from ingestion.filings import source_name
from utils import config
from vectorstore.faiss_store import VectorStore
from vectorstore.metadata_store import DocumentRecord

pytestmark = pytest.mark.roadmap("2.2")

FILINGS = [  # accession, ticker, cik, form, fiscal_year, fiscal_period
    ("0000320193-24-000123", "AAPL", "0000320193", "10-K", 2024, "FY"),
    ("0000320193-25-000071", "AAPL", "0000320193", "10-Q", 2025, "Q3"),
    ("0000320193-25-000079", "AAPL", "0000320193", "10-K", 2025, "FY"),
    ("0000950170-25-000100", "MSFT", "0000789019", "10-K", 2025, "FY"),
]
PER_FILING = 60
VOCAB = ("revenue net sales services iphone mac cloud azure margin operating income tax dividend "
         "repurchase segment americas europe china liquidity cash debt employees risk supply "
         "inflation currency lease goodwill impairment research development wearables").split()


def corpus(seed=7):
    """Chunks in ingest order, with unit vectors and a section on half of them."""
    rng = np.random.default_rng(seed)
    chunks, vectors, sections = [], [], []
    for accession, ticker, _cik, form, year, period in FILINGS:
        name = source_name({"ticker": ticker, "form": form, "fiscal_year": year, "fiscal_period": period})
        for n in range(PER_FILING):
            words = rng.choice(VOCAB, size=int(rng.integers(20, 60))).tolist()
            if n % 9 == 0:
                words += ["xenoberry"] * int(rng.integers(1, 4))   # a rare word BM25 should love
            text = " ".join(words)
            chunks.append(Chunk(
                text=text, source=name, page=1, chunk_id=f"{name.replace(' ', '_')}_chunk{n}",
                start_word=n * 40, end_word=n * 40 + len(words), start_char=n * 300, end_char=n * 300 + len(text),
                ticker=ticker, accession=accession, form=form, fiscal_year=year, fiscal_period=period,
            ))
            v = rng.normal(size=config.EMBED_DIM).astype("float32")
            vectors.append(v / np.linalg.norm(v))
            sections.append("MD&A" if n % 2 else None)
    return chunks, np.stack(vectors), sections


CHUNKS, VECTORS, SECTIONS = corpus()


def faiss_store(keep=lambda chunk: True) -> VectorStore:
    store = VectorStore()
    for name in dict.fromkeys(c.source for c in CHUNKS):
        rows = [i for i, c in enumerate(CHUNKS) if c.source == name and keep(c)]
        if rows:
            store.add(VECTORS[rows], [CHUNKS[i] for i in rows],
                      DocumentRecord(name=name, content_hash="x", pages=1, chunk_count=len(rows)))
    return store


def query_vector(near: int, noise: float, seed: int) -> np.ndarray:
    """A question vector close to chunk `near`, so similarities are realistic."""
    rng = np.random.default_rng(seed)
    v = VECTORS[near] + noise * rng.normal(size=config.EMBED_DIM).astype("float32") / np.sqrt(config.EMBED_DIM)
    return (v / np.linalg.norm(v)).astype("float32")


@pytest.fixture(scope="module")
def db(neon_migrated):
    conn = neon_migrated
    conn.execute("TRUNCATE filings, tables, eval_runs CASCADE")
    conn.execute(
        "INSERT INTO filings (accession, ticker, cik, company, form, fiscal_year, fiscal_period,"
        " filed, primary_document, sha256) VALUES " + ", ".join(["(%s, %s, %s, 'Co', %s, %s, %s,"
                                                                   " '2025-01-01', 'f.htm', %s)"] * len(FILINGS)),
        [x for f in FILINGS for x in (*f, "b" * 64)],
    )
    with conn.cursor() as cur:   # executemany pipelines the rows: a few round trips, not 240
        cur.executemany(
            "INSERT INTO chunks (chunk_id, accession, ticker, cik, form, fiscal_year, fiscal_period, section,"
            " page, start_char, end_char, start_word, end_word, words, text, embedding)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [(c.chunk_id, c.accession, c.ticker, {f[0]: f[2] for f in FILINGS}[c.accession], c.form,
              c.fiscal_year, c.fiscal_period, section, c.page, c.start_char, c.end_char, c.start_word,
              c.end_word, len(c.text.split()), c.text, v)
             for c, v, section in zip(CHUNKS, VECTORS, SECTIONS)],
        )
    return conn


def keyword_index(conn, scope: Filters | None = None) -> KeywordIndex:
    clause, params = where(scope)
    rows = conn.execute(f"SELECT id, text FROM chunks{clause} ORDER BY id", params).fetchall()
    return KeywordIndex([r[0] for r in rows], [r[1] for r in rows])


def assert_same(expected, got):
    assert [c.chunk_id for c, _ in got] == [c.chunk_id for c, _ in expected]
    assert [c for c, _ in got] == [c for c, _ in expected], "every Chunk field must survive Postgres"
    assert np.allclose([s for _, s in got], [s for _, s in expected], atol=1e-5)


# --- Filters and the keyword index (no database) ------------------------------

def test_filters_become_sql_conditions():
    assert where(Filters()) == ("", [])
    assert not Filters()
    clause, params = where(Filters(accessions=["a", "b"]), Filters(tickers=("AAPL",), fiscal_years=[2025]))
    assert clause == " WHERE accession = ANY(%s) AND ticker = ANY(%s) AND fiscal_year = ANY(%s)"
    assert params == [["a", "b"], ["AAPL"], [2025]]
    assert hash(Filters(forms=["10-K"])) == hash(Filters(forms=("10-K",))), "usable as a cache key"


def test_a_single_value_is_refused_rather_than_split_into_letters():
    with pytest.raises(TypeError, match="list of values"):
        Filters(tickers="AAPL")


def test_snapshot_scope_is_its_accessions():
    snapshot = {"filings": [{"accession": f[0]} for f in FILINGS]}
    assert Filters.snapshot(snapshot).accessions == tuple(f[0] for f in FILINGS)


def test_keyword_scores_are_bm25_by_id():
    index = KeywordIndex([10, 20, 30], ["cash and debt", "xenoberry pie", "cash cash"])
    assert index.scores("xenoberry", [20, 10]).tolist()[1] == 0
    assert index.scores("xenoberry", [20, 10])[0] > 0
    with pytest.raises(LookupError, match="Rebuild it"):
        index.scores("cash", [40])


# --- The same answers as FAISS -----------------------------------------------

CASES = [  # (query text, near chunk, noise, k, min_similarity)
    (None, 5, 1.0, 10, None),                                    # meaning only
    ("xenoberry revenue", 17, 1.0, 10, None),                    # hybrid, every chunk
    ("cloud azure margin", 130, 2.0, 5, None),
    ("xenoberry", 200, 3.0, 8, 0.4),                             # floor, with keyword rescue
    ("supply inflation currency", 61, 1.5, 10, 0.3),
    ("services wearables", 99, 0.5, 300, None),                  # k beyond the corpus
]


@pytest.mark.parametrize("text, near, noise, k, floor", CASES)
def test_search_matches_the_faiss_store(db, text, near, noise, k, floor):
    probe = query_vector(near, noise, seed=near)
    expected = faiss_store().search(probe, k, query=text, min_similarity=floor)
    got = PgVectorStore(db, keyword=keyword_index(db)).search(probe, k, query=text, min_similarity=floor)
    assert got, "the case should return something"
    assert_same(expected, got)


@pytest.mark.parametrize("text, near, noise, k, floor", CASES[1:5])
def test_search_over_a_pool_matches_the_faiss_store(db, monkeypatch, text, near, noise, k, floor):
    """Above HYBRID_EXACT_LIMIT both stores blend over a candidate pool."""
    monkeypatch.setattr(config, "HYBRID_EXACT_LIMIT", 100)
    probe = query_vector(near, noise, seed=near)
    expected = faiss_store().search(probe, k, query=text, min_similarity=floor)
    got = PgVectorStore(db, keyword=keyword_index(db)).search(probe, k, query=text, min_similarity=floor)
    assert_same(expected, got)


def test_a_scoped_store_matches_a_faiss_store_of_just_those_filings(db):
    """The evaluation's view: only the pinned filings, as if nothing else existed."""
    scope = Filters(accessions=[f[0] for f in FILINGS[:3]])
    expected_store = faiss_store(keep=lambda c: c.accession in scope.accessions)
    store = PgVectorStore(db, scope=scope, keyword=keyword_index(db, scope))

    assert len(store) == len(expected_store) == 3 * PER_FILING
    assert store.sources == expected_store.sources
    assert store.total_words == expected_store.total_words
    probe = query_vector(190, 1.0, seed=1)   # nearest to an MSFT chunk, which is out of scope
    assert_same(expected_store.search(probe, 10, query="xenoberry cash"),
                store.search(probe, 10, query="xenoberry cash"))


def test_the_whole_store_reports_what_faiss_reports(db):
    store, expected = PgVectorStore(db), faiss_store()
    assert (len(store), store.sources, store.total_words) == (len(expected), expected.sources,
                                                              expected.total_words)


# --- Filters ------------------------------------------------------------------

def test_a_q4_question_searches_the_10k_and_the_q3_10q(db):
    store = PgVectorStore(db, keyword=keyword_index(db))
    q4 = Filters(tickers=["AAPL"], fiscal_years=[2025], fiscal_periods=["FY", "Q3"])
    assert store.count(q4) == 2 * PER_FILING
    found = store.search(query_vector(0, 1.0, seed=3), 500, query="revenue", filters=q4)
    assert len(found) == 2 * PER_FILING
    assert {c.source for c, _ in found} == {"AAPL 10-K FY2025", "AAPL 10-Q Q32025"}


def test_filters_by_form_and_section(db):
    store = PgVectorStore(db)
    found = store.search(query_vector(0, 1.0, seed=4), 500, filters=Filters(forms=["10-Q"]))
    assert {c.form for c, _ in found} == {"10-Q"}
    mdna = Filters(sections=["MD&A"])
    assert store.count(mdna) == sum(s == "MD&A" for s in SECTIONS), "chunks with no section are left out"


def test_nothing_matching_returns_nothing(db):
    store = PgVectorStore(db, keyword=keyword_index(db))
    assert store.search(query_vector(0, 1.0, seed=5), 5, query="cash", filters=Filters(tickers=["JPM"])) == []
    empty = PgVectorStore(db, scope=Filters(accessions=["0000000000-00-000000"]))
    assert len(empty) == 0 and empty.sources == [] and empty.total_words == 0
