"""Baseline ingestion of the snapshot (ROADMAP 1.3).

Offline tests cover the minimal HTML loader, the chunk offsets and metadata,
and indexing a tiny made-up snapshot with a stand-in embedder. The last test
checks the real index in data/snapshot_index/ against evaluation/snapshot.json.
"""

import hashlib

import numpy as np
import pytest

from evaluation import snapshot as snap
from ingestion import document_loader, filings
from ingestion.chunker import chunk_pages
from ingestion.document_loader import html_text
from utils import config
from vectorstore.faiss_store import VectorStore

pytestmark = pytest.mark.roadmap("1.3")


# --- The minimal HTML loader ---------------------------------------------------------

def test_hidden_xbrl_facts_scripts_and_styles_are_not_text():
    text = html_text(
        "<html><head><title>t</title><style>p {color: red}</style><script>var x = 1;</script></head><body>"
        '<div style="display: none"><ix:header><ix:hidden>dei:FACT 0000320193</ix:hidden></ix:header></div>'
        "<p>Visible.</p></body></html>"
    )
    assert text == "Visible."


def test_entities_and_no_break_spaces_become_plain_text():
    assert html_text("<p>Management&#8217;s&#160;Discussion &amp; Analysis</p>") == \
        "Management’s Discussion & Analysis"


def test_each_block_is_a_line_and_inline_tags_are_not():
    assert html_text("<p>Revenue grew <b>8</b><span>%</span>.</p><div>Next<br/>line</div>") == \
        "Revenue grew 8%.\nNext\nline"


def test_a_table_row_is_one_line_with_its_cells_joined():
    text = html_text(
        "<table><tr><td><div><span>Net sales</span></div></td><td></td><td><p>$</p></td>"
        "<td>391,035</td></tr><tr><td>Cost of sales</td><td>(210,352)</td></tr></table>"
    )
    assert text == "Net sales | $ | 391,035\nCost of sales | (210,352)"


def test_a_stray_closing_tag_does_not_swallow_the_rest():
    assert html_text("<p>One</span></p><p>Two</p>") == "One\nTwo"


def test_htm_files_load_as_one_page(tmp_path):
    path = tmp_path / "filing.htm"
    path.write_bytes(b"<p>FORM 10-K</p><p>Item 1. Business</p>")
    assert document_loader.load_pages(path) == [(1, "FORM 10-K\nItem 1. Business")]


# --- Chunk offsets -------------------------------------------------------------------

def test_every_chunk_knows_where_it_sits_in_the_text():
    text = "\n".join(f"Paragraph {i}. " + "word " * (17 + i % 30) for i in range(120))
    chunks = chunk_pages([(1, text)], source="AAPL 10-K FY2026")
    assert len(chunks) > 5
    for chunk in chunks:
        assert chunk.start_char >= 0
        assert text[chunk.start_char:chunk.end_char] == chunk.text


# --- Indexing a snapshot -------------------------------------------------------------

def fake_embed(texts, **_):
    rng = np.random.default_rng(len(texts))
    vectors = rng.normal(size=(len(texts), config.EMBED_DIM)).astype("float32")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def document(words: int, tag: str) -> bytes:
    rows = "".join(f"<p>{tag} paragraph {i}. " + "revenue " * 40 + "</p>" for i in range(words // 40))
    return f"<html><body>{rows}</body></html>".encode()


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """Two companies, two filings each, written where the snapshot says they live."""
    monkeypatch.setattr(filings.embedder, "embed", fake_embed)
    cache = tmp_path / "filings"
    pinned = []
    for ticker in ("AAA", "BBB"):
        for n, (form, period) in enumerate((("10-Q", "Q1"), ("10-K", "FY"))):
            content = document(800, f"{ticker}-{period}")
            pinned.append({
                "ticker": ticker, "cik": "0000000001", "company": ticker, "form": form,
                "accession": f"{ticker}-{n}", "filing_date": "", "report_date": "",
                "primary_document": f"{ticker.lower()}-{period}.htm", "fiscal_year_end": "1231",
                "fiscal_year": 2025, "fiscal_period": period, "period_source": "cover",
                "sha256": hashlib.sha256(content).hexdigest(), "size": len(content),
            })
            path = cache / ticker / "2025" / period / f"{ticker.lower()}-{period}.htm"
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
    snapshot = {"version": 7, "companies": ["AAA", "BBB"], "years": 1, "filings": pinned}
    return snapshot, cache, tmp_path / "index"


def test_every_filing_is_indexed_with_its_metadata(tiny):
    snapshot, cache, directory = tiny
    store, report = filings.index_snapshot(snapshot, directory, cache)
    assert sorted(store.documents) == ["AAA 10-K FY2025", "AAA 10-Q Q12025", "BBB 10-K FY2025", "BBB 10-Q Q12025"]
    assert report.chunks_added == len(store) > 0
    for chunk in store.chunks:
        record = next(f for f in snapshot["filings"] if filings.source_name(f) == chunk.source)
        assert (chunk.ticker, chunk.accession, chunk.form, chunk.fiscal_year, chunk.fiscal_period) == (
            record["ticker"], record["accession"], record["form"], record["fiscal_year"], record["fiscal_period"])


def test_the_saved_index_keeps_the_metadata_and_a_manifest(tiny):
    snapshot, cache, directory = tiny
    filings.index_snapshot(snapshot, directory, cache)
    loaded = VectorStore.load(directory)
    assert loaded is not None and {c.fiscal_period for c in loaded.chunks} == {"Q1", "FY"}
    manifest = filings.load_manifest(directory)
    assert manifest["snapshot_version"] == 7 and manifest["filings"] == 4
    assert manifest["chunks"] == len(loaded) and manifest["embed_model"] == config.EMBED_MODEL


def test_chunk_offsets_point_into_the_filings_own_text(tiny):
    snapshot, cache, directory = tiny
    store, _ = filings.index_snapshot(snapshot, directory, cache)
    pinned = snapshot["filings"][0]
    text = filings.load_filing_text(snap.paths(snapshot, cache)[0], pinned["sha256"])
    for chunk in (c for c in store.chunks if c.accession == pinned["accession"]):
        assert text[chunk.start_char:chunk.end_char] == chunk.text


def test_a_rerun_indexes_nothing(tiny):
    snapshot, cache, directory = tiny
    first, _ = filings.index_snapshot(snapshot, directory, cache)
    again, report = filings.index_snapshot(snapshot, directory, cache)
    assert report.indexed == [] and len(report.skipped) == 4
    assert len(again) == len(first)


def test_an_interrupted_run_resumes_where_it_stopped(tiny, monkeypatch):
    snapshot, cache, directory = tiny
    calls = []

    def embed_then_fail(texts, **kw):
        calls.append(len(texts))
        if len(calls) == 3:               # the first filing of the second company
            raise KeyboardInterrupt
        return fake_embed(texts)

    monkeypatch.setattr(filings.embedder, "embed", embed_then_fail)
    with pytest.raises(KeyboardInterrupt):
        filings.index_snapshot(snapshot, directory, cache)
    assert sorted(VectorStore.load(directory).documents) == ["AAA 10-K FY2025", "AAA 10-Q Q12025"]

    monkeypatch.setattr(filings.embedder, "embed", fake_embed)
    store, report = filings.index_snapshot(snapshot, directory, cache)
    assert len(report.skipped) == 2 and len(report.indexed) == 2 and len(store.documents) == 4


def test_a_filing_that_is_not_the_pinned_one_is_refused(tiny):
    snapshot, cache, directory = tiny
    snap.paths(snapshot, cache)[0].write_bytes(b"<p>something else</p>")
    with pytest.raises(ValueError, match="pinned hash"):
        filings.index_snapshot(snapshot, directory, cache)


# --- The real index ----------------------------------------------------------------

def test_the_snapshot_is_indexed():
    snapshot = snap.load()
    directory = filings.index_dir(snapshot)
    store = VectorStore.load(directory)
    if store is None:
        pytest.skip(f"no index at {directory}: run  python -m ingestion.filings  (from server/)")

    manifest = filings.load_manifest(directory)
    assert manifest["snapshot_version"] == snapshot["version"]
    pinned = {filings.source_name(f): f for f in snapshot["filings"]}
    assert {n: r.content_hash for n, r in store.documents.items()} == {n: f["sha256"] for n, f in pinned.items()}

    per_document = {}
    for chunk in store.chunks:
        per_document[chunk.source] = per_document.get(chunk.source, 0) + 1
        f = pinned[chunk.source]
        assert (chunk.accession, chunk.form, chunk.fiscal_year, chunk.fiscal_period) == (
            f["accession"], f["form"], f["fiscal_year"], f["fiscal_period"]), chunk.chunk_id
        assert chunk.start_char >= 0, chunk.chunk_id
    assert all(per_document[n] >= 50 for n in pinned), "a filing yielded almost no chunks"

    smallest = min(snapshot["filings"], key=lambda f: f["size"])
    path = snap.paths({"filings": [smallest]})[0]
    text = filings.load_filing_text(path, smallest["sha256"])
    for chunk in (c for c in store.chunks if c.accession == smallest["accession"]):
        assert text[chunk.start_char:chunk.end_char] == chunk.text
