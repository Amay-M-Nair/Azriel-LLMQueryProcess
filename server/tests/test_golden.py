"""The golden set (ROADMAP 1.4): its schema, its checks, and the review that approves it.

The offline tests use a made-up filing. The last tests read the real files:
every drafted candidate must check out against the pinned filings, and the
"done when" (every approved case valid, 60-100 of them, at least 8 of each
type and 20 from 10-Qs, all verified by hand) is skipped, not failed, while
the review is still under way, so 1.4 stays unticked until it is finished.
"""

import json
from datetime import date

import pytest
from pydantic import ValidationError

from evaluation import golden
from evaluation.golden import GoldenCase, QuestionType

pytestmark = pytest.mark.roadmap("1.4")

ACCESSION = "0001234567-26-000001"
TEXT = "Net sales\nTotal net sales | 1,000 | 900\nThe Company had approximately 5,000 employees.\nTotal net sales | 1,000 | 900"
PINNED = {ACCESSION: {"accession": ACCESSION, "ticker": "NWND", "form": "10-Q"}}


def span(quote, start=None):
    start = TEXT.index(quote) if start is None else start
    return {"accession": ACCESSION, "start_char": start, "end_char": start + len(quote), "quote": quote}


def case(**overrides):
    base = {
        "id": "nwnd-employees", "type": "text_lookup",
        "question": "How many employees does Northwind have?",
        "answer": "About 5,000.",
        "spans": [span("approximately 5,000 employees")],
        "about": [{"ticker": "NWND", "fiscal_year": 2026, "fiscal_period": "FY"}],
    }
    return GoldenCase.model_validate(base | overrides)


# --- Schema ------------------------------------------------------------------------------

def test_a_well_formed_case_validates():
    assert case().type is QuestionType.TEXT_LOOKUP


def test_a_span_must_be_as_long_as_its_quote():
    with pytest.raises(ValidationError, match="not 5 characters"):
        case(spans=[{"accession": ACCESSION, "start_char": 0, "end_char": 9, "quote": "Net s"}])


def test_unanswerable_cases_cite_nothing_and_the_rest_cite_something():
    with pytest.raises(ValidationError, match="unanswerable"):
        case(type="unanswerable")
    with pytest.raises(ValidationError, match="unanswerable"):
        case(spans=[])
    assert case(type="unanswerable", spans=[]).spans == []


def test_only_follow_ups_have_history():
    turn = [{"question": "How big is Northwind?", "answer": "Mid-sized."}]
    with pytest.raises(ValidationError, match="follow-up"):
        case(type="follow_up")
    with pytest.raises(ValidationError, match="follow-up"):
        case(history=turn)
    assert case(type="follow_up", history=turn).history


def test_a_cross_period_case_is_about_two_periods():
    one = [{"ticker": "NWND", "fiscal_year": 2026, "fiscal_period": "FY"}]
    with pytest.raises(ValidationError, match="two periods"):
        case(type="cross_period", about=one)
    two = one + [{"ticker": "NWND", "fiscal_year": 2025, "fiscal_period": "FY"}]
    assert len(case(type="cross_period", about=two).about) == 2


def test_a_summary_draws_on_two_spans():
    with pytest.raises(ValidationError, match="two spans"):
        case(type="summary")


def test_unknown_fields_and_bad_ids_are_refused():
    with pytest.raises(ValidationError):
        case(confidence=0.9)
    with pytest.raises(ValidationError):
        case(id="Not An Id")


def test_cases_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "golden.jsonl"
    cases = [case(), case(id="nwnd-other", verified={"by": "Reviewer", "on": "2026-11-26"})]
    golden.save(cases, path)
    assert golden.load(path) == cases
    assert golden.load(path)[1].verified.on == date(2026, 11, 26)


def test_a_bad_line_is_reported_with_its_number(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(case().model_dump_json() + "\n{\"id\": \"broken\"}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        golden.load(path)


# --- Checks against the filings ------------------------------------------------------------

def test_a_quote_that_is_not_at_its_offsets_is_a_problem():
    moved = case(spans=[span("approximately 5,000 employees", start=0)])
    assert golden.problems([moved], text_of=lambda a: TEXT, pinned=PINNED)
    assert golden.problems([case()], text_of=lambda a: TEXT, pinned=PINNED) == []


def test_unknown_filings_companies_and_repeated_ids_are_problems():
    stranger = case(spans=[{**span("approximately 5,000 employees"), "accession": "0009999999-26-000001"}])
    elsewhere = case(id="nwnd-x", about=[{"ticker": "ZZZ", "fiscal_year": 2026}])
    found = golden.problems([stranger, elsewhere, case(id="nwnd-x")], text_of=lambda a: TEXT, pinned=PINNED)
    assert any("not in the snapshot" in p for p in found)
    assert any("ZZZ" in p for p in found)
    assert any("used 2 times" in p for p in found)


def test_coverage_counts_types_10q_cases_and_verification():
    cases = [case(), case(id="nwnd-none", type="unanswerable", spans=[],
                          verified={"by": "Reviewer", "on": "2026-11-26"})]
    cov = golden.coverage(cases, pinned=PINNED)
    assert cov["cases"] == 2 and cov["from_10q"] == 1 and cov["verified"] == 1
    assert cov["by_type"]["text_lookup"] == 1 and cov["by_type"]["unanswerable"] == 1


def test_shortfalls_name_what_is_missing():
    gaps = golden.shortfalls([case()], pinned=PINNED)
    assert any("1 cases" in g for g in gaps)
    assert any("summary" in g for g in gaps)
    assert any("not verified" in g for g in gaps)


def test_locate_finds_a_unique_quote_and_refuses_an_ambiguous_one(monkeypatch):
    monkeypatch.setattr(golden, "_pinned", lambda: PINNED)
    monkeypatch.setattr(golden, "filing_text", lambda a: TEXT)
    draft = {"id": "x", "spans": [{"accession": ACCESSION, "quote": "approximately 5,000 employees"}]}
    located = golden.locate(draft)["spans"][0]
    assert TEXT[located["start_char"]:located["end_char"]] == "approximately 5,000 employees"
    with pytest.raises(ValueError, match="more than once"):
        golden.locate({"id": "x", "spans": [{"accession": ACCESSION, "quote": "Total net sales | 1,000"}]})
    with pytest.raises(ValueError, match="not found"):
        golden.locate({"id": "x", "spans": [{"accession": ACCESSION, "quote": "dividend"}]})


# --- The review page's writes --------------------------------------------------------------

@pytest.fixture
def review(tmp_path, monkeypatch):
    from evaluation import review_app

    monkeypatch.setattr(golden, "GOLDEN", tmp_path / "golden.jsonl")
    monkeypatch.setattr(golden, "CANDIDATES", tmp_path / "golden_candidates.jsonl")
    monkeypatch.setattr(review_app, "REJECTED", tmp_path / "golden_rejected.jsonl")
    golden.save([case(id="nwnd-a"), case(id="nwnd-b")], golden.CANDIDATES)
    return review_app


def test_approving_writes_the_case_with_the_reviewer_and_any_corrected_answer(review):
    review.approve(case(id="nwnd-b"), "About 5,000 people.", "Reviewer", "checked")
    approved = golden.load(golden.GOLDEN)
    assert [c.id for c in approved] == ["nwnd-b"]
    assert approved[0].answer == "About 5,000 people."
    assert approved[0].verified.by == "Reviewer" and approved[0].verified.note == "checked"
    assert golden.load(golden.CANDIDATES)[1].answer == "About 5,000.", "the candidates file is never changed"


def test_approved_cases_keep_the_candidates_order(review):
    review.approve(case(id="nwnd-b"), "b", "Reviewer", "")
    review.approve(case(id="nwnd-a"), "a", "Reviewer", "")
    assert [c.id for c in golden.load(golden.GOLDEN)] == ["nwnd-a", "nwnd-b"]


def test_rejecting_records_the_reason_and_undoes_an_approval(review):
    review.approve(case(id="nwnd-a"), "a", "Reviewer", "")
    review.reject(case(id="nwnd-a"), "Reviewer", "the figure is the wrong column")
    assert golden.load(golden.GOLDEN) == []
    row = json.loads(review.REJECTED.read_text(encoding="utf-8"))
    assert row["id"] == "nwnd-a" and row["note"] == "the figure is the wrong column"
    review.approve(case(id="nwnd-a"), "a", "Reviewer", "")
    assert review.rejected() == {}, "approving after rejecting clears the rejection"


# --- The real files ------------------------------------------------------------------------

def test_every_drafted_candidate_checks_out_against_the_filings():
    candidates = golden.load(golden.CANDIDATES)
    assert candidates, "no candidates drafted"
    try:
        found = golden.problems(candidates)
    except FileNotFoundError:
        pytest.skip("the pinned filings are not cached: run  python -m evaluation.snapshot")
    assert found == []


def test_the_golden_set_is_complete_and_verified():
    cases = golden.load(golden.GOLDEN)
    try:
        found = golden.problems(cases)
    except FileNotFoundError:
        pytest.skip("the pinned filings are not cached: run  python -m evaluation.snapshot")
    assert found == [], "approved cases that do not check out"
    gaps = golden.shortfalls(cases)
    if gaps:
        pytest.skip("golden set not finished: " + "; ".join(gaps))
