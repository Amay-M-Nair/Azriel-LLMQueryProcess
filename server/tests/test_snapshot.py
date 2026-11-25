"""The pinned snapshot (ROADMAP 1.2): how it picks, and that it reproduces.

The first half runs offline against a fictional company (NWND, calendar fiscal
year) served by `httpx.MockTransport`. The last test checks the real
evaluation/snapshot.json against the filings cached in data/filings/.
"""

import hashlib
import json
from datetime import date

import httpx
import pytest

from evaluation import snapshot as snap
from ingestion import edgar
from ingestion.edgar import EdgarClient, Throttle
from utils import config

pytestmark = pytest.mark.roadmap("1.2")

AGENT = "Test Runner tests@example.com"
CIK = "0001234567"
TODAY = date(2026, 11, 25)

# (fiscal year, period, report date, filing date) for a calendar-year filer.
# 2022 to 2025 are complete; 2026 has its three 10-Qs and no 10-K yet.
QUARTERS = {"Q1": ("03-31", "05-01"), "Q2": ("06-30", "08-01"), "Q3": ("09-30", "11-01")}


def history(years=range(2022, 2027), drop=()):
    rows = []
    for fy in years:
        for period, (end, filed) in QUARTERS.items():
            rows.append((fy, period, f"{fy}-{end}", f"{fy}-{filed}"))
        if fy < 2026:
            rows.append((fy, "FY", f"{fy}-12-31", f"{fy + 1}-02-15"))
    return [r for r in rows if (r[0], r[1]) not in drop]


def document(fy, period, form, salt=""):
    return (f'<html><body><ix:nonNumeric name="dei:DocumentFiscalYearFocus">{fy}</ix:nonNumeric>'
            f'<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus">{period}</ix:nonNumeric>'
            f'<p>{form} for fiscal {fy} {period}{salt}</p></body></html>').encode()


class FakeEdgar:
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: r[3], reverse=True)
        self.documents = {}
        recent = {k: [] for k in ("accessionNumber", "filingDate", "reportDate", "form", "primaryDocument")}
        for n, (fy, period, report, filed) in enumerate(self.rows):
            form = "10-K" if period == "FY" else "10-Q"
            accession = f"{CIK}-{filed[2:4]}-{n:06d}"
            name = f"nwnd-{report.replace('-', '')}.htm"
            for key, value in zip(recent, (accession, filed, report, form, name)):
                recent[key].append(value)
            url = f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{accession.replace('-', '')}/{name}"
            self.documents[url] = document(fy, period, form)
        self.submissions = {"name": "Northwind Instruments Inc.", "fiscalYearEnd": "1231",
                            "filings": {"recent": recent, "files": []}}

    def handler(self, request):
        url = str(request.url)
        if url == edgar.TICKERS_URL:
            return httpx.Response(200, json={"0": {"cik_str": int(CIK), "ticker": "NWND", "title": "Northwind"}})
        if url == edgar.SUBMISSIONS_URL.format(cik=CIK):
            return httpx.Response(200, json=self.submissions)
        if url in self.documents:
            return httpx.Response(200, content=self.documents[url])
        return httpx.Response(404)


def client_for(sec, cache_dir):
    return EdgarClient(agent=AGENT, cache_dir=cache_dir, transport=httpx.MockTransport(sec.handler),
                       throttle=Throttle(1000, sleep=lambda s: None), sleep=lambda s: None)


def built(sec, cache_dir):
    with client_for(sec, cache_dir) as client:
        return snap.with_version(snap.build(client, ("NWND",), today=TODAY), None, today=TODAY)


# --- Picking ---------------------------------------------------------------------------

def test_the_three_newest_complete_years_are_pinned(tmp_path):
    snapshot = built(FakeEdgar(history()), tmp_path)
    picked = [(f["fiscal_year"], f["fiscal_period"]) for f in snapshot["filings"]]
    assert picked == [(y, p) for y in (2023, 2024, 2025) for p in snap.PERIODS]
    assert {f["form"] for f in snapshot["filings"] if f["fiscal_period"] == "FY"} == {"10-K"}


def test_a_year_still_in_progress_is_left_out(tmp_path):
    snapshot = built(FakeEdgar(history()), tmp_path)
    assert 2026 not in {f["fiscal_year"] for f in snapshot["filings"]}


def test_a_year_missing_a_10q_is_not_complete(tmp_path):
    snapshot = built(FakeEdgar(history(drop={(2024, "Q2")})), tmp_path)
    assert sorted({f["fiscal_year"] for f in snapshot["filings"]}) == [2022, 2023, 2025]


def test_too_few_complete_years_is_an_error(tmp_path):
    with pytest.raises(snap.SnapshotError, match="only 2 complete"):
        built(FakeEdgar(history(years=range(2024, 2027))), tmp_path)


def test_quarters_out_of_order_mean_a_misread_period():
    def download(period, report):
        filing = edgar.Filing("NWND", CIK, "N", "10-K" if period == "FY" else "10-Q", period, "", report,
                              "d.htm", "1231")
        return edgar.Download(filing, 2025, period, "cover", None, "", 0)

    periods = {"Q1": download("Q1", "2025-03-31"), "Q2": download("Q2", "2025-09-30"),
               "Q3": download("Q3", "2025-06-30"), "FY": download("FY", "2025-12-31")}
    assert snap.order_problems(2025, periods)


def test_every_pin_carries_what_reproducing_it_needs(tmp_path):
    pinned = built(FakeEdgar(history()), tmp_path)["filings"][0]
    for field in ("accession", "sha256", "primary_document", "cik", "fiscal_year", "fiscal_period"):
        assert pinned[field]
    assert snap.filing_of(pinned).url.startswith("https://www.sec.gov/Archives/edgar/data/1234567/")


# --- Versions ------------------------------------------------------------------------

def test_the_first_snapshot_is_version_one(tmp_path):
    assert built(FakeEdgar(history()), tmp_path)["version"] == 1


def test_rebuilding_the_same_filings_keeps_the_version(tmp_path):
    old = built(FakeEdgar(history()), tmp_path)
    with client_for(FakeEdgar(history()), tmp_path) as client:
        again = snap.with_version(snap.build(client, ("NWND",), today=TODAY), old)
    assert again is old


def test_a_new_complete_year_bumps_the_version(tmp_path):
    old = built(FakeEdgar(history()), tmp_path)
    rows = history() + [(2026, "FY", "2026-12-31", "2027-02-15")]
    with client_for(FakeEdgar(rows), tmp_path / "later") as client:
        new = snap.with_version(snap.build(client, ("NWND",), today=date(2027, 3, 1)), old)
    assert new["version"] == 2
    assert sorted({f["fiscal_year"] for f in new["filings"]}) == [2024, 2025, 2026]


# --- Reproducing -----------------------------------------------------------------------

def test_fetching_into_an_empty_folder_reproduces_every_file(tmp_path):
    sec = FakeEdgar(history())
    snapshot = built(sec, tmp_path / "first")
    with client_for(sec, tmp_path / "fresh") as client:
        checks = snap.fetch(client, snapshot)
    assert len(checks) == 12 and all(c.ok for c in checks)
    for path, pinned in zip(snap.paths(snapshot, tmp_path / "fresh"), snapshot["filings"]):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pinned["sha256"]


def test_a_filing_that_changed_upstream_is_reported(tmp_path):
    sec = FakeEdgar(history())
    snapshot = built(sec, tmp_path / "first")
    url = snap.filing_of(snapshot["filings"][0]).url
    sec.documents[url] = document(2023, "Q1", "10-Q", salt=" (restated)")
    with client_for(sec, tmp_path / "fresh") as client:
        checks = snap.fetch(client, snapshot)
    bad = [c for c in checks if not c.ok]
    assert len(bad) == 1 and "sha256" in bad[0].problem


def test_a_snapshot_round_trips_through_its_file(tmp_path):
    snapshot = built(FakeEdgar(history()), tmp_path)
    snap.save(snapshot, tmp_path / "snapshot.json")
    assert snap.load(tmp_path / "snapshot.json") == snapshot


# --- The real snapshot -----------------------------------------------------------------

def test_the_real_snapshot_is_well_formed_and_reproduces_from_the_cache():
    snapshot = snap.load()
    filings = snapshot["filings"]
    assert snapshot["version"] >= 1
    assert 5 <= len(snapshot["companies"]) <= 8
    assert len({f["accession"] for f in filings}) == len(filings), "an accession is pinned twice"
    for ticker in snapshot["companies"]:
        rows = [(f["fiscal_year"], f["fiscal_period"], f["form"]) for f in filings if f["ticker"] == ticker]
        years = sorted({y for y, _, _ in rows})
        assert len(years) == snapshot["years"] == 3, f"{ticker}: {years}"
        assert sorted(rows) == sorted((y, p, "10-K" if p == "FY" else "10-Q")
                                      for y in years for p in snap.PERIODS), ticker

    missing = [p for p in snap.paths(snapshot) if not p.exists()]
    if missing:
        pytest.skip(f"{len(missing)} pinned filings are not cached: run  python -m evaluation.snapshot")
    for path, pinned in zip(snap.paths(snapshot), filings):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pinned["sha256"], path


def test_the_snapshot_file_is_in_the_repository_not_in_data():
    assert snap.SNAPSHOT.parent == config.PROJECT_ROOT / "evaluation"
    assert json.loads(snap.SNAPSHOT.read_text(encoding="utf-8"))["filings"]
