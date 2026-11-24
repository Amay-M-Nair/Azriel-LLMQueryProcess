"""The EDGAR fetcher (ROADMAP 1.1): what it asks for, what it keeps, how it fails.

No network. `httpx.MockTransport` answers every request from tests/fixtures/edgar,
which is shaped like EDGAR's responses and trimmed to the fields the fetcher
reads. The company in it, Northwind Instruments (NWND), is fictional; its
fiscal year ends in late September, so nothing here can pass by assuming a
December year end.

tests/test_edgar_live.py is the other half of 1.1: real filings, downloaded.
"""

import json
import logging
import os
from datetime import date
from pathlib import Path

import httpx
import pytest

from ingestion import edgar
from ingestion.edgar import EdgarClient, EdgarError, EdgarNotReady, Throttle
from utils import config, logs

pytestmark = pytest.mark.roadmap("1.1")

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"
AGENT = "Test Runner tests@example.com"


def filing_document(form: str, fiscal_year: str | None, period: str | None) -> bytes:
    """A primary document, cut down to the inline-XBRL cover facts that matter."""
    if fiscal_year is None:                          # older than inline XBRL: no tags at all
        return f"<html><body><p>FORM {form}</p><p>Annual report</p></body></html>".encode()
    return f"""<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>
<div style="display:none"><ix:header><ix:hidden>
<ix:nonNumeric contextRef="c-1" name="dei:DocumentFiscalYearFocus">{fiscal_year}</ix:nonNumeric>
<ix:nonNumeric name='dei:DocumentFiscalPeriodFocus' contextRef="c-1"><span style="x">{period}</span></ix:nonNumeric>
</ix:hidden></ix:header></div>
<p>FORM <ix:nonNumeric contextRef="c-1" name="dei:DocumentType">{form}</ix:nonNumeric></p>
<p>Revenue &amp; other income</p>
</body></html>""".encode()


ARCHIVE = "https://www.sec.gov/Archives/edgar/data/1234567"
DOCUMENTS = {
    f"{ARCHIVE}/000123456726000031/nwnd-20260627.htm": filing_document("10-Q", "2026", "Q3"),
    f"{ARCHIVE}/000123456726000012/nwnd-20260328.htm": filing_document("10-Q", "2026", "Q2"),
    f"{ARCHIVE}/000123456726000004/nwnd-20251227.htm": filing_document("10-Q", "2026", "Q1"),
    f"{ARCHIVE}/000123456725000051/nwnd-20250927.htm": filing_document("10-K", "2025", "FY"),
    f"{ARCHIVE}/000123456725000033/nwnd-20250628.htm": filing_document("10-Q", "2025", "Q3"),
    f"{ARCHIVE}/000123456724000049/nwnd-20240928.htm": filing_document("10-K", None, None),
}


class Edgar:
    """A stand-in for EDGAR: serves the fixtures and records every request.

    `script` lets a test put responses in front of a URL's real one, e.g.
    [503, 200] to make the first try fail.
    """

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.script: dict[str, list] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        queued = self.script.get(url)
        if queued:
            outcome = queued.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, httpx.Response):
                return outcome
        if url == edgar.TICKERS_URL:
            return httpx.Response(200, content=(FIXTURES / "company_tickers.json").read_bytes())
        if url.startswith("https://data.sec.gov/submissions/"):
            path = FIXTURES / url.rsplit("/", 1)[1]
            if path.exists():
                return httpx.Response(200, content=path.read_bytes())
        if url in DOCUMENTS:
            return httpx.Response(200, content=DOCUMENTS[url])
        return httpx.Response(404)

    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


@pytest.fixture
def sec():
    return Edgar()


@pytest.fixture
def slept():
    return []


@pytest.fixture
def captured():
    """Every line the groundedqa loggers write during a test, as a handler would see it.

    caplog cannot see these: logs.setup() stops them propagating to the root.
    """
    logs.setup()
    logger = logging.getLogger("groundedqa")
    lines = []

    class Collect(logging.Handler):
        def emit(self, record):
            lines.append(self.format(record))

    handler = Collect()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        yield lines
    finally:
        logger.removeHandler(handler)


@pytest.fixture
def client(sec, slept, tmp_path):
    with EdgarClient(agent=AGENT, cache_dir=tmp_path, transport=httpx.MockTransport(sec.handler),
                     throttle=Throttle(1000, sleep=lambda s: None), sleep=slept.append) as c:
        yield c


# --- The User-Agent ------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", "   ", "Jane Doe", "jane@example.com", "Jane Doe jane@example"])
def test_a_user_agent_without_a_name_and_an_email_is_refused(monkeypatch, value):
    monkeypatch.setenv("SEC_USER_AGENT", value)
    with pytest.raises(EdgarNotReady, match="SEC_USER_AGENT"):
        EdgarClient()


@pytest.mark.parametrize("value", ["Jane Doe jane@example.com", "jane@example.com Jane Doe",
                                   "Example Research LLC ops@example.co.uk"])
def test_a_name_and_an_email_is_accepted(monkeypatch, value):
    monkeypatch.setenv("SEC_USER_AGENT", value)
    assert edgar.user_agent() == value


def test_every_request_names_who_is_making_it(client, sec):
    client.download(client.list_filings("NWND")[0])
    assert len(sec.requests) == 3
    assert all(r.headers["User-Agent"] == AGENT for r in sec.requests)


def test_the_user_agent_never_reaches_the_log(monkeypatch, captured):
    monkeypatch.setenv("SEC_USER_AGENT", AGENT)
    edgar.log.warning("refused, agent was %s", AGENT)
    assert captured and not any("tests@example.com" in line for line in captured)


# --- Ticker -> CIK -------------------------------------------------------------------

def test_a_ticker_becomes_a_ten_digit_cik(client):
    assert client.cik("NWND") == "0001234567"
    assert client.cik(" nwnd ") == "0001234567"


def test_share_classes_are_written_the_way_edgar_writes_them(client):
    assert client.cik("EXH.B") == client.cik("exh-b") == "0007654321"


def test_an_unknown_ticker_says_so(client):
    with pytest.raises(EdgarError, match="ZZZZ"):
        client.cik("ZZZZ")


def test_the_ticker_map_is_fetched_once_and_then_read_from_disk(client, sec):
    client.cik("NWND")
    client.cik("EXH-B")
    assert sec.urls().count(edgar.TICKERS_URL) == 1


def test_a_stale_ticker_map_is_fetched_again(client, sec, tmp_path):
    client.cik("NWND")
    week_old = (config.EDGAR_TICKER_MAP_MAX_AGE_DAYS + 1) * 86400
    path = tmp_path / "company_tickers.json"
    os.utime(path, (path.stat().st_atime - week_old, path.stat().st_mtime - week_old))
    client.cik("NWND")
    assert sec.urls().count(edgar.TICKERS_URL) == 2


# --- Listing -------------------------------------------------------------------------

def test_only_10k_and_10q_are_listed_newest_first(client):
    filings = client.list_filings("NWND")
    assert [(f.form, f.report_date) for f in filings] == [
        ("10-Q", "2026-06-27"), ("10-Q", "2026-03-28"), ("10-Q", "2025-12-27"), ("10-K", "2025-09-27"),
    ]


def test_amendments_are_listed_in_the_log_but_not_fetched(client, captured):
    filings = client.list_filings("NWND")
    assert all(f.form in edgar.FORMS for f in filings)
    assert any("1 amendment(s)" in line for line in captured)


def test_forms_can_be_narrowed(client):
    assert [f.form for f in client.list_filings("NWND", forms=("10-K",))] == ["10-K"]


def test_without_since_only_the_recent_block_is_read(client, sec):
    client.list_filings("NWND")
    assert not any("submissions-0" in u for u in sec.urls())


def test_since_reads_older_pages_but_only_those_that_reach_back_far_enough(client, sec):
    filings = client.list_filings("NWND", since=date(2024, 10, 1))
    assert [f.report_date for f in filings][-2:] == ["2025-06-28", "2024-09-28"]
    assert any(u.endswith("submissions-001.json") for u in sec.urls())
    assert not any(u.endswith("submissions-002.json") for u in sec.urls()), "a page from before `since`"


def test_since_drops_filings_older_than_it(client):
    filings = client.list_filings("NWND", since=date(2025, 1, 1))
    assert min(f.filing_date for f in filings) >= "2025-01-01"


def test_a_filing_knows_its_archive_url(client):
    tenk = client.list_filings("NWND", forms=("10-K",))[0]
    assert tenk.url == f"{ARCHIVE}/000123456725000051/nwnd-20250927.htm"


# --- Downloading and the cache ---------------------------------------------------------

def test_a_download_is_filed_under_the_period_its_cover_declares(client, tmp_path):
    q3, q2, q1, tenk = client.list_filings("NWND")
    done = client.download(q3)
    assert (done.fiscal_year, done.fiscal_period, done.period_source) == (2026, "Q3", "cover")
    assert done.path == tmp_path / "NWND" / "2026" / "Q3" / "nwnd-20260627.htm"
    assert done.path.read_bytes() == DOCUMENTS[q3.url]

    # Filed in calendar 2025, but the cover says fiscal 2025 FY: the folder follows the cover.
    assert client.download(tenk).path.parent == tmp_path / "NWND" / "2025" / "FY"
    # Filed in calendar 2026 for a quarter ending December 2025: fiscal 2026 Q1.
    assert client.download(q1).path.parent == tmp_path / "NWND" / "2026" / "Q1"


def test_the_record_beside_each_filing_carries_its_metadata_and_hash(client):
    done = client.download(client.list_filings("NWND")[0])
    record = json.loads((done.path.parent / "filing.json").read_text(encoding="utf-8"))
    assert record["accession"] == "0001234567-26-000031"
    assert record["form"] == "10-Q"
    assert (record["fiscal_year"], record["fiscal_period"]) == (2026, "Q3")
    assert record["sha256"] == done.sha256 == edgar._sha256(done.path.read_bytes())
    assert record["fetched_at"]


def test_a_cached_filing_is_not_fetched_again(client, sec):
    filing = client.list_filings("NWND")[0]
    client.download(filing)
    before = len(sec.requests)
    assert client.download(filing).fiscal_period == "Q3"
    assert len(sec.requests) == before


def test_a_cached_filing_that_changed_on_disk_is_fetched_again(client, sec):
    filing = client.list_filings("NWND")[0]
    done = client.download(filing)
    done.path.write_bytes(b"tampered")
    again = client.download(filing)
    assert sec.urls().count(filing.url) == 2
    assert again.path.read_bytes() == DOCUMENTS[filing.url]


def test_no_half_written_files_are_left_behind(client, tmp_path):
    client.download(client.list_filings("NWND")[0])
    assert not list(tmp_path.rglob("*.part"))


def test_latest_lists_only_what_is_not_cached(client):
    q3, q2, q1, tenk = client.list_filings("NWND")
    client.download(tenk)
    client.download(q1)
    assert [f.accession for f in client.latest("NWND")] == [q3.accession, q2.accession]
    for filing in client.latest("NWND"):
        client.download(filing)
    assert client.latest("NWND") == []


def test_two_filings_claiming_the_same_period_is_an_error(client, sec):
    q3, q2, *_ = client.list_filings("NWND")
    client.download(q3)
    sec.script[q2.url] = [httpx.Response(200, content=filing_document("10-Q", "2026", "Q3"))]
    with pytest.raises(EdgarError, match="misread"):
        client.download(q2)


# --- Working out the fiscal period -----------------------------------------------------

def test_a_filing_without_cover_tags_is_placed_by_dates_and_says_so(client, captured):
    old_10k = client.list_filings("NWND", since=date(2024, 10, 1))[-1]
    done = client.download(old_10k)
    assert (done.fiscal_year, done.fiscal_period, done.period_source) == (2024, "FY", "dates")
    assert any("placed by dates" in line for line in captured)


@pytest.mark.parametrize("form, report_date, year_end, expected", [
    # Fiscal year ending late September (Apple-like), 52/53-week dates.
    ("10-K", "2025-09-27", "0927", (2025, "FY")),
    ("10-K", "2024-09-28", "0927", (2024, "FY")),
    ("10-Q", "2024-12-28", "0927", (2025, "Q1")),
    ("10-Q", "2025-03-29", "0927", (2025, "Q2")),
    ("10-Q", "2025-06-28", "0927", (2025, "Q3")),
    # Calendar fiscal year.
    ("10-K", "2025-12-31", "1231", (2025, "FY")),
    ("10-Q", "2025-03-31", "1231", (2025, "Q1")),
    ("10-Q", "2025-09-30", "1231", (2025, "Q3")),
    # A 52/53-week year that runs into the first days of January is named for the year before.
    ("10-K", "2026-01-03", "0103", (2025, "FY")),
    ("10-Q", "2025-04-05", "0103", (2025, "Q1")),
    # A late-January year end is named for the year it ends in.
    ("10-K", "2026-01-31", "0131", (2026, "FY")),
    ("10-Q", "2025-10-31", "0131", (2026, "Q3")),
])
def test_period_from_dates(form, report_date, year_end, expected):
    assert edgar.period_from_dates(form, report_date, year_end) == expected


def test_period_from_dates_needs_a_fiscal_year_end():
    with pytest.raises(EdgarError, match="fiscal year end"):
        edgar.period_from_dates("10-Q", "2025-06-30", "")


def test_cover_facts_are_read_through_formatting_and_either_quote_style():
    facts = edgar.cover_facts(filing_document("10-Q", "2026", "Q2").decode())
    assert facts == {"DocumentType": "10-Q", "DocumentFiscalYearFocus": "2026", "DocumentFiscalPeriodFocus": "Q2"}


def test_a_10q_cover_with_an_impossible_period_falls_back_to_dates():
    filing = edgar.Filing("NWND", "0001234567", "Northwind", "10-Q", "x", "2025-08-01", "2025-06-28",
                          "d.htm", "0927")
    assert edgar.fiscal_period(filing, filing_document("10-Q", "2025", "FY").decode()) == (2025, "Q3", "dates")


# --- The SEC's rules: rate, retries, refusals --------------------------------------------

def test_requests_are_spaced_to_at_most_ten_a_second():
    now = [0.0]
    stamps = []

    def sleep(seconds):
        now[0] += seconds

    throttle = Throttle(config.EDGAR_MAX_REQUESTS_PER_SECOND, clock=lambda: now[0], sleep=sleep)
    for _ in range(25):
        throttle.wait()
        stamps.append(now[0])
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= 0.1 - 1e-9
    assert sum(1 for t in stamps if t < 1.0) <= 10


def test_the_client_uses_the_configured_rate_by_default(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", AGENT)
    with EdgarClient() as c:
        assert c.throttle.interval >= 1 / config.EDGAR_MAX_REQUESTS_PER_SECOND


def test_a_server_error_is_retried_with_backoff(client, sec, slept):
    sec.script[edgar.TICKERS_URL] = [httpx.Response(503), httpx.Response(502)]
    assert client.cik("NWND") == "0001234567"
    assert slept == [config.EDGAR_RETRY_BACKOFF, config.EDGAR_RETRY_BACKOFF * 2]


def test_retry_after_is_honoured_and_capped(client, sec, slept):
    sec.script[edgar.TICKERS_URL] = [httpx.Response(429, headers={"Retry-After": "3"}),
                                     httpx.Response(429, headers={"Retry-After": "9999"})]
    client.cik("NWND")
    assert slept == [3.0, config.EDGAR_MAX_WAIT]


def test_a_dropped_connection_is_retried(client, sec, slept):
    sec.script[edgar.TICKERS_URL] = [httpx.ConnectError("reset")]
    assert client.cik("NWND") == "0001234567"
    assert len(slept) == 1


def test_retries_give_up_with_the_reason(client, sec):
    sec.script[edgar.TICKERS_URL] = [httpx.Response(503)] * config.EDGAR_RETRIES
    with pytest.raises(EdgarError, match=f"after {config.EDGAR_RETRIES} tries.*HTTP 503"):
        client.cik("NWND")


def test_a_403_is_never_retried_and_points_at_the_user_agent(client, sec, slept):
    sec.script[edgar.TICKERS_URL] = [httpx.Response(403)]
    with pytest.raises(EdgarError, match="SEC_USER_AGENT"):
        client.cik("NWND")
    assert slept == [] and len(sec.requests) == 1


def test_a_missing_company_record_is_a_clear_error(client, sec):
    sec.script[edgar.SUBMISSIONS_URL.format(cik="0001234567")] = [httpx.Response(404)]
    with pytest.raises(EdgarError, match="404"):
        client.list_filings("NWND")


# --- CLI -----------------------------------------------------------------------------------

def test_the_cli_explains_a_missing_user_agent(monkeypatch, capsys):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert edgar.main(["NWND"]) == 1
    assert "SEC_USER_AGENT" in capsys.readouterr().out
