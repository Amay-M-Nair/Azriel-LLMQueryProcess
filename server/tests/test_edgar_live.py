"""ROADMAP 1.1, the half recorded responses cannot prove: real filings download.

Runs only when SEC_USER_AGENT is set in .env, and is skipped otherwise, which
leaves 1.1 unticked: a check that did not run proved nothing. Downloads go to
the real cache (data/filings/), so after the first run this costs two small
requests: the ticker map (weekly) and Apple's submissions record.

Apple is the subject because its fiscal year ends in late September, so a
fetcher that assumed December would file its quarters under the wrong year.
"""

import pytest

from ingestion import edgar
from ingestion.edgar import EdgarClient, EdgarError, EdgarNotReady

pytestmark = pytest.mark.roadmap("1.1")


@pytest.fixture(scope="module")
def client():
    try:
        edgar.user_agent()
    except EdgarNotReady as exc:
        pytest.skip(f"no real EDGAR check without it: {str(exc).splitlines()[0]}")
    with EdgarClient() as c:
        yield c


def latest(client, form):
    try:
        filings = client.list_filings("AAPL", forms=(form,))
    except EdgarError as exc:
        if "ConnectError" in str(exc) or "ConnectTimeout" in str(exc):
            pytest.skip(f"EDGAR is unreachable from here: {exc}")
        raise
    assert filings, f"EDGAR lists no {form} for AAPL in its recent block"
    return filings[0]


def test_a_real_10k_downloads_and_files_itself_by_its_cover(client):
    done = client.download(latest(client, "10-K"))
    assert done.fiscal_period == "FY" and done.period_source == "cover"
    assert done.size > 200_000, "too small to be a 10-K"
    assert done.path.exists() and done.path.parent.name == "FY"


def test_a_real_10q_downloads_and_files_itself_by_its_cover(client):
    done = client.download(latest(client, "10-Q"))
    assert done.fiscal_period in ("Q1", "Q2", "Q3") and done.period_source == "cover"
    assert done.size > 100_000, "too small to be a 10-Q"
