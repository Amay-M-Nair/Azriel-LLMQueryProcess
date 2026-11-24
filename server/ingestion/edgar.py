"""SEC EDGAR: ticker -> CIK -> filings -> the 10-K and 10-Q documents themselves.

    ticker   AAPL
      |      company_tickers.json (cached, refreshed weekly)
    CIK      0000320193
      |      data.sec.gov/submissions/CIK0000320193.json
    filings  every 10-K and 10-Q, newest first; amendments listed and left out
      |      www.sec.gov/Archives/edgar/data/320193/{accession}/{primary document}
    files    data/filings/AAPL/2025/FY/aapl-20250927.htm  + filing.json

Every filing is filed under the fiscal year and period the filing itself
declares on its cover (the inline-XBRL `dei:DocumentFiscalYearFocus` and
`dei:DocumentFiscalPeriodFocus` tags). Dates alone cannot say: Apple's fiscal
2025 ended in September 2025, Target's fiscal 2025 ends in early 2026, and
Walmart calls the year ending January 2026 fiscal 2026. Only a filing with no
cover tags (older than inline XBRL) falls back to working it out from the
report date and the company's fiscal year end, and says so in its
`period_source`.

The SEC's rules for automated clients are kept here rather than discovered:

  - every request names who is making it (SEC_USER_AGENT, a name and a
    contact email). Without one the SEC answers 403.
  - at most 10 requests per second. A client that goes faster is locked out
    with a 403, not slowed down, so a 403 is never retried: retrying a
    lockout only extends it.
  - 429, 5xx and dropped connections are retried with backoff, honouring
    Retry-After when the SEC sends one.

    python -m ingestion.edgar AAPL                       # list recent 10-K / 10-Q
    python -m ingestion.edgar AAPL --since 2023-01-01    # further back
    python -m ingestion.edgar AAPL --latest              # only what is not cached yet
    python -m ingestion.edgar AAPL --latest --download   # ... and fetch it
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import html
import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from utils import config, logs

log = logs.get(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

FORMS = ("10-K", "10-Q")
AMENDMENTS = {"10-K/A", "10-Q/A"}
PERIODS = {"10-K": ("FY",), "10-Q": ("Q1", "Q2", "Q3")}

RETRYABLE = {429, 500, 502, 503, 504}

# A name and an email, which is what the SEC asks for: "Sample Company Name
# AdminContact@<sample company domain>.com".
USER_AGENT_SHAPE = re.compile(r"\S.*\s\S+@\S+\.\S+|\S+@\S+\.\S+\s.*\S")

# An inline-XBRL fact on the cover page, e.g.
#   <ix:nonNumeric name="dei:DocumentFiscalPeriodFocus" contextRef="c-1">Q3</ix:nonNumeric>
COVER_FACT = r"<(\w+):nonNumeric\b[^>]*\bname=[\"']dei:{tag}[\"'][^>]*>(.*?)</\1:nonNumeric>"
TAG = re.compile(r"<[^>]+>")


class EdgarError(Exception):
    """EDGAR could not give us what was asked for. The message says why."""


class EdgarNotReady(EdgarError):
    """Nothing can be fetched until SEC_USER_AGENT is set."""


@dataclass(frozen=True)
class Filing:
    """One 10-K or 10-Q as EDGAR lists it, before anything is downloaded."""

    ticker: str
    cik: str                    # ten digits, zero-padded
    company: str
    form: str                   # "10-K" | "10-Q"
    accession: str              # "0000320193-25-000079"
    filing_date: str            # ISO date it was filed
    report_date: str            # ISO date of the period it reports on
    primary_document: str       # "aapl-20250927.htm"
    fiscal_year_end: str        # "0927", from the company's submissions record

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=int(self.cik), accession=self.accession.replace("-", ""), document=self.primary_document,
        )


@dataclass(frozen=True)
class Download:
    """A filing on disk, filed under the fiscal year and period it declares."""

    filing: Filing
    fiscal_year: int
    fiscal_period: str          # "FY" | "Q1" | "Q2" | "Q3"
    period_source: str          # "cover" (dei tags) | "dates" (fallback)
    path: Path
    sha256: str
    size: int

    def record(self) -> dict:
        return {
            **asdict(self.filing),
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
            "period_source": self.period_source,
            "sha256": self.sha256,
            "size": self.size,
        }


def user_agent() -> str:
    """SEC_USER_AGENT, checked for the name-and-email shape the SEC asks for."""
    value = (os.environ.get("SEC_USER_AGENT") or "").strip()
    if not value:
        raise EdgarNotReady(
            "SEC_USER_AGENT is not set. EDGAR refuses automated requests that do not say "
            "who is making them: add your name and a contact email to .env, e.g.\n"
            "  SEC_USER_AGENT=Jane Doe jane@example.com"
        )
    if not USER_AGENT_SHAPE.fullmatch(value):
        raise EdgarNotReady(
            "SEC_USER_AGENT needs a name and a contact email, e.g. "
            "'Jane Doe jane@example.com'. The SEC refuses requests that lack either."
        )
    return value


class Throttle:
    """Spaces requests so there are never more than `per_second` in a second.

    Shared by every request a client makes, from any thread. Even spacing
    rather than bursts: the SEC counts requests per second, and a burst of
    ten followed by a pause is ten in a much shorter window.

    The spacing is 5% wider than 1/per_second. Exactly 1/per_second puts the
    eleventh request at the one-second mark, where float rounding or the
    SEC's clock can count it inside the same second.
    """

    MARGIN = 1.05

    def __init__(self, per_second: float, clock=time.monotonic, sleep=time.sleep):
        self.interval = self.MARGIN / per_second
        self.clock = clock
        self.sleep = sleep
        self._next = float("-inf")
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self.clock()
            if now < self._next:
                self.sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


# --- Working out the fiscal period -------------------------------------------------

def cover_facts(document: str) -> dict[str, str]:
    """The cover-page dei facts the period is read from, where present."""
    facts = {}
    for tag in ("DocumentType", "DocumentFiscalYearFocus", "DocumentFiscalPeriodFocus", "DocumentPeriodEndDate"):
        if m := re.search(COVER_FACT.format(tag=tag), document, re.DOTALL | re.IGNORECASE):
            text = html.unescape(TAG.sub("", m.group(2))).strip()
            if text:
                facts[tag] = text
    return facts


def _on(year: int, mmdd: str) -> date:
    month, day = int(mmdd[:2]), int(mmdd[2:])
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def period_from_dates(form: str, report_date: str, fiscal_year_end: str) -> tuple[int, str]:
    """Fiscal year and period worked out from dates, for filings with no cover tags.

    The fiscal year is named for the calendar year it ends in, except a year
    ending in the first week of January: that is a 52/53-week year that ran a
    few days over, and is named for the year before. Some retailers name their
    years differently again, which is why this is only the fallback.
    """
    if not re.fullmatch(r"\d{4}", fiscal_year_end or ""):
        raise EdgarError(f"no fiscal year end on record to place a {form} dated {report_date}")
    reported = date.fromisoformat(report_date)
    slack = timedelta(days=10)          # 52/53-week years end on a weekday, not a date
    end = next(e for e in (_on(reported.year, fiscal_year_end), _on(reported.year + 1, fiscal_year_end))
               if e >= reported - slack)
    fiscal_year = end.year - 1 if (end.month == 1 and end.day <= 7) else end.year
    if form == "10-K":
        return fiscal_year, "FY"
    quarter = round((reported - _on(end.year - 1, fiscal_year_end)).days / 91.3)
    if quarter not in (1, 2, 3):
        raise EdgarError(f"a 10-Q dated {report_date} does not fall in Q1-Q3 of a year ending {end}")
    return fiscal_year, f"Q{quarter}"


def fiscal_period(filing: Filing, document: str) -> tuple[int, str, str]:
    """(fiscal_year, fiscal_period, source) for a downloaded filing."""
    facts = cover_facts(document)
    year, period = facts.get("DocumentFiscalYearFocus", ""), facts.get("DocumentFiscalPeriodFocus", "").upper()
    if re.fullmatch(r"\d{4}", year) and period in PERIODS[filing.form]:
        return int(year), period, "cover"
    if filing.form == "10-K" and re.fullmatch(r"\d{4}", year):
        log.warning("%s %s declares period %r on a 10-K; filing it as FY", filing.ticker, filing.accession, period)
        return int(year), "FY", "cover"
    fiscal_year, fiscal_period_ = period_from_dates(filing.form, filing.report_date, filing.fiscal_year_end)
    log.warning("%s %s has no usable cover tags (%s); placed by dates as %s %s",
                filing.ticker, filing.accession, facts or "none", fiscal_year, fiscal_period_)
    return fiscal_year, fiscal_period_, "dates"


# --- The client ----------------------------------------------------------------------

class EdgarClient:
    """Lists and downloads filings, under the SEC's rules, into a local cache."""

    def __init__(
        self,
        agent: str | None = None,
        cache_dir: Path | None = None,
        transport: httpx.BaseTransport | None = None,
        throttle: Throttle | None = None,
        sleep=time.sleep,
    ):
        self.agent = agent or user_agent()
        self.cache_dir = Path(cache_dir or config.FILINGS_DIR)
        self.throttle = throttle or Throttle(config.EDGAR_MAX_REQUESTS_PER_SECOND)
        self.sleep = sleep
        self.http = httpx.Client(
            headers={"User-Agent": self.agent},
            timeout=config.EDGAR_TIMEOUT,
            transport=transport,
            follow_redirects=True,
        )

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.http.close()

    # -- HTTP ----------------------------------------------------------------

    def get(self, url: str) -> httpx.Response:
        """GET under the rate limit, retrying what is worth retrying."""
        problem = ""
        for attempt in range(1, config.EDGAR_RETRIES + 1):
            self.throttle.wait()
            retry_after = None
            try:
                response = self.http.get(url)
            except httpx.TransportError as exc:
                problem = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status == 200:
                    return response
                if status == 403:
                    raise EdgarError(
                        f"EDGAR refused {url} (403). It does this when the User-Agent does not "
                        "name a person and an email (check SEC_USER_AGENT), and when a client has "
                        "gone over 10 requests a second, in which case wait ten minutes."
                    )
                if status == 404:
                    raise EdgarError(f"EDGAR has nothing at {url} (404)")
                if status not in RETRYABLE:
                    raise EdgarError(f"EDGAR answered {status} for {url}")
                problem = f"HTTP {status}"
                retry_after = _retry_after(response)
            if attempt == config.EDGAR_RETRIES:
                break
            wait = retry_after if retry_after is not None else config.EDGAR_RETRY_BACKOFF * 2 ** (attempt - 1)
            wait = min(wait, config.EDGAR_MAX_WAIT)
            log.warning("EDGAR %s for %s; retry %d/%d in %.1fs",
                        problem, url, attempt, config.EDGAR_RETRIES - 1, wait)
            self.sleep(wait)
        raise EdgarError(f"EDGAR did not answer {url} after {config.EDGAR_RETRIES} tries ({problem})")

    def get_json(self, url: str) -> dict:
        try:
            return self.get(url).json()
        except ValueError as exc:
            raise EdgarError(f"EDGAR sent something that is not JSON from {url}: {exc}") from exc

    # -- Ticker -> CIK -------------------------------------------------------

    def tickers(self) -> dict[str, tuple[str, str]]:
        """Ticker -> (ten-digit CIK, company name), from a cached copy when fresh."""
        path = self.cache_dir / "company_tickers.json"
        max_age = config.EDGAR_TICKER_MAP_MAX_AGE_DAYS * 86400
        if path.exists() and time.time() - path.stat().st_mtime < max_age:
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = self.get_json(TICKERS_URL)
            _write_atomic(path, json.dumps(raw).encode("utf-8"))
        return {
            normalise_ticker(row["ticker"]): (f"{int(row['cik_str']):010d}", row["title"])
            for row in raw.values()
        }

    def cik(self, ticker: str) -> str:
        found = self.tickers().get(normalise_ticker(ticker))
        if found is None:
            raise EdgarError(f"EDGAR has no company with the ticker {ticker!r}")
        return found[0]

    # -- Listing -------------------------------------------------------------

    def list_filings(self, ticker: str, forms=FORMS, since: date | None = None) -> list[Filing]:
        """Every 10-K / 10-Q for `ticker`, newest first.

        Without `since`, only EDGAR's `recent` block is read: at least the last
        year of filings, which is where anything new appears. With it, older
        pages are read too, but only those that reach back past `since`.
        """
        ticker = normalise_ticker(ticker)
        cik = self.cik(ticker)
        record = self.get_json(SUBMISSIONS_URL.format(cik=cik))
        company = record.get("name") or self.tickers()[ticker][1]
        fiscal_year_end = record.get("fiscalYearEnd") or ""

        blocks = [record.get("filings", {}).get("recent", {})]
        if since is not None:
            for page in record.get("filings", {}).get("files", []):
                if page.get("filingTo", "") >= since.isoformat():
                    blocks.append(self.get_json(SUBMISSIONS_PAGE_URL.format(name=page["name"])))

        filings, amendments = {}, 0
        for block in blocks:
            for row in _rows(block):
                form = row.get("form", "")
                if form in AMENDMENTS and form[:-2] in forms:
                    amendments += 1
                    continue
                if form not in forms:
                    continue
                if since is not None and row.get("filingDate", "") < since.isoformat():
                    continue
                if not row.get("primaryDocument"):
                    log.warning("%s %s %s has no primary document; skipped", ticker, form, row.get("accessionNumber"))
                    continue
                filings[row["accessionNumber"]] = Filing(
                    ticker=ticker, cik=cik, company=company, form=form,
                    accession=row["accessionNumber"], filing_date=row.get("filingDate", ""),
                    report_date=row.get("reportDate", ""), primary_document=row["primaryDocument"],
                    fiscal_year_end=fiscal_year_end,
                )
        if amendments:
            log.info("%s: %d amendment(s) (10-K/A, 10-Q/A) listed and not fetched", ticker, amendments)
        return sorted(filings.values(), key=lambda f: (f.filing_date, f.accession), reverse=True)

    def latest(self, ticker: str, forms=FORMS) -> list[Filing]:
        """Filings EDGAR lists that are not in the cache yet, newest first."""
        have = self.cached(ticker)
        return [f for f in self.list_filings(ticker, forms) if f.accession not in have]

    # -- Downloading -----------------------------------------------------------

    def cached(self, ticker: str) -> dict[str, Download]:
        """Accession -> Download for everything already on disk for `ticker`."""
        found = {}
        for record_path in (self.cache_dir / normalise_ticker(ticker)).glob("*/*/filing.json"):
            data = json.loads(record_path.read_text(encoding="utf-8"))
            filing = Filing(**{k: data[k] for k in Filing.__dataclass_fields__})
            found[filing.accession] = Download(
                filing=filing, fiscal_year=data["fiscal_year"], fiscal_period=data["fiscal_period"],
                period_source=data["period_source"], path=record_path.parent / filing.primary_document,
                sha256=data["sha256"], size=data["size"],
            )
        return found

    def download(self, filing: Filing) -> Download:
        """The filing's primary document on disk, fetched only if it is not there already."""
        have = self.cached(filing.ticker).get(filing.accession)
        if have is not None:
            if have.path.exists() and _sha256(have.path.read_bytes()) == have.sha256:
                return have
            log.warning("%s %s: cached copy is missing or changed; fetching again", filing.ticker, filing.accession)

        content = self.get(filing.url).content
        fiscal_year, period, source = fiscal_period(filing, content.decode("utf-8", errors="replace"))
        folder = self.cache_dir / filing.ticker / str(fiscal_year) / period
        record_path = folder / "filing.json"
        if record_path.exists():
            other = json.loads(record_path.read_text(encoding="utf-8"))
            if other["accession"] != filing.accession:
                raise EdgarError(
                    f"{filing.ticker} {filing.accession} and {other['accession']} both place themselves "
                    f"in fiscal {fiscal_year} {period}; one of them is misread"
                )
        done = Download(filing=filing, fiscal_year=fiscal_year, fiscal_period=period, period_source=source,
                        path=folder / filing.primary_document, sha256=_sha256(content), size=len(content))
        _write_atomic(done.path, content)
        record = done.record() | {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _write_atomic(record_path, json.dumps(record, indent=2).encode("utf-8"))
        log.info("%s %s %s -> %s/%s (%d KB, period from %s)", filing.ticker, filing.form, filing.accession,
                 fiscal_year, period, len(content) // 1024, source)
        return done


# --- Helpers --------------------------------------------------------------------------

def normalise_ticker(ticker: str) -> str:
    """EDGAR writes share classes with a dash: BRK.B is BRK-B."""
    return ticker.strip().upper().replace(".", "-")


def _rows(block: dict) -> list[dict]:
    """EDGAR's column arrays ({"form": [...], "accessionNumber": [...]}) as rows."""
    columns = {k: v for k, v in block.items() if isinstance(v, list)}
    count = len(columns.get("accessionNumber", []))
    return [{k: v[i] if i < len(v) else None for k, v in columns.items()} for i in range(count)]


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "")
    try:
        return max(0.0, float(value))
    except ValueError:
        return None                    # an HTTP date, or nothing: use the backoff


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_atomic(path: Path, content: bytes) -> None:
    """Write via a temporary file, so an interrupted run never leaves half a filing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


# --- CLI ----------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--forms", nargs="+", default=list(FORMS), choices=list(FORMS))
    parser.add_argument("--since", type=date.fromisoformat, help="reach back to this filing date (YYYY-MM-DD)")
    parser.add_argument("--latest", action="store_true", help="only filings not cached yet")
    parser.add_argument("--download", action="store_true", help="fetch what is listed")
    args = parser.parse_args(argv)
    logs.setup()

    try:
        client = EdgarClient()
    except EdgarNotReady as exc:
        print(exc)
        return 1

    failed = False
    with client:
        for ticker in args.tickers:
            try:
                filings = (client.latest(ticker, tuple(args.forms)) if args.latest
                           else client.list_filings(ticker, tuple(args.forms), since=args.since))
                print(f"{normalise_ticker(ticker)}: {len(filings)} filing(s)")
                for filing in filings:
                    line = f"  {filing.form:5} {filing.report_date}  filed {filing.filing_date}  {filing.accession}"
                    if args.download:
                        done = client.download(filing)
                        line += f"  -> {done.fiscal_year} {done.fiscal_period} ({done.period_source})"
                    print(line)
            except EdgarError as exc:        # one company failing does not stop the rest
                print(f"{normalise_ticker(ticker)}: {exc}")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
