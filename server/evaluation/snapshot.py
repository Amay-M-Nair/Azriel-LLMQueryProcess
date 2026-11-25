"""The pinned snapshot (ROADMAP 1.2): exactly which filings the evaluation runs on.

The live index follows EDGAR as new filings arrive. The evaluation does not:
it runs on the filings listed in evaluation/snapshot.json, pinned by
accession number and SHA-256, so a score moves only when the code does.

A snapshot holds, per company, its three newest *complete* fiscal years: a
year counts once its 10-K and all three of its 10-Qs are filed. A year still
in progress is left out rather than half-included, so every company
contributes the same shape: 3 x (Q1, Q2, Q3, FY) = 12 filings.

    python -m evaluation.snapshot                  # fetch the pinned filings, check every hash
    python -m evaluation.snapshot --fresh          # ... into an empty folder, proving it reproduces
    python -m evaluation.snapshot --build          # re-pick from EDGAR (default companies)
    python -m evaluation.snapshot --build AAPL KO  # ... for these companies

Rebuilding writes a new `version` only if the pinned filings changed. Every
evaluation result records the version it ran on.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ingestion.edgar import Download, EdgarClient, EdgarError, EdgarNotReady, Filing, normalise_ticker
from utils import config, logs

log = logs.get(__name__)

SNAPSHOT = config.PROJECT_ROOT / "evaluation" / "snapshot.json"

# Six industries and five kinds of fiscal year, so nothing downstream can be
# tuned to one company's layout or one calendar:
#   AAPL  hardware      ends late September (52/53-week)
#   MSFT  software      ends 30 June
#   JPM   bank          calendar year; financial statements unlike anyone else's
#   JNJ   healthcare    52/53-week year ending near 31 December
#   WMT   retail        ends 31 January, named for the year it ends in
#   CVX   energy        calendar year. (Not XOM: see ROADMAP Findings, 1.2.)
DEFAULT_COMPANIES = ("AAPL", "MSFT", "JPM", "JNJ", "WMT", "CVX")
YEARS = 3
PERIODS = ("Q1", "Q2", "Q3", "FY")


class SnapshotError(Exception):
    """The snapshot could not be built, or does not reproduce."""


@dataclass
class Check:
    """One pinned filing checked against what was fetched."""

    filing: Filing
    ok: bool
    problem: str = ""


# --- Building ----------------------------------------------------------------------

def complete_years(downloads: list[Download]) -> dict[int, dict[str, Download]]:
    """Fiscal year -> {period: download}, only for years with all four periods."""
    years: dict[int, dict[str, Download]] = {}
    for d in downloads:
        years.setdefault(d.fiscal_year, {})[d.fiscal_period] = d
    return {y: periods for y, periods in years.items() if set(periods) == set(PERIODS)}


def order_problems(year: int, periods: dict[str, Download]) -> list[str]:
    """A fiscal year whose quarters do not run Q1 < Q2 < Q3 < FY was misread."""
    dates = [periods[p].filing.report_date for p in PERIODS]
    problems = []
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        problems.append(f"fiscal {year} periods are out of order: " +
                        ", ".join(f"{p} {d}" for p, d in zip(PERIODS, dates)))
    span = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days
    if not 150 <= span <= 300:          # Q1 end to year end is ~9 months
        problems.append(f"fiscal {year} runs {span} days from Q1 to FY end; expected about 270")
    return problems


def pick(ticker: str, downloads: list[Download], years: int = YEARS) -> list[Download]:
    """The newest `years` complete fiscal years for one company, checked."""
    complete = complete_years(downloads)
    chosen = sorted(complete, reverse=True)[:years]
    if len(chosen) < years:
        have = sorted({d.fiscal_year for d in downloads})
        raise SnapshotError(
            f"{ticker}: only {len(chosen)} complete fiscal year(s) {sorted(chosen)} among {have}; need {years}"
        )
    problems = [p for y in chosen for p in order_problems(y, complete[y])]
    if problems:
        raise SnapshotError(f"{ticker}: " + "; ".join(problems))
    return [complete[y][p] for y in sorted(chosen) for p in PERIODS]


def entry(d: Download) -> dict:
    record = d.record()
    return {k: record[k] for k in (
        "ticker", "cik", "company", "form", "accession", "filing_date", "report_date",
        "primary_document", "fiscal_year_end", "fiscal_year", "fiscal_period", "period_source",
        "sha256", "size",
    )}


def build(client: EdgarClient, tickers=DEFAULT_COMPANIES, years: int = YEARS,
          today: date | None = None) -> dict:
    """Pick each company's newest complete fiscal years from EDGAR and pin them.

    Everything since a year before the oldest wanted year is downloaded, since
    a filing's fiscal year is only known once its cover has been read.
    """
    today = today or date.today()
    since = date(today.year - years - 1, 1, 1)
    filings = []
    for ticker in tickers:
        ticker = normalise_ticker(ticker)
        listed = client.list_filings(ticker, since=since)
        downloads = [client.download(f) for f in listed]
        chosen = pick(ticker, downloads, years)
        log.info("%s: fiscal %s", ticker, ", ".join(str(y) for y in sorted({d.fiscal_year for d in chosen})))
        filings.extend(entry(d) for d in chosen)
    return {"companies": [normalise_ticker(t) for t in tickers], "years": years, "filings": filings}


def with_version(new: dict, old: dict | None, today: date | None = None) -> dict:
    """Keep the old version if nothing pinned changed; otherwise the next one."""
    def key(snapshot):
        return sorted((f["accession"], f["sha256"]) for f in snapshot["filings"])

    if old and key(old) == key(new) and old.get("companies") == new["companies"]:
        return old
    version = (old or {}).get("version", 0) + 1
    created = (today or date.today()).isoformat()
    return {"version": version, "created": created, **new}


# --- Loading and checking -------------------------------------------------------------

def load(path: Path = SNAPSHOT) -> dict:
    if not path.exists():
        raise SnapshotError(f"no snapshot at {path}: run  python -m evaluation.snapshot --build")
    return json.loads(path.read_text(encoding="utf-8"))


def save(snapshot: dict, path: Path = SNAPSHOT) -> None:
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8", newline="\n")


def filing_of(pinned: dict) -> Filing:
    return Filing(**{k: pinned[k] for k in Filing.__dataclass_fields__})


def fetch(client: EdgarClient, snapshot: dict) -> list[Check]:
    """Fetch every pinned filing (from cache where it is there) and check it matches."""
    checks = []
    for pinned in snapshot["filings"]:
        filing = filing_of(pinned)
        try:
            done = client.download(filing)
        except EdgarError as exc:
            checks.append(Check(filing, False, str(exc)))
            continue
        problems = []
        if done.sha256 != pinned["sha256"]:
            problems.append(f"sha256 {done.sha256[:12]} != pinned {pinned['sha256'][:12]}")
        if (done.fiscal_year, done.fiscal_period) != (pinned["fiscal_year"], pinned["fiscal_period"]):
            problems.append(f"read as {done.fiscal_year} {done.fiscal_period}, "
                            f"pinned as {pinned['fiscal_year']} {pinned['fiscal_period']}")
        checks.append(Check(filing, not problems, "; ".join(problems)))
    return checks


def paths(snapshot: dict, cache_dir: Path | None = None) -> list[Path]:
    """Where each pinned filing lives in the cache, in snapshot order."""
    root = Path(cache_dir or config.FILINGS_DIR)
    return [root / f["ticker"] / str(f["fiscal_year"]) / f["fiscal_period"] / f["primary_document"]
            for f in snapshot["filings"]]


# --- CLI ----------------------------------------------------------------------------

def summary(snapshot: dict) -> str:
    lines = [f"snapshot v{snapshot['version']} ({snapshot['created']}): "
             f"{len(snapshot['companies'])} companies, {len(snapshot['filings'])} filings"]
    for ticker in snapshot["companies"]:
        rows = [f for f in snapshot["filings"] if f["ticker"] == ticker]
        years = sorted({f["fiscal_year"] for f in rows})
        size = sum(f["size"] for f in rows) / 1e6
        lines.append(f"  {ticker:5} fiscal {years[0]}-{years[-1]}  {len(rows)} filings  {size:.1f} MB")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", nargs="*", metavar="TICKER",
                        help="re-pick from EDGAR (no tickers: the default companies)")
    parser.add_argument("--fresh", action="store_true", help="fetch into an empty folder, not the cache")
    args = parser.parse_args(argv)
    logs.setup()

    try:
        if args.build is not None:
            with EdgarClient() as client:
                new = build(client, tuple(args.build) or DEFAULT_COMPANIES)
            old = load() if SNAPSHOT.exists() else None
            snapshot = with_version(new, old)
            save(snapshot)
            print(summary(snapshot))
            print("unchanged" if snapshot is old else f"wrote {SNAPSHOT.name} v{snapshot['version']}")
            return 0

        snapshot = load()
        with tempfile.TemporaryDirectory() as tmp:
            with EdgarClient(cache_dir=Path(tmp) if args.fresh else None) as client:
                checks = fetch(client, snapshot)
    except (EdgarNotReady, SnapshotError, EdgarError) as exc:
        print(exc)
        return 1

    bad = [c for c in checks if not c.ok]
    print(summary(snapshot))
    where = "into an empty folder" if args.fresh else "from the cache"
    print(f"{len(checks) - len(bad)}/{len(checks)} filings match their pinned hash ({where})")
    for c in bad:
        print(f"  MISMATCH {c.filing.ticker} {c.filing.form} {c.filing.accession}: {c.problem}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
