"""The golden set (ROADMAP 1.4): hand-verified questions over the pinned snapshot.

Every case says which passages answer it as **text spans**: an accession
number plus character offsets into that filing's text, with the quoted text
itself alongside. Offsets are what retrieval is scored against (a chunk is
relevant if it overlaps a span), and they survive re-chunking and the move to
Postgres, because they point into the filing rather than into an index. The
quote is what makes them survive a change of parser too: if 4.1 changes the
text, each span can be found again by its quote.

Offsets are into `document_loader.html_text` of the pinned filing, the text
1.3 indexes. `TEXT_VERSION` names it, and `check` fails if a quote is no
longer at its offsets.

Two files, one schema:
  evaluation/golden_candidates.jsonl   drafted, not yet checked by a person
  evaluation/golden.jsonl              approved by hand; this is the golden set

    python -m evaluation.golden check                          # validate both files against the filings
    python -m evaluation.golden search AAPL FY2026 "net sales" # find passages to quote
    python -m evaluation.golden locate drafts.jsonl            # fill in offsets from quotes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evaluation import snapshot as snap
from ingestion.filings import load_filing_text, source_name
from utils import config

GOLDEN = config.PROJECT_ROOT / "evaluation" / "golden.jsonl"
CANDIDATES = config.PROJECT_ROOT / "evaluation" / "golden_candidates.jsonl"
TEXT_VERSION = "html_text-1.3"

MIN_CASES, MAX_CASES = 60, 101   # 101: every reviewed draft was approved (2026-11-26)
MIN_PER_TYPE = 8
MIN_FROM_10Q = 20


class QuestionType(StrEnum):
    TABLE_LOOKUP = "table_lookup"      # a figure read off a financial table
    TEXT_LOOKUP = "text_lookup"        # a fact stated in prose
    FOLLOW_UP = "follow_up"            # only makes sense after `history`
    CROSS_PERIOD = "cross_period"      # year over year, quarter over quarter, or a derived Q4
    DEFINITION = "definition"          # what a term means in this filing
    SUMMARY = "summary"                # drawn from several passages
    UNANSWERABLE = "unanswerable"      # the snapshot does not say; the right answer is to decline


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Span(_Strict):
    """A passage that answers the question: filing_text[start_char:end_char] == quote."""

    accession: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    start_char: int = Field(ge=0)
    end_char: int
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def _length_matches(self):
        if self.end_char - self.start_char != len(self.quote):
            raise ValueError(f"span {self.start_char}:{self.end_char} is not {len(self.quote)} characters long")
        return self


class Period(_Strict):
    """A company and period the question is about. `fiscal_period` None means the whole year."""

    ticker: str
    fiscal_year: int
    fiscal_period: str | None = Field(default=None, pattern=r"^(FY|Q1|Q2|Q3|Q4)$")


class Turn(_Strict):
    question: str
    answer: str


class Verification(_Strict):
    by: str = Field(min_length=1)
    on: date
    note: str = ""


class GoldenCase(_Strict):
    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    type: QuestionType
    question: str = Field(min_length=5)
    history: list[Turn] = []
    answer: str = Field(min_length=1)          # the reference answer
    spans: list[Span] = []
    about: list[Period] = Field(min_length=1)  # what the question asks about
    calculation: str = ""                      # how a derived figure is worked out, if it is
    verified: Verification | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _shape_fits_type(self):
        t = self.type
        if (t is QuestionType.UNANSWERABLE) != (not self.spans):
            raise ValueError("an unanswerable case has no spans, and every other case has at least one")
        if (t is QuestionType.FOLLOW_UP) != bool(self.history):
            raise ValueError("a follow-up has history, and no other type does")
        if t is QuestionType.CROSS_PERIOD and len({(p.ticker, p.fiscal_year, p.fiscal_period) for p in self.about}) < 2:
            raise ValueError("a cross-period case is about at least two periods")
        if t is QuestionType.SUMMARY and len(self.spans) < 2:
            raise ValueError("a summary draws on at least two spans")
        return self


# --- Files -----------------------------------------------------------------------------

def load(path: Path = GOLDEN) -> list[GoldenCase]:
    """Every case in a .jsonl file. Raises with the line number on the first bad one."""
    if not path.exists():
        return []
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                cases.append(GoldenCase.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"{path.name} line {number}: {exc}") from exc
    return cases


def save(cases: list[GoldenCase], path: Path) -> None:
    lines = [case.model_dump_json(exclude_defaults=True) for case in cases]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")


# --- Checking against the filings ----------------------------------------------------------

@lru_cache(maxsize=None)
def _pinned() -> dict[str, dict]:
    return {f["accession"]: f for f in snap.load()["filings"]}


@lru_cache(maxsize=16)
def filing_text(accession: str) -> str:
    pinned = _pinned()[accession]
    return load_filing_text(snap.paths({"filings": [pinned]})[0], pinned["sha256"])


def problems(cases: list[GoldenCase], text_of=filing_text, pinned: dict | None = None) -> list[str]:
    """Everything wrong with a set of cases that the schema alone cannot see."""
    pinned = _pinned() if pinned is None else pinned
    found = []
    ids = Counter(c.id for c in cases)
    found += [f"id {i!r} is used {n} times" for i, n in ids.items() if n > 1]
    for case in cases:
        for span in case.spans:
            if span.accession not in pinned:
                found.append(f"{case.id}: {span.accession} is not in the snapshot")
                continue
            text = text_of(span.accession)
            if text[span.start_char:span.end_char] != span.quote:
                found.append(f"{case.id}: the quote is not at {span.start_char}:{span.end_char} in {span.accession}")
        for period in case.about:
            if period.ticker not in {f["ticker"] for f in pinned.values()}:
                found.append(f"{case.id}: {period.ticker} is not a snapshot company")
        uncited = uncited_figures(case)
        if uncited:
            found.append(f"{case.id}: no cited passage or calculation gives {', '.join(uncited)}")
    return found


# A reference answer may say only what its cited passages say. Prose claims
# need a person to check, but figures can be checked mechanically: every
# number in the answer must appear in a cited quote or in the calculation,
# allowing for rounding and for millions written as billions (482,975 -> 483.0).
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?!\d)")
_YEAR = re.compile(r"(19|20)\d\d")
_DAY = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+$")
_NUMBER_WORDS = {w: float(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def _numbers(text: str) -> list[tuple[float, int, str, int]]:
    """(value, decimal places, as written, position) for each number in `text`."""
    return [(float(m.group(1).replace(",", "") + (f".{m.group(2)}" if m.group(2) else "")),
             len(m.group(2) or ""), m.group(0), m.start()) for m in _NUMBER.finditer(text)]


def uncited_figures(case: GoldenCase) -> list[str]:
    """Figures in the reference answer that no cited quote or the calculation gives."""
    if not case.spans:
        return []
    source = " ".join([s.quote for s in case.spans] + [case.calculation])
    values = [v for v, _, _, _ in _numbers(source)]
    values += [_NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", source.lower()) if w in _NUMBER_WORDS]
    missing = []
    for value, places, written, at in _numbers(case.answer):
        if not places and (_YEAR.fullmatch(written) or _DAY.search(case.answer[:at])):
            continue  # a year or a day of the month, not a figure
        tolerance = 0.5 * 10 ** -places + 1e-9
        if not any(abs(v * scale - value) <= tolerance for v in values for scale in (1, 1e-3, 1e3)):
            missing.append(written)
    return missing


def coverage(cases: list[GoldenCase], pinned: dict | None = None) -> dict:
    """The counts the 1.4 "done when" is stated in."""
    pinned = _pinned() if pinned is None else pinned
    forms = {a: f["form"] for a, f in pinned.items()}
    from_10q = sum(1 for c in cases if any(forms.get(s.accession) == "10-Q" for s in c.spans))
    return {
        "cases": len(cases),
        "by_type": {t.value: sum(1 for c in cases if c.type is t) for t in QuestionType},
        "from_10q": from_10q,
        "verified": sum(1 for c in cases if c.verified is not None),
    }


def shortfalls(cases: list[GoldenCase], pinned: dict | None = None) -> list[str]:
    """What stands between these cases and a finished golden set."""
    cov = coverage(cases, pinned)
    out = []
    if not MIN_CASES <= cov["cases"] <= MAX_CASES:
        out.append(f"{cov['cases']} cases; the set needs {MIN_CASES}-{MAX_CASES}")
    out += [f"{n} {t} case(s); each type needs {MIN_PER_TYPE}" for t, n in cov["by_type"].items() if n < MIN_PER_TYPE]
    if cov["from_10q"] < MIN_FROM_10Q:
        out.append(f"{cov['from_10q']} cases answered from 10-Qs; the set needs {MIN_FROM_10Q}")
    if cov["verified"] < cov["cases"]:
        out.append(f"{cov['cases'] - cov['verified']} case(s) not verified by hand")
    return out


# --- Writing cases: finding and locating quotes -------------------------------------------

def resolve(filing: str) -> str:
    """An accession from either an accession or a name like "AAPL 10-K FY2026"."""
    if filing in _pinned():
        return filing
    by_name = {source_name(f): a for a, f in _pinned().items()}
    if filing not in by_name:
        raise ValueError(f"no pinned filing called {filing!r}")
    return by_name[filing]


def locate(draft: dict) -> dict:
    """Fill each span's accession and offsets from its `filing` and `quote`.

    A quote that appears more than once is an error: a span has to say which
    passage it means, so the quote has to be long enough to be unique.
    """
    spans = []
    for span in draft.get("spans", []):
        accession = resolve(span.get("filing") or span["accession"])
        text = filing_text(accession)
        quote = span["quote"]
        start = text.find(quote)
        if start == -1:
            raise ValueError(f"{draft['id']}: quote not found in {accession}: {quote[:80]!r}")
        if text.find(quote, start + 1) != -1:
            raise ValueError(f"{draft['id']}: quote appears more than once in {accession}; lengthen it: {quote[:80]!r}")
        spans.append({"accession": accession, "start_char": start, "end_char": start + len(quote), "quote": quote})
    return {**draft, "spans": spans}


def search(ticker: str, period: str, pattern: str, context: int = 1, limit: int = 20) -> list[str]:
    """Lines of one filing matching `pattern` (case-insensitive), with neighbouring lines."""
    match = re.fullmatch(r"(FY|Q1|Q2|Q3)(\d{4})", period.upper())
    if not match:
        raise ValueError("period is like FY2026 or Q3 2025 written Q32025")
    form = "10-K" if match.group(1) == "FY" else "10-Q"
    accession = resolve(f"{ticker.upper()} {form} {match.group(1)}{match.group(2)}")
    lines = filing_text(accession).split("\n")
    regex = re.compile(pattern, re.IGNORECASE)
    out = []
    for i, line in enumerate(lines):
        if regex.search(line):
            block = lines[max(0, i - context): i + context + 1]
            out.append(f"--- line {i}\n" + "\n".join(block))
            if len(out) >= limit:
                break
    return out


# --- CLI ----------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate golden.jsonl and golden_candidates.jsonl")
    s = sub.add_parser("search", help="find lines in a filing")
    s.add_argument("ticker")
    s.add_argument("period", help="FY2026, Q12025, ...")
    s.add_argument("pattern")
    s.add_argument("--context", type=int, default=1)
    s.add_argument("--limit", type=int, default=20)
    loc = sub.add_parser("locate", help="fill offsets in a drafts file and append to the candidates")
    loc.add_argument("drafts", type=Path)
    args = parser.parse_args(argv)

    if args.command == "search":
        for block in search(args.ticker, args.period, args.pattern, args.context, args.limit):
            print(block)
        return 0

    if args.command == "locate":
        existing = load(CANDIDATES)
        taken = {c.id for c in existing} | {c.id for c in load(GOLDEN)}
        added, failed = [], 0
        for line in args.drafts.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            draft = json.loads(line)
            try:
                case = GoldenCase.model_validate(locate(draft))
            except (ValueError, ValidationError) as exc:
                print(f"  FAILED {draft.get('id')}: {exc}")
                failed += 1
                continue
            if case.id in taken:
                print(f"  FAILED {case.id}: id already used")
                failed += 1
                continue
            taken.add(case.id)
            added.append(case)
        save(existing + added, CANDIDATES)
        print(f"{len(added)} candidate(s) added, {failed} failed; {len(existing) + len(added)} in {CANDIDATES.name}")
        return 1 if failed else 0

    status = 0
    for path in (CANDIDATES, GOLDEN):
        try:
            cases = load(path)
        except ValueError as exc:
            print(exc)
            status = 1
            continue
        bad = problems(cases)
        cov = coverage(cases)
        print(f"{path.name}: {cov['cases']} cases, {cov['from_10q']} from 10-Qs, {cov['verified']} verified")
        print("  " + ", ".join(f"{t} {n}" for t, n in cov["by_type"].items()))
        for p in bad:
            print(f"  PROBLEM {p}")
        status |= bool(bad)
    golden = load(GOLDEN)
    for gap in shortfalls(golden):
        print(f"  golden set: {gap}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
