"""Review drafted golden-set cases by hand (ROADMAP 1.4).

    ../.venv/Scripts/streamlit.exe run evaluation/review_app.py      (from server/)

One candidate at a time: the question, the proposed answer, and each cited
passage highlighted inside its filing's text, so it can be checked against
the source in seconds. Approve it (optionally correcting the answer), reject
it with a reason, or skip it for later.

  approved   -> evaluation/golden.jsonl, with your name and today's date
  rejected   -> evaluation/golden_rejected.jsonl, with your reason
  candidates -> evaluation/golden_candidates.jsonl is never modified here

Everything is written the moment you click, so closing the page loses nothing.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # server/, when run by streamlit

import streamlit as st  # noqa: E402

from evaluation import golden  # noqa: E402
from evaluation import snapshot as snap  # noqa: E402
from ingestion.filings import source_name  # noqa: E402

REJECTED = golden.GOLDEN.parent / "golden_rejected.jsonl"
CONTEXT_BEFORE = 300     # characters of the filing shown above the highlight ...
CONTEXT_AFTER = 800      # ... and below it, which is where table headers' rows continue


# --- State, all of it on disk ------------------------------------------------------------

def rejected() -> dict[str, dict]:
    if not REJECTED.exists():
        return {}
    rows = [json.loads(line) for line in REJECTED.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def save_rejected(rows: dict[str, dict]) -> None:
    REJECTED.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()),
                        encoding="utf-8", newline="\n")


def approve(case: golden.GoldenCase, answer: str, reviewer: str, note: str) -> None:
    approved = [c for c in golden.load(golden.GOLDEN) if c.id != case.id]
    data = case.model_dump(mode="json", exclude_defaults=True)
    data["answer"] = answer.strip()
    data["verified"] = {"by": reviewer, "on": date.today().isoformat(), **({"note": note} if note else {})}
    approved.append(golden.GoldenCase.model_validate(data))
    order = {c.id: i for i, c in enumerate(golden.load(golden.CANDIDATES))}
    golden.save(sorted(approved, key=lambda c: order.get(c.id, len(order))), golden.GOLDEN)
    rows = rejected()
    if rows.pop(case.id, None) is not None:
        save_rejected(rows)


def reject(case: golden.GoldenCase, reviewer: str, note: str) -> None:
    rows = rejected()
    rows[case.id] = {"id": case.id, "by": reviewer, "on": date.today().isoformat(), "note": note}
    save_rejected(rows)
    approved = golden.load(golden.GOLDEN)
    if any(c.id == case.id for c in approved):
        golden.save([c for c in approved if c.id != case.id], golden.GOLDEN)


def default_reviewer() -> str:
    try:
        return subprocess.check_output(["git", "config", "user.name"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


# --- Showing a passage in its filing -----------------------------------------------------

@st.cache_data(show_spinner="Reading the filing...")
def text_of(accession: str) -> str:
    return golden.filing_text(accession)


@st.cache_data
def names() -> dict[str, str]:
    return {f["accession"]: source_name(f) for f in snap.load()["filings"]}


def passage_html(text: str, start: int, end: int) -> str:
    lo, hi = max(0, start - CONTEXT_BEFORE), min(len(text), end + CONTEXT_AFTER)

    def lines(fragment: str) -> str:
        return html.escape(fragment).replace("\n", "<br>")

    return (
        "<div style='font-family: ui-monospace, monospace; font-size: 0.85rem; line-height: 1.5; "
        "max-height: 26rem; overflow-y: auto; padding: 0.75rem; border: 1px solid #8884; border-radius: 6px'>"
        f"{'…' if lo else ''}{lines(text[lo:start])}"
        f"<mark style='background: #fde68a; color: #111; padding: 0 2px'>{lines(text[start:end])}</mark>"
        f"{lines(text[end:hi])}{'…' if hi < len(text) else ''}</div>"
    )


def search_box(case: golden.GoldenCase) -> None:
    """For an unanswerable case: look for the thing the filing supposedly does not say."""
    tickers = {p.ticker for p in case.about}
    options = {n: a for a, n in names().items() if n.split()[0] in tickers}
    with st.expander("Search the filings to confirm they do not say", expanded=True):
        col1, col2 = st.columns([2, 1])
        pattern = col1.text_input("Words or regex", key=f"q-{case.id}")
        ordered = sorted(options)
        # Start on the filing the question is about, or the newest 10-K of that company.
        first = case.about[0]
        wanted = [f"{first.ticker} 10-K FY{first.fiscal_year}",
                  f"{first.ticker} 10-Q {first.fiscal_period}{first.fiscal_year}"]
        fallback = max((n for n in ordered if " 10-K " in n), key=lambda n: n[-4:], default=ordered[0])
        start = next((n for n in wanted if n in options), fallback)
        chosen = col2.selectbox("Filing", ordered, index=ordered.index(start), key=f"f-{case.id}")
        if pattern and chosen:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                st.error(f"Not a valid pattern: {exc}")
                return
            hits = [line for line in text_of(options[chosen]).split("\n") if regex.search(line)]
            st.caption(f"{len(hits)} matching line(s) in {chosen}")
            for line in hits[:15]:
                st.text(line[:400])


# --- Page --------------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Golden set review", layout="wide")
    candidates = golden.load(golden.CANDIDATES)
    approved = {c.id: c for c in golden.load(golden.GOLDEN)}
    refused = rejected()
    pending = [c for c in candidates if c.id not in approved and c.id not in refused]

    with st.sidebar:
        st.header("Golden set review")
        reviewer = st.text_input("Your name", value=st.session_state.get("reviewer") or default_reviewer())
        st.session_state["reviewer"] = reviewer
        st.metric("Left to review", len(pending))
        st.write(f"Approved {len(approved)} · Rejected {len(refused)} · Drafted {len(candidates)}")
        st.progress((len(approved) + len(refused)) / max(len(candidates), 1))
        cov = golden.coverage(list(approved.values()))
        st.caption("Approved so far, by type (each needs 8):")
        st.write({t: n for t, n in cov["by_type"].items()})
        st.caption(f"From 10-Qs: {cov['from_10q']} (needs {golden.MIN_FROM_10Q})")
        show = st.radio("Show", ["Pending", "Approved", "Rejected", "All"], horizontal=True)

    pool = {"Pending": pending,
            "Approved": [c for c in candidates if c.id in approved],
            "Rejected": [c for c in candidates if c.id in refused],
            "All": candidates}[show]
    if not pool:
        st.success("Nothing here." if show != "Pending" else
                   "Every candidate has been reviewed. Run  python -m evaluation.golden check  to see where the set stands.")
        return

    labels = [f"{c.id}  ({c.type.value})" for c in pool]
    index = st.session_state.get("index", 0) % len(pool)
    picked = st.selectbox("Case", range(len(pool)), index=index, format_func=lambda i: labels[i])
    case = approved.get(pool[picked].id, pool[picked]) if show != "Pending" else pool[picked]

    status = "approved" if case.id in approved else "rejected" if case.id in refused else "pending"
    st.caption(f"{case.type.value} · about " + ", ".join(
        f"{p.ticker} {p.fiscal_period or ''} {p.fiscal_year}".replace("  ", " ") for p in case.about) + f" · {status}")
    if case.id in refused:
        st.warning(f"Rejected: {refused[case.id]['note']}")

    for turn in case.history:
        st.markdown(f"**Earlier question:** {turn.question}  \n**Earlier answer:** {turn.answer}")
    st.subheader(case.question)
    answer = st.text_area("Reference answer (edit it if it is wrong or unclear)", value=case.answer,
                          key=f"a-{case.id}", height=100)
    if case.calculation:
        st.markdown(f"**Calculation:** `{case.calculation}`")
    if case.notes:
        st.info(case.notes)

    if case.spans:
        st.markdown(f"**Cited passages ({len(case.spans)})** - the highlighted text is what the answer rests on:")
        for n, span in enumerate(case.spans, start=1):
            st.markdown(f"{n}. **{names().get(span.accession, span.accession)}** · characters {span.start_char:,}-{span.end_char:,}")
            st.markdown(passage_html(text_of(span.accession), span.start_char, span.end_char), unsafe_allow_html=True)
    else:
        st.markdown("**No passage is cited: the right answer is to decline.** Check that the filing really does not say.")
        search_box(case)

    note = st.text_input("Note (required to reject; optional to approve)", key=f"n-{case.id}")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", use_container_width=True, disabled=not reviewer):
        approve(case, answer, reviewer, note)
        st.session_state["index"] = picked
        st.rerun()
    if col2.button("Reject", use_container_width=True, disabled=not (reviewer and note)):
        reject(case, reviewer, note)
        st.session_state["index"] = picked
        st.rerun()
    if col3.button("Skip for now", use_container_width=True):
        st.session_state["index"] = picked + 1
        st.rerun()
    if not reviewer:
        st.caption("Enter your name in the sidebar to approve or reject.")


if __name__ == "__main__":       # streamlit runs the script as __main__; tests import it
    main()
