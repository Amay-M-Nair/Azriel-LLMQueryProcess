"""ROADMAP.md is generated where it can be, and checked where it cannot.

The failure this guards against is a roadmap that says one thing while the
code says another: a box ticked over a failing check, or a results table older
than the pipeline it describes.
"""

import subprocess

import pytest

from tools import roadmap
from utils import config

SAMPLE = """# Roadmap

<!-- status:start -->
old
<!-- status:end -->

<!-- results:start -->
old
<!-- results:end -->

## Phase 0 — Foundations
- [x] **0.1 First thing**
- [ ] **0.2 Second thing**

## Phase 1 — Data and baseline (doc step 1)
- [ ] **1.1 Fetch**
"""

pytestmark = pytest.mark.roadmap("0.4")


def test_parses_phases_and_boxes():
    phases = roadmap.parse(SAMPLE)
    assert [(p.number, p.title) for p in phases] == [(0, "Foundations"), (1, "Data and baseline")]
    assert [(s.id, s.done) for s in phases[0].subphases] == [("0.1", True), ("0.2", False)]


def test_outcomes_tick_and_untick_only_what_ran():
    text = roadmap.apply_outcomes(SAMPLE, {"0.1": False, "0.2": True})
    phases = roadmap.parse(text)
    assert [s.done for s in phases[0].subphases] == [False, True]
    assert phases[1].subphases[0].done is False, "a subphase with no test changed"


def test_status_counts_and_names_the_next_step():
    body = roadmap.render_status(roadmap.parse(SAMPLE))
    assert "| 0 Foundations | 1/2 | 0.2 Second thing |" in body
    assert "| 1 Data and baseline | 0/1 | 1.1 Fetch |" in body


def test_render_is_idempotent():
    once = roadmap.render(SAMPLE, results=[])
    assert roadmap.render(once, results=[]) == once


def test_results_rows_carry_their_run():
    rows = roadmap.render_results([{
        "configuration": "Baseline",
        "date": "2026-11-24", "commit": "abcdef123456", "snapshot": "snap-1",
        "metrics": {"recall_at_5": {"value": 0.5, "ci": [0.4, 0.6]}, "cost_per_query": 0.0012},
    }])
    assert "| Baseline | 0.500 [0.400–0.600] |" in rows
    assert "$0.0012" in rows
    assert "2026-11-24 · abcdef1 · snap-1" in rows


def test_empty_results_keep_a_baseline_placeholder():
    assert roadmap.NO_RESULTS in roadmap.render_results([])


def test_the_real_roadmap_parses_completely():
    phases = roadmap.parse(roadmap.ROADMAP.read_text(encoding="utf-8"))
    assert [p.number for p in phases] == list(range(10))
    ids = [s.id for p in phases for s in p.subphases]
    assert len(ids) == len(set(ids)), "a subphase id is used twice"
    for phase in phases:
        assert all(s.id.startswith(f"{phase.number}.") for s in phase.subphases)


def test_results_are_not_older_than_the_pipeline():
    """Once results exist, a pipeline change without a fresh run fails here."""
    results = roadmap.load_results()
    if not results:
        return  # nothing to be stale yet: the baseline is subphase 1.8
    try:
        last_change = int(subprocess.check_output(
            ["git", "log", "-1", "--format=%ct", "--",
             "server/backend", "server/ingestion", "server/vectorstore"],
            cwd=config.REPO_ROOT, text=True,
        ).strip() or 0)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    newest = max(int(r.get("timestamp", 0)) for r in results)
    assert newest >= last_change, (
        "the pipeline changed after the last evaluation run - "
        "rerun  python -m evaluation.run_eval  (from server/)"
    )
