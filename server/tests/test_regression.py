"""The current pipeline stays within evaluation/thresholds.json (ROADMAP 1.9).

The last test is the harness itself: it reads the run the thresholds name and
fails on any shortfall. The rest check the rule on hand-made runs.
"""

import json
import math

import pytest

from evaluation import regression

pytestmark = pytest.mark.roadmap("1.9")


def run(value=0.5, low=0.4, cost=0.0005, p95=10.0, snapshot="v1", judged=89):
    metrics = {k: {"value": value, "ci": [low, value + 0.1], "n": 89} for k in regression.FLOORED}
    metrics |= {"cost_per_query": cost, "p95_seconds": p95}
    return {"configuration": "Baseline (current pipeline)", "snapshot": snapshot, "date": "2026-11-27",
            "commit": "abc", "questions": {"all": 101, "answerable": 89, "judged": judged}, "metrics": metrics}


def test_floors_are_the_lower_ends_of_the_intervals_and_ceilings_have_headroom():
    t = regression.thresholds_from(run(low=0.4567), "baseline")
    assert t["floors"]["recall_at_5"] == 0.456, "rounded down, never up"
    assert t["ceilings"] == {"cost_per_query": 0.00075, "p95_seconds": 20.0}


def test_the_run_the_thresholds_came_from_passes():
    assert regression.check(run(), regression.thresholds_from(run(), "baseline")) == []


def test_a_drop_below_a_floor_fails():
    t = regression.thresholds_from(run(), "baseline")
    problems = regression.check(run(value=0.39), t)
    assert any("recall_at_5 0.390 is below its floor 0.400" in p for p in problems)


def test_a_cost_or_latency_blowout_fails_and_unknown_cost_fails():
    t = regression.thresholds_from(run(), "baseline")
    assert any("cost_per_query" in p for p in regression.check(run(cost=0.001), t))
    assert any("p95_seconds" in p for p in regression.check(run(p95=25.0), t))
    assert any("cost_per_query is unknown" in p for p in regression.check(run(cost=None), t))


def test_another_snapshot_or_ungraded_cases_are_not_comparable():
    t = regression.thresholds_from(run(), "baseline")
    assert any("snapshot" in p for p in regression.check(run(snapshot="v2"), t))
    assert any("88 of 89" in p for p in regression.check(run(judged=88), t))


def test_a_metric_with_no_interval_gets_no_floor():
    r = run()
    r["metrics"]["faithfulness"]["ci"] = [math.nan, math.nan]
    assert "faithfulness" not in regression.thresholds_from(r, "baseline")["floors"]


def test_the_current_pipeline_clears_its_thresholds():
    if not regression.THRESHOLDS.exists():
        pytest.skip("no thresholds yet: run  python -m evaluation.regression --set baseline  (from server/)")
    thresholds = json.loads(regression.THRESHOLDS.read_text(encoding="utf-8"))
    problems = regression.check(regression.load_result(thresholds["result"]), thresholds)
    assert not problems, "\n".join(problems)
