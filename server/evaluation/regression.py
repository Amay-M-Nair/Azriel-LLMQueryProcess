"""Floors and ceilings the current pipeline must stay within (ROADMAP 1.9).

    python -m evaluation.regression --set baseline    # thresholds from results/baseline.json
    python -m evaluation.regression                   # check that run against them

The rule, set once from a run and changed only on purpose:

  floors    each quality metric's floor is the lower end of that run's 95%
            bootstrap interval. Falling below it is more than resampling the
            questions would explain.
  ceilings  $/query may grow by half and p95 latency may double before it
            fails: latency moves with the network and the machine.
  same set  a run on another snapshot, or with cases left ungraded, fails:
            its numbers are not comparable.

tests/test_regression.py runs the check, and tests/test_roadmap.py already
fails when the pipeline changes after the last run, so a change has to be
re-measured, and the new numbers have to clear the floors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from utils import config

THRESHOLDS = config.PROJECT_ROOT / "evaluation" / "thresholds.json"
RESULTS_DIR = config.PROJECT_ROOT / "evaluation" / "results"

FLOORED = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", "faithfulness", "answer_relevancy",
           "context_precision", "context_recall", "answer_correctness")
COST_HEADROOM = 1.5
LATENCY_HEADROOM = 2.0


def thresholds_from(result: dict, name: str) -> dict:
    m = result["metrics"]
    floors = {}
    for key in FLOORED:
        low = (m.get(key) or {}).get("ci", [math.nan])[0]
        if low is not None and not math.isnan(low):
            floors[key] = math.floor(low * 1000) / 1000
    return {
        "result": name,
        "configuration": result["configuration"],
        "snapshot": result["snapshot"],
        "set_from": {"date": result["date"], "commit": result["commit"]},
        "floors": floors,
        "ceilings": {"cost_per_query": round(m["cost_per_query"] * COST_HEADROOM, 6),
                     "p95_seconds": round(m["p95_seconds"] * LATENCY_HEADROOM, 1)},
    }


def check(result: dict, thresholds: dict) -> list[str]:
    """Every way the run falls short; empty when it passes."""
    problems = []
    if result["snapshot"] != thresholds["snapshot"]:
        problems.append(f"run is on snapshot {result['snapshot']}, thresholds on {thresholds['snapshot']}: "
                        "set new thresholds on purpose")
    q = result["questions"]
    if q["judged"] < q["answerable"]:
        problems.append(f"only {q['judged']} of {q['answerable']} answerable cases were graded")
    m = result["metrics"]
    for key, floor in thresholds["floors"].items():
        value = (m.get(key) or {}).get("value")
        if value is None or math.isnan(value):
            problems.append(f"{key} is missing")
        elif value < floor:
            problems.append(f"{key} {value:.3f} is below its floor {floor:.3f}")
    for key, ceiling in thresholds["ceilings"].items():
        value = m.get(key)
        if value is None:
            problems.append(f"{key} is unknown")
        elif value > ceiling:
            problems.append(f"{key} {value:.4g} is above its ceiling {ceiling:.4g}")
    return problems


def load_result(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", metavar="RESULT", help="write thresholds from results/RESULT.json")
    args = parser.parse_args(argv)
    if args.set:
        thresholds = thresholds_from(load_result(args.set), args.set)
        THRESHOLDS.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {THRESHOLDS.name} from {args.set}.json")
        for key, floor in thresholds["floors"].items():
            print(f"  {key:20} >= {floor:.3f}")
        for key, ceiling in thresholds["ceilings"].items():
            print(f"  {key:20} <= {ceiling:g}")
        return 0
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems = check(load_result(thresholds["result"]), thresholds)
    for p in problems:
        print(f"  FAIL {p}")
    print("passes" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
