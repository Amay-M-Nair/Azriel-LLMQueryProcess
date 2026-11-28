"""Keep ROADMAP.md true without anyone remembering to edit it.

Two blocks in ROADMAP.md are generated, never typed:

  status   one row per phase: how many subphases are done, and what is next.
           A subphase's box is ticked by its "done when" test - any test
           marked `@pytest.mark.roadmap("x.y")`. When the suite runs, the hook
           in server/conftest.py hands the outcomes to `apply_outcomes`: all
           marked tests passed -> ticked; any failed or skipped -> unticked.
           Subphases with no test in the run keep the box they had.

  results  the ablation table, rendered from evaluation/results/*.json. Each
           file is one configuration's run; `python -m evaluation.evaluate`
           writes them (Phase 1).

    python -m tools.roadmap           # re-render both blocks from what is there
    python -m tools.roadmap --check   # exit 1 if ROADMAP.md is out of date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from utils import config

ROADMAP = config.REPO_ROOT / "ROADMAP.md"
RESULTS_DIR = config.PROJECT_ROOT / "evaluation" / "results"

PHASE = re.compile(r"^## Phase (\d+) — (.+?)\s*(?:\(.*\))?\s*$")
SUBPHASE = re.compile(r"^- \[( |x)\] \*\*(\d+\.\d+) (.+?)\*\*\s*$")
BLOCK = "<!-- {name}:start -->\n{body}\n<!-- {name}:end -->"

# Columns of the ablation table: (heading, key in the results JSON, format).
COLUMNS = [
    ("Recall@5", "recall_at_5", "{:.3f}"),
    ("MRR", "mrr", "{:.3f}"),
    ("nDCG@10", "ndcg_at_10", "{:.3f}"),
    ("Faithfulness", "faithfulness", "{:.3f}"),
    ("Unsupported claims", "unsupported_claim_rate", "{:.1%}"),
    ("Correct refusals", "refusal_accuracy", "{:.1%}"),
    ("$/query", "cost_per_query", "${:.4f}"),
    ("p95 latency", "p95_seconds", "{:.1f}s"),
]
NO_RESULTS = (
    "No runs yet. The baseline (1.8) comes first; every later subphase adds a "
    "row, including rows where nothing improved."
)


@dataclass
class Subphase:
    id: str
    title: str
    done: bool


@dataclass
class Phase:
    number: int
    title: str
    subphases: list[Subphase]


def parse(text: str) -> list[Phase]:
    """Every phase and subphase, with its current box."""
    phases: list[Phase] = []
    for line in text.splitlines():
        if m := PHASE.match(line):
            phases.append(Phase(int(m.group(1)), m.group(2).strip(), []))
        elif (m := SUBPHASE.match(line)) and phases:
            phases[-1].subphases.append(Subphase(m.group(2), m.group(3), m.group(1) == "x"))
    return phases


def apply_outcomes(text: str, outcomes: dict[str, bool]) -> str:
    """Tick or untick each subphase named in `outcomes` (id -> all passed)."""
    def tick(m: re.Match) -> str:
        sub_id = m.group(2)
        if sub_id not in outcomes:
            return m.group(0)
        box = "x" if outcomes[sub_id] else " "
        return f"- [{box}] **{sub_id} {m.group(3)}**"

    return "\n".join(
        SUBPHASE.sub(tick, line) for line in text.split("\n")
    )


def render_status(phases: list[Phase]) -> str:
    rows = ["| Phase | Done | Next |", "|---|---|---|"]
    for phase in phases:
        done = sum(s.done for s in phase.subphases)
        pending = [s for s in phase.subphases if not s.done]
        upcoming = f"{pending[0].id} {pending[0].title}" if pending else "complete"
        rows.append(f"| {phase.number} {phase.title} | {done}/{len(phase.subphases)} | {upcoming} |")
    return "\n".join(rows)


def load_results(directory: Path = RESULTS_DIR) -> list[dict]:
    """Results files in run order (the `order` field, then file name)."""
    if not directory.exists():
        return []
    results = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "configuration" in data:          # other JSON (e.g. the memory spike) is not a row
            results.append(data)
    return sorted(results, key=lambda r: (r.get("order", 0), r["configuration"]))


def _cell(value, fmt: str) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):          # {"value": x, "ci": [lo, hi]}
        main = fmt.format(value["value"])
        if value.get("ci"):
            lo, hi = value["ci"]
            return f"{main} [{fmt.format(lo)}–{fmt.format(hi)}]"
        return main
    return fmt.format(value)


def render_results(results: list[dict]) -> str:
    header = "| Configuration | " + " | ".join(c[0] for c in COLUMNS) + " | Run |"
    rule = "|---" * (len(COLUMNS) + 2) + "|"
    if not results:
        empty = "| Baseline (current pipeline) |" + " |" * (len(COLUMNS) + 1)
        return f"{NO_RESULTS}\n\n{header}\n{rule}\n{empty}"
    rows = [header, rule]
    for r in results:
        metrics = r.get("metrics", {})
        cells = [_cell(metrics.get(key), fmt) for _, key, fmt in COLUMNS]
        run = " · ".join(x for x in (r.get("date"), r.get("commit", "")[:7], r.get("snapshot")) if x)
        rows.append(f"| {r['configuration']} | " + " | ".join(cells) + f" | {run} |")
    return "\n".join(rows)


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"<!-- {name}:start -->.*?<!-- {name}:end -->", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"ROADMAP.md has no {name} block")
    return pattern.sub(lambda _m: BLOCK.format(name=name, body=body), text, count=1)


def render(text: str, results: list[dict] | None = None) -> str:
    """Both generated blocks, recomputed from the boxes and the results files."""
    text = replace_block(text, "status", render_status(parse(text)))
    return replace_block(text, "results", render_results(load_results() if results is None else results))


def update(outcomes: dict[str, bool] | None = None, path: Path = ROADMAP) -> bool:
    """Apply test outcomes and re-render. Returns True if the file changed."""
    if not path.exists():
        return False
    before = path.read_text(encoding="utf-8")
    after = render(apply_outcomes(before, outcomes or {}))
    if after != before:
        path.write_text(after, encoding="utf-8")
    return after != before


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if out of date")
    args = parser.parse_args(argv)
    current = ROADMAP.read_text(encoding="utf-8")
    expected = render(current)
    if args.check:
        if expected != current:
            print("ROADMAP.md is out of date: run  python -m tools.roadmap")
            return 1
        print("ROADMAP.md is up to date")
        return 0
    if expected != current:
        ROADMAP.write_text(expected, encoding="utf-8")
        print("ROADMAP.md updated")
    else:
        print("ROADMAP.md already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
