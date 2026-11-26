"""Run the pipeline over the golden set and write one Results row (ROADMAP 1.8).

    python -m evaluation.evaluate                          # the Baseline row
    python -m evaluation.evaluate --answers-only           # generate answers, judge nothing
    python -m evaluation.evaluate --name "Hybrid (RRF)" --out hybrid_rrf --order 45

Four steps, each resumable from its cache:

  1. answer   every case through the pipeline as it runs (answers.py caches each).
  2. retrieve score retrieval against the golden spans (metrics.py). Recall@5 is
              scored on the passages the answer model was given. Recall@10 and
              nDCG@10 come from a top-10 search with the same retriever and the
              same query: above 20,000 chunks the candidate pool grows with k,
              so a top-10 list can order its first five differently.
  3. judge    RAGAS on the answerable cases (judge.py). Unanswerable cases are
              left to refusal accuracy (5.4).
  4. write    evaluation/results/{out}.json: every metric as a mean with a 95%
              bootstrap interval, plus $/query and p50/p95 latency (1.7). The
              Results block of ROADMAP.md is re-rendered from it.

Retrieval scores are free. The judge is about $1.10 for 89 cases, and answers
about $0.04 on Groq.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from evaluation import answers, golden, judge, metrics
from evaluation import snapshot as snap
from ingestion.pipeline import retrieve
from utils import config, logs

RESULTS_DIR = config.PROJECT_ROOT / "evaluation" / "results"
DEEP_K = 10


# --- Retrieval --------------------------------------------------------------------------

def span_ranges(case: golden.GoldenCase) -> list[metrics.Range]:
    return [metrics.Range(s.accession, s.start_char, s.end_char) for s in case.spans]


def chunk_ranges(contexts: list[dict]) -> list[metrics.Range | None]:
    """Each retrieved chunk as a stretch of its filing; None if it was never located in it."""
    return [metrics.Range(c["accession"], c["start_char"], c["end_char"])
            if c.get("accession") and 0 <= c["start_char"] < c["end_char"] else None
            for c in contexts]


def _located(ranges: list[metrics.Range | None]) -> list[metrics.Range]:
    # An unlocated chunk keeps its rank as a miss: a range that covers nothing.
    return [r if r is not None else metrics.Range("", 0, 1) for r in ranges]


def deep_contexts(store, record: dict) -> list[dict]:
    """The top 10 for the same query, or nothing when the pipeline did not retrieve."""
    if record["route"] != "retrieval" or not record["contexts"]:
        return []
    return [{"accession": c.accession, "start_char": c.start_char, "end_char": c.end_char}
            for c, _score in retrieve(store, record["search_query"], k=DEEP_K)]


def retrieval_scores(case: golden.GoldenCase, record: dict, deep: list[dict]) -> dict[str, float]:
    """Recall@5 on what the answer model saw; recall@10, MRR and nDCG@10 on the top 10."""
    spans = span_ranges(case)
    seen = _located(chunk_ranges(record["contexts"]))
    ranked = _located(chunk_ranges(deep))
    five = metrics.score_question(seen, spans, ks=(5,))
    ten = metrics.score_question(ranked, spans, ks=(10,))
    return {"recall_at_5": five["recall_at_5"], "recall_at_10": ten["recall_at_10"],
            "reciprocal_rank": ten["reciprocal_rank"], "ndcg_at_10": ten["ndcg_at_10"]}


# --- Aggregation ------------------------------------------------------------------------

def with_ci(values: list[float | None]) -> dict:
    """{"value", "ci", "n"} over the values that exist (None and NaN are left out, and n says so)."""
    data = np.array([np.nan if v is None else v for v in values], dtype=float)
    value, low, high = metrics.bootstrap_ci(data)
    return {"value": value, "ci": [low, high], "n": int(np.sum(~np.isnan(data)))}


def by_type(cases: list[golden.GoldenCase], per_case: dict[str, dict], names: list[str]) -> dict:
    """Each metric's mean per question type, for finding where the pipeline fails."""
    out = {}
    for t in sorted({c.type.value for c in cases}):
        ids = [c.id for c in cases if c.type.value == t]
        row = {"n": len(ids)}
        for name in names:
            vals = [per_case[i].get(name) for i in ids]
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            row[name] = float(np.mean(vals)) if vals else None
        out[t] = row
    return out


def commit() -> tuple[str, bool]:
    """The commit the run measured, and whether the working tree had changes on top."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, text=True).strip()
        # Tracked files only: an untracked file (this run's own results file, say) is
        # not code the run imported.
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "server"],
                                             cwd=config.REPO_ROOT, text=True).strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return "", True


def build_result(name: str, order: int, cases: list[golden.GoldenCase], records: dict[str, dict],
                 retrieval: dict[str, dict], grades: dict[str, dict], judge_name: str | None,
                 snapshot: dict, answer_set: str, judge_errors: list[dict] = ()) -> dict:
    """The results file: what the Results table reads, and the detail behind it.

    A case counts as judged only if every metric was graded without an error.
    """
    answerable = judge.answerable(cases)
    errored = {e["case_id"] for e in judge_errors}
    per_case = {c.id: {**retrieval.get(c.id, {}), **grades.get(c.id, {})} for c in cases}
    retrieval_names = ["recall_at_5", "recall_at_10", "reciprocal_rank", "ndcg_at_10"]
    out_metrics = {}
    for key in retrieval_names:
        out_metrics["mrr" if key == "reciprocal_rank" else key] = with_ci(
            [retrieval[c.id][key] for c in answerable])
    if grades:
        for m in judge.METRICS:
            out_metrics[m] = with_ci([grades.get(c.id, {}).get(m) for c in answerable])
    cost = answers.summarize([records[c.id] for c in cases])
    out_metrics["cost_per_query"] = cost["cost_per_query"]
    out_metrics["p50_seconds"] = cost["p50_seconds"]
    out_metrics["p95_seconds"] = cost["p95_seconds"]
    head, dirty = commit()
    answer_models = Counter(u["model"] for r in records.values() for u in r["usage"] if u.get("stage") == "answer")
    return {
        "configuration": name,
        "order": order,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "timestamp": int(time.time()),
        "commit": head,
        "uncommitted_changes": dirty,
        "snapshot": f"v{snapshot['version']}",
        "answers": answer_set,
        "judge": judge_name,
        "questions": {"all": len(cases), "answerable": len(answerable),
                      "judged": sum(1 for c in answerable if c.id in grades and c.id not in errored)},
        "judge_errors": [{"case_id": e["case_id"], "metric": e["metric"], "error": e["error"]}
                         for e in judge_errors],
        "metrics": out_metrics,
        "cost": cost,
        "routes": dict(Counter(r["route"] for r in records.values())),
        "answer_models": dict(answer_models),
        "by_type": by_type(answerable, per_case, retrieval_names + (list(judge.METRICS) if grades else [])),
        "per_case": per_case,
    }


# --- The run ----------------------------------------------------------------------------

def warm_up(store) -> None:
    """Load the embedder and the index before anything is timed.

    The first retrieval takes about 16 s while models load; timing it would
    put start-up, not the pipeline, into p95.
    """
    retrieve(store, "warm-up query", k=1)


RATE_LIMIT_WAIT = 60      # seconds; Groq's limits are per minute
RATE_LIMIT_TRIES = 10


def degraded(record: dict) -> str | None:
    """Why an answer did not run the pipeline as configured, or None if it did.

    The classifier and the rewriter swallow a failed call and fall back to
    rules, so a rate limit there leaves an answer that looks normal but was
    made on a lesser path.
    """
    if "call failed" in (record.get("intent_reason") or ""):
        return f"the classifier call failed ({record['intent_reason']})"
    if record.get("rewrite_reason") == "rewrite call failed":
        return "the rewrite call failed"
    return None


def answer_one(case, store, answer_dir, sleep=time.sleep) -> dict:
    """One case's answer, as the pipeline runs it.

    Groq's per-minute rate limit is waited out, whether it stopped the answer
    or quietly degraded the classifier or the rewriter. A degraded answer is
    thrown away and made again.
    """
    from backend.providers.base import ProviderError

    for attempt in range(RATE_LIMIT_TRIES):
        try:
            record = answers.answer(case, store, answer_dir)
        except ProviderError as exc:
            if "rate limit" not in str(exc).lower() or attempt == RATE_LIMIT_TRIES - 1:
                raise
            print(f"  rate limited; waiting {RATE_LIMIT_WAIT}s", flush=True)
            sleep(RATE_LIMIT_WAIT)
            continue
        why = degraded(record)
        if why is None:
            return record
        (answer_dir / f"{case.id}.json").unlink(missing_ok=True)
        if attempt == RATE_LIMIT_TRIES - 1:
            raise RuntimeError(f"{case.id}: {why}, {RATE_LIMIT_TRIES} times")
        print(f"  {why}; waiting {RATE_LIMIT_WAIT}s and answering again", flush=True)
        sleep(RATE_LIMIT_WAIT)
    raise AssertionError("unreachable")


def answer_all(cases, store, answer_dir) -> dict[str, dict]:
    records = {}
    for i, case in enumerate(cases, 1):
        cached = (answer_dir / f"{case.id}.json").exists()
        records[case.id] = answer_one(case, store, answer_dir)
        r = records[case.id]
        cost = "?" if r["cost"] is None else f"${r['cost']:.5f}"
        took = "cached" if cached else f"{r['seconds']:5.1f}s {cost}"
        print(f"  [{i:3}/{len(cases)}] {case.id:44} {r['route']:9} {took}", flush=True)
    return records


def judge_all(cases, records) -> tuple[dict[str, dict], str, judge.GradeReport]:
    samples = [judge.sample_for(c, records[c.id]) for c in judge.answerable(cases)]
    metric_set, calls = judge.make_judge()
    name = judge.judge_name()
    cache = judge.ScoreCache(config.JUDGE_CACHE_DIR / "scores.jsonl")
    done = [0]

    def progress(message):
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  graded {done[0]} so far ({message})", flush=True)

    report = asyncio.run(judge.grade(samples, metric_set, cache, name, calls, on_progress=progress))
    return report.scores(), name, report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="Baseline (current pipeline)", help="the Results row's label")
    parser.add_argument("--out", default="baseline", help="results file name, without .json")
    parser.add_argument("--order", type=int, default=0, help="the row's position in the Results table")
    parser.add_argument("--answers-only", action="store_true", help="generate answers and stop")
    parser.add_argument("--no-judge", action="store_true", help="write the row without RAGAS")
    args = parser.parse_args(argv)
    logs.setup()
    # Every answer from its stage's own model. In the app a busy model falls back
    # to the next; here that would make the score depend on rate-limit timing
    # (it did: one baseline answer came from 20b), so a busy model is waited out.
    config.GROQ_FALLBACK_MODELS = []

    snapshot = snap.load()
    cases = golden.load(golden.GOLDEN)
    store = answers.load_store(snapshot)
    answer_set = answers.fingerprint(snapshot)
    answer_dir = config.ANSWER_CACHE_DIR / answer_set
    print(f"{len(cases)} cases, snapshot v{snapshot['version']}, answers {answer_set}", flush=True)

    warm_up(store)
    started = time.monotonic()
    print("answering", flush=True)
    records = answer_all(cases, store, answer_dir)
    print(f"answered in {time.monotonic() - started:.0f}s", flush=True)
    if args.answers_only:
        return 0

    print("scoring retrieval", flush=True)
    retrieval = {c.id: retrieval_scores(c, records[c.id], deep_contexts(store, records[c.id]))
                 for c in judge.answerable(cases)}

    grades, judge_name, report = {}, None, None
    if not args.no_judge:
        print(f"judging {len(judge.answerable(cases))} answerable cases", flush=True)
        grades, judge_name, report = judge_all(cases, records)
        print(f"  {len(report.rows)} grades: {report.cache_hits} from cache, {report.judge_calls} judge calls, "
              f"{len(report.errors)} errors", flush=True)
        for row in report.errors:
            print(f"  ERROR {row['case_id']} / {row['metric']}: {row['error'][:300]}", flush=True)

    result = build_result(args.name, args.order, cases, records, retrieval, grades, judge_name,
                          snapshot, answer_set, report.errors if report else [])
    path = RESULTS_DIR / f"{args.out}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    from tools import roadmap
    roadmap.main([])

    m = result["metrics"]
    show = lambda k, f: "—" if m.get(k) is None or (isinstance(m[k], dict) and np.isnan(m[k]["value"])) \
        else (f.format(m[k]["value"]) + f" [{f.format(m[k]['ci'][0])}–{f.format(m[k]['ci'][1])}]"
              if isinstance(m[k], dict) else f.format(m[k]))
    print(f"\n{args.name} -> {path.name}")
    for key in ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", *(judge.METRICS if grades else ())):
        print(f"  {key:20} {show(key, '{:.3f}')}")
    print(f"  {'$/query':20} {show('cost_per_query', '${:.5f}')}")
    print(f"  {'p50 / p95':20} {show('p50_seconds', '{:.1f}s')} / {show('p95_seconds', '{:.1f}s')}")
    print(f"  routes {result['routes']}; answer models {result['answer_models']}")
    if result["uncommitted_changes"]:
        print("  NOTE: server/ had uncommitted changes; the row names a commit that is not exactly what ran")
    return 1 if report and report.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
