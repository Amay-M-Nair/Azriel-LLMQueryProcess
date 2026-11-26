"""RAGAS with DeepSeek on AWS Bedrock as the judge, every grade cached (ROADMAP 1.6).

Five RAGAS metrics, scored per answer from 0 to 1:

  faithfulness        are the answer's claims supported by the retrieved passages?
  answer_relevancy    does the answer address the question that was asked?
  context_precision   are the passages that matter ranked above the ones that do not?
  context_recall      do the passages hold what the reference answer needs?
  answer_correctness  does the answer agree with the golden reference answer?

Grades are cached at two levels, both under data/judge_cache/:
  scores.jsonl   one line per (answer, metric, judge): a rerun of the same
                 answers costs no calls at all.
  calls/{model}/ RAGAS's own cache of each judge call, so a metric interrupted
                 halfway resumes without repeating calls. One per judge model:
                 RAGAS keys it by prompt alone.

A throttled call is waited out. Any other failure is recorded and not cached,
so the next run tries it again.

    python -m evaluation.judge --ping           # one tiny call: is the judge reachable?
    python -m evaluation.judge                  # grade 5 answered cases, one per type
    python -m evaluation.judge --ids a b c      # these cases
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluation import golden
from utils import config, logs

log = logs.get(__name__)

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness")
REPORT = config.PROJECT_ROOT / "evaluation" / "results" / "ragas_check.json"

# What each RAGAS metric is given.
ARGUMENTS: dict[str, Callable[["Sample"], dict]] = {
    "faithfulness": lambda s: {"user_input": s.user_input, "response": s.response,
                               "retrieved_contexts": list(s.contexts)},
    "answer_relevancy": lambda s: {"user_input": s.user_input, "response": s.response},
    "context_precision": lambda s: {"user_input": s.user_input, "reference": s.reference,
                                    "retrieved_contexts": list(s.contexts)},
    "context_recall": lambda s: {"user_input": s.user_input, "retrieved_contexts": list(s.contexts),
                                 "reference": s.reference},
    "answer_correctness": lambda s: {"user_input": s.user_input, "response": s.response,
                                     "reference": s.reference},
}

# Settings that change what a metric's grade means. Passed to the metric, and part
# of every grade's key, so changing one re-grades that metric only.
METRIC_SETTINGS: dict[str, dict] = {
    # RAGAS's default strictness=3 sends one prompt three times and averages. The
    # call cache answers the second and third from the first, so 3 would only
    # pretend to average: 1 says what actually happens.
    "answer_relevancy": {"strictness": 1},
    # Facts only. RAGAS's default gives 25% to embedding similarity with the
    # reference, which pays a wrong figure on the right topic: an answer that put
    # Apple's Services at $35B (the filing says $109B) scored 0.45.
    "answer_correctness": {"weights": [1.0, 0.0]},
}


# What the passage metrics are when the pipeline retrieved nothing (a question it
# routed to "clarify", say), which RAGAS refuses to score. Context precision and
# recall are 0: no passage was useful, none held what the reference needs.
# Faithfulness does not apply: a reply with no passages makes no claim from them.
# None is left out of the mean, and the result's n says how many were.
NO_PASSAGES: dict[str, float | None] = {"context_precision": 0.0, "context_recall": 0.0, "faithfulness": None}


@dataclass(frozen=True)
class Sample:
    """One answer to grade: the question as asked, what came back, and the golden reference."""

    case_id: str
    user_input: str
    response: str
    contexts: tuple[str, ...]
    reference: str

    def key(self, metric: str, judge: str) -> str:
        blob = json.dumps([metric, METRIC_SETTINGS.get(metric, {}), judge, self.user_input, self.response,
                           list(self.contexts), self.reference])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def user_input_for(case: golden.GoldenCase) -> str:
    """The question as the judge sees it: a follow-up carries the conversation before it."""
    if not case.history:
        return case.question
    turns = "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in case.history)
    return f"{turns}\nUser: {case.question}"


def sample_for(case: golden.GoldenCase, answer: dict, contexts: int = config.JUDGE_CONTEXTS) -> Sample:
    return Sample(
        case_id=case.id,
        user_input=user_input_for(case),
        response=answer["response"],
        contexts=tuple(c["text"] for c in answer["contexts"][:contexts]),
        reference=case.answer,
    )


# --- The score cache ---------------------------------------------------------------------

class ScoreCache:
    """Grades already given, one JSON line each, keyed by answer, metric and judge."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["key"]] = row
        self.hits = 0

    def get(self, key: str) -> dict | None:
        row = self.rows.get(key)
        if row is not None:
            self.hits += 1
        return row

    def put(self, row: dict) -> None:
        self.rows[row["key"]] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- Grading -----------------------------------------------------------------------------

def is_throttled(exc: BaseException) -> bool:
    """A 429 or 503 anywhere in the chain: instructor wraps the endpoint's error in its own."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if re.search(r"\b(429|503)\b|Throttl|RateLimit|Too many requests|ServiceUnavailable",
                     f"{type(exc).__name__}: {exc}", re.IGNORECASE):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@dataclass
class GradeReport:
    rows: list[dict] = field(default_factory=list)       # one per (case, metric)
    cache_hits: int = 0
    judge_calls: int = 0

    @property
    def errors(self) -> list[dict]:
        return [r for r in self.rows if r.get("error")]

    def scores(self) -> dict[str, dict[str, float | None]]:
        out: dict[str, dict[str, float | None]] = {}
        for row in self.rows:
            out.setdefault(row["case_id"], {})[row["metric"]] = row.get("value")
        return out


async def grade(samples: list[Sample], metrics: dict[str, Any], cache: ScoreCache, judge: str,
                calls: Callable[[], int] = lambda: 0, sleep=asyncio.sleep,
                on_progress: Callable[[str], None] | None = None,
                concurrency: int = config.JUDGE_CONCURRENCY) -> GradeReport:
    """Score every sample on every metric, from the cache where possible.

    Up to `concurrency` grades run at once. Rows come back in sample-then-metric
    order whatever order they finish in.
    """
    gate = asyncio.Semaphore(concurrency)
    before_hits, before_calls = cache.hits, calls()

    async def one(sample: Sample, name: str, metric: Any) -> dict:
        key = sample.key(name, judge)
        cached = cache.get(key)
        if cached is not None:
            return cached | {"cached": True}
        row = {"key": key, "case_id": sample.case_id, "metric": name, "judge": judge}
        if not sample.contexts and name in NO_PASSAGES:
            row |= {"value": NO_PASSAGES[name], "reason": "no passages were retrieved",
                    "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            cache.put(row)
            return row | {"cached": False}
        async with gate:
            if on_progress:
                on_progress(f"grading {sample.case_id}: {name}")
            for attempt in range(config.JUDGE_RETRIES + 1):
                try:
                    result = await metric.ascore(**ARGUMENTS[name](sample))
                except Exception as exc:          # the judge's failures are data, not crashes
                    if is_throttled(exc) and attempt < config.JUDGE_RETRIES:
                        wait = 10 * 2 ** attempt
                        log.info("judge throttled; waiting %ds", wait)
                        await sleep(wait)
                        continue
                    return row | {"value": None, "error": logs.redact(f"{type(exc).__name__}: {exc}")[:500],
                                  "cached": False}
                value = result.value
                row |= {"value": None if value is None else float(value),
                        "reason": (getattr(result, "reason", None) or "")[:1000],
                        "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                cache.put(row)
                return row | {"cached": False}

    rows = await asyncio.gather(*(one(s, name, m) for s in samples for name, m in metrics.items()))
    return GradeReport(rows=list(rows), cache_hits=cache.hits - before_hits, judge_calls=calls() - before_calls)


# --- The real judge ---------------------------------------------------------------------

def judge_name() -> str:
    """What a grade is keyed by: RAGAS version, judge model, embedding model."""
    import ragas

    return f"ragas-{ragas.__version__}/{config.JUDGE_MODEL}/{config.JUDGE_EMBED_MODEL}"


def calls_cache_dir(cache_dir: Path = config.JUDGE_CACHE_DIR) -> Path:
    """RAGAS's call cache, one per judge model.

    RAGAS keys a cached call by its prompt alone, not by the model. A shared
    cache would hand a new judge the old judge's replies to the same prompts
    (it did: the first DeepSeek run reused Gemini's), mixing two judges in one grade.
    """
    return Path(cache_dir) / "calls" / config.JUDGE_MODEL


def bedrock_key() -> str:
    key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
    if not key:
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is not set. "
                           "Create a Bedrock API key in the Bedrock console and add it to .env.")
    return key


def make_judge(cache_dir: Path = config.JUDGE_CACHE_DIR) -> tuple[dict[str, Any], Callable[[], int]]:
    """The five RAGAS metrics on the Bedrock judge, and a counter of the judge calls made."""
    import httpx
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerCorrectness, AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness,
    )

    counted = []

    async def count(response):
        counted.append(response.status_code)

    client = AsyncOpenAI(api_key=bedrock_key(), base_url=config.JUDGE_BASE_URL, max_retries=0,
                         http_client=httpx.AsyncClient(timeout=180, event_hooks={"response": [count]}))
    cache = DiskCacheBackend(cache_dir=str(calls_cache_dir(cache_dir)))
    llm = llm_factory(config.JUDGE_MODEL, provider="openai", client=client, cache=cache,
                      max_tokens=config.JUDGE_MAX_TOKENS)
    embeddings = HuggingFaceEmbeddings(model=config.JUDGE_EMBED_MODEL, cache=cache)
    settings = METRIC_SETTINGS
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, **settings["answer_relevancy"]),
        "context_precision": ContextPrecisionWithReference(llm=llm, name="context_precision"),
        "context_recall": ContextRecall(llm=llm),
        "answer_correctness": AnswerCorrectness(llm=llm, **settings["answer_correctness"]),
    }
    return metrics, lambda: len(counted)


# --- CLI ----------------------------------------------------------------------------

def answerable(cases: list[golden.GoldenCase]) -> list[golden.GoldenCase]:
    """The cases RAGAS can grade: those with a cited answer.

    An unanswerable case's right answer is a refusal. RAGAS would mark it down
    (nothing for faithfulness to check, a "noncommittal" answer for relevancy),
    so declining is scored on its own, by refusal accuracy (ROADMAP 5.4).
    """
    return [c for c in cases if c.spans]


def default_cases(cases: list[golden.GoldenCase], n: int = 5) -> list[golden.GoldenCase]:
    """The first answerable case of each type, in a fixed order, until there are n."""
    order = [golden.QuestionType.TABLE_LOOKUP, golden.QuestionType.TEXT_LOOKUP, golden.QuestionType.CROSS_PERIOD,
             golden.QuestionType.DEFINITION, golden.QuestionType.SUMMARY, golden.QuestionType.FOLLOW_UP]
    picked = []
    for t in order:
        found = next((c for c in answerable(cases) if c.type is t), None)
        if found:
            picked.append(found)
    return picked[:n]


def record(report: GradeReport, judge: str, answers_from: str, path: Path = REPORT) -> dict:
    """Append this run to evaluation/results/ragas_check.json and return the whole file."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    data["judge"] = judge
    data["answers"] = answers_from
    data["scores"] = report.scores()
    data["runs"].append({
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": judge,
        "graded": len(report.rows),
        "cache_hits": report.cache_hits,
        "judge_calls": report.judge_calls,
        "errors": len(report.errors),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return data


def ping() -> int:
    """One tiny call to the judge model; if it is refused, list the models the key can reach."""
    from openai import OpenAI

    client = OpenAI(api_key=bedrock_key(), base_url=config.JUDGE_BASE_URL, max_retries=0, timeout=60)
    try:
        reply = client.chat.completions.create(
            model=config.JUDGE_MODEL, max_tokens=20,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}])
    except Exception as exc:
        print(f"{config.JUDGE_MODEL} refused: {logs.redact(str(exc))[:400]}")
        try:
            print("models this key can reach:\n  " + "\n  ".join(sorted(m.id for m in client.models.list())))
        except Exception as inner:
            print(f"could not list models either: {logs.redact(str(inner))[:300]}")
        return 1
    usage = reply.usage
    print(f"{config.JUDGE_MODEL} answered {reply.choices[0].message.content!r} "
          f"({usage.prompt_tokens} tokens in, {usage.completion_tokens} out)")
    return 0


def main(argv: list[str]) -> int:
    from evaluation import answers

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", nargs="*", help="case ids (default: one answered case per type, 5 in all)")
    parser.add_argument("--ping", action="store_true", help="one tiny call: is the judge reachable?")
    args = parser.parse_args(argv)
    logs.setup()
    if args.ping:
        return ping()

    pool = golden.load(golden.GOLDEN)
    chosen = [c for c in pool if c.id in set(args.ids)] if args.ids else default_cases(pool)
    skipped = [c.id for c in chosen if not c.spans]
    if skipped:
        print(f"  not graded by RAGAS (unanswerable, scored by refusal accuracy in 5.4): {', '.join(skipped)}")
    chosen = answerable(chosen)
    store = answers.load_store()
    answer_dir = config.ANSWER_CACHE_DIR / answers.fingerprint()
    samples = []
    for case in chosen:
        print(f"  answering {case.id}", flush=True)
        samples.append(sample_for(case, answers.answer(case, store, answer_dir)))

    metrics, calls = make_judge()
    judge = judge_name()
    cache = ScoreCache(config.JUDGE_CACHE_DIR / "scores.jsonl")
    started = time.monotonic()
    report = asyncio.run(grade(samples, metrics, cache, judge, calls,
                               on_progress=lambda m: print(f"  {m}", flush=True)))
    record(report, judge, answer_dir.name)

    print(f"\nanswers {answer_dir.name}; judge {config.JUDGE_MODEL}")
    print(f"{'case':38}" + "".join(f"{m[:12]:>13}" for m in METRICS))
    for case_id, scores in report.scores().items():
        cells = "".join(f"{'—' if scores.get(m) is None else format(scores[m], '.2f'):>13}" for m in METRICS)
        print(f"{case_id:38}{cells}")
    print(f"\n{len(report.rows)} grades: {report.cache_hits} from cache, {report.judge_calls} judge calls, "
          f"{time.monotonic() - started:.0f}s")
    for row in report.errors:
        print(f"  ERROR {row['case_id']} / {row['metric']}: {row['error'][:200]}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
