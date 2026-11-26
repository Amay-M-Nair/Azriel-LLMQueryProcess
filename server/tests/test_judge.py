"""The RAGAS judge (ROADMAP 1.6): what it is given, what it caches, how it stops.

Offline tests replace RAGAS metrics with fakes that count their calls. The
last test reads evaluation/results/ragas_check.json, written by real runs of
`python -m evaluation.judge`: the "done when" is five cases graded on every
metric, then a rerun served entirely from the cache.
"""

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

from evaluation import golden, judge
from evaluation.judge import Sample, ScoreCache
from utils import config

pytestmark = pytest.mark.roadmap("1.6")

JUDGE = "ragas-test/judge-test"


def case(**overrides):
    base = {
        "id": "nwnd-sales", "type": "table_lookup", "question": "What were Northwind's net sales in 2026?",
        "answer": "$1,000 million.",
        "spans": [{"accession": "0001234567-26-000001", "start_char": 0, "end_char": 5, "quote": "Total"}],
        "about": [{"ticker": "NWND", "fiscal_year": 2026, "fiscal_period": "FY"}],
    }
    return golden.GoldenCase.model_validate(base | overrides)


def sample(case_id="nwnd-sales", response="Net sales were $1,000 million [1]."):
    return Sample(case_id, "What were Northwind's net sales in 2026?", response,
                  ("Total net sales | 1,000 | 900",), "$1,000 million.")


class FakeMetric:
    """Answers with a fixed score; can be told to fail first."""

    def __init__(self, value=0.8, failures=()):
        self.value = value
        self.failures = list(failures)
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(value=self.value, reason="because")


def fakes(**overrides):
    metrics = {name: FakeMetric() for name in judge.METRICS}
    metrics.update(overrides)
    return metrics


def calls_of(metrics):
    return sum(len(m.calls) for m in metrics.values())


def run(samples, metrics, cache, sleep=None):
    async def no_wait(seconds):
        waited.append(seconds)

    waited = []
    report = asyncio.run(judge.grade(samples, metrics, cache, JUDGE, calls=lambda: calls_of(metrics),
                                     sleep=sleep or no_wait))
    return report, waited


# --- What the judge is given -----------------------------------------------------------------

def test_a_plain_question_is_passed_as_asked():
    assert judge.user_input_for(case()) == "What were Northwind's net sales in 2026?"


def test_a_follow_up_carries_its_conversation():
    follow = case(type="follow_up", question="And the year before?",
                  history=[{"question": "Net sales in 2026?", "answer": "$1,000 million."}])
    assert judge.user_input_for(follow) == "User: Net sales in 2026?\nAssistant: $1,000 million.\nUser: And the year before?"


def test_a_sample_takes_the_answer_the_top_contexts_and_the_golden_reference():
    answer = {"response": "It was $1,000 million [1].",
              "contexts": [{"text": f"chunk {i}"} for i in range(8)]}
    s = judge.sample_for(case(), answer)
    assert s.response == "It was $1,000 million [1]."
    assert s.contexts == tuple(f"chunk {i}" for i in range(config.JUDGE_CONTEXTS))
    assert s.reference == "$1,000 million."


def test_each_metric_is_given_exactly_the_arguments_ragas_asks_for():
    from ragas.metrics.collections import (
        AnswerCorrectness, AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness,
    )

    classes = {"faithfulness": Faithfulness, "answer_relevancy": AnswerRelevancy,
               "context_precision": ContextPrecisionWithReference, "context_recall": ContextRecall,
               "answer_correctness": AnswerCorrectness}
    for name, cls in classes.items():
        wanted = set(inspect.signature(cls.ascore).parameters) - {"self"}
        assert set(judge.ARGUMENTS[name](sample())) == wanted, name


def test_a_grade_is_keyed_by_everything_that_could_change_it():
    s = sample()
    assert s.key("faithfulness", JUDGE) == sample().key("faithfulness", JUDGE)
    assert s.key("faithfulness", JUDGE) != s.key("context_recall", JUDGE)
    assert s.key("faithfulness", JUDGE) != s.key("faithfulness", "another-judge")
    assert s.key("faithfulness", JUDGE) != sample(response="Something else.").key("faithfulness", JUDGE)


# --- Caching ---------------------------------------------------------------------------------

def test_every_metric_is_graded_once_and_a_rerun_costs_nothing(tmp_path):
    samples = [sample("a", "Answer A."), sample("b", "Answer B.")]
    first_metrics = fakes()
    first, _ = run(samples, first_metrics, ScoreCache(tmp_path / "scores.jsonl"))
    assert len(first.rows) == 10 and first.judge_calls == 10 and first.cache_hits == 0
    assert first.scores()["a"]["faithfulness"] == 0.8

    again_metrics = fakes()
    again, _ = run(samples, again_metrics, ScoreCache(tmp_path / "scores.jsonl"))
    assert again.cache_hits == 10 and again.judge_calls == 0 and calls_of(again_metrics) == 0
    assert again.scores() == first.scores()


def test_a_changed_answer_is_graded_again(tmp_path):
    run([sample()], fakes(), ScoreCache(tmp_path / "scores.jsonl"))
    metrics = fakes()
    report, _ = run([sample(response="A different answer.")], metrics, ScoreCache(tmp_path / "scores.jsonl"))
    assert report.cache_hits == 0 and calls_of(metrics) == 5


# --- Failures ----------------------------------------------------------------------------------

THROTTLED = RuntimeError("Error code: 429 - ThrottlingException: Too many requests, please wait before trying again.")


def test_throttling_is_found_even_when_wrapped():
    assert judge.is_throttled(THROTTLED)
    assert judge.is_throttled(RuntimeError("Error code: 503 - ServiceUnavailableException"))
    assert not judge.is_throttled(ValueError("could not parse the judge's output"))
    try:
        try:
            raise THROTTLED
        except RuntimeError as inner:
            raise ValueError("instructor gave up") from inner
    except ValueError as wrapped:
        assert judge.is_throttled(wrapped), "a 429 wrapped by instructor is still found"


def test_a_throttled_call_is_waited_out(tmp_path):
    metrics = fakes(faithfulness=FakeMetric(failures=[THROTTLED, THROTTLED]))
    report, waited = run([sample()], metrics, ScoreCache(tmp_path / "scores.jsonl"))
    assert waited == [10, 20] and report.scores()["nwnd-sales"]["faithfulness"] == 0.8 and not report.errors


def test_throttling_that_never_clears_is_recorded_not_cached(tmp_path):
    metrics = fakes(faithfulness=FakeMetric(failures=[THROTTLED] * (config.JUDGE_RETRIES + 1)))
    report, _ = run([sample()], metrics, ScoreCache(tmp_path / "scores.jsonl"))
    assert len(report.errors) == 1 and len(report.rows) == 5, "the other metrics still graded"
    assert len(ScoreCache(tmp_path / "scores.jsonl").rows) == 4, "the failure is not cached"


def test_any_other_failure_is_recorded_and_tried_again_next_run(tmp_path):
    metrics = fakes(answer_relevancy=FakeMetric(failures=[ValueError("bad JSON from the judge")]))
    report, _ = run([sample()], metrics, ScoreCache(tmp_path / "scores.jsonl"))
    assert len(report.errors) == 1 and "bad JSON" in report.errors[0]["error"]
    assert report.scores()["nwnd-sales"]["answer_relevancy"] is None

    retry_metrics = fakes()
    retried, _ = run([sample()], retry_metrics, ScoreCache(tmp_path / "scores.jsonl"))
    assert calls_of(retry_metrics) == 1 and retried.scores()["nwnd-sales"]["answer_relevancy"] == 0.8


def test_the_judge_is_not_the_family_that_writes_the_answers():
    answer_models = {config.GROQ_MODEL, config.GROQ_CHEAP_MODEL, *config.GROQ_FALLBACK_MODELS}
    model = config.JUDGE_MODEL
    assert "gpt" not in model.lower() and "openai" not in model.lower(), model
    assert model not in answer_models
    assert not model.endswith("-latest"), "an alias would drift"


# --- Choosing cases and recording runs -----------------------------------------------------

def test_the_default_cases_are_one_answerable_case_per_type():
    pool = golden.load(golden.GOLDEN)
    picked = judge.default_cases(pool)
    assert len(picked) == 5 and len({c.type for c in picked}) == 5 and all(c.spans for c in picked)


def test_each_run_is_appended_to_the_report(tmp_path):
    path = tmp_path / "ragas_check.json"
    report = judge.GradeReport(rows=[{"case_id": "a", "metric": "faithfulness", "value": 1.0}], judge_calls=2)
    judge.record(report, JUDGE, "abc", path)
    judge.record(judge.GradeReport(rows=report.rows, cache_hits=1), JUDGE, "abc", path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [r["judge_calls"] for r in data["runs"]] == [2, 0]
    assert data["scores"] == {"a": {"faithfulness": 1.0}}


# --- The real thing ----------------------------------------------------------------------------

def test_five_cases_were_graded_and_the_rerun_came_from_the_cache():
    if not judge.REPORT.exists():
        pytest.skip("no judge run yet: run  python -m evaluation.judge  (from server/), twice")
    data = json.loads(judge.REPORT.read_text(encoding="utf-8"))
    runs = [r for r in data["runs"] if r.get("judge") == data["judge"]]
    complete = {c: s for c, s in data["scores"].items()
                if all(s.get(m) is not None for m in judge.METRICS)}
    if len(complete) < 5:
        pytest.skip(f"{len(complete)} of 5 cases fully graded so far: run the judge again")
    paid = [i for i, r in enumerate(runs) if r["judge_calls"] > 0]
    assert paid, "no run ever called the judge"
    reruns = [r for r in runs[paid[-1] + 1:] if r["graded"] >= 25 and r["judge_calls"] == 0 and not r["errors"]]
    if not reruns:
        pytest.skip("graded, but not yet rerun: run the judge once more to show the cache")
    assert reruns[-1]["cache_hits"] == reruns[-1]["graded"]
