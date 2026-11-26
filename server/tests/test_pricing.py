"""What a call costs, and what a set of answers cost (ROADMAP 1.7)."""

import pytest

from backend import pricing
from evaluation import answers
from utils import config

pytestmark = pytest.mark.roadmap("1.7")


def test_a_call_is_priced_from_the_table_with_cached_input_cheaper():
    usage = {"model": config.GROQ_MODEL, "prompt_tokens": 1000, "cached_tokens": 200, "completion_tokens": 500}
    assert pricing.call_cost(usage) == pytest.approx((800 * 0.15 + 200 * 0.075 + 500 * 0.60) / 1_000_000)


def test_every_model_the_pipeline_uses_has_a_price():
    used = {config.GROQ_MODEL, config.GROQ_CHEAP_MODEL, *config.GROQ_FALLBACK_MODELS,
            *(model for model, _effort in config.GROQ_STAGES.values())}
    assert used <= set(config.PRICES), f"unpriced: {sorted(used - set(config.PRICES))}"


def test_an_unknown_model_is_unpriced_not_free():
    assert pricing.call_cost({"model": "nobody/knows", "prompt_tokens": 10, "completion_tokens": 10}) is None


def record(seconds, cost, prompt=1000):
    return {"seconds": seconds, "cost": cost,
            "tokens": {"prompt_tokens": prompt, "cached_tokens": 0, "completion_tokens": 100, "reasoning_tokens": 50}}


def test_a_run_reports_cost_per_query_and_p50_p95_latency():
    records = [record(s, 0.001) for s in range(1, 21)]     # 1s ... 20s
    s = answers.summarize(records)
    assert s["questions"] == 20 and s["cost_per_query"] == pytest.approx(0.001)
    assert s["cost_total"] == pytest.approx(0.02)
    assert s["p50_seconds"] == pytest.approx(10.5) and s["p95_seconds"] == pytest.approx(19.05)
    assert s["tokens_per_query"]["prompt_tokens"] == 1000
    assert s["prices_checked"] == config.PRICES_CHECKED


def test_one_unknown_cost_makes_the_run_cost_unknown():
    s = answers.summarize([record(1, 0.001), record(2, None)])
    assert s["cost_per_query"] is None and s["cost_unknown"] == 1


def test_an_answer_record_carries_what_it_cost(tmp_path, monkeypatch):
    from backend.query_processor import Plan, QueryTrace
    from evaluation import golden
    from utils.preprocessing import preprocess

    trace = QueryTrace(query=preprocess("q"), api_calls=1, timings={"retrieve": 0.5, "answer": 2.0},
                       usage=[{"model": config.GROQ_MODEL, "prompt_tokens": 100, "cached_tokens": 0,
                               "completion_tokens": 10, "reasoning_tokens": 5, "stage": "answer"}])
    seen = {}

    def fake_process(question, store, history, provider):
        seen["provider"] = provider
        return Plan(trace, answer="It was $1 [1].")

    monkeypatch.setattr(answers, "process", fake_process)
    case = golden.load(golden.GOLDEN)[0]
    rec = answers.answer(case, store=None, cache_dir=tmp_path)
    assert seen["provider"] is not None, "the eval runs the configured provider, never none"
    assert rec["seconds"] == 2.5 and rec["usage"] == trace.usage
    assert rec["cost"] == pytest.approx((100 * 0.15 + 10 * 0.60) / 1_000_000)
    assert set(answers.summarize([rec])) >= {"cost_per_query", "p50_seconds", "p95_seconds"}
