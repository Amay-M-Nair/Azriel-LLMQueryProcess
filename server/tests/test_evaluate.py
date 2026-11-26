"""The golden-set run that writes a Results row (ROADMAP 1.8).

Offline tests build the row from hand-made answers and grades. The last test
reads evaluation/results/baseline.json, written by  python -m evaluation.evaluate:
the "done when" is the Baseline row filled, with confidence intervals.
"""

import json

import pytest

from evaluation import evaluate, golden
from tools import roadmap
from utils import config

pytestmark = pytest.mark.roadmap("1.8")

ACC = "0001234567-26-000001"


def case(id="nwnd-sales", type="table_lookup", spans=((100, 200),)):
    return golden.GoldenCase.model_validate({
        "id": id, "type": type, "question": "What were Northwind's net sales in 2026?",
        "answer": "$1,000 million." if spans else "The filings do not say.",
        "spans": [{"accession": ACC, "start_char": s, "end_char": e, "quote": "x" * (e - s)} for s, e in spans],
        "about": [{"ticker": "NWND", "fiscal_year": 2026, "fiscal_period": "FY"}],
    })


def ctx(start, end, accession=ACC):
    return {"accession": accession, "start_char": start, "end_char": end, "text": "t"}


def record(contexts, route="retrieval", seconds=2.0, cost=0.0004):
    return {"route": route, "contexts": contexts, "search_query": "q", "seconds": seconds, "cost": cost,
            "usage": [{"stage": "answer", "model": config.GROQ_MODEL}],
            "tokens": {"prompt_tokens": 100, "cached_tokens": 0, "completion_tokens": 10, "reasoning_tokens": 5}}


def test_recall_at_5_is_what_the_model_saw_and_the_rest_come_from_the_top_10():
    seen = [ctx(0, 50), ctx(500, 600)]                          # the answer model missed the span
    deep = [ctx(0, 50), ctx(500, 600), ctx(110, 190)]           # the top 10 has it at rank 3
    s = evaluate.retrieval_scores(case(), record(seen), deep)
    assert s["recall_at_5"] == 0.0
    assert s["recall_at_10"] == 1.0 and s["reciprocal_rank"] == pytest.approx(1 / 3)


def test_an_unlocated_chunk_keeps_its_rank_as_a_miss():
    deep = [{"accession": ACC, "start_char": -1, "end_char": -1}, ctx(100, 200)]
    s = evaluate.retrieval_scores(case(), record([]), deep)
    assert s["reciprocal_rank"] == 0.5, "the hit is at rank 2, not promoted to 1"


def test_no_retrieval_scores_zero_rather_than_disappearing():
    assert evaluate.deep_contexts(store=None, record=record([], route="clarify")) == []
    s = evaluate.retrieval_scores(case(), record([], route="clarify"), [])
    assert s["recall_at_5"] == 0.0 and s["recall_at_10"] == 0.0


def test_the_row_has_every_column_with_intervals_and_leaves_refusals_out():
    cases = [case("a"), case("b", spans=((300, 400),)), case("no", type="unanswerable", spans=())]
    records = {"a": record([ctx(100, 200)], seconds=1.0), "b": record([ctx(0, 10)], seconds=3.0),
               "no": record([], route="retrieval", seconds=2.0)}
    retrieval = {c.id: evaluate.retrieval_scores(c, records[c.id], records[c.id]["contexts"]) for c in cases[:2]}
    grades = {"a": {m: 1.0 for m in evaluate.judge.METRICS}, "b": {m: 0.0 for m in evaluate.judge.METRICS}}
    result = evaluate.build_result("Baseline (current pipeline)", 0, cases, records, retrieval, grades,
                                   "judge-x", {"version": 1}, "abc")
    m = result["metrics"]
    for key in ("recall_at_5", "mrr", "ndcg_at_10", "faithfulness", "answer_correctness"):
        assert set(m[key]) == {"value", "ci", "n"} and m[key]["n"] == 2, key
    assert m["recall_at_5"]["value"] == 0.5
    assert m["cost_per_query"] == pytest.approx(0.0004) and m["p95_seconds"] == pytest.approx(2.9)
    assert result["questions"] == {"all": 3, "answerable": 2, "judged": 2}
    assert result["by_type"]["table_lookup"]["n"] == 2
    row = roadmap.render_results([result])
    assert "Baseline (current pipeline)" in row and "0.500 [" in row and "$0.0004" in row


def test_a_missing_grade_is_left_out_and_counted():
    v = evaluate.with_ci([1.0, None, 0.0])
    assert v["value"] == 0.5 and v["n"] == 2


def test_the_baseline_row_is_filled_with_confidence_intervals():
    path = evaluate.RESULTS_DIR / "baseline.json"
    if not path.exists():
        pytest.skip("no baseline run yet: run  python -m evaluation.evaluate  (from server/)")
    result = json.loads(path.read_text(encoding="utf-8"))
    m = result["metrics"]
    for key in ("recall_at_5", "mrr", "ndcg_at_10", "faithfulness"):
        assert m[key]["ci"][0] <= m[key]["value"] <= m[key]["ci"][1], key
    assert m["cost_per_query"] is not None and m["p95_seconds"] is not None
    assert result["questions"]["judged"] == result["questions"]["answerable"], "every answerable case graded"


def test_a_rate_limit_is_waited_out_and_other_errors_are_not(monkeypatch, tmp_path):
    from backend.providers.base import ProviderError

    replies = [ProviderError("Groq rate limit reached on: x"), {"ok": True}]

    def fake_answer(case, store, answer_dir):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    waited = []
    monkeypatch.setattr(evaluate.answers, "answer", fake_answer)
    assert evaluate.answer_one(case(), None, tmp_path, sleep=waited.append) == {"ok": True}
    assert waited == [evaluate.RATE_LIMIT_WAIT]

    monkeypatch.setattr(evaluate.answers, "answer",
                        lambda *a: (_ for _ in ()).throw(ProviderError("Groq rejected the API key")))
    with pytest.raises(ProviderError):
        evaluate.answer_one(case(), None, tmp_path, sleep=waited.append)


def test_an_answer_whose_classifier_quietly_failed_is_made_again(monkeypatch, tmp_path):
    bad = {"intent_reason": "classifier call failed: Groq rate limit reached", "rewrite_reason": None}
    good = {"intent_reason": "the model said so", "rewrite_reason": "rewritten by the model"}
    replies = [bad, good]
    (tmp_path / "nwnd-sales.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate.answers, "answer", lambda *a: replies.pop(0))
    waited = []
    assert evaluate.answer_one(case(), None, tmp_path, sleep=waited.append) is good
    assert waited == [evaluate.RATE_LIMIT_WAIT]
    assert not (tmp_path / "nwnd-sales.json").exists(), "the degraded answer was thrown away"
    assert evaluate.degraded({"rewrite_reason": "rewrite call failed"})
    assert evaluate.degraded({"intent_reason": "classifier did not return usable JSON"}) is None, \
        "a bad reply is the pipeline's own behaviour, not a degraded run"


def test_a_case_with_a_judge_error_does_not_count_as_judged():
    cases = [case("a")]
    records = {"a": record([ctx(100, 200)])}
    retrieval = {"a": evaluate.retrieval_scores(cases[0], records["a"], records["a"]["contexts"])}
    grades = {"a": {m: 1.0 for m in evaluate.judge.METRICS} | {"faithfulness": None}}
    result = evaluate.build_result("x", 0, cases, records, retrieval, grades, "j", {"version": 1}, "abc",
                                   [{"case_id": "a", "metric": "faithfulness", "error": "boom"}])
    assert result["questions"]["judged"] == 0 and result["judge_errors"][0]["error"] == "boom"
