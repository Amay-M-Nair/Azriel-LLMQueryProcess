"""The evaluation metrics (ROADMAP 1.5), checked against examples worked by hand.

Each expected value below is derived in its comment, or comes from a
published worked example, rather than from running the code under test.
"""

from math import log2

import numpy as np
import pytest

from evaluation import metrics as m
from evaluation.metrics import Range

pytestmark = pytest.mark.roadmap("1.5")

A = "0000000001-26-000001"
B = "0000000001-26-000002"


# --- Matching chunks to evidence ---------------------------------------------------------

def test_a_chunk_holding_most_of_a_short_span_covers_it():
    span = Range(A, 100, 150)                              # a 50-character table row
    assert m.covers(Range(A, 120, 400), span)              # overlap 30 of 50 = 0.6
    assert not m.covers(Range(A, 140, 400), span)          # overlap 10 of 50 = 0.2: a graze
    assert not m.covers(Range(A, 150, 400), span)          # touching is not overlapping


def test_a_chunk_inside_a_long_span_covers_it():
    span = Range(A, 0, 2000)                               # a paragraph longer than a chunk
    assert m.covers(Range(A, 500, 900), span)              # overlap 400 of the chunk's 400
    assert not m.covers(Range(A, 1900, 2300), span)        # overlap 100 of 400 = 0.25


def test_a_chunk_from_another_filing_never_covers():
    assert not m.covers(Range(B, 100, 150), Range(A, 100, 150))


def test_the_coverage_threshold_can_be_changed():
    assert m.covers(Range(A, 140, 400), Range(A, 100, 150), min_coverage=0.2)


def test_an_empty_range_is_refused():
    with pytest.raises(ValueError):
        Range(A, 10, 10)


# --- One question, worked by hand --------------------------------------------------------
#
# Two spans, S1 and S2. Five chunks retrieved:
#   rank 1  covers nothing
#   rank 2  covers S1
#   rank 3  covers S1 again (an overlapping neighbour)
#   rank 4  covers S2
#   rank 5  covers nothing
#
# recall@1 = 0/2, recall@3 = 1/2, recall@5 = 2/2
# RR       = 1/2 (first hit at rank 2)
# gains    = [0, 1, 0, 1, 0]   (rank 3 adds nothing new)
# DCG      = 1/log2(3) + 1/log2(5)
# IDCG     = 1/log2(2) + 1/log2(3)       (both spans in the first two ranks)
# nDCG     = (0.63093 + 0.43068) / (1 + 0.63093) = 0.65092

S1, S2 = Range(A, 1000, 1060), Range(B, 5000, 5080)
RETRIEVED = [Range(A, 0, 800), Range(A, 990, 1800), Range(A, 1010, 1900), Range(B, 4900, 5600), Range(B, 9000, 9800)]


def test_the_coverage_matrix_of_the_worked_example():
    matrix = m.coverage_matrix(RETRIEVED, [S1, S2])
    assert matrix.tolist() == [[False, False], [True, False], [True, False], [False, True], [False, False]]
    assert m.novel_gains(matrix).tolist() == [0, 1, 0, 1, 0]


def test_recall_at_k_of_the_worked_example():
    matrix = m.coverage_matrix(RETRIEVED, [S1, S2])
    assert [m.recall_at_k(matrix, k) for k in (1, 2, 3, 4, 5, 10)] == [0, 0.5, 0.5, 1, 1, 1]


def test_reciprocal_rank_of_the_worked_example():
    assert m.reciprocal_rank(m.coverage_matrix(RETRIEVED, [S1, S2])) == 0.5


def test_ndcg_of_the_worked_example():
    expected = (1 / log2(3) + 1 / log2(5)) / (1 / log2(2) + 1 / log2(3))
    assert m.ndcg_for(m.coverage_matrix(RETRIEVED, [S1, S2]), 10) == pytest.approx(expected)
    assert expected == pytest.approx(0.65092, abs=1e-5)


def test_ndcg_is_cut_at_k():
    # With k = 3 only rank 2 counts: DCG = 1/log2(3); IDCG@3 is still both spans.
    expected = (1 / log2(3)) / (1 + 1 / log2(3))
    assert m.ndcg_for(m.coverage_matrix(RETRIEVED, [S1, S2]), 3) == pytest.approx(expected)


def test_a_perfect_ranking_scores_one_and_a_miss_scores_zero():
    perfect = m.coverage_matrix([Range(A, 1000, 1060), Range(B, 5000, 5080)], [S1, S2])
    assert m.recall_at_k(perfect, 2) == 1 and m.reciprocal_rank(perfect) == 1 and m.ndcg_for(perfect, 10) == 1
    miss = m.coverage_matrix([Range(A, 0, 500)], [S1, S2])
    assert m.recall_at_k(miss, 10) == 0 and m.reciprocal_rank(miss) == 0 and m.ndcg_for(miss, 10) == 0


def test_covering_one_span_many_times_cannot_push_ndcg_past_one():
    many = [Range(A, 1000 + i, 1060 + i) for i in range(10)]
    assert m.ndcg_for(m.coverage_matrix(many, [S1]), 10) == 1.0


def test_a_question_with_no_spans_has_no_retrieval_score():
    scores = m.score_question(RETRIEVED, [])
    assert all(np.isnan(v) for v in scores.values())


def test_score_question_reports_every_metric():
    scores = m.score_question(RETRIEVED, [S1, S2])
    assert scores["recall_at_1"] == 0 and scores["recall_at_5"] == 1
    assert scores["reciprocal_rank"] == 0.5
    assert scores["ndcg_at_10"] == pytest.approx(0.65092, abs=1e-5)


# --- Published worked examples ---------------------------------------------------------------

def test_dcg_and_ndcg_match_the_textbook_example():
    # Graded relevance in rank order 3, 2, 3, 0, 1, 2:
    #   DCG6  = 3 + 2/log2(3) + 3/log2(4) + 0 + 1/log2(6) + 2/log2(7) = 6.861
    #   ideal = 3, 3, 3, 2, 2, 2 -> IDCG6 = 8.740, nDCG6 = 0.785
    ranked = [3, 2, 3, 0, 1, 2]
    assert m.dcg(ranked, 6) == pytest.approx(6.861, abs=1e-3)
    assert m.dcg([3, 3, 3, 2, 2, 2], 6) == pytest.approx(8.740, abs=1e-3)
    assert m.ndcg_at_k(ranked, [3, 3, 3, 2, 2, 2, 1, 0], 6) == pytest.approx(0.785, abs=1e-3)


def test_mrr_matches_the_textbook_example():
    # First relevant answers at ranks 3, 2 and 1: MRR = (1/3 + 1/2 + 1) / 3 = 11/18.
    def first_hit_at(rank):
        return m.reciprocal_rank(np.array([[r == rank] for r in range(1, 6)]))

    assert np.mean([first_hit_at(3), first_hit_at(2), first_hit_at(1)]) == pytest.approx(11 / 18)


def test_ndcg_is_undefined_when_nothing_is_relevant():
    assert np.isnan(m.ndcg_at_k([0, 0, 0], [0, 0], 3))


# --- Bootstrap confidence intervals -------------------------------------------------------

def test_a_constant_has_no_uncertainty():
    assert m.bootstrap_ci([0.7] * 30) == pytest.approx((0.7, 0.7, 0.7))


def test_questions_a_metric_does_not_apply_to_are_left_out():
    value, low, high = m.bootstrap_ci([1.0, float("nan"), 3.0], resamples=500)
    assert value == 2.0 and 1.0 <= low <= high <= 3.0


def test_the_interval_is_reproducible():
    values = np.random.default_rng(1).random(50)
    assert m.bootstrap_ci(values) == m.bootstrap_ci(values)


def test_the_interval_matches_the_normal_approximation_for_a_large_sample():
    # For a mean of n = 400 draws with sd ~1, the 95% half-width is about
    # 1.96 * sd / sqrt(n) = 0.098. A percentile bootstrap should land close.
    values = np.random.default_rng(7).normal(0.0, 1.0, size=400)
    value, low, high = m.bootstrap_ci(values)
    expected = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    assert low < value < high
    assert (high - low) / 2 == pytest.approx(expected, rel=0.15)


def test_more_questions_give_a_narrower_interval():
    rng = np.random.default_rng(3)
    _, lo_small, hi_small = m.bootstrap_ci(rng.random(20))
    _, lo_large, hi_large = m.bootstrap_ci(rng.random(500))
    assert hi_large - lo_large < hi_small - lo_small


def test_a_paired_difference_sees_through_noise_shared_by_both_runs():
    # b is a plus exactly 0.05 on every question: the questions' own spread
    # cancels out, so the paired interval is exactly 0.05.
    a = np.random.default_rng(5).random(60)
    assert m.paired_difference(a, a + 0.05) == pytest.approx((0.05, 0.05, 0.05))
    with pytest.raises(ValueError, match="same questions"):
        m.paired_difference([1, 2], [1, 2, 3])


def test_summarize_reports_value_interval_and_count_per_metric():
    per_question = [{"recall_at_5": 1.0, "reciprocal_rank": 1.0},
                    {"recall_at_5": 0.0, "reciprocal_rank": 0.5},
                    {"recall_at_5": float("nan"), "reciprocal_rank": float("nan")}]
    summary = m.summarize(per_question, resamples=500)
    assert summary["recall_at_5"]["value"] == 0.5 and summary["recall_at_5"]["n"] == 2
    assert summary["mrr"]["value"] == 0.75, "reciprocal rank is reported as its mean, MRR"
    low, high = summary["mrr"]["ci"]
    assert 0.5 <= low <= 0.75 <= high <= 1.0


# --- Cohen's kappa ----------------------------------------------------------------------------

def labels(counts):
    """Two raters' labels from a 2x2 table {(a, b): count}."""
    a = [x for (x, _), n in counts.items() for _ in range(n)]
    b = [y for (_, y), n in counts.items() for _ in range(n)]
    return a, b


def test_kappa_matches_the_textbook_example():
    # 50 items: both yes 20, A yes / B no 5, A no / B yes 10, both no 15.
    # p_o = 35/50 = 0.7. A says yes 25/50, B 30/50: p_e = 0.5*0.6 + 0.5*0.4 = 0.5.
    # kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
    a, b = labels({("yes", "yes"): 20, ("yes", "no"): 5, ("no", "yes"): 10, ("no", "no"): 15})
    assert m.cohens_kappa(a, b) == pytest.approx(0.4)


@pytest.mark.parametrize("table, expected", [
    # Same 60% raw agreement, different chance agreement, so different kappa.
    # p_e = 0.6*0.7 + 0.4*0.3 = 0.54 -> kappa = 0.06 / 0.46
    ({("y", "y"): 45, ("y", "n"): 15, ("n", "y"): 25, ("n", "n"): 15}, 0.06 / 0.46),
    # p_e = 0.6*0.3 + 0.4*0.7 = 0.46 -> kappa = 0.14 / 0.54
    ({("y", "y"): 25, ("y", "n"): 35, ("n", "y"): 5, ("n", "n"): 35}, 0.14 / 0.54),
])
def test_kappa_corrects_for_chance(table, expected):
    assert m.cohens_kappa(*labels(table)) == pytest.approx(expected)


def test_kappa_at_its_extremes():
    assert m.cohens_kappa([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0           # perfect agreement
    assert m.cohens_kappa([0, 1, 0, 1], [1, 0, 1, 0]) == -1.0          # p_o 0, p_e 0.5
    assert np.isnan(m.cohens_kappa(["x"] * 5, ["x"] * 5))              # one label throughout


def test_kappa_handles_more_than_two_labels():
    # 3 labels, 6 items. Agree on 4: p_o = 4/6.
    # A: s s p p u u -> 1/3 each.  B: s p p p u s -> s 2/6, p 3/6, u 1/6.
    # p_e = 1/3 * (2/6 + 3/6 + 1/6) = 1/3.  kappa = (4/6 - 1/3) / (2/3) = 0.5
    a = ["supported", "supported", "partly", "partly", "unsupported", "unsupported"]
    b = ["supported", "partly", "partly", "partly", "unsupported", "supported"]
    assert m.cohens_kappa(a, b) == pytest.approx(0.5)


def test_kappa_agrees_with_scikit_learn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(11)
    for _ in range(20):
        a = rng.integers(0, 4, 80)
        b = np.where(rng.random(80) < 0.5, a, rng.integers(0, 4, 80))   # agrees about half the time
        assert m.cohens_kappa(a, b) == pytest.approx(sklearn.cohen_kappa_score(a, b))


def test_kappa_has_a_confidence_interval():
    a, b = labels({("yes", "yes"): 20, ("yes", "no"): 5, ("no", "yes"): 10, ("no", "no"): 15})
    value, low, high = m.kappa_ci(a, b)
    assert value == pytest.approx(0.4) and low < 0.4 < high


def test_kappa_refuses_mismatched_raters():
    with pytest.raises(ValueError):
        m.cohens_kappa([1, 2], [1])
