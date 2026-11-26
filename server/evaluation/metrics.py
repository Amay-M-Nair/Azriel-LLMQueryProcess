"""Exact evaluation metrics, from scratch in NumPy (ROADMAP 1.5).

Retrieval is scored against the golden set's **evidence**, not against chunks.
A golden case lists the spans of filing text that answer it; a retrieved
chunk *covers* a span when they overlap enough (`covers`). Scoring at the
level of evidence is what lets the numbers survive a change of chunking: a
smaller chunk size cannot score better just by cutting one answer into more
pieces, because each span counts once however many chunks cover it.

For one question, with `k` retrieved chunks in rank order:

  recall@k   the share of the question's spans covered by the top k chunks.
  RR         1 / the rank of the first chunk that covers any span (0 if none);
             MRR is its mean over questions.
  nDCG@k     each rank earns a gain of 1 if its chunk covers a span that no
             higher-ranked chunk covered, discounted by log2(rank + 1); divided
             by the best possible, every span covered in the first ranks. A
             second chunk on the same span earns nothing, so nDCG cannot
             exceed 1.

Questions with no spans (the unanswerable ones) have no retrieval score:
there is nothing to find, and how well a system declines is measured
separately (5.4).

Across questions, each metric is reported as a mean with a bootstrap 95%
confidence interval: resample the questions with replacement, recompute the
mean, and take the 2.5th and 97.5th percentiles. Two configurations run on
the same questions are compared with a paired bootstrap of their difference,
which is what decides whether an ablation row is an improvement or noise.

Cohen's kappa measures agreement between two raters beyond what chance alone
would produce. 7.5 uses it to calibrate the RAGAS judge against hand labels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_MIN_COVERAGE = 0.5
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261126


@dataclass(frozen=True)
class Range:
    """A stretch of one filing's text: [start, end) in characters."""

    accession: str
    start: int
    end: int

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(f"empty or reversed range {self.start}:{self.end}")


# --- Matching chunks to evidence ---------------------------------------------------------

def covers(chunk: Range, span: Range, min_coverage: float = DEFAULT_MIN_COVERAGE) -> bool:
    """Does this chunk hold enough of this span to count as having found it?

    The overlap is measured against the shorter of the two. A chunk that
    holds at least `min_coverage` of a short span (a table row) has found it.
    A chunk that lies mostly inside a long span (a paragraph longer than a
    chunk) has found part of it, and that is a hit too. A chunk that only
    grazes the edge of a span, as overlapping windows do, has not.
    """
    if chunk.accession != span.accession:
        return False
    overlap = min(chunk.end, span.end) - max(chunk.start, span.start)
    if overlap <= 0:
        return False
    return overlap / min(chunk.end - chunk.start, span.end - span.start) >= min_coverage


def coverage_matrix(retrieved: Sequence[Range], spans: Sequence[Range],
                    min_coverage: float = DEFAULT_MIN_COVERAGE) -> np.ndarray:
    """A (ranks x spans) boolean matrix: does the chunk at each rank cover each span?"""
    matrix = np.zeros((len(retrieved), len(spans)), dtype=bool)
    for r, chunk in enumerate(retrieved):
        for s, span in enumerate(spans):
            matrix[r, s] = covers(chunk, span, min_coverage)
    return matrix


def novel_gains(matrix: np.ndarray) -> np.ndarray:
    """1 at each rank whose chunk covers a span no higher rank covered, else 0."""
    seen = np.zeros(matrix.shape[1], dtype=bool)
    gains = np.zeros(matrix.shape[0])
    for r, row in enumerate(matrix):
        new = row & ~seen
        gains[r] = 1.0 if new.any() else 0.0
        seen |= row
    return gains


# --- Per-question metrics -----------------------------------------------------------------

def recall_at_k(matrix: np.ndarray, k: int) -> float:
    """Share of spans covered by the top k ranks. NaN when there are no spans."""
    if matrix.shape[1] == 0:
        return float("nan")
    return float(matrix[:k].any(axis=0).mean())


def reciprocal_rank(matrix: np.ndarray) -> float:
    """1 / rank of the first rank covering any span; 0 if none does; NaN with no spans."""
    if matrix.shape[1] == 0:
        return float("nan")
    hits = np.flatnonzero(matrix.any(axis=1))
    return float(1.0 / (hits[0] + 1)) if hits.size else 0.0


def dcg(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain of the first k ranks: sum gain_i / log2(i + 1), i from 1."""
    g = np.asarray(gains, dtype=float)[:k]
    return float(np.sum(g / np.log2(np.arange(2, g.size + 2))))


def ndcg_at_k(gains: Sequence[float], ideal_gains: Sequence[float], k: int) -> float:
    """DCG@k of `gains` over DCG@k of `ideal_gains` sorted best first. NaN if nothing is relevant."""
    best = dcg(np.sort(np.asarray(ideal_gains, dtype=float))[::-1], k)
    return dcg(gains, k) / best if best > 0 else float("nan")


def ndcg_for(matrix: np.ndarray, k: int) -> float:
    """nDCG@k where every span is worth one unit of gain, earned once."""
    if matrix.shape[1] == 0:
        return float("nan")
    return ndcg_at_k(novel_gains(matrix), np.ones(matrix.shape[1]), k)


def score_question(retrieved: Sequence[Range], spans: Sequence[Range], ks: Sequence[int] = (1, 5, 10),
                   ndcg_k: int = 10, min_coverage: float = DEFAULT_MIN_COVERAGE) -> dict[str, float]:
    """Every retrieval metric for one question. All NaN when it has no spans."""
    matrix = coverage_matrix(retrieved, spans, min_coverage)
    scores = {f"recall_at_{k}": recall_at_k(matrix, k) for k in ks}
    scores["reciprocal_rank"] = reciprocal_rank(matrix)
    scores[f"ndcg_at_{ndcg_k}"] = ndcg_for(matrix, ndcg_k)
    return scores


# --- Across questions: means with confidence intervals ------------------------------------

def bootstrap_ci(values: Sequence[float], statistic: Callable[[np.ndarray], float] = np.mean,
                 resamples: int = DEFAULT_RESAMPLES, confidence: float = 0.95,
                 seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """(statistic, low, high): a percentile bootstrap interval over the items.

    NaN items (questions a metric does not apply to) are dropped first.
    `values` may be 2-D, (items x columns), when the statistic needs several
    numbers per item, such as two raters' labels for kappa; whole rows are
    resampled together.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[~np.isnan(data)]
    else:
        data = data[~np.isnan(data).any(axis=1)]
    n = len(data)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(data))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n, size=(resamples, n))
    if data.ndim == 1 and statistic is np.mean:
        stats = data[picks].mean(axis=1)                # vectorised fast path
    else:
        stats = np.array([statistic(data[p]) for p in picks])
    stats = stats[~np.isnan(stats)]
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(stats, [tail, 100 - tail])
    return point, float(low), float(high)


def paired_difference(a: Sequence[float], b: Sequence[float], resamples: int = DEFAULT_RESAMPLES,
                      confidence: float = 0.95, seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """mean(b - a) with a bootstrap interval, over questions both configurations scored.

    If the interval excludes 0, b differs from a by more than resampling the
    questions would explain.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a paired comparison needs both configurations scored on the same questions")
    return bootstrap_ci(b - a, resamples=resamples, confidence=confidence, seed=seed)


def summarize(per_question: Sequence[dict[str, float]], resamples: int = DEFAULT_RESAMPLES,
              seed: int = DEFAULT_SEED) -> dict[str, dict]:
    """{metric: {"value", "ci": [low, high], "n"}} in the shape the Results table reads."""
    out = {}
    for name in per_question[0] if per_question else []:
        values = np.array([q[name] for q in per_question], dtype=float)
        value, low, high = bootstrap_ci(values, resamples=resamples, seed=seed)
        key = "mrr" if name == "reciprocal_rank" else name
        out[key] = {"value": value, "ci": [low, high], "n": int(np.sum(~np.isnan(values)))}
    return out


# --- Agreement between raters ----------------------------------------------------------

def cohens_kappa(a: Sequence, b: Sequence) -> float:
    """Agreement between two raters' labels beyond chance: (p_o - p_e) / (1 - p_e).

    p_o is how often they agree; p_e is how often they would agree by chance,
    each labelling at their own observed rates. NaN when p_e is 1: both used a
    single label and the same one throughout, so agreement says nothing.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("kappa needs two equally long, non-empty lists of labels")
    labels = np.union1d(a, b)
    index_a, index_b = np.searchsorted(labels, a), np.searchsorted(labels, b)
    table = np.zeros((labels.size, labels.size))
    np.add.at(table, (index_a, index_b), 1)
    table /= a.size
    observed = np.trace(table)
    expected = float(table.sum(axis=1) @ table.sum(axis=0))
    return float("nan") if np.isclose(expected, 1.0) else float((observed - expected) / (1 - expected))


def kappa_ci(a: Sequence, b: Sequence, resamples: int = 2_000, confidence: float = 0.95,
             seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """Cohen's kappa with a bootstrap interval, resampling the rated items."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    pairs = np.column_stack([np.searchsorted(labels, a), np.searchsorted(labels, b)]).astype(float)
    return bootstrap_ci(pairs, statistic=lambda p: cohens_kappa(p[:, 0], p[:, 1]),
                        resamples=resamples, confidence=confidence, seed=seed)
