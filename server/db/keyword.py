"""Keyword search held in memory, small enough for a 512 MB server (ROADMAP 2.3).

`vectorstore.keyword.BM25` keeps one Python tuple per (word, chunk) pair. Over
the snapshot's 31,119 chunks that is 2 million tuples and about 290 MB, four
times the ~75 MB of headroom 0.7 left. `CompactBM25` keeps the same numbers in
flat numpy arrays (about a tenth of the size) and computes the same scores,
bit for bit: the same tokenizer, the same idf, and the same float32
operations in the same order, only applied to arrays instead of one posting at
a time. `tests/test_db_keyword.py` checks the equality on the real chunks.

`KeywordIndex` puts database ids on it, and `KeywordIndex.load` builds it from
Postgres at startup, reading chunks in batches so all their text is never in
memory at once.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter
from typing import Iterable, Iterator, Sequence

import numpy as np
import psycopg

from vectorstore.keyword import B, K1, tokenize


def _view(values: array, dtype) -> np.ndarray:
    """A numpy view of an `array.array`, without copying it."""
    return np.frombuffer(values, dtype=dtype) if len(values) else np.zeros(0, dtype=dtype)


class CompactBM25:
    """`BM25` with its postings in compressed-sparse-row form:
    `docs[starts[t]:starts[t + 1]]` are the chunks containing term t, and
    `freqs` the same slice's term counts."""

    def __init__(self, documents: Iterable[str]):
        vocabulary: dict[str, int] = {}
        doc_of, term_of, freq_of = array("i"), array("i"), array("i")
        lengths = array("f")
        for index, text in enumerate(documents):
            tokens = tokenize(text)
            lengths.append(len(tokens))
            for word, freq in Counter(tokens).items():
                doc_of.append(index)
                term_of.append(vocabulary.setdefault(word, len(vocabulary)))
                freq_of.append(freq)

        self.vocabulary = vocabulary
        self.count = len(lengths)
        self.lengths = np.array(lengths, dtype=np.float32)
        # The same expression BM25 uses, on the same float32 array in the same
        # order, so the mean (and every score built on it) is identical.
        self.avg_length = float(self.lengths.mean()) if self.count else 0.0

        # Regroup the postings by term. Each source array goes as soon as it
        # has been used: at full size each is 8 MB and the sort order 16 MB,
        # and a server with little headroom pays for the peak, not the result.
        terms = _view(term_of, np.int32)
        del term_of
        df = np.bincount(terms, minlength=len(vocabulary))
        order = np.argsort(terms, kind="stable")   # stable: chunks stay in index order per term
        del terms
        docs = _view(doc_of, np.int32)
        del doc_of
        self.docs = docs[order]
        del docs
        freqs = _view(freq_of, np.int32)
        del freq_of
        self.freqs = freqs[order]
        del freqs, order
        self.starts = np.concatenate(([0], np.cumsum(df))).astype(np.int64)
        # Python floats from math.log, exactly as BM25 computes them.
        self.idf = np.array(
            [math.log(1 + (self.count - n + 0.5) / (n + 0.5)) for n in df.tolist()], dtype=np.float64)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score for every document, in index order: `BM25.scores`, exactly.

        Each step mirrors BM25's scalar arithmetic under NumPy 2's rules, where
        a Python number meeting a float32 takes float32's type:
        - `lengths[i] / avg`, `B * x`, `(1 - B) + ...`, `K1 * norm` and
          `freq + ...` are float32 operations;
        - `idf * (freq * (K1 + 1))` is computed in double, then rounded to
          float32 when it meets the float32 denominator;
        - the division and the running sum are float32.
        Each chunk appears once per term, so `result[docs] += ...` adds once.
        """
        result = np.zeros(self.count, dtype=np.float32)
        if not self.count:
            return result
        avg = np.float32(self.avg_length or 1)
        for word in tokenize(query):
            term = self.vocabulary.get(word)
            if term is None:
                continue
            span = slice(self.starts[term], self.starts[term + 1])
            docs, freqs = self.docs[span], self.freqs[span]
            norm = np.float32(1 - B) + np.float32(B) * (self.lengths[docs] / avg)
            denominator = freqs.astype(np.float32) + np.float32(K1) * norm
            numerator = (float(self.idf[term]) * (freqs.astype(np.float64) * (K1 + 1))).astype(np.float32)
            result[docs] += numerator / denominator
        return result

    def nbytes(self) -> int:
        """Memory held by the index: the arrays plus an estimate of the
        vocabulary dict (its table, keys and int values)."""
        import sys

        arrays = sum(a.nbytes for a in (self.lengths, self.docs, self.freqs, self.starts, self.idf))
        words = sys.getsizeof(self.vocabulary) + sum(sys.getsizeof(w) + 28 for w in self.vocabulary)
        return arrays + words


class KeywordIndex:
    """BM25 over a fixed set of chunks, addressed by their database ids.

    BM25's statistics (document frequencies, average length) come from the
    whole set, so scores depend on which chunks are in it, never on a search's
    filters: the same as the FAISS store, which scores against everything.
    """

    def __init__(self, ids: Sequence[int], texts: Iterable[str]):
        self.bm25 = CompactBM25(texts)
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) != self.bm25.count:
            raise ValueError(f"{len(ids)} ids but {self.bm25.count} texts")
        # Sorted ids and a binary search instead of a dict: 31k dict entries
        # cost ~3 MB, these two arrays ~0.5 MB.
        self._order = np.argsort(ids, kind="stable")
        self._sorted = ids[self._order]

    @classmethod
    def load(cls, conn: psycopg.Connection, scope=None, batch: int = 5000) -> KeywordIndex:
        """Build from the chunks in Postgres (within `scope`, a `db.store.Filters`),
        reading `batch` rows at a time by id, so the texts stream through."""
        from db.store import where

        clause, params = where(scope)
        ids = array("q")

        def texts() -> Iterator[str]:
            last = 0
            while True:
                rows = conn.execute(
                    f"SELECT id, text FROM chunks{clause}{' AND' if clause else ' WHERE'} id > %s"
                    " ORDER BY id LIMIT %s", [*params, last, batch],
                ).fetchall()
                for chunk_id, text in rows:
                    ids.append(chunk_id)
                    yield text
                if len(rows) < batch:
                    return
                last = rows[-1][0]

        bm25 = CompactBM25(texts())
        index = cls.__new__(cls)
        index.bm25 = bm25
        sorted_ids = np.array(ids, dtype=np.int64)   # ORDER BY id: already ascending
        index._order = np.arange(len(sorted_ids))
        index._sorted = sorted_ids
        return index

    def __len__(self) -> int:
        return self.bm25.count

    def positions(self, ids: Sequence[int]) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        at = np.searchsorted(self._sorted, ids)
        found = at < len(self._sorted)
        found[found] = self._sorted[at[found]] == ids[found]
        if not found.all():
            # A chunk added since this index was built. Scoring it 0 would
            # quietly rank it below everything; better to say so.
            raise LookupError(f"chunk id {int(ids[~found][0])} is not in the keyword index; "
                              "it was built before that chunk was added. Rebuild it.")
        return self._order[at]

    def scores(self, query: str, ids: Sequence[int]) -> np.ndarray:
        return self.bm25.scores(query)[self.positions(ids)]

    def nbytes(self) -> int:
        return self.bm25.nbytes() + self._order.nbytes + self._sorted.nbytes
