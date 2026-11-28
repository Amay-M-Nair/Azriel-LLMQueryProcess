"""Search over the chunks in Postgres (ROADMAP 2.2).

`PgVectorStore` answers the questions the pipeline asks of the FAISS
`VectorStore` - `len(store)`, `store.sources`, `store.total_words` and
`store.search(...)` - with the same answers, plus filters by company, form,
fiscal year, fiscal period, section and filing.

Search is an exact scan (no vector index), and ranks by inner product
(`<#>`), the metric FAISS's `IndexFlatIP` uses. The vectors are unit length, so
it equals the cosine similarity, without the division `<=>` would add. Ties are
broken by row id, which is ingest order: the order FAISS holds the chunks in.

The keyword half of hybrid search is a `KeywordIndex`, BM25 over the same
chunks, kept in memory (2.3 loads it from Postgres at startup). The blend is
the FAISS store's, line for line; `tests/test_db_store.py` runs both stores
over the same chunks and checks they return the same thing.

Two kinds of filter:
- `scope`, fixed for the store's lifetime, limits everything it reports. The
  evaluation scopes to the pinned snapshot's filings, so filings the refresh
  job (2.7) adds to the live index never move a score.
- `filters`, per search, narrows further (4.3 fills them from the question).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np
import psycopg

from ingestion.chunker import Chunk
from utils import config
from vectorstore.keyword import BM25

# What a Chunk is rebuilt from, in order.
CHUNK_COLUMNS = ("id, chunk_id, accession, ticker, form, fiscal_year, fiscal_period, page,"
                 " start_char, end_char, start_word, end_word, text")


@dataclass(frozen=True)
class Filters:
    """Each field, when not empty, keeps only chunks whose value is one of
    those listed; the fields combine with AND.

    Lists rather than single values, because some questions span filings: a
    Q4 question needs that year's 10-K and its Q3 10-Q (fiscal_years=(2025,),
    fiscal_periods=("FY", "Q3")), a year-over-year one two fiscal years.
    """

    tickers: tuple[str, ...] = ()
    forms: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    fiscal_periods: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    accessions: tuple[str, ...] = ()

    COLUMNS = {"tickers": "ticker", "forms": "form", "fiscal_years": "fiscal_year",
               "fiscal_periods": "fiscal_period", "sections": "section", "accessions": "accession"}

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (str, int)):
                raise TypeError(f"Filters.{f.name} takes a list of values, not one: {value!r}")
            object.__setattr__(self, f.name, tuple(value))

    @classmethod
    def snapshot(cls, snapshot: dict) -> Filters:
        """Exactly the pinned snapshot's filings."""
        return cls(accessions=tuple(f["accession"] for f in snapshot["filings"]))

    def clauses(self) -> tuple[list[str], list]:
        """SQL conditions and their parameters. `= ANY(list)` never matches a
        NULL, so a section filter leaves out chunks with no section yet."""
        sql, params = [], []
        for name, column in self.COLUMNS.items():
            values = getattr(self, name)
            if values:
                sql.append(f"{column} = ANY(%s)")
                params.append(list(values))
        return sql, params

    def __bool__(self) -> bool:
        return any(getattr(self, name) for name in self.COLUMNS)


def where(*filters: Filters | None) -> tuple[str, list]:
    sql, params = [], []
    for f in filters:
        if f:
            s, p = f.clauses()
            sql += s
            params += p
    return (" WHERE " + " AND ".join(sql) if sql else ""), params


class KeywordIndex:
    """BM25 over a fixed set of chunks, addressed by their database ids.

    BM25's statistics (document frequencies, average length) come from the
    whole set, so scores depend on which chunks are in it, never on a search's
    filters: the same as the FAISS store, which scores against everything.
    """

    def __init__(self, ids: Sequence[int], texts: Sequence[str]):
        if len(ids) != len(texts):
            raise ValueError(f"{len(ids)} ids but {len(texts)} texts")
        self.bm25 = BM25(list(texts))
        self.position = {int(i): n for n, i in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.position)

    def scores(self, query: str, ids: Sequence[int]) -> np.ndarray:
        try:
            rows = [self.position[int(i)] for i in ids]
        except KeyError as missing:
            # A chunk added since this index was built. Scoring it 0 would
            # quietly rank it below everything; better to say so.
            raise LookupError(f"chunk id {missing} is not in the keyword index; "
                              "it was built before that chunk was added. Rebuild it.") from None
        return self.bm25.scores(query)[rows]


class PgVectorStore:
    def __init__(self, conn: psycopg.Connection, scope: Filters | None = None,
                 keyword: KeywordIndex | None = None):
        """`conn` must have pgvector's types registered (`db.connection.connect`
        does that). Without `keyword`, search ranks by meaning alone."""
        self.conn = conn
        self.scope = scope or Filters()
        self.keyword = keyword
        self._counts: dict[Filters, int] = {}

    # --- What the pipeline reads -------------------------------------------

    def count(self, filters: Filters | None = None) -> int:
        """Chunks in scope that pass `filters`. Cached per filter: the store
        reads an index that ingest writes, and is rebuilt after an ingest."""
        key = filters or Filters()
        if key not in self._counts:
            clause, params = where(self.scope, key)
            self._counts[key] = self.conn.execute(f"SELECT count(*) FROM chunks{clause}", params).fetchone()[0]
        return self._counts[key]

    def __len__(self) -> int:
        return self.count()

    @property
    def sources(self) -> list[str]:
        """Each filing in scope, named as in citations ("AAPL 10-K FY2026"),
        in the order it was ingested."""
        from ingestion.filings import source_name

        clause, params = where(self.scope)
        rows = self.conn.execute(
            "SELECT ticker, form, fiscal_year, fiscal_period FROM chunks" + clause +
            " GROUP BY accession, ticker, form, fiscal_year, fiscal_period ORDER BY min(id)", params,
        ).fetchall()
        return [source_name({"ticker": t, "form": f, "fiscal_year": y, "fiscal_period": p})
                for t, f, y, p in rows]

    @property
    def total_words(self) -> int:
        clause, params = where(self.scope)
        return int(self.conn.execute(f"SELECT coalesce(sum(words), 0) FROM chunks{clause}", params)
                   .fetchone()[0])

    # --- Searching -----------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        query: str | None = None,
        min_similarity: float | None = None,
        filters: Filters | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Rank chunks by meaning, and by keyword overlap when a query is given.

        The same contract as `VectorStore.search`: the score returned is
        always the cosine similarity, while the keyword signal only changes
        the order; `min_similarity` drops chunks unlike the question unless
        their keyword score rescues them.
        """
        total = self.count(filters)
        if total == 0:
            return []

        k = min(k, total)
        hybrid = bool(query) and self.keyword is not None

        # The candidate pool, sized exactly as the FAISS store sizes it.
        if hybrid and total <= config.HYBRID_EXACT_LIMIT:
            pool = total
        elif hybrid:
            pool = min(total, max(k * config.HYBRID_POOL_MULTIPLIER, config.HYBRID_POOL_MIN))
        else:
            pool = k

        probe = np.ascontiguousarray(query_vector, dtype="float32").reshape(-1)
        clause, params = where(self.scope, filters)
        rows = self.conn.execute(
            f"SELECT id, -(embedding <#> %s) FROM chunks{clause} ORDER BY embedding <#> %s, id LIMIT %s",
            [probe, *params, probe, pool],
        ).fetchall()
        if not rows:
            return []
        ids = np.array([r[0] for r in rows], dtype="int64")
        cosine = np.array([r[1] for r in rows], dtype="float32")

        # From here on, the FAISS store's blend, unchanged.
        lexical = None
        if hybrid:
            raw = self.keyword.scores(query, ids)
            if raw.max() > 0:
                lexical = raw / raw.max()
                spread = cosine.max() - cosine.min()
                dense = (cosine - cosine.min()) / spread if spread else cosine * 0
                ranking = (
                    (1 - config.KEYWORD_WEIGHT) * dense
                    + config.KEYWORD_WEIGHT * lexical
                )
                order = np.argsort(-ranking)
                cosine, ids, lexical = cosine[order], ids[order], lexical[order]

        if min_similarity is not None:
            related = cosine >= min_similarity
            if lexical is not None:
                related |= lexical >= config.KEYWORD_RESCUE
            cosine, ids = cosine[related], ids[related]

        chosen = [int(i) for i in ids[:k]]
        chunks = self.chunks_by_id(chosen)
        return [(chunks[i], float(score)) for i, score in zip(chosen, cosine[:k])]

    def chunks_by_id(self, ids: Sequence[int]) -> dict[int, Chunk]:
        if not ids:
            return {}
        rows = self.conn.execute(f"SELECT {CHUNK_COLUMNS} FROM chunks WHERE id = ANY(%s)", [list(ids)])
        return {row[0]: to_chunk(row) for row in rows.fetchall()}


def to_chunk(row) -> Chunk:
    """A row of CHUNK_COLUMNS as the Chunk the FAISS store would hold."""
    from ingestion.filings import source_name

    (_id, chunk_id, accession, ticker, form, year, period, page,
     start_char, end_char, start_word, end_word, text) = row
    return Chunk(
        text=text,
        source=source_name({"ticker": ticker, "form": form, "fiscal_year": year, "fiscal_period": period}),
        page=page, chunk_id=chunk_id,
        start_word=start_word, end_word=end_word, start_char=start_char, end_char=end_char,
        ticker=ticker, accession=accession, form=form,
        fiscal_year=year if year is not None else 0,
        fiscal_period=period if period is not None else "",
    )
