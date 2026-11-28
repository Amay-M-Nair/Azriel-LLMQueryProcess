"""The Neon schema and its migration runner (ROADMAP 2.1).

Done when: migrations apply to a fresh Neon branch. Each database test gets a
throwaway schema on the Neon `test` branch (TEST_DATABASE_URL), applies the
migrations into it from nothing, and drops it afterwards. Without
TEST_DATABASE_URL those tests skip, and a skip leaves 2.1 unticked.

Tests never read DATABASE_URL: that is the live index.
"""

import shutil

import numpy as np
import psycopg
import pytest

from db import connection, migrate

pytestmark = pytest.mark.roadmap("2.1")

ACCESSION = "0000320193-25-000079"
SHA = "a" * 64


@pytest.fixture
def fresh(neon_schema):
    with neon_schema("t21") as conn:
        yield conn


def tables_in(conn) -> set[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
    ).fetchall()
    return {name for (name,) in rows}


def add_filing(conn, accession=ACCESSION, form="10-K", year=2025, period="FY"):
    conn.execute(
        "INSERT INTO filings (accession, cik, ticker, company, form, fiscal_year, fiscal_period,"
        " filed, primary_document, sha256)"
        " VALUES (%s, '0000320193', 'AAPL', 'Apple Inc.', %s, %s, %s, '2025-10-31', 'aapl.htm', %s)",
        (accession, form, year, period, SHA),
    )


def add_chunk(conn, chunk_id, vector, accession=ACCESSION, year=2025, period="FY", form="10-K"):
    conn.execute(
        "INSERT INTO chunks (chunk_id, accession, ticker, cik, form, fiscal_year, fiscal_period,"
        " start_char, end_char, start_word, end_word, words, text, embedding)"
        " VALUES (%s, %s, 'AAPL', '0000320193', %s, %s, %s, 0, 10, 0, 2, 2, 'some text', %s)",
        (chunk_id, accession, form, year, period, vector),
    )


def unit(*values) -> np.ndarray:
    v = np.zeros(384, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


# --- Discovering migration files (no database) -------------------------------

def test_the_shipped_migrations_are_well_formed():
    labels = [m.label for m in migrate.discover()]
    assert labels[0] == "001_schema"


@pytest.mark.parametrize("names, problem", [
    (["001_a.sql", "001_b.sql"], "two migrations numbered 001"),
    (["001_a.sql", "003_c.sql"], "without gaps"),
    (["1_a.sql"], "named NNN_name.sql"),
])
def test_a_bad_set_of_files_is_refused(tmp_path, names, problem):
    for name in names:
        (tmp_path / name).write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(migrate.MigrationError, match=problem):
        migrate.discover(tmp_path)


def test_tests_can_never_reach_the_live_database(monkeypatch):
    monkeypatch.setenv(connection.LIVE, "postgresql://u:p@live/db")
    monkeypatch.setenv(connection.TEST, "postgresql://u:p@live/db")
    with pytest.raises(connection.DatabaseNotConfigured, match="same as"):
        connection.url(connection.TEST)


# --- Applying them to a fresh schema on the Neon test branch -----------------

def test_migrations_apply_to_a_fresh_branch(fresh):
    assert tables_in(fresh) == set()
    assert migrate.migrate(fresh) == ["001_schema", "002_chunk_words"]
    assert tables_in(fresh) == {"schema_migrations", "filings", "chunks", "tables", "eval_runs"}
    dims = fresh.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()[0]
    assert dims == 384
    assert migrate.status(fresh) == ["applied  001_schema", "applied  002_chunk_words"]


def test_a_second_run_applies_nothing(fresh):
    migrate.migrate(fresh)
    assert migrate.migrate(fresh) == []


def test_an_edited_migration_is_refused(fresh, tmp_path):
    shutil.copy(migrate.MIGRATIONS_DIR / "001_schema.sql", tmp_path)
    migrate.migrate(fresh, tmp_path)
    with open(tmp_path / "001_schema.sql", "a", encoding="utf-8") as f:
        f.write("\n-- an afterthought\n")
    with pytest.raises(migrate.MigrationError, match="changed after it was applied"):
        migrate.migrate(fresh, tmp_path)


def test_a_failing_migration_leaves_nothing_behind(fresh, tmp_path):
    shutil.copy(migrate.MIGRATIONS_DIR / "001_schema.sql", tmp_path)
    (tmp_path / "002_broken.sql").write_text(
        "CREATE TABLE half_done (id int);\nSELECT no_such_column FROM filings;\n", encoding="utf-8")
    with pytest.raises(migrate.MigrationError, match="002_broken.sql failed"):
        migrate.migrate(fresh, tmp_path)
    assert "half_done" not in tables_in(fresh)
    assert sorted(migrate.applied(fresh)) == [1], "001 is kept, the broken 002 is not recorded"


# --- What the schema itself allows -------------------------------------------

@pytest.fixture
def schema(neon_migrated):
    neon_migrated.execute("TRUNCATE filings, tables, eval_runs CASCADE")
    return neon_migrated


def test_form_is_free_text_but_known_forms_carry_their_period(schema):
    add_filing(schema)
    # An 8-K (later) needs no migration and has no fiscal period.
    add_filing(schema, accession="0000320193-25-000080", form="8-K", year=None, period=None)
    for accession, form, year, period in [
        ("0000320193-25-000081", "10-K", None, None),   # a 10-K must say its fiscal year
        ("0000320193-25-000082", "10-Q", 2025, "Q4"),   # there is no Q4 10-Q
        ("0000320193-25-000083", "10-K", 2025, "Q1"),   # a 10-K covers the year
    ]:
        with pytest.raises(psycopg.errors.CheckViolation):
            add_filing(schema, accession=accession, form=form, year=year, period=period)


def test_a_chunk_cannot_disagree_with_its_filing(schema):
    add_filing(schema)
    add_chunk(schema, "ok", unit(1))
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        add_chunk(schema, "wrong-year", unit(1), year=2024)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        add_chunk(schema, "no-filing", unit(1), accession="0000000000-25-000001")


def test_vectors_must_have_384_dimensions(schema):
    add_filing(schema)
    with pytest.raises(psycopg.errors.DataException):
        add_chunk(schema, "short", np.ones(383, dtype=np.float32))


def test_deleting_a_filing_deletes_its_chunks_and_tables(schema):
    add_filing(schema)
    add_chunk(schema, "c1", unit(1))
    schema.execute("INSERT INTO tables (accession, data) VALUES (%s, '{}')", (ACCESSION,))
    schema.execute("DELETE FROM filings WHERE accession = %s", (ACCESSION,))
    assert schema.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert schema.execute("SELECT count(*) FROM tables").fetchone()[0] == 0


def test_vectors_round_trip_and_rank_by_cosine(schema):
    """What 2.2's exact search will rely on: `<=>` is cosine distance, so
    1 - distance is the same cosine similarity FAISS's inner product gives."""
    add_filing(schema)
    vectors = {"near": unit(1, 0.1), "middle": unit(1, 1), "far": unit(0, 1)}
    for chunk_id, vector in vectors.items():
        add_chunk(schema, chunk_id, vector)
    query = unit(1)
    rows = schema.execute(
        "SELECT chunk_id, 1 - (embedding <=> %s) AS cosine, embedding FROM chunks ORDER BY embedding <=> %s",
        (query, query),
    ).fetchall()
    assert [r[0] for r in rows] == ["near", "middle", "far"]
    for chunk_id, cosine, stored in rows:
        assert np.array_equal(stored.to_numpy(), vectors[chunk_id]), "float32 survives the round trip exactly"
        assert cosine == pytest.approx(float(vectors[chunk_id] @ query), abs=1e-6)
