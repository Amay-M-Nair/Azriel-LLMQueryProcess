"""Fixtures for tests that need the Neon test branch (TEST_DATABASE_URL).

Each test module gets brand-new schemas on the `test` branch and drops them
afterwards, so tests start from nothing and never see each other's rows.
Without TEST_DATABASE_URL these tests skip, and a skipped roadmap check
counts as not done. DATABASE_URL, the live index, is never read here.
"""

import uuid
from contextlib import contextmanager

import pytest


@pytest.fixture(scope="session")
def neon_admin():
    """One connection for creating and dropping schemas, shared: each new
    connection to Neon costs a second or two."""
    from db import connection

    try:
        connection.url(connection.TEST)
    except connection.DatabaseNotConfigured as error:
        pytest.skip(str(error))
    with connection.connect(connection.TEST, register_vector=False) as conn:
        conn.autocommit = True
        yield conn


@pytest.fixture(scope="session")
def neon_schema(neon_admin):
    """A factory: `with neon_schema("t22") as conn:` gives a connection whose
    search path starts in a new, empty schema, dropped on exit."""
    from db import connection

    @contextmanager
    def make(prefix: str):
        name = f"{prefix}_{uuid.uuid4().hex[:10]}"
        neon_admin.execute(f'CREATE SCHEMA "{name}"')
        try:
            with connection.connect(connection.TEST, search_path=name, register_vector=False) as conn:
                yield conn
        finally:
            neon_admin.execute(f'DROP SCHEMA "{name}" CASCADE')

    return make


@pytest.fixture(scope="module")
def neon_migrated(neon_schema):
    """One fully migrated schema for a whole module, vector types registered,
    in autocommit."""
    from pgvector.psycopg import register_vector

    from db import migrate

    with neon_schema("tm") as conn:
        migrate.migrate(conn)
        register_vector(conn)
        yield conn
