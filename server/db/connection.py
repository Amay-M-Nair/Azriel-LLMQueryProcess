"""Opening a connection to Neon.

Two databases, never confused:
- `DATABASE_URL`: the live index (Neon's `production` branch).
- `TEST_DATABASE_URL`: the `test` branch, the only one tests may touch.

Both are direct (unpooled) connection strings. Migrations need a direct
connection, and at this size nothing needs a pooler yet.
"""

import os

import psycopg

from utils.logs import redact

LIVE = "DATABASE_URL"
TEST = "TEST_DATABASE_URL"


class DatabaseNotConfigured(RuntimeError):
    pass


def url(variable: str = LIVE) -> str:
    value = (os.environ.get(variable) or "").strip()
    if not value:
        raise DatabaseNotConfigured(f"{variable} is not set. Put the Neon connection string in .env.")
    if variable == TEST and value == (os.environ.get(LIVE) or "").strip():
        # A test that wipes its schema must never be able to reach the live data.
        raise DatabaseNotConfigured(f"{TEST} is the same as {LIVE}. Point it at the Neon test branch.")
    return value


def connect(variable: str = LIVE, *, search_path: str | None = None,
            register_vector: bool = True) -> psycopg.Connection:
    """A connection with pgvector's types registered, so a numpy array goes in
    as a `vector` and comes back as one.

    `search_path` puts the connection in another schema first (tests use a
    throwaway schema each). `public` stays on the path, since that is where the
    `vector` extension lives.

    `register_vector=False` is for before the extension exists: registering
    looks the type up and fails on an empty database.
    """
    options = f"-c search_path={search_path},public" if search_path else None
    try:
        conn = psycopg.connect(url(variable), options=options, connect_timeout=15)
    except psycopg.OperationalError as error:
        # libpq echoes the connection string in some errors; never print the password.
        raise psycopg.OperationalError(redact(str(error))) from None
    if register_vector:
        from pgvector.psycopg import register_vector as register

        register(conn)
    return conn
