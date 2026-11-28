"""Apply the schema migrations in `db/migrations/` (ROADMAP 2.1).

    python -m db.migrate              # the live database (DATABASE_URL)
    python -m db.migrate --test       # the Neon test branch (TEST_DATABASE_URL)
    python -m db.migrate --status     # list what is applied, change nothing

A migration is a file `NNN_name.sql`, applied once, in number order, each in
its own transaction: Postgres DDL is transactional, so a migration that fails
halfway leaves nothing behind and is not recorded.

`schema_migrations` records each applied file with a SHA-256 of its text. A
file edited after it was applied is refused rather than silently skipped: the
database would no longer match what the file says. Change the schema with a
new migration instead.

Two runners at once (say, two deploys) are kept apart by an advisory lock.
"""

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
FILE_NAME = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")
LOCK_ID = 2_001_202_611   # any constant; the same one for every runner

LEDGER = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     integer     PRIMARY KEY,
    name        text        NOT NULL,
    sha256      text        NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()

    @property
    def label(self) -> str:
        return f"{self.version:03d}_{self.name}"


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Every migration file, in order. A stray .sql file or a reused number is
    an error, not something to guess about."""
    found: dict[int, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        match = FILE_NAME.match(path.name)
        if not match:
            raise MigrationError(f"{path.name}: migration files are named NNN_name.sql")
        version = int(match.group(1))
        if version in found:
            raise MigrationError(f"two migrations numbered {version:03d}: "
                                 f"{found[version].label}.sql and {path.name}")
        found[version] = Migration(version, match.group(2), path.read_text(encoding="utf-8"))
    versions = sorted(found)
    if versions and versions != list(range(1, len(versions) + 1)):
        raise MigrationError(f"migration numbers must run 001, 002, ... without gaps; found {versions}")
    return [found[v] for v in versions]


def applied(conn: psycopg.Connection) -> dict[int, tuple[str, str]]:
    """version -> (name, sha256) of what this database has applied."""
    exists = conn.execute("SELECT to_regclass('schema_migrations') IS NOT NULL").fetchone()[0]
    if not exists:
        return {}
    rows = conn.execute("SELECT version, name, sha256 FROM schema_migrations").fetchall()
    return {version: (name, sha) for version, name, sha in rows}


def check_history(migrations: list[Migration], done: dict[int, tuple[str, str]]) -> None:
    files = {m.version: m for m in migrations}
    for version, (name, sha) in sorted(done.items()):
        migration = files.get(version)
        if migration is None:
            raise MigrationError(f"the database has migration {version:03d}_{name}, "
                                 "but its file is gone. Restore it; never delete an applied migration.")
        if migration.sha256 != sha:
            raise MigrationError(f"{migration.label}.sql changed after it was applied. "
                                 "Put the change in a new migration instead.")


def migrate(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply whatever is pending. Returns the labels applied, in order."""
    migrations = discover(directory)
    # Each `conn.transaction()` below must be a real transaction, committed on
    # its own, not a savepoint inside one the caller left open. So close any
    # open one and switch to autocommit, where transaction() means BEGIN/COMMIT.
    if not conn.autocommit:
        conn.commit()
        conn.autocommit = True
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
        conn.execute(LEDGER)
    newly = []
    for migration in migrations:
        with conn.transaction():
            # Re-read under the lock each time: another runner may have got here first.
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
            done = applied(conn)
            check_history(migrations, done)
            if migration.version in done:
                continue
            try:
                conn.execute(migration.sql)
            except psycopg.Error as error:
                raise MigrationError(f"{migration.label}.sql failed, nothing from it was kept: "
                                     f"{error}") from error
            conn.execute(
                "INSERT INTO schema_migrations (version, name, sha256) VALUES (%s, %s, %s)",
                (migration.version, migration.name, migration.sha256),
            )
        newly.append(migration.label)
    return newly


def status(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    migrations = discover(directory)
    done = applied(conn)
    check_history(migrations, done)
    return [f"{'applied' if m.version in done else 'pending'}  {m.label}" for m in migrations]


def main(argv: list[str] | None = None) -> int:
    from db import connection

    parser = argparse.ArgumentParser(prog="python -m db.migrate", description=__doc__.split("\n\n")[0])
    parser.add_argument("--test", action="store_true", help="the Neon test branch (TEST_DATABASE_URL)")
    parser.add_argument("--status", action="store_true", help="list applied and pending, change nothing")
    args = parser.parse_args(argv)

    variable = connection.TEST if args.test else connection.LIVE
    try:
        with connection.connect(variable, register_vector=False) as conn:
            if args.status:
                print("\n".join(status(conn)) or "no migrations")
                return 0
            newly = migrate(conn)
    except (MigrationError, connection.DatabaseNotConfigured) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{variable}: applied {', '.join(newly)}" if newly else f"{variable}: already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
