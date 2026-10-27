"""Database connection and schema management."""
from __future__ import annotations

import psycopg

from .config import SCHEMA_PATH, database_url

# Drop order respects foreign keys (CASCADE handles the rest).
TABLES = (
    "tm_transfers",
    "tm_appearances",
    "tm_player_valuations",
    "tm_players",
    "tm_clubs",
    "tm_competitions",
    "shots",
    "event_tags",
    "events",
    "substitutions",
    "lineups",
    "match_teams",
    "matches",
    "players",
    "teams",
    "competitions",
    "tags",
    "etl_loads",
)


def connect(url: str | None = None) -> psycopg.Connection:
    """Open an autocommit connection; callers group work with `conn.transaction()`."""
    return psycopg.connect(url or database_url(), autocommit=True)


def init_db(conn: psycopg.Connection) -> None:
    """Create all tables and indexes that do not exist yet."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def reset_db(conn: psycopg.Connection) -> None:
    """Drop every table this project owns. Irreversible."""
    conn.execute("DROP TABLE IF EXISTS " + ", ".join(TABLES) + " CASCADE")
