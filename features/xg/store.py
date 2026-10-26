"""Write scored shots to PostgreSQL (table `shots`, view `team_match_xg` in db/schema.sql)."""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import psycopg

from .shots import SHOT_COLUMNS

DB_COLUMNS = SHOT_COLUMNS + ("xg", "xg_model")
# pandas turns an integer column with a missing value (a shot without a player) into floats,
# which PostgreSQL rejects for INTEGER columns ("25437.0"); these are cast back explicitly.
INT_COLUMNS = frozenset({"event_id", "match_id", "team_id", "player_id", "score_diff"})


def _rows(shots: pd.DataFrame) -> Iterator[tuple]:
    """Plain Python values per row: int / float / bool / str, None for missing."""
    columns = []
    for col in DB_COLUMNS:
        s = shots[col]
        if col in INT_COLUMNS:
            s = s.astype("Int64")
        columns.append(s.astype(object).where(s.notna(), None))
    return zip(*columns)


def write_shots(conn: psycopg.Connection, shots: pd.DataFrame) -> int:
    """Replace the whole `shots` table with these rows (one transaction)."""
    n = 0
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM shots")
        with cur.copy(f"COPY shots ({', '.join(DB_COLUMNS)}) FROM STDIN") as cp:
            for row in _rows(shots):
                cp.write_row(row)
                n += 1
    return n
