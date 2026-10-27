"""Load the downloaded Transfermarkt files into the tm_* tables.

Each file replaces its whole table inside one transaction (TRUNCATE + COPY) and is skipped when its
SHA-256 matches the last successful load, like the Wyscout loader.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import psycopg

from ..wyscout.load import FileResult
from .download import sha256
from .tables import SPEC, TABLES, db_table, read_raw

SOURCE = "transfermarkt"
Log = Callable[[str], None]


def _rows(df: pd.DataFrame, table: str):
    """Plain Python values for COPY: None for missing, datetime.date for dates."""
    columns = []
    for col, kind in SPEC[table].items():
        s = df[col]
        if kind == "date":
            values = [None if pd.isna(v) else v.date() for v in s]
        elif kind == "int":
            values = [None if pd.isna(v) else int(v) for v in s]
        elif kind == "num":
            values = [None if pd.isna(v) else float(v) for v in s]
        else:
            values = [None if v is None else str(v) for v in s]
        columns.append(values)
    return zip(*columns)


def load_table(conn: psycopg.Connection, raw_dir: Path, table: str, force: bool = False, log: Log = print) -> FileResult:
    path = Path(raw_dir) / f"{table}.csv.gz"
    name = path.name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m etl download-transfermarkt`")
    digest = sha256(path)
    if not force:
        row = conn.execute("SELECT sha256 FROM etl_loads WHERE source = %s AND file_name = %s",
                           (SOURCE, name)).fetchone()
        if row and row[0] == digest:
            log(f"  skip   {name} (unchanged)")
            return FileResult(name, "skipped")

    log(f"  load   {name}")
    started = time.perf_counter()
    df = read_raw(raw_dir, table)
    cols = list(SPEC[table])
    with conn.transaction():
        conn.execute(f"TRUNCATE {db_table(table)}")
        with conn.cursor() as cur, cur.copy(f"COPY {db_table(table)} ({', '.join(cols)}) FROM STDIN") as cp:
            for r in _rows(df, table):
                cp.write_row(r)
        conn.execute(
            "INSERT INTO etl_loads (source, file_name, sha256, rows_loaded) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (source, file_name) DO UPDATE "
            "SET sha256 = EXCLUDED.sha256, rows_loaded = EXCLUDED.rows_loaded, loaded_at = now()",
            (SOURCE, name, digest, len(df)),
        )
    seconds = time.perf_counter() - started
    log(f"         {len(df):,} rows in {seconds:.1f}s")
    return FileResult(name, "loaded", len(df), seconds)


def load_transfermarkt(conn: psycopg.Connection, raw_dir: Path, force: bool = False, log: Log = print) -> list[FileResult]:
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir} (run `python -m etl download-transfermarkt` first)")
    return [load_table(conn, raw_dir, t, force, log) for t in TABLES]
