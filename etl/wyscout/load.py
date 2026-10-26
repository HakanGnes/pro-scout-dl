"""Load the Wyscout public dataset from a raw directory into PostgreSQL.

Idempotent by design:
  * Every source file is loaded inside its own transaction.
  * A file whose SHA-256 matches the last successful load is skipped
    (pass force=True to reload anyway).
  * Reference tables are upserted; per-match children (teams, lineups,
    substitutions, events, tags) are replaced for the matches in the file.

The raw directory layout is flexible: files are found by name anywhere below it
(e.g. data/raw/wyscout/events/events_England.json).
"""
from __future__ import annotations

import csv
import hashlib
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ijson
import psycopg

from . import parse as p

SOURCE = "wyscout"
Log = Callable[[str], None]


@dataclass
class FileResult:
    file: str
    status: str  # 'loaded' | 'skipped' | 'missing'
    rows: int = 0
    seconds: float = 0.0
    note: str = ""


# ---------------------------------------------------------------- helpers


def _iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the items of a top-level JSON array without loading the whole file."""
    with path.open("rb") as f:
        yield from ijson.items(f, "item", use_float=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_one(raw_dir: Path, name: str) -> Path | None:
    hits = sorted(raw_dir.rglob(name))
    return hits[0] if hits else None


def _upsert(cur: psycopg.Cursor, table: str, cols: tuple[str, ...], pk: tuple[str, ...], rows: list[tuple]) -> int:
    if not rows:
        return 0
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in pk)
    conflict = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
    cur.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
        f"ON CONFLICT ({', '.join(pk)}) {conflict}",
        rows,
    )
    return len(rows)


def _copy(cur: psycopg.Cursor, table: str, cols: tuple[str, ...], rows: list[tuple]) -> int:
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
        for row in rows:
            cp.write_row(row)
    return len(rows)


# ---------------------------------------------------------------- per-file loaders
# Each returns (rows loaded, note) and runs inside the caller's transaction.


def _load_tags(conn: psycopg.Connection, path: Path, log: Log) -> tuple[int, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in (p.parse_tag(raw) for raw in csv.DictReader(f)) if r]
    with conn.cursor() as cur:
        return _upsert(cur, "tags", p.TAG_COLS, ("tag_id",), rows), ""


def _reference_loader(table: str, cols: tuple[str, ...], parser: Callable[[dict], tuple]):
    def load(conn: psycopg.Connection, path: Path, log: Log) -> tuple[int, str]:
        rows = [parser(d) for d in _iter_json_array(path)]
        bad = sum(1 for r in rows if r[0] is None)
        rows = [r for r in rows if r[0] is not None]
        with conn.cursor() as cur:
            n = _upsert(cur, table, cols, (cols[0],), rows)
        return n, f"{bad} records without id skipped" if bad else ""
    return load


_load_competitions = _reference_loader("competitions", p.COMPETITION_COLS, p.parse_competition)
_load_teams = _reference_loader("teams", p.TEAM_COLS, p.parse_team)
_load_players = _reference_loader("players", p.PLAYER_COLS, p.parse_player)


def _load_matches(conn: psycopg.Connection, path: Path, log: Log) -> tuple[int, str]:
    records = [p.parse_match(d) for d in _iter_json_array(path)]
    records = [r for r in records if r.match[0] is not None]
    if not records:
        return 0, "file contains no matches"

    match_ids = [r.match[0] for r in records]
    competition_ids = {r.match[1] for r in records if r.match[1] is not None}
    team_ids = {t[1] for r in records for t in r.teams}

    with conn.cursor() as cur:
        # Placeholders keep foreign keys valid if a reference file lacks an id.
        cur.execute("SELECT competition_id FROM competitions WHERE competition_id = ANY(%s)", (list(competition_ids),))
        missing_comp = competition_ids - {row[0] for row in cur.fetchall()}
        cur.execute("SELECT team_id FROM teams WHERE team_id = ANY(%s)", (list(team_ids),))
        missing_teams = team_ids - {row[0] for row in cur.fetchall()}
        cur.executemany(
            "INSERT INTO competitions (competition_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            [(c, f"Unknown competition {c}") for c in sorted(missing_comp)],
        )
        cur.executemany(
            "INSERT INTO teams (team_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            [(t, f"Unknown team {t}") for t in sorted(missing_teams)],
        )

        _upsert(cur, "matches", p.MATCH_COLS, ("match_id",), [r.match for r in records])
        # Replace per-match children (cascades to lineups and substitutions).
        cur.execute("DELETE FROM match_teams WHERE match_id = ANY(%s)", (match_ids,))
        _copy(cur, "match_teams", p.MATCH_TEAM_COLS, [t for r in records for t in r.teams])
        _copy(cur, "lineups", p.LINEUP_COLS, [x for r in records for x in r.lineups])
        # Substitutions can repeat in the raw data; de-duplicate on the primary key.
        subs = {(s[0], s[2], s[3]): s for r in records for s in r.substitutions}
        _copy(cur, "substitutions", p.SUBSTITUTION_COLS, list(subs.values()))

    notes = []
    if missing_comp:
        notes.append(f"{len(missing_comp)} competition placeholder(s)")
    if missing_teams:
        notes.append(f"{len(missing_teams)} team placeholder(s)")
    return len(records), "; ".join(notes)


def _load_events(conn: psycopg.Connection, path: Path, log: Log) -> tuple[int, str]:
    cols = ", ".join(p.EVENT_COLS)
    staged = bad = 0
    with conn.cursor() as cur, tempfile.TemporaryFile("w+", encoding="utf-8", newline="") as tag_buf:
        cur.execute("CREATE TEMP TABLE stage_events (LIKE events) ON COMMIT DROP")
        cur.execute("CREATE TEMP TABLE stage_event_tags (event_id BIGINT, tag_id INTEGER) ON COMMIT DROP")

        # Pass 1: stream events into staging; tags go to a temp file (one COPY at a time).
        with cur.copy(f"COPY stage_events ({cols}) FROM STDIN") as cp:
            for raw in _iter_json_array(path):
                row, tags = p.parse_event(raw)
                if not p.event_is_loadable(row):
                    bad += 1
                    continue
                cp.write_row(row)
                for tag in tags:
                    tag_buf.write(f"{row[0]}\t{tag}\n")
                staged += 1
                if staged % 250_000 == 0:
                    log(f"    {path.name}: {staged:,} events staged")

        tag_buf.seek(0)
        with cur.copy("COPY stage_event_tags (event_id, tag_id) FROM STDIN") as cp:
            while chunk := tag_buf.read(1 << 20):
                cp.write(chunk)

        # Pass 2: replace the events of every match present in the file.
        cur.execute(
            "SELECT count(*) FROM stage_events s "
            "WHERE NOT EXISTS (SELECT 1 FROM matches m WHERE m.match_id = s.match_id)"
        )
        no_match = cur.fetchone()[0]
        cur.execute("DELETE FROM events WHERE match_id IN (SELECT DISTINCT match_id FROM stage_events)")
        cur.execute(
            f"INSERT INTO events ({cols}) SELECT {cols} FROM stage_events s "
            "WHERE EXISTS (SELECT 1 FROM matches m WHERE m.match_id = s.match_id) "
            "ON CONFLICT (event_id) DO NOTHING"
        )
        inserted = cur.rowcount
        cur.execute(
            "INSERT INTO event_tags (event_id, tag_id) "
            "SELECT DISTINCT t.event_id, t.tag_id FROM stage_event_tags t "
            "JOIN stage_events s USING (event_id) "
            "JOIN matches m ON m.match_id = s.match_id "
            "ON CONFLICT DO NOTHING"
        )
        tag_rows = cur.rowcount

    duplicates = staged - no_match - inserted
    notes = [f"{tag_rows:,} tags"]
    if bad:
        notes.append(f"{bad:,} malformed skipped")
    if no_match:
        notes.append(f"{no_match:,} without a loaded match skipped")
    if duplicates:
        notes.append(f"{duplicates:,} duplicate ids skipped")
    return inserted, "; ".join(notes)


# ---------------------------------------------------------------- orchestration


def _run_file(
    conn: psycopg.Connection,
    raw_dir: Path,
    path: Path,
    loader: Callable[[psycopg.Connection, Path, Log], tuple[int, str]],
    force: bool,
    log: Log,
) -> FileResult:
    name = path.relative_to(raw_dir).as_posix()
    digest = _sha256(path)
    if not force:
        row = conn.execute(
            "SELECT sha256 FROM etl_loads WHERE source = %s AND file_name = %s", (SOURCE, name)
        ).fetchone()
        if row and row[0] == digest:
            log(f"  skip   {name} (unchanged)")
            return FileResult(name, "skipped")

    log(f"  load   {name}")
    started = time.perf_counter()
    with conn.transaction():
        rows, note = loader(conn, path, log)
        conn.execute(
            "INSERT INTO etl_loads (source, file_name, sha256, rows_loaded) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (source, file_name) DO UPDATE "
            "SET sha256 = EXCLUDED.sha256, rows_loaded = EXCLUDED.rows_loaded, loaded_at = now()",
            (SOURCE, name, digest, rows),
        )
    seconds = time.perf_counter() - started
    log(f"         {rows:,} rows in {seconds:.1f}s" + (f" ({note})" if note else ""))
    return FileResult(name, "loaded", rows, seconds, note)


def load_wyscout(conn: psycopg.Connection, raw_dir: Path, force: bool = False, log: Log = print) -> list[FileResult]:
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir} (run `python -m etl download-wyscout` first)")

    results: list[FileResult] = []
    reference = (
        ("tags2name.csv", _load_tags, False),
        ("competitions.json", _load_competitions, True),
        ("teams.json", _load_teams, True),
        ("players.json", _load_players, True),
    )
    for name, loader, required in reference:
        path = _find_one(raw_dir, name)
        if path is None:
            if required:
                raise FileNotFoundError(f"{name} not found under {raw_dir}")
            log(f"  miss   {name} (optional)")
            results.append(FileResult(name, "missing"))
            continue
        results.append(_run_file(conn, raw_dir, path, loader, force, log))

    match_files = sorted(raw_dir.rglob("matches_*.json"))
    event_files = sorted(raw_dir.rglob("events_*.json"))
    if not match_files:
        raise FileNotFoundError(f"No matches_*.json files under {raw_dir}")
    for path in match_files:  # all matches before any events (foreign keys)
        results.append(_run_file(conn, raw_dir, path, _load_matches, force, log))
    for path in event_files:
        results.append(_run_file(conn, raw_dir, path, _load_events, force, log))

    if any(r.status == "loaded" for r in results):
        log("  analyze (refreshing planner statistics)")
        conn.execute("ANALYZE")
    return results
