"""End-to-end: load the fictional sample into a real PostgreSQL database."""
from __future__ import annotations

import shutil

from etl.quality import FAIL, OK, WARN, has_failures, run_checks
from etl.wyscout.load import load_wyscout

QUIET = lambda _msg: None  # noqa: E731

COUNTED = ("competitions", "teams", "players", "tags", "matches", "match_teams",
           "lineups", "substitutions", "events", "event_tags")


def _counts(conn):
    return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in COUNTED}


EXPECTED = {
    "competitions": 1, "teams": 3, "players": 9, "tags": 9, "matches": 2, "match_teams": 4,
    "lineups": 17, "substitutions": 1, "events": 39,
}


def test_load_counts_and_values(conn, fixture_dir):
    results = load_wyscout(conn, fixture_dir, log=QUIET)
    assert all(r.status == "loaded" for r in results)

    counts = _counts(conn)
    assert {k: counts[k] for k in EXPECTED} == EXPECTED
    assert counts["event_tags"] > counts["events"]

    name = conn.execute("SELECT short_name FROM players WHERE player_id = 201").fetchone()[0]
    assert name == "D. Öztürk"
    null_players = conn.execute("SELECT count(*) FROM events WHERE player_id IS NULL").fetchone()[0]
    assert null_players == 1


def test_reload_is_idempotent(conn, fixture_dir):
    load_wyscout(conn, fixture_dir, log=QUIET)
    first = _counts(conn)

    again = load_wyscout(conn, fixture_dir, log=QUIET)
    assert all(r.status == "skipped" for r in again if r.status != "missing")

    forced = load_wyscout(conn, fixture_dir, force=True, log=QUIET)
    assert all(r.status == "loaded" for r in forced)
    assert _counts(conn) == first


def test_changed_file_replaces_its_matches(conn, fixture_dir, tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(fixture_dir, raw)
    load_wyscout(conn, raw, log=QUIET)

    # Drop the last event from the file: only that event should disappear.
    events_file = raw / "events" / "events_Sample.json"
    import json
    events = json.loads(events_file.read_text(encoding="utf-8"))
    dropped = events.pop()
    events_file.write_text(json.dumps(events), encoding="utf-8")

    results = {r.file: r.status for r in load_wyscout(conn, raw, log=QUIET)}
    assert results["events/events_Sample.json"] == "loaded"
    assert results["players.json"] == "skipped"
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 38
    assert conn.execute("SELECT count(*) FROM events WHERE event_id = %s", (dropped["id"],)).fetchone()[0] == 0


def test_quality_checks(conn, fixture_dir):
    load_wyscout(conn, fixture_dir, log=QUIET)
    checks = {c.name: c for c in run_checks(conn)}

    assert not has_failures(list(checks.values()))
    assert checks["two teams per match"].status == OK
    assert checks["coordinates within 0-100"].status == OK
    # Penalty and own goal count, goalkeeper "goal" tags do not: 4/4 team scores match.
    assert checks["scores rebuilt from events"].status == OK, checks["scores rebuilt from events"].detail
    # The sample deliberately contains player 105, who is not in players.json.
    assert checks["players known in players.json"].status == WARN
    # A 39-event sample is far below a real match.
    assert checks["events per match"].status == WARN


def test_coordinate_tolerance(conn, fixture_dir):
    load_wyscout(conn, fixture_dir, log=QUIET)
    status = lambda: {c.name: c for c in run_checks(conn)}["coordinates within 0-100"].status  # noqa: E731
    event_id = conn.execute("SELECT min(event_id) FROM events").fetchone()[0]

    conn.execute("UPDATE events SET end_y = 101 WHERE event_id = %s", (event_id,))
    assert status() == WARN   # recording artefact seen in the real data
    conn.execute("UPDATE events SET start_x = -1 WHERE event_id = %s", (event_id,))
    assert status() == WARN
    conn.execute("UPDATE events SET end_y = 150 WHERE event_id = %s", (event_id,))
    assert status() == FAIL   # genuinely broken


def test_quality_fails_on_empty_database(conn):
    checks = {c.name: c for c in run_checks(conn)}
    assert checks["data present"].status == FAIL
    assert checks["tm data present"].status == "WARN"  # Transfermarkt not loaded is a warning, not a failure
