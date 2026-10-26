"""Data-quality checks over the loaded database.

Statuses: OK (as expected), INFO (context, nothing to fix), WARN (worth a look,
does not block the next phase), FAIL (the data is not usable as is).
"""
from __future__ import annotations

from dataclasses import dataclass

import psycopg

OK, INFO, WARN, FAIL = "OK", "INFO", "WARN", "FAIL"

# A full Wyscout match has roughly 1,500–1,900 events; far fewer suggests truncation.
MIN_EVENTS_PER_MATCH = 1000
# Share of (match, team) scores that events must reproduce before we call it OK.
SCORE_AGREEMENT_OK = 0.98
# Coordinates this far past a pitch line are a known Wyscout recording artefact, not corruption.
COORD_TOLERANCE = 1

# Wyscout ids used below.
SHOT, FREE_KICK = 10, 3
FREE_KICK_SHOT, PENALTY = 33, 35
TAG_GOAL, TAG_OWN_GOAL = 101, 102


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _one(conn: psycopg.Connection, sql: str, params: tuple = ()) -> tuple:
    return conn.execute(sql, params).fetchone()


def _all(conn: psycopg.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    return conn.execute(sql, params).fetchall()


def run_checks(conn: psycopg.Connection) -> list[Check]:
    checks: list[Check] = []

    counts = {
        t: _one(conn, f"SELECT count(*) FROM {t}")[0]
        for t in ("competitions", "teams", "players", "matches", "lineups", "events", "event_tags", "tags")
    }
    checks.append(Check("row counts", INFO, ", ".join(f"{t}={n:,}" for t, n in counts.items())))
    if counts["matches"] == 0:
        checks.append(Check("data present", FAIL, "no matches loaded; run `python -m etl load-wyscout`"))
        return checks

    # Every match must have exactly one home and one away team.
    bad = _one(conn, """
        SELECT count(*) FROM (
            SELECT m.match_id FROM matches m LEFT JOIN match_teams mt USING (match_id)
            GROUP BY m.match_id
            HAVING count(mt.team_id) <> 2
                OR count(*) FILTER (WHERE mt.side = 'home') <> 1
        ) x""")[0]
    checks.append(Check("two teams per match", FAIL if bad else OK,
                        f"{bad:,} matches without exactly one home and one away team" if bad else "all matches"))

    # Matches with no events, per competition.
    rows = _all(conn, """
        SELECT c.name, count(*) FILTER (WHERE NOT EXISTS (
                   SELECT 1 FROM events e WHERE e.match_id = m.match_id)),
               count(*)
        FROM matches m JOIN competitions c USING (competition_id)
        GROUP BY c.name ORDER BY c.name""")
    missing = [(name, n, total) for name, n, total in rows if n]
    if not missing:
        checks.append(Check("events for every match", OK, f"{counts['matches']:,} matches"))
    else:
        detail = "; ".join(f"{name}: {n:,}/{total:,} without events" for name, n, total in missing)
        checks.append(Check("events for every match", WARN, detail))

    if counts["events"] == 0:
        checks.append(Check("events", WARN, "no events loaded; skipping event checks"))
        return checks

    n, lo, med, hi, few = _one(conn, f"""
        SELECT count(*), min(n), percentile_cont(0.5) WITHIN GROUP (ORDER BY n), max(n),
               count(*) FILTER (WHERE n < {MIN_EVENTS_PER_MATCH})
        FROM (SELECT match_id, count(*) AS n FROM events GROUP BY match_id) x""")
    checks.append(Check(
        "events per match", WARN if few else OK,
        f"min {lo:,} / median {med:,.0f} / max {hi:,} over {n:,} matches"
        + (f"; {few:,} below {MIN_EVENTS_PER_MATCH:,}" if few else ""),
    ))

    checks.append(_coordinate_check(conn))

    wrong_team = _one(conn, """
        SELECT count(*) FROM events e
        WHERE NOT EXISTS (SELECT 1 FROM match_teams mt
                          WHERE mt.match_id = e.match_id AND mt.team_id = e.team_id)""")[0]
    checks.append(Check("event team plays in the match", FAIL if wrong_team else OK,
                        f"{wrong_team:,} events" if wrong_team else "all events"))

    no_player = _one(conn, "SELECT count(*) FROM events WHERE player_id IS NULL")[0]
    checks.append(Check("events without a player", INFO,
                        f"{no_player:,} ({no_player / counts['events']:.2%}); expected for ball-out and similar events"))

    orphan_events = _one(conn, """
        SELECT count(DISTINCT e.player_id) FROM events e
        WHERE e.player_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM players p WHERE p.player_id = e.player_id)""")[0]
    orphan_lineups = _one(conn, """
        SELECT count(DISTINCT l.player_id) FROM lineups l
        WHERE NOT EXISTS (SELECT 1 FROM players p WHERE p.player_id = l.player_id)""")[0]
    checks.append(Check("players known in players.json", WARN if orphan_events or orphan_lineups else OK,
                        f"{orphan_events:,} unknown player ids in events, {orphan_lineups:,} in lineups"))

    if counts["tags"]:
        unknown_tags = _all(conn, """
            SELECT DISTINCT t.tag_id FROM event_tags t
            WHERE NOT EXISTS (SELECT 1 FROM tags g WHERE g.tag_id = t.tag_id) ORDER BY 1 LIMIT 10""")
        checks.append(Check("tag ids described in tags2name.csv", WARN if unknown_tags else OK,
                            "unknown: " + ", ".join(str(r[0]) for r in unknown_tags) if unknown_tags else "all tags"))

    checks.append(_score_reconciliation(conn))
    return checks


def _out_of_range(conn: psycopg.Connection, lo: int, hi: int) -> int:
    return _one(conn, f"""
        SELECT count(*) FROM events
        WHERE start_x NOT BETWEEN {lo} AND {hi} OR start_y NOT BETWEEN {lo} AND {hi}
           OR end_x   NOT BETWEEN {lo} AND {hi} OR end_y   NOT BETWEEN {lo} AND {hi}""")[0]


def _coordinate_check(conn: psycopg.Connection) -> Check:
    """0-100 is the pitch. Wyscout records a handful of balls just past a line as -1 or 101
    (the next event then starts at the same point), so ±1 is a WARN; anything beyond is a FAIL.
    The raw values are kept; feature code clips to 0-100."""
    gross = _out_of_range(conn, -COORD_TOLERANCE, 100 + COORD_TOLERANCE)
    if gross:
        return Check("coordinates within 0-100", FAIL, f"{gross:,} events more than {COORD_TOLERANCE} unit outside")
    slight = _out_of_range(conn, 0, 100)
    if slight:
        return Check("coordinates within 0-100", WARN,
                     f"{slight:,} events at most {COORD_TOLERANCE} unit outside (kept raw; clip in features)")
    return Check("coordinates within 0-100", OK, "all events")


def _score_reconciliation(conn: psycopg.Connection) -> Check:
    """Rebuild each team's score from goal-tagged events and compare with the official score.

    Goals for a team = its goal-tagged shots, free-kick shots and penalties
    (shoot-out excluded) + own goals by the opponent.
    """
    rows = _all(conn, f"""
        WITH goal_events AS (
            SELECT e.match_id, e.team_id AS team_id
            FROM events e JOIN event_tags t ON t.event_id = e.event_id AND t.tag_id = {TAG_GOAL}
            WHERE e.period <> 'P'
              AND (e.type_id = {SHOT} OR (e.type_id = {FREE_KICK} AND e.subtype_id IN ({FREE_KICK_SHOT}, {PENALTY})))
            UNION ALL
            SELECT e.match_id, mt.team_id
            FROM events e
            JOIN event_tags t ON t.event_id = e.event_id AND t.tag_id = {TAG_OWN_GOAL}
            JOIN match_teams mt ON mt.match_id = e.match_id AND mt.team_id <> e.team_id
            WHERE e.period <> 'P'
        ),
        rebuilt AS (SELECT match_id, team_id, count(*) AS goals FROM goal_events GROUP BY 1, 2)
        SELECT mt.match_id, mt.team_id, mt.score, coalesce(r.goals, 0)
        FROM match_teams mt
        JOIN (SELECT DISTINCT match_id FROM events) em USING (match_id)
        LEFT JOIN rebuilt r USING (match_id, team_id)""")
    if not rows:
        return Check("scores rebuilt from events", INFO, "no matches with events")
    mismatches = [r for r in rows if r[2] != r[3]]
    share = 1 - len(mismatches) / len(rows)
    detail = f"{share:.1%} of {len(rows):,} team scores match"
    if mismatches:
        detail += "; e.g. " + ", ".join(f"match {m} team {t}: {s} vs {g}" for m, t, s, g in mismatches[:3])
    return Check("scores rebuilt from events", OK if share >= SCORE_AGREEMENT_OK else WARN, detail)


def format_report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    return "\n".join(f"[{c.status:<4}] {c.name:<{width}}  {c.detail}" for c in checks)


def has_failures(checks: list[Check]) -> bool:
    return any(c.status == FAIL for c in checks)
