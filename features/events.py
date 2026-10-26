"""Event and match access shared by the feature builders (xG, passing networks).

Each source yields the same objects, so feature code is written once and the
tests can check that raw JSON and PostgreSQL agree:
  * `iter_matches_raw` / `iter_matches_db`   -> (competition label, events of one match)
  * `load_contexts_raw` / `load_contexts_db` -> {match_id: MatchContext}

Competition labels differ by source (raw: file label "England"; database:
competition name "English first division"); each is consistent within its source.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

PERIOD_ORDER = {"1H": 0, "2H": 1, "E1": 2, "E2": 3, "P": 4}


class Event(NamedTuple):
    event_id: int
    match_id: int
    team_id: int
    player_id: int | None
    period: str
    event_sec: float
    type_id: int | None
    subtype_id: int | None
    x: int | None
    y: int | None
    tags: frozenset[int]
    end_x: int | None = None
    end_y: int | None = None


def sort_key(e: Event) -> tuple:
    return (PERIOD_ORDER.get(e.period, 9), e.event_sec, e.event_id)


# ---------------------------------------------------------------- events


def iter_matches_raw(raw_dir: Path, competitions: list[str] | None = None) -> Iterator[tuple[str, list[Event]]]:
    """(competition file label, events of one match) from data/raw/wyscout/**/events_*.json."""
    import ijson

    from etl.wyscout.parse import EVENT_COLS, event_is_loadable, parse_event

    idx = {c: i for i, c in enumerate(EVENT_COLS)}
    for path in sorted(Path(raw_dir).rglob("events_*.json")):
        label = path.stem.removeprefix("events_")
        if competitions and label not in competitions:
            continue
        matches: dict[int, list[Event]] = defaultdict(list)
        with path.open("rb") as f:
            for raw in ijson.items(f, "item", use_float=True):
                row, tags = parse_event(raw)
                if not event_is_loadable(row):
                    continue
                matches[row[idx["match_id"]]].append(Event(
                    row[idx["event_id"]], row[idx["match_id"]], row[idx["team_id"]], row[idx["player_id"]],
                    row[idx["period"]], row[idx["event_sec"]], row[idx["type_id"]], row[idx["subtype_id"]],
                    row[idx["start_x"]], row[idx["start_y"]], frozenset(tags),
                    row[idx["end_x"]], row[idx["end_y"]],
                ))
        for match_id in sorted(matches):
            yield label, matches.pop(match_id)


_DB_EVENTS = """
    SELECT c.name, e.event_id, e.match_id, e.team_id, e.player_id, e.period, e.event_sec,
           e.type_id, e.subtype_id, e.start_x, e.start_y,
           coalesce((SELECT array_agg(t.tag_id) FROM event_tags t WHERE t.event_id = e.event_id), '{}'),
           e.end_x, e.end_y
    FROM events e
    JOIN matches m USING (match_id)
    JOIN competitions c USING (competition_id)
    ORDER BY e.match_id
"""


def iter_matches_db(conn) -> Iterator[tuple[str, list[Event]]]:
    """(competition name, events of one match), streamed with a server-side cursor."""
    current: list[Event] = []
    label = None
    with conn.transaction(), conn.cursor(name="feature_events") as cur:
        cur.itersize = 50_000
        cur.execute(_DB_EVENTS)
        for comp, *row in cur:
            ev = Event(*row[:10], frozenset(row[10]), row[11], row[12])
            if current and ev.match_id != current[0].match_id:
                yield label, current
                current = []
            label = comp
            current.append(ev)
    if current:
        yield label, current


# ---------------------------------------------------------------- match context


@dataclass
class MatchContext:
    match_id: int
    competition: str
    is_international: bool
    gameweek: int | None
    sides: dict[int, str]                                     # team_id -> 'home' | 'away'
    lineups: list[tuple[int, int, bool, int | None]] = field(default_factory=list)
    # (team_id, player_id, is_starter, red_card_minute)
    substitutions: list[tuple[int, int, int, int | None]] = field(default_factory=list)
    # (team_id, player_in, player_out, minute)
    roles: dict[int, str] = field(default_factory=dict)       # player_id -> 'GK' | 'DF' | 'MD' | 'FW'


def load_contexts_raw(raw_dir: Path) -> dict[int, MatchContext]:
    from etl.wyscout.parse import parse_match, parse_player

    raw_dir = Path(raw_dir)
    competition_type: dict[int, str | None] = {}
    comp_file = next(iter(sorted(raw_dir.rglob("competitions.json"))), None)
    if comp_file:
        for c in json.loads(comp_file.read_text(encoding="utf-8")):
            competition_type[c.get("wyId")] = c.get("type")
    roles: dict[int, str] = {}
    players_file = next(iter(sorted(raw_dir.rglob("players.json"))), None)
    if players_file:
        for d in json.loads(players_file.read_text(encoding="utf-8")):
            row = parse_player(d)
            if row[0] is not None and row[8]:
                roles[row[0]] = row[8]

    contexts: dict[int, MatchContext] = {}
    for path in sorted(raw_dir.rglob("matches_*.json")):
        label = path.stem.removeprefix("matches_")
        for d in json.loads(path.read_text(encoding="utf-8")):
            rec = parse_match(d)
            match_id, competition_id, gameweek = rec.match[0], rec.match[1], rec.match[4]
            ctx = MatchContext(
                match_id=match_id,
                competition=label,
                is_international=competition_type.get(competition_id) == "international",
                gameweek=gameweek or None,
                sides={t[1]: t[2] for t in rec.teams},
                lineups=[(l[1], l[2], l[3], l[7]) for l in rec.lineups],
                substitutions=[(s[1], s[2], s[3], s[4]) for s in rec.substitutions],
            )
            ctx.roles = {pid: roles[pid] for _, pid, _, _ in ctx.lineups if pid in roles}
            contexts[match_id] = ctx
    return contexts


def load_contexts_db(conn) -> dict[int, MatchContext]:
    contexts: dict[int, MatchContext] = {}
    for match_id, comp, comp_type, gameweek in conn.execute(
        "SELECT m.match_id, c.name, c.type, m.gameweek FROM matches m JOIN competitions c USING (competition_id)"
    ):
        contexts[match_id] = MatchContext(match_id, comp, comp_type == "international", gameweek or None, {})
    for match_id, team_id, side in conn.execute("SELECT match_id, team_id, side FROM match_teams"):
        contexts[match_id].sides[team_id] = side
    roles = dict(conn.execute("SELECT player_id, role_code FROM players WHERE role_code IS NOT NULL").fetchall())
    for match_id, team_id, player_id, is_starter, red in conn.execute(
        "SELECT match_id, team_id, player_id, is_starter, red_card_minute FROM lineups ORDER BY match_id, team_id, player_id"
    ):
        ctx = contexts[match_id]
        ctx.lineups.append((team_id, player_id, is_starter, red))
        if player_id in roles:
            ctx.roles[player_id] = roles[player_id]
    for match_id, team_id, p_in, p_out, minute in conn.execute(
        "SELECT match_id, team_id, player_in, player_out, minute FROM substitutions ORDER BY match_id, team_id, minute"
    ):
        contexts[match_id].substitutions.append((team_id, p_in, p_out, minute))
    return contexts
