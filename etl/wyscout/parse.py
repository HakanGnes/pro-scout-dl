"""Pure functions turning raw Wyscout JSON records into database rows.

The public dataset has a few quirks these parsers absorb:
  * Missing values appear as None, "" or the literal string "null".
  * Many names are double-escaped (the JSON holds "\\u00e9", so after decoding
    the string still contains a literal backslash sequence).
  * playerId 0 means "no player" (ball out of play, etc.).
  * Some events have a single position; subEventId can be "".
  * Card fields in lineups hold the minute of the card, "0" meaning none.

Every parser returns plain tuples in the column order declared next to it,
so the loader can feed them to COPY or executemany directly.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, NamedTuple

_NULLS = {"", "null", "none", "nan"}
_ESCAPED = re.compile(r"\\u([0-9a-fA-F]{4})")


# ---------------------------------------------------------------- scalars


def is_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in _NULLS)


def to_int(value: Any) -> int | None:
    if is_null(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    if is_null(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fix_text(value: Any) -> str | None:
    """Decode literal \\uXXXX sequences and normalise blanks to None."""
    if is_null(value):
        return None
    text = str(value)
    if "\\u" in text:
        text = _ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), text)
        try:  # re-join surrogate pairs produced by escaped astral characters
            text = text.encode("utf-16", "surrogatepass").decode("utf-16")
        except UnicodeError:
            pass
    text = text.strip()
    return text or None


def to_date(value: Any) -> date | None:
    if is_null(value):
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def to_utc(value: Any) -> datetime | None:
    """Parse Wyscout's `dateutc` ("2018-05-13 14:00:00")."""
    if is_null(value):
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def area_name(value: Any) -> str | None:
    return fix_text(value.get("name")) if isinstance(value, dict) else None


def _card_minute(value: Any) -> int | None:
    minute = to_int(value)
    return minute if minute else None  # 0 / null = no card


# ---------------------------------------------------------------- reference data

TAG_COLS = ("tag_id", "label", "description")
COMPETITION_COLS = ("competition_id", "name", "area_name", "format", "type")
TEAM_COLS = ("team_id", "name", "official_name", "city", "area_name", "type")
PLAYER_COLS = (
    "player_id", "short_name", "first_name", "middle_name", "last_name",
    "birth_date", "birth_area", "passport_area", "role_code", "role_name",
    "height_cm", "weight_kg", "foot", "current_team_id", "current_national_team_id",
)


def parse_tag(row: dict[str, Any]) -> tuple | None:
    """One row of tags2name.csv (columns Tag, Label, Description; any case)."""
    row = {str(k).strip().lower(): v for k, v in row.items()}
    tag_id = to_int(row.get("tag"))
    label = fix_text(row.get("label"))
    if tag_id is None or label is None:
        return None
    return (tag_id, label, fix_text(row.get("description")))


def parse_competition(d: dict[str, Any]) -> tuple:
    return (
        to_int(d.get("wyId")),
        fix_text(d.get("name")) or f"Competition {d.get('wyId')}",
        area_name(d.get("area")),
        fix_text(d.get("format")),
        fix_text(d.get("type")),
    )


def parse_team(d: dict[str, Any]) -> tuple:
    return (
        to_int(d.get("wyId")),
        fix_text(d.get("name")) or f"Team {d.get('wyId')}",
        fix_text(d.get("officialName")),
        fix_text(d.get("city")),
        area_name(d.get("area")),
        fix_text(d.get("type")),
    )


def parse_player(d: dict[str, Any]) -> tuple:
    role = d.get("role") if isinstance(d.get("role"), dict) else {}
    height, weight = to_int(d.get("height")), to_int(d.get("weight"))
    return (
        to_int(d.get("wyId")),
        fix_text(d.get("shortName")),
        fix_text(d.get("firstName")),
        fix_text(d.get("middleName")),
        fix_text(d.get("lastName")),
        to_date(d.get("birthDate")),
        area_name(d.get("birthArea")),
        area_name(d.get("passportArea")),
        fix_text(role.get("code2")),
        fix_text(role.get("name")),
        height or None,  # 0 = unknown
        weight or None,
        fix_text(d.get("foot")),
        to_int(d.get("currentTeamId")) or None,
        to_int(d.get("currentNationalTeamId")) or None,
    )


# ---------------------------------------------------------------- matches

MATCH_COLS = (
    "match_id", "competition_id", "season_id", "round_id", "gameweek", "kickoff_utc",
    "status", "duration", "winner_team_id", "venue", "label",
)
MATCH_TEAM_COLS = ("match_id", "team_id", "side", "score", "score_ht", "score_et", "score_p", "coach_id")
LINEUP_COLS = (
    "match_id", "team_id", "player_id", "is_starter", "goals", "own_goals",
    "yellow_card_minute", "red_card_minute",
)
SUBSTITUTION_COLS = ("match_id", "team_id", "player_in", "player_out", "minute")


class MatchRecord(NamedTuple):
    match: tuple
    teams: list[tuple]
    lineups: list[tuple]
    substitutions: list[tuple]


def parse_match(d: dict[str, Any]) -> MatchRecord:
    match_id = to_int(d.get("wyId"))
    match = (
        match_id,
        to_int(d.get("competitionId")),
        to_int(d.get("seasonId")),
        to_int(d.get("roundId")),
        to_int(d.get("gameweek")),
        to_utc(d.get("dateutc")),
        fix_text(d.get("status")),
        fix_text(d.get("duration")),
        to_int(d.get("winner")) or None,  # 0 = draw
        fix_text(d.get("venue")),
        fix_text(d.get("label")),
    )

    teams: list[tuple] = []
    lineups: list[tuple] = []
    subs: list[tuple] = []
    teams_data = d.get("teamsData") if isinstance(d.get("teamsData"), dict) else {}
    for key, td in teams_data.items():
        if not isinstance(td, dict):
            continue
        team_id = to_int(td.get("teamId")) or to_int(key)
        side = fix_text(td.get("side"))
        teams.append((
            match_id, team_id, side if side in ("home", "away") else None,
            to_int(td.get("score")), to_int(td.get("scoreHT")),
            to_int(td.get("scoreET")), to_int(td.get("scoreP")),
            to_int(td.get("coachId")) or None,
        ))

        formation = td.get("formation") if isinstance(td.get("formation"), dict) else {}
        seen: set[int] = set()
        for is_starter, group in ((True, formation.get("lineup")), (False, formation.get("bench"))):
            for p in group if isinstance(group, list) else []:
                player_id = to_int(p.get("playerId")) if isinstance(p, dict) else None
                if not player_id or player_id in seen:
                    continue
                seen.add(player_id)
                lineups.append((
                    match_id, team_id, player_id, is_starter,
                    to_int(p.get("goals")) or 0,  # "null" = no goals
                    to_int(p.get("ownGoals")) or 0,
                    _card_minute(p.get("yellowCards")),
                    _card_minute(p.get("redCards")),
                ))

        raw_subs = formation.get("substitutions")
        for s in raw_subs if isinstance(raw_subs, list) else []:  # may be the string "null"
            if not isinstance(s, dict):
                continue
            p_in, p_out = to_int(s.get("playerIn")), to_int(s.get("playerOut"))
            if p_in and p_out:
                subs.append((match_id, team_id, p_in, p_out, to_int(s.get("minute"))))

    return MatchRecord(match, teams, lineups, subs)


# ---------------------------------------------------------------- events

EVENT_COLS = (
    "event_id", "match_id", "team_id", "player_id", "period", "event_sec",
    "type_id", "type_name", "subtype_id", "subtype_name",
    "start_x", "start_y", "end_x", "end_y",
)


def parse_event(d: dict[str, Any]) -> tuple[tuple, list[int]]:
    """Return (event row, sorted unique tag ids)."""
    positions = d.get("positions") if isinstance(d.get("positions"), list) else []
    start = positions[0] if len(positions) > 0 and isinstance(positions[0], dict) else {}
    end = positions[1] if len(positions) > 1 and isinstance(positions[1], dict) else {}

    tags: set[int] = set()
    for t in d.get("tags") if isinstance(d.get("tags"), list) else []:
        tag_id = to_int(t.get("id")) if isinstance(t, dict) else None
        if tag_id is not None:
            tags.add(tag_id)

    row = (
        to_int(d.get("id")),
        to_int(d.get("matchId")),
        to_int(d.get("teamId")),
        to_int(d.get("playerId")) or None,  # 0 = no player
        fix_text(d.get("matchPeriod")),
        to_float(d.get("eventSec")),
        to_int(d.get("eventId")),
        fix_text(d.get("eventName")),
        to_int(d.get("subEventId")),
        fix_text(d.get("subEventName")),
        to_int(start.get("x")),
        to_int(start.get("y")),
        to_int(end.get("x")),
        to_int(end.get("y")),
    )
    return row, sorted(tags)


def event_is_loadable(row: tuple) -> bool:
    """Rows missing a NOT NULL column (id, match, team, period, time) are skipped."""
    return all(row[i] is not None for i in (0, 1, 2, 4, 5))
