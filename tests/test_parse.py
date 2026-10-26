from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from etl.wyscout import parse as p


@pytest.mark.parametrize("value, expected", [
    (None, None), ("", None), ("null", None), ("NULL", None), (" ", None),
    (0, 0), ("0", 0), ("34", 34), (12.0, 12), ("7.0", 7), ("abc", None),
])
def test_to_int(value, expected):
    assert p.to_int(value) == expected


@pytest.mark.parametrize("value, expected", [
    ("Mart\\u00edn", "Martín"),          # literal backslash sequence, as in players.json
    ("Y\\u0131lmaz", "Yılmaz"),
    ("\\u00d6zt\\u00fcrk", "Öztürk"),
    ("Martín", "Martín"),                # already decoded: untouched
    ("  K. Walker ", "K. Walker"),
    ("\\ud83d\\ude00", "\U0001F600"),    # escaped surrogate pair is re-joined
    ("null", None), ("", None), (None, None),
])
def test_fix_text(value, expected):
    assert p.fix_text(value) == expected


def test_dates():
    assert p.to_date("1994-03-12") == date(1994, 3, 12)
    assert p.to_date(None) is None and p.to_date("not a date") is None
    assert p.to_utc("2017-08-11 18:45:00") == datetime(2017, 8, 11, 18, 45, tzinfo=timezone.utc)


def _load(fixture_dir, name):
    return json.loads((fixture_dir / name).read_text(encoding="utf-8"))


def test_parse_player(fixture_dir):
    rows = {r[0]: r for r in map(p.parse_player, _load(fixture_dir, "players.json"))}
    row = dict(zip(p.PLAYER_COLS, rows[101]))
    assert row["short_name"] == "M. Yılmaz"
    assert row["first_name"] == "Martín"
    assert row["middle_name"] is None
    assert row["birth_date"] == date(1994, 3, 12)
    assert row["birth_area"] == "Turkey"
    assert row["role_code"] == "MD"
    assert row["current_national_team_id"] is None  # "null"

    unknown = dict(zip(p.PLAYER_COLS, rows[104]))
    assert unknown["height_cm"] is None and unknown["foot"] is None  # 0 and "" mean unknown
    assert rows[202][p.PLAYER_COLS.index("birth_date")] is None


def test_parse_match(fixture_dir):
    record = p.parse_match(_load(fixture_dir, "matches/matches_Sample.json")[0])
    match = dict(zip(p.MATCH_COLS, record.match))
    assert match["match_id"] == 9001 and match["winner_team_id"] == 1609
    assert match["kickoff_utc"] == datetime(2017, 8, 11, 18, 45, tzinfo=timezone.utc)

    sides = {t[1]: (t[2], t[3]) for t in record.teams}
    assert sides == {1609: ("home", 2), 1631: ("away", 1)}

    lineups = {row[2]: dict(zip(p.LINEUP_COLS, row)) for row in record.lineups}
    assert len(lineups) == 9
    assert lineups[104]["is_starter"] is False
    assert lineups[101]["goals"] == 1 and lineups[103]["goals"] == 0      # "null" goals -> 0
    assert lineups[102]["yellow_card_minute"] == 34                       # card field = minute
    assert lineups[103]["yellow_card_minute"] is None                     # "0" = no card
    assert lineups[203]["red_card_minute"] == 88

    assert record.substitutions == [(9001, 1609, 104, 103, 70)]           # other side: "null"


def test_parse_event_quirks(fixture_dir):
    raw = {e["subEventName"] or e["eventName"]: e for e in _load(fixture_dir, "events/events_Sample.json")}

    ball_out, _ = p.parse_event(raw["Ball out of the field"])
    row = dict(zip(p.EVENT_COLS, ball_out))
    assert row["player_id"] is None                     # playerId 0
    assert (row["end_x"], row["end_y"]) == (None, None)  # single position

    offside, tags = p.parse_event(raw["Offside"])
    assert offside[p.EVENT_COLS.index("subtype_id")] is None  # subEventId ""
    assert offside[p.EVENT_COLS.index("subtype_name")] is None
    assert tags == []

    goal = next(e for e in _load(fixture_dir, "events/events_Sample.json")
                if e["eventName"] == "Shot" and {"id": 101} in e["tags"])
    shot, tags = p.parse_event(goal)
    assert tags == sorted(tags) and 101 in tags
    assert p.event_is_loadable(shot)


def test_event_without_id_is_not_loadable():
    row, _ = p.parse_event({"matchId": 1, "teamId": 2, "matchPeriod": "1H", "eventSec": 1.0})
    assert not p.event_is_loadable(row)


def test_parse_tag():
    assert p.parse_tag({"Tag": "101", "Label": "goal", "Description": "Goal"}) == (101, "goal", "Goal")
    assert p.parse_tag({"tag": "x", "label": "goal"}) is None
