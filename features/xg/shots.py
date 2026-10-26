"""Shot extraction and pre-shot features, independent of where events come from.

`extract_shots` is a pure function over one match's events in time order. The
events come from `features.events` (raw Wyscout JSON or PostgreSQL), so the same
logic runs on both sources and the tests can check that they agree.

Only information available *before* the ball is struck is used. Wyscout tags
that describe the outcome are excluded on purpose:
  * 201 "opportunity"   — present on 909 of 914 Premier League goals (leaks the label)
  * 1801/1802 accuracy  — on target or not
  * 1201–1223           — where the shot ended up
  * 2101 "blocked"      — what happened after the strike
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from features.events import Event, iter_matches_db, iter_matches_raw, sort_key  # noqa: F401 (re-exported)

# Pitch size used to convert Wyscout's 0–100 percentages to metres.
PITCH_LENGTH_M, PITCH_WIDTH_M, GOAL_WIDTH_M = 105.0, 68.0, 7.32

PERIOD_START_MIN = {"1H": 0, "2H": 45, "E1": 90, "E2": 105}

# Wyscout event / sub-event ids (eventid2name.csv).
DUEL, FREE_KICK, OTHERS, PASS, SAVE, SHOT = 1, 3, 7, 8, 9, 10
AIR_DUEL, ATTACKING_DUEL = 10, 11
CORNER, FREE_KICK_PLAIN, FREE_KICK_CROSS, FREE_KICK_SHOT, PENALTY, THROW_IN = 30, 31, 32, 33, 35, 36
ACCELERATION = 70
CROSS, SMART_PASS = 80, 86

# Wyscout tag ids (tags2name.csv).
TAG_GOAL, TAG_OWN_GOAL = 101, 102
TAG_HEAD_BODY = 403
TAG_THROUGH = 901
TAG_COUNTER = 1901

# Wyscout occasionally records a team's possession mirrored (a corner at x=1, the keeper
# saving at x=0 right after). A shot from more than ~73 m out is such a record, not a real
# shot: its coordinates are flipped back. 25 of 45,869 shots in the 2017/18 data.
MIRROR_X_BELOW = 30

# How far back to look for the shooting team's last action and for set pieces.
LOOKBACK_EVENTS = 6
REBOUND_WINDOW_S = 5.0
SET_PIECE_WINDOW_S = 15.0

SET_PIECES = {CORNER: "corner", FREE_KICK_PLAIN: "free_kick", FREE_KICK_CROSS: "free_kick", THROW_IN: "throw_in"}


@dataclass
class Shot:
    event_id: int
    match_id: int
    team_id: int
    player_id: int | None
    period: str
    minute: float
    shot_type: str            # 'open_play' | 'free_kick' | 'penalty'
    x: float                  # 0–100, clipped
    y: float
    distance_m: float         # to the centre of the goal
    angle_rad: float          # angle subtended by the goal mouth
    is_header: bool
    is_counter: bool
    assist_type: str          # see _assist_type
    set_piece: str            # 'none' | 'corner' | 'free_kick' | 'throw_in'
    same_player_prev: bool    # shooter also made the team's previous action (carry, dribble)
    seconds_since_prev: float
    score_diff: int           # shooter's goal difference before the shot
    coords_mirrored: bool     # coordinates flipped back, see MIRROR_X_BELOW
    is_goal: bool


SHOT_COLUMNS = tuple(Shot.__dataclass_fields__)


# ---------------------------------------------------------------- geometry


def _clip(v: float | None) -> float:
    return min(100.0, max(0.0, float(v if v is not None else 0.0)))


def shot_geometry(x: float, y: float) -> tuple[float, float]:
    """Distance (m) to the goal centre and the angle (rad) the goal mouth subtends."""
    dx = (100.0 - x) * PITCH_LENGTH_M / 100.0
    dy = abs(50.0 - y) * PITCH_WIDTH_M / 100.0
    distance = math.hypot(dx, dy)
    half = GOAL_WIDTH_M / 2
    angle = math.atan2(GOAL_WIDTH_M * dx, dx * dx + dy * dy - half * half)
    if angle < 0:  # standing inside the goal mouth's circle
        angle += math.pi
    return distance, angle


# ---------------------------------------------------------------- context


def _assist_type(shot: Event, window: list[Event]) -> tuple[str, Event | None]:
    """Classify how the chance was created from the events just before the shot.

    `window` holds up to LOOKBACK_EVENTS earlier events of the same period, oldest first.
    Wyscout logs every duel twice (once per team), so the opponent's mirror duel is skipped.
    """
    recent = [e for e in window if shot.event_sec - e.event_sec <= REBOUND_WINDOW_S]
    # A save or a shot by the same team moments earlier: second ball after a shot.
    for e in reversed(recent[-3:]):
        if (e.type_id == SAVE and e.team_id != shot.team_id) or (e.type_id == SHOT and e.team_id == shot.team_id):
            return "rebound", next((o for o in reversed(window) if o.team_id == shot.team_id), None)

    # The latest event that is not the opponent's mirror of a duel.
    last = next((e for e in reversed(window) if not (e.team_id != shot.team_id and e.type_id == DUEL)), None)
    own = next((e for e in reversed(window) if e.team_id == shot.team_id), None)
    if last is not None and last.team_id != shot.team_id:
        return "loose_ball", own  # clearance, touch, interception just before the shot
    if own is None:
        return "other", None

    if own.type_id == PASS:
        if own.subtype_id == CROSS:
            return "cross", own
        if own.subtype_id == SMART_PASS or TAG_THROUGH in own.tags:
            return "through_ball", own
        return "pass", own
    if own.type_id == FREE_KICK:
        if own.subtype_id in (CORNER, FREE_KICK_CROSS):
            return "cross", own
        return "pass", own
    if own.type_id == DUEL:
        return ("aerial" if own.subtype_id == AIR_DUEL else "dribble"), own
    if own.type_id == OTHERS and own.subtype_id == ACCELERATION:
        return "dribble", own
    return "other", own


def _set_piece(shot: Event, window: list[Event]) -> str:
    for e in reversed(window):
        if shot.event_sec - e.event_sec > SET_PIECE_WINDOW_S:
            break
        if e.team_id == shot.team_id and e.type_id == FREE_KICK and e.subtype_id in SET_PIECES:
            return SET_PIECES[e.subtype_id]
    return "none"


def _is_goal(e: Event) -> bool:
    return TAG_GOAL in e.tags and (e.type_id == SHOT or (e.type_id == FREE_KICK and e.subtype_id in (FREE_KICK_SHOT, PENALTY)))


# ---------------------------------------------------------------- extraction


def extract_shots(events: list[Event]) -> list[Shot]:
    """All shots of one match (shoot-outs excluded), with pre-shot features."""
    events = sorted(events, key=sort_key)
    score: dict[int, int] = defaultdict(int)
    teams = {e.team_id for e in events}
    shots: list[Shot] = []

    for i, e in enumerate(events):
        if e.period == "P":
            break
        is_shot = e.type_id == SHOT
        is_fk_shot = e.type_id == FREE_KICK and e.subtype_id in (FREE_KICK_SHOT, PENALTY)
        if is_shot or is_fk_shot:
            window = [w for w in events[max(0, i - LOOKBACK_EVENTS):i] if w.period == e.period]
            assist, own = _assist_type(e, window) if is_shot else ("set_piece", None)
            x, y = _clip(e.x), _clip(e.y)
            mirrored = x < MIRROR_X_BELOW and not (is_fk_shot and e.subtype_id == PENALTY)
            if mirrored:
                x, y = 100.0 - x, 100.0 - y
            distance, angle = shot_geometry(x, y)
            opponent = next((t for t in teams if t != e.team_id), None)
            shots.append(Shot(
                event_id=e.event_id,
                match_id=e.match_id,
                team_id=e.team_id,
                player_id=e.player_id,
                period=e.period,
                minute=round(PERIOD_START_MIN.get(e.period, 0) + e.event_sec / 60.0, 2),
                shot_type="open_play" if is_shot else ("penalty" if e.subtype_id == PENALTY else "free_kick"),
                x=x,
                y=y,
                distance_m=round(distance, 3),
                angle_rad=round(angle, 5),
                is_header=TAG_HEAD_BODY in e.tags,
                is_counter=TAG_COUNTER in e.tags,
                assist_type=assist,
                set_piece=_set_piece(e, window) if is_shot else "none",
                same_player_prev=bool(own and own.player_id and own.player_id == e.player_id),
                seconds_since_prev=round(min(30.0, max(0.0, e.event_sec - own.event_sec)) if own else 30.0, 3),
                score_diff=max(-3, min(3, score[e.team_id] - (score[opponent] if opponent else 0))),
                coords_mirrored=mirrored,
                is_goal=TAG_GOAL in e.tags,
            ))
        # Update the score after the event (own goals count for the opponent).
        if _is_goal(e):
            score[e.team_id] += 1
        elif TAG_OWN_GOAL in e.tags:
            for t in teams - {e.team_id}:
                score[t] += 1
    return shots


def shots_to_records(shots: list[Shot]) -> list[dict]:
    return [asdict(s) for s in shots]


def collect_shots(matches: Iterator[tuple[str, list[Event]]]) -> list[dict]:
    """Shot records for every match, each tagged with its competition."""
    records: list[dict] = []
    for label, events in matches:
        for rec in shots_to_records(extract_shots(events)):
            rec["competition"] = label
            records.append(rec)
    return records
