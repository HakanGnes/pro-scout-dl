from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import pytest

from features.xg import shots as S
from features.xg.shots import Event, extract_shots, shot_geometry

A, B = 1, 2  # team ids


def ev(i, team, type_id, subtype_id=None, sec=None, x=50, y=50, tags=(), player=None, period="1H"):
    return Event(i, 99, team, player if player is not None else team * 10, period,
                 float(i if sec is None else sec), type_id, subtype_id, x, y, frozenset(tags))


def shot_of(events, event_id):
    return next(s for s in extract_shots(events) if s.event_id == event_id)


# ---------------------------------------------------------------- geometry


def test_geometry():
    d, a = shot_geometry(100 - 11 / 1.05, 50)  # penalty spot
    assert d == pytest.approx(11.0, abs=1e-6)
    assert math.degrees(a) == pytest.approx(36.8, abs=0.1)
    assert shot_geometry(100, 50)[1] == pytest.approx(math.pi)  # on the goal line
    # Further out or wider = smaller angle.
    assert shot_geometry(80, 50)[1] < shot_geometry(90, 50)[1]
    assert shot_geometry(90, 10)[1] < shot_geometry(90, 50)[1]


# ---------------------------------------------------------------- context rules


def test_cross_then_header():
    events = [ev(1, A, S.PASS, S.CROSS, x=90, y=5), ev(2, A, S.SHOT, 100, x=94, y=48, tags=[S.TAG_HEAD_BODY], player=11)]
    s = shot_of(events, 2)
    assert (s.assist_type, s.is_header, s.same_player_prev) == ("cross", True, False)
    assert s.seconds_since_prev == pytest.approx(1.0)


def test_through_ball_by_tag_or_smart_pass():
    assert shot_of([ev(1, A, S.PASS, 85, tags=[S.TAG_THROUGH]), ev(2, A, S.SHOT, 100, x=88)], 2).assist_type == "through_ball"
    assert shot_of([ev(1, A, S.PASS, S.SMART_PASS), ev(2, A, S.SHOT, 100, x=88)], 2).assist_type == "through_ball"


def test_mirror_duel_is_skipped():
    """Wyscout logs a duel for both teams; the opponent's half must not turn a dribble into a loose ball."""
    events = [ev(1, A, S.DUEL, S.ATTACKING_DUEL, player=10), ev(2, B, S.DUEL, 12, sec=1.0),
              ev(3, A, S.SHOT, 100, x=85, player=10)]
    s = shot_of(events, 3)
    assert (s.assist_type, s.same_player_prev) == ("dribble", True)


def test_opponent_touch_is_loose_ball():
    events = [ev(1, A, S.PASS, S.CROSS), ev(2, B, S.OTHERS, 71), ev(3, A, S.SHOT, 100, x=82)]
    assert shot_of(events, 3).assist_type == "loose_ball"


def test_rebound_after_save():
    events = [ev(1, A, S.SHOT, 100, sec=10, x=88), ev(2, B, S.SAVE, 91, sec=10.5, x=1),
              ev(3, A, S.SHOT, 100, sec=12, x=95)]
    assert shot_of(events, 3).assist_type == "rebound"
    late = [ev(1, A, S.SHOT, 100, sec=10, x=88), ev(2, B, S.SAVE, 91, sec=10.5),
            ev(3, A, S.PASS, 85, sec=20), ev(4, A, S.SHOT, 100, sec=21, x=95)]
    assert shot_of(late, 4).assist_type == "pass"  # outside the rebound window


def test_set_piece_window():
    corner = [ev(1, A, S.FREE_KICK, S.CORNER, sec=100, x=100, y=0), ev(2, A, S.SHOT, 100, sec=103, x=95)]
    assert shot_of(corner, 2).set_piece == "corner"
    assert shot_of(corner, 2).assist_type == "cross"  # a corner is delivered like a cross
    later = [ev(1, A, S.FREE_KICK, S.CORNER, sec=100), ev(2, A, S.PASS, 85, sec=110), ev(3, A, S.SHOT, 100, sec=120, x=80)]
    assert shot_of(later, 3).set_piece == "none"


def test_penalty_and_free_kick_shots():
    events = [ev(1, A, S.FREE_KICK, S.PENALTY, x=89, tags=[S.TAG_GOAL]), ev(2, A, S.FREE_KICK, S.FREE_KICK_SHOT, x=75)]
    pen, fk = extract_shots(events)
    assert (pen.shot_type, pen.assist_type, pen.is_goal) == ("penalty", "set_piece", True)
    assert fk.shot_type == "free_kick"


def test_mirrored_coordinates_are_flipped():
    s = shot_of([ev(1, A, S.SHOT, 100, x=3, y=60)], 1)
    assert s.coords_mirrored and (s.x, s.y) == (97.0, 40.0)
    assert s.distance_m == pytest.approx(math.hypot(3 * 1.05, 10 * 0.68), abs=1e-3)  # 7.5 m, not 98 m
    normal = shot_of([ev(1, A, S.SHOT, 100, x=45, y=50)], 1)  # a genuine long shot
    assert not normal.coords_mirrored


def test_score_diff_counts_goals_and_own_goals():
    events = [
        ev(1, A, S.SHOT, 100, x=90, tags=[S.TAG_GOAL]),          # A 1-0
        ev(2, B, S.SHOT, 100, x=90),                             # B shoots at 0-1
        ev(3, B, S.OTHERS, 71, tags=[S.TAG_OWN_GOAL]),           # own goal by B: A 2-0
        ev(4, A, S.SHOT, 100, x=90),                             # A shoots at 2-0
        ev(5, B, S.SAVE, 91, tags=[S.TAG_GOAL]),                 # keeper event tagged goal: not a goal
        ev(6, B, S.SHOT, 100, x=90),
    ]
    diffs = {s.event_id: s.score_diff for s in extract_shots(events)}
    assert diffs == {1: 0, 2: -1, 4: 2, 6: -2}


def test_shootout_excluded_and_order_independent():
    events = [ev(1, A, S.SHOT, 100, x=90, period="2H"), ev(2, A, S.FREE_KICK, S.PENALTY, period="P")]
    assert [s.event_id for s in extract_shots(list(reversed(events)))] == [1]


# ---------------------------------------------------------------- sources


def test_raw_source_on_fixture(fixture_dir):
    records = S.collect_shots(S.iter_matches_raw(fixture_dir))
    # Match 9001: two open-play goals + a penalty; 9002: one missed shot. Goalkeeper events are not shots.
    assert len(records) == 4
    assert sum(r["is_goal"] for r in records) == 3
    assert all(r["competition"] == "Sample" for r in records)
    pen = next(r for r in records if r["shot_type"] == "penalty")
    assert pen["is_goal"] and pen["score_diff"] == 0      # taken at 1-1
    late = next(r for r in records if r["match_id"] == 9002)
    assert late["score_diff"] == 1                         # after the opponent's own goal


def test_db_source_matches_raw(conn, fixture_dir):
    from etl.wyscout.load import load_wyscout

    load_wyscout(conn, fixture_dir, log=lambda _m: None)
    key = lambda r: {k: v for k, v in r.items() if k != "competition"}  # noqa: E731
    raw = sorted(map(key, S.collect_shots(S.iter_matches_raw(fixture_dir))), key=lambda r: r["event_id"])
    db = sorted(map(key, S.collect_shots(S.iter_matches_db(conn))), key=lambda r: r["event_id"])
    assert db == raw


# ---------------------------------------------------------------- model


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(2, 35, n)
    df = pd.DataFrame({
        "event_id": np.arange(n), "match_id": rng.integers(0, 200, n), "competition": rng.choice(list("ABC"), n),
        "shot_type": rng.choice(["open_play"] * 18 + ["free_kick", "penalty"], n),
        "distance_m": d, "angle_rad": np.clip(2 / d, 0, 3), "x": 100 - d, "y": rng.uniform(30, 70, n),
        "seconds_since_prev": rng.uniform(0, 5, n), "minute": rng.uniform(0, 95, n), "score_diff": rng.integers(-2, 3, n),
        "is_header": rng.random(n) < 0.15, "is_counter": rng.random(n) < 0.05, "same_player_prev": rng.random(n) < 0.3,
        "assist_type": rng.choice(["pass", "cross", "rebound", "dribble"], n), "set_piece": "none",
    })
    df["is_goal"] = rng.random(n) < 1 / (1 + np.exp(0.18 * d - 0.5))
    df.loc[df["shot_type"] == "penalty", "is_goal"] = rng.random((df["shot_type"] == "penalty").sum()) < 0.75
    return df


def test_evaluate_and_final_model_roundtrip():
    import joblib

    from features.xg.model import MODEL_NAMES, bootstrap_log_loss_diff, evaluate, fit_final

    df = _synthetic()
    ev = evaluate(df)
    assert set(ev.pooled.index) == set(MODEL_NAMES)
    assert ev.best in ("lr_geometry", "lr_full", "lgbm")
    # Every real model beats the constant rate on data with a strong distance effect.
    assert (ev.pooled.drop("baseline")["log_loss"] < ev.pooled.loc["baseline", "log_loss"]).all()
    assert ev.oof.filter(like="p_").notna().all().all()

    mean, lo, hi = bootstrap_log_loss_diff(ev, df["match_id"], "lr_geometry", "baseline", n_boot=200)
    assert lo <= mean <= hi < 0

    final = fit_final(df, "lgbm")
    buf = io.BytesIO()
    joblib.dump(final, buf)
    buf.seek(0)
    loaded = joblib.load(buf)
    xg = loaded.predict(df)
    assert ((xg > 0) & (xg < 1)).all()
    pens = (df["shot_type"] == "penalty").to_numpy()
    assert np.allclose(xg[pens], final.penalty_xg)
    near, far = df.assign(distance_m=5.0, angle_rad=0.9, x=95.0), df.assign(distance_m=30.0, angle_rad=0.15, x=71.0)
    assert loaded.predict(near)[~pens].mean() > loaded.predict(far)[~pens].mean()


def test_report_is_written(tmp_path):
    from features.xg.model import evaluate, fit_final
    from features.xg.report import write_report

    df = _synthetic(2500)
    ev = evaluate(df, models=("baseline", "lr_geometry", "lr_full", "lgbm"))
    final = fit_final(df, "lgbm")
    path = write_report(ev, final, df, tmp_path, {"lgbm - lr_full": (-0.001, -0.002, 0.0)})
    text = path.read_text(encoding="utf-8")
    assert "Leave one competition out" in text and "lgbm - lr_full" in text
    for png in ("calibration.png", "xg_map.png", "feature_importance.png"):
        assert (tmp_path / png).stat().st_size > 10_000


# ---------------------------------------------------------------- database


def test_store_and_team_view(conn, fixture_dir):
    from etl.wyscout.load import load_wyscout
    from features.xg.store import write_shots

    load_wyscout(conn, fixture_dir, log=lambda _m: None)
    shots = pd.DataFrame(S.collect_shots(S.iter_matches_db(conn)))
    shots["xg"] = 0.25
    shots["xg_model"] = "test"
    assert write_shots(conn, shots) == len(shots)
    assert write_shots(conn, shots) == len(shots)  # replaces, does not append
    assert conn.execute("SELECT count(*) FROM shots").fetchone()[0] == len(shots)

    # A shot without a player (Wyscout playerId 0) turns player_id into a float column in pandas.
    shots.loc[0, "player_id"] = None
    assert shots["player_id"].dtype == float
    assert write_shots(conn, shots) == len(shots)
    stored = conn.execute("SELECT player_id FROM shots WHERE event_id = %s", (int(shots.loc[0, "event_id"]),)).fetchone()
    assert stored == (None,)
    other = conn.execute("SELECT pg_typeof(player_id)::text, player_id FROM shots WHERE player_id IS NOT NULL LIMIT 1").fetchone()
    assert other[0] == "integer" and isinstance(other[1], int)

    rows = conn.execute("SELECT match_id, team_id, goals, shots, xg, npxg FROM team_match_xg ORDER BY 1, 2").fetchall()
    assert len(rows) == 4  # every team of every match, even without shots
    home = next(r for r in rows if r[0] == 9001 and r[1] == 1609)
    assert home[2:] == (2, 2, pytest.approx(0.5), pytest.approx(0.25))  # 1 open-play shot + 1 penalty
