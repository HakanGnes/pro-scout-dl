from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from features.events import Event, MatchContext
from features.pass_networks import dataset
from features.pass_networks.graph import (
    BASELINE_FEATURES, NODE_FEATURES, build_team_graphs, minutes_played,
)

A, B = 1, 2
ACC, INACC = 1801, 1802
F = {name: i for i, name in enumerate(NODE_FEATURES)}


def ev(i, team, player, type_id=8, sub=85, sec=None, x=50, y=50, ex=None, ey=None, tags=(ACC,), period="1H"):
    return Event(i, 7, team, player, period, float(i if sec is None else sec), type_id, sub, x, y,
                 frozenset(tags), ex if ex is not None else x, ey if ey is not None else y)


def ctx(**kw) -> MatchContext:
    base = dict(
        match_id=7, competition="Test", is_international=False, gameweek=1, sides={A: "home", B: "away"},
        lineups=[(A, 11, True, None), (A, 12, True, None), (A, 13, True, None), (A, 14, False, None),
                 (B, 21, True, None), (B, 22, True, None)],
        substitutions=[], roles={11: "GK", 12: "MD", 13: "FW", 14: "FW", 21: "DF", 22: "MD"},
    )
    base.update(kw)
    return MatchContext(**base)


def graph_of(events, c=None, team=A):
    return next(g for g in build_team_graphs(events, c or ctx()) if g.team_id == team)


def node(g, player):
    return g.x[g.player_ids.index(player)]


def edges(g):
    return {(g.player_ids[s], g.player_ids[t]): round(math.expm1(a[0]), 4)  # counts are stored as float32 logs
            for (s, t), a in zip(g.edge_index.T, g.edge_attr)}


# ---------------------------------------------------------------- minutes


def test_minutes_with_substitution_and_red_card():
    c = ctx(substitutions=[(A, 14, 13, 60)], lineups=[(A, 11, True, None), (A, 12, True, 80),
                                                     (A, 13, True, None), (A, 14, False, None)])
    m = minutes_played(c, A, 90.0)
    assert m[11] == (90.0, True)
    assert m[12] == (80.0, True)         # sent off in the 80th minute
    assert m[13] == (60.0, True)         # replaced at 60'
    assert m[14] == (30.0, False)
    unused = ctx(lineups=[(A, 11, True, None), (A, 15, False, None)])
    assert 15 not in minutes_played(unused, A, 90.0)  # an unused substitute is not on the pitch


# ---------------------------------------------------------------- receivers and edges


def test_receiver_is_next_teammate():
    g = graph_of([ev(1, A, 12), ev(2, A, 13, type_id=7, sub=72)])
    assert edges(g) == {(12, 13): 1.0}
    assert node(g, 13)[F["log_received"]] == pytest.approx(math.log1p(1))


def test_receiver_skips_opponent_duel_mirror():
    g = graph_of([ev(1, A, 12), ev(2, B, 21, type_id=1, sub=12, tags=()), ev(3, A, 13, type_id=1, sub=11, tags=())])
    assert edges(g) == {(12, 13): 1.0}


def test_no_receiver_cases():
    same_player = [ev(1, A, 12), ev(2, A, 12, type_id=7, sub=70)]
    inaccurate = [ev(1, A, 12, tags=(INACC,)), ev(2, A, 13)]
    new_period = [ev(1, A, 12), ev(2, A, 13, period="2H")]
    far = [ev(1, A, 12), ev(2, B, 21), ev(3, B, 22), ev(4, B, 21), ev(5, A, 13)]
    for events in (same_player, inaccurate, new_period, far):
        assert (12, 13) not in edges(graph_of(events))


def test_pass_zones_and_progression():
    events = [
        ev(1, A, 12, x=60, y=50, ex=75, ey=50),   # into the final third, +15.75 m: progressive
        ev(2, A, 13, x=80, y=50, ex=90, ey=50),   # into the box
        ev(3, A, 12, x=90, y=50, ex=95, ey=50),   # already in the box: not an entry
        ev(4, A, 11, x=10, y=50, ex=15, ey=50),   # short pass at the back
        ev(5, A, 13),
    ]
    g = graph_of(events)
    assert math.expm1(node(g, 12)[F["log_final_third"]]) == pytest.approx(1)
    assert math.expm1(node(g, 13)[F["log_box"]]) == pytest.approx(1)
    assert math.expm1(node(g, 12)[F["log_box"]]) == pytest.approx(0)
    assert math.expm1(node(g, 12)[F["log_progressive"]]) == pytest.approx(1)
    assert g.stats["box_entries"] == 1 and g.stats["final_third_entries"] == 1
    assert node(g, 11)[F["role_gk"]] == 1 and node(g, 13)[F["role_fw"]] == 1


def test_shots_are_never_features():
    base = [ev(1, A, 12), ev(2, A, 13, type_id=7, sub=72, x=60)]
    with_shot = base + [ev(3, A, 13, type_id=10, sub=100, x=95, tags=(101,)),
                        ev(4, A, 12, type_id=3, sub=35, x=89, tags=(101,))]
    g0, g1 = graph_of(base), graph_of(with_shot)
    np.testing.assert_array_equal(g0.x, g1.x)
    assert g0.stats == g1.stats


def test_graph_shape_and_ordering():
    g = graph_of([ev(1, A, 13), ev(2, A, 12), ev(3, A, 13)])
    assert g.player_ids == sorted(g.player_ids)
    assert g.x.shape == (len(g.player_ids), len(NODE_FEATURES)) and g.x.dtype == np.float32
    assert g.edge_index.shape[0] == 2 and g.edge_index.shape[1] == g.edge_attr.shape[0]
    assert g.edge_index.max() < len(g.player_ids)
    assert set(BASELINE_FEATURES) <= set(g.stats)
    # Started players with no events still appear (minutes > 0); their position is the team average.
    assert 11 in g.player_ids


# ---------------------------------------------------------------- dataset


def _graphs(comp, weeks, intl=False):
    c = ctx(competition=comp, is_international=intl)
    out = []
    for w in weeks:
        c.gameweek, c.match_id = w, w
        out.extend(build_team_graphs([ev(1, A, 12), ev(2, A, 13)], c))
    return out


def test_time_split():
    graphs = _graphs("L", range(1, 39)) + _graphs("Cup", [1, 2], intl=True)
    counts = dataset.assign_splits(graphs)
    # 38 weeks: cutoff ceil(0.7 * 38) = 27 -> train weeks 1-22, val 23-27, test 28-38; two graphs per match.
    assert counts == {"train": 44, "val": 10, "test": 22, "test_intl": 4}
    for g in graphs:  # both teams of a match share the split
        assert g.split == next(h.split for h in graphs if h.match_id == g.match_id and h.competition == g.competition)


class _ConstantXG:
    def predict(self, shots):
        return np.full(len(shots), 0.2)


def test_targets_exclude_penalties():
    graphs = _graphs("L", [1])
    shots = pd.DataFrame({"match_id": [1, 1, 1], "team_id": [A, A, B],
                          "shot_type": ["open_play", "penalty", "free_kick"]})
    dataset.attach_targets(graphs, shots, _ConstantXG())
    target = {g.team_id: g.target for g in graphs}
    assert target == {A: pytest.approx(0.2), B: pytest.approx(0.2)}


def test_raw_and_db_graphs_agree(conn, fixture_dir):
    from etl.wyscout.load import load_wyscout
    from features.events import iter_matches_db, iter_matches_raw, load_contexts_db, load_contexts_raw

    load_wyscout(conn, fixture_dir, log=lambda _m: None)
    raw, _ = dataset.build(iter_matches_raw(fixture_dir), load_contexts_raw(fixture_dir), log=lambda _m: None)
    db, _ = dataset.build(iter_matches_db(conn), load_contexts_db(conn), log=lambda _m: None)
    key = lambda g: (g.match_id, g.team_id)  # noqa: E731
    assert len(raw) == len(db) == 4
    for a, b in zip(sorted(raw, key=key), sorted(db, key=key)):
        assert a.player_ids == b.player_ids
        np.testing.assert_allclose(a.x, b.x)
        np.testing.assert_array_equal(a.edge_index, b.edge_index)
        assert a.stats == pytest.approx(b.stats)
        assert (a.gameweek, a.is_international) == (b.gameweek, b.is_international)


def test_save_and_load(tmp_path):
    graphs = _graphs("L", [1, 2])
    dataset.save(graphs, tmp_path / "g.pkl", {"source": "test"})
    loaded, meta = dataset.load(tmp_path / "g.pkl")
    assert meta == {"source": "test"} and len(loaded) == 4
    np.testing.assert_array_equal(loaded[0].x, graphs[0].x)


def test_rich_edges_extend_the_phase1_edges():
    from features.pass_networks.graph import EDGE_FEATURES_RICH

    events = [
        ev(1, A, 12, x=60, y=20, ex=75, ey=30), ev(2, A, 13, x=70, y=40, ex=72, ey=40),    # 12 -> 13 ends in final third
        ev(3, A, 12, x=80, y=50, ex=90, ey=50), ev(4, A, 13, x=90, y=50, ex=92, ey=52),    # 12 -> 13 ends in the box
        ev(5, A, 11, x=10, y=50, ex=20, ey=50), ev(6, A, 12, type_id=7, sub=72),            # 11 -> 12 at the back
    ]
    g = graph_of(events)
    assert g.edge_attr_rich.shape == (g.edge_index.shape[1], len(EDGE_FEATURES_RICH))
    np.testing.assert_array_equal(g.edge_attr_rich[:, :2], g.edge_attr)   # Phase 1 features untouched
    rich = {(g.player_ids[s], g.player_ids[t]): r for (s, t), r in zip(g.edge_index.T, g.edge_attr_rich)}
    share_ft, share_box, start_x, start_y = rich[(12, 13)][2:]
    assert share_ft == pytest.approx(1.0)          # both passes end at x >= 66.7
    assert share_box == pytest.approx(0.5)         # one of the two ends in the box
    assert start_x == pytest.approx((0.60 + 0.80) / 2) and start_y == pytest.approx((0.20 + 0.50) / 2)
    assert list(rich[(11, 12)][2:4]) == [0.0, 0.0]
