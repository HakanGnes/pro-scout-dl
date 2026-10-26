from __future__ import annotations

import math

import numpy as np
import pytest

from features.pass_networks.graph import BASELINE_FEATURES, EDGE_FEATURES, NODE_FEATURES, TeamGraph
from sim.bundle import Bundle, build_bundle
from sim.profiles import COUNT_FEATURES, build_profiles
from sim.swap import Simulator, candidates_by_role, swap_graph

F = {name: i for i, name in enumerate(NODE_FEATURES)}
ROLE = {"GK": "role_gk", "DF": "role_df", "MD": "role_md", "FW": "role_fw"}


def node(role="FW", share=1.0, starter=True, box=2.0, passes=30.0, acc=0.8, x=0.6, y=0.5):
    v = np.zeros(len(NODE_FEATURES), dtype=np.float32)
    v[F[ROLE[role]]] = 1
    v[F["minutes_share"]], v[F["is_starter"]] = share, float(starter)
    v[F["log_box"]], v[F["log_passes"]] = math.log1p(box), math.log1p(passes)
    for feat in ("log_received", "log_final_third", "log_progressive", "log_touches"):
        v[F[feat]] = math.log1p(10.0)
    v[F["pass_accuracy"]], v[F["median_x"]], v[F["median_y"]] = acc, x, y
    return v


def graph(match, team, players: dict[int, np.ndarray], week=1, intl=False):
    ids = sorted(players)
    x = np.stack([players[p] for p in ids])
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_attr = np.ones((2, len(EDGE_FEATURES)), dtype=np.float32)
    box = sum(math.expm1(float(r[F["log_box"]])) for r in x)
    stats = {f: 1.0 for f in BASELINE_FEATURES} | {"box_entries": box}
    return TeamGraph(match, team, "L", week, intl, "home", ids, x, edge_index, edge_attr, stats, 0.1 * box, "train")


class LinearEnsemble:
    """Stand-in for the networks: npxG = 0.1 × box entries (two 'models' that differ by 10%)."""
    nets = [("a", None), ("b", None)]

    def predict(self, graphs):
        box = np.array([sum(math.expm1(float(r[F["log_box"]])) for r in g.x) for g in graphs])
        return np.vstack([0.1 * box, 0.11 * box])


def league():
    """Team 1: forward 11 (2 box entries a match) + midfielder 12. Team 2: forward 21 (6 a match), keeper 22."""
    graphs = []
    for m in range(1, 11):
        graphs.append(graph(m, 1, {11: node("FW", box=2), 12: node("MD", box=1)}, week=m))
        graphs.append(graph(m, 2, {21: node("FW", box=6), 22: node("GK", box=0)}, week=m))
    return graphs


# ---------------------------------------------------------------- profiles


def test_profile_rates_are_per_full_match():
    graphs = [graph(1, 1, {7: node(box=2, share=1.0)}), graph(2, 1, {7: node(box=1, share=0.5)}),
              graph(3, 1, {7: node(box=9, share=0.05)})]   # a 4-minute cameo is ignored
    p = build_profiles(graphs)[7]
    assert p.matches == 2 and p.match_share == pytest.approx(1.5)
    assert p.rates["log_box"] == pytest.approx(3 / 1.5)      # (2 + 1) entries in 1.5 matches
    assert p.minutes == pytest.approx(135)
    assert p.role == "FW"


def test_profile_accuracy_is_pooled_over_passes():
    graphs = [graph(1, 1, {7: node(passes=40, acc=0.9)}), graph(2, 1, {7: node(passes=10, acc=0.5)})]
    assert build_profiles(graphs)[7].pass_accuracy == pytest.approx((36 + 5) / 50)


def test_international_matches_are_left_out():
    graphs = [graph(1, 1, {7: node(box=2)}), graph(2, 99, {7: node(box=20)}, intl=True)]
    assert build_profiles(graphs)[7].rates["log_box"] == pytest.approx(2)


def test_node_features_scale_with_minutes():
    p = build_profiles([graph(1, 1, {7: node(box=4)})])[7]
    half = p.node_features(0.5, False)
    assert math.expm1(half[F["log_box"]]) == pytest.approx(2)
    assert (half[F["minutes_share"]], half[F["is_starter"]], half[F["role_fw"]]) == (0.5, 0.0, 1.0)


# ---------------------------------------------------------------- swaps


def test_swap_graph_replaces_only_that_node():
    graphs = league()
    profiles = build_profiles(graphs)
    g = graphs[0]
    s = swap_graph(g, 11, profiles[21])
    assert s.player_ids == [21, 12] and g.player_ids == [11, 12]  # same slot; the original is untouched
    np.testing.assert_array_equal(s.x[1], g.x[1])
    assert math.expm1(s.x[0, F["log_box"]]) == pytest.approx(6)
    np.testing.assert_array_equal(s.edge_index, g.edge_index)


def test_swap_delta_and_uncertainty():
    graphs = league()
    sim = Simulator(graphs, build_profiles(graphs), LinearEnsemble())
    r = sim.swap(1, 11, 21)                      # 2 -> 6 box entries a match
    assert r.matches == 10 and r.skipped == 0
    assert r.delta == pytest.approx((0.4 + 0.44) / 2)
    assert r.se_matches == pytest.approx(0, abs=1e-9)   # identical matches
    assert r.sd_models > 0                       # the two 'models' disagree by 10%
    assert r.delta_per_90 == pytest.approx(r.delta)


def test_self_swap_is_neutral_for_a_regular_starter():
    graphs = league()
    sim = Simulator(graphs, build_profiles(graphs), LinearEnsemble())
    assert sim.swap(1, 11, 11).delta == pytest.approx(0, abs=1e-6)


def test_candidate_already_in_the_team_is_skipped():
    graphs = league()
    sim = Simulator(graphs, build_profiles(graphs), LinearEnsemble())
    r = sim.swap(1, 11, 12)                      # 12 already plays next to 11 in every match
    assert r.matches == 0 and r.skipped == 10


def test_rank_orders_candidates_and_filters_by_role():
    graphs = league()
    profiles = build_profiles(graphs)
    sim = Simulator(graphs, profiles, LinearEnsemble())
    assert candidates_by_role(profiles, "FW", 0, exclude_team=1) == [21]
    table, results = sim.rank(1, 11, [22, 21])
    assert list(table["player_in"]) == [21, 22]
    assert results[22].delta < 0 < results[21].delta
    with pytest.raises(ValueError):
        sim.swap(2, 11, 21)                      # 11 never played for team 2


def test_team_players_lists_season_minutes():
    sim = Simulator(league(), {}, LinearEnsemble())
    t = sim.team_players(1)
    assert set(t["player_id"]) == {11, 12} and list(t["matches"]) == [10, 10]


# ---------------------------------------------------------------- bundle


def test_bundle_name_search_ignores_case_and_accents():
    b = Bundle({}, {1: {"name": "D. Öztürk"}, 2: {"name": "K. Yılmaz"}, 3: {"name": "M. Öztürk Jr"}}, {})
    assert b.find("d. ozturk") == [1]             # an exact match wins over partial ones
    assert b.find("ozturk") == [1, 3]
    assert b.find("YILMAZ") == [2]
    assert b.player_name(9) == "player 9"


def test_bundle_records_league_competition():
    graphs = league()
    b = build_bundle(graphs, {}, {1: {"name": "One"}, 2: {"name": "Two"}})
    assert b.teams[1]["competition"] == "L" and set(b.profiles) == {11, 12, 21, 22}
    assert set(COUNT_FEATURES) == set(b.profiles[11].rates)


# ---------------------------------------------------------------- end to end with real networks


def test_cli_with_trained_networks(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from features.pass_networks import dataset
    from models.gnn.__main__ import main as gnn_main
    from sim import bundle as B
    from sim.__main__ import main as sim_main

    rng = np.random.default_rng(0)
    graphs = []
    for m in range(76):
        for team, squad in ((1, range(100, 115)), (2, range(200, 215))):
            players = {p: node(["GK", "DF", "MD", "FW"][min(3, (p % 100) // 4)], box=float(rng.integers(0, 4)))
                       for p in squad}
            g = graph(m, team, players, week=m % 38 + 1)
            g.edge_index = np.array([[k, (k + 1) % 15] for k in range(15)], dtype=np.int64).T
            g.edge_attr = np.ones((15, len(EDGE_FEATURES)), dtype=np.float32)
            g.match_id = m * 10 + (0 if m < 38 else 5)
            graphs.append(g)
    dataset.assign_splits(graphs)
    ds, model, bundle_path = tmp_path / "g.pkl", tmp_path / "m.pt", tmp_path / "b.pkl"
    dataset.save(graphs, ds, {"xg_model": "test"})
    assert gnn_main(["evaluate", "--dataset", str(ds), "--report-dir", str(tmp_path / "r"), "--model-out", str(model),
                     "--seeds", "1", "--max-epochs", "3", "--device", "cpu"]) == 0
    B.save(build_bundle(graphs, {113: {"name": "Striker One"}, 213: {"name": "Striker Two"}},
                        {1: {"name": "Alpha"}, 2: {"name": "Beta"}}), bundle_path)
    common = ["--dataset", str(ds), "--bundle", str(bundle_path), "--model", str(model)]
    capsys.readouterr()
    assert sim_main(common + ["swap", "--team", "alpha", "--out", "striker one", "--in", "striker two"]) == 0
    out = capsys.readouterr().out
    assert "Alpha: replacing Striker One" in out and "npxG per match over" in out
    assert sim_main(common + ["swap", "--team", "Alpha", "--out", "113", "--rank", "--min-minutes", "0"]) == 0
    assert "candidates (FW" in capsys.readouterr().out


def test_sanity_checks_pass_on_a_sensible_model(tmp_path):
    from sim.checks import run_all, write_report

    graphs = []
    for m in range(1, 21):  # 20 full matches = 1,800 minutes for every regular
        for team, fwd_box in ((1, 2), (2, 4), (3, 6), (4, 8)):
            graphs.append(graph(m, team, {team * 10 + 1: node("FW", box=fwd_box), team * 10 + 2: node("GK", box=0),
                                          team * 10 + 3: node("MD", box=1)}, week=m))
    bundle = build_bundle(graphs, {}, {t: {"name": f"T{t}"} for t in (1, 2, 3, 4)})
    sim = Simulator(graphs, bundle.profiles, LinearEnsemble())
    results = run_all(sim, bundle)
    assert [r.name for r in results] == ["Self-swap", "Forward → goalkeeper", "Candidate ranking"]
    assert all(r.passed for r in results), [r.summary for r in results]
    write_report(results, tmp_path / "README.md", {"n_models": 2})
    assert "| Self-swap | pass |" in (tmp_path / "README.md").read_text(encoding="utf-8")
