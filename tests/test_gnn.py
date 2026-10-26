"""Neural models on passing networks. Skipped when PyTorch / PyTorch Geometric are not installed."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from features.pass_networks import dataset  # noqa: E402
from features.pass_networks.graph import BASELINE_FEATURES, EDGE_FEATURES, NODE_FEATURES, TeamGraph  # noqa: E402


def synthetic_graphs(n_matches=120, seed=0) -> list[TeamGraph]:
    """Random passing networks whose target depends on node features and on edges between attackers."""
    rng = np.random.default_rng(seed)
    graphs = []
    for m in range(n_matches):
        week = m % 38 + 1
        for team in (1, 2):
            n = int(rng.integers(12, 16))
            strength = rng.random()  # how dangerous this team was in this match
            x = rng.random((n, len(NODE_FEATURES))).astype(np.float32)
            x[:, NODE_FEATURES.index("log_box")] *= strength
            src, dst = np.where(rng.random((n, n)) < 0.5)
            keep = src != dst
            edge_index = np.vstack([src[keep], dst[keep]]).astype(np.int64)
            edge_attr = rng.random((edge_index.shape[1], len(EDGE_FEATURES))).astype(np.float32)
            attack_links = float((x[edge_index[0], 9] * x[edge_index[1], 9] * edge_attr[:, 0]).mean())
            target = float(2.5 * strength + 1.0 * attack_links + rng.normal(0, 0.1))
            box = x[:, NODE_FEATURES.index("log_box")]
            stats = {f: float(rng.random()) for f in BASELINE_FEATURES} | {"box_entries": float(box.sum())}
            graphs.append(TeamGraph(m, team, "L", week, False, "home" if team == 1 else "away",
                                    list(range(n)), x, edge_index, edge_attr, stats, max(0.0, target)))
    dataset.assign_splits(graphs)
    return graphs


def test_models_are_permutation_invariant_and_non_negative():
    from torch_geometric.data import Batch

    from models.gnn.net import Normalizer, make_net

    graphs = synthetic_graphs(10)
    norm = Normalizer(graphs)
    g = graphs[0]
    perm = np.random.default_rng(1).permutation(len(g.player_ids))
    inv = np.argsort(perm)
    shuffled = TeamGraph(**{**g.__dict__, "x": g.x[perm], "edge_index": inv[g.edge_index]})
    for arch in ("deepsets", "gnn"):
        torch.manual_seed(0)
        net = make_net(arch, hidden=16, layers=2, dropout=0.0).eval()
        a = net(Batch.from_data_list([norm.to_data(g)]))
        b = net(Batch.from_data_list([norm.to_data(shuffled)]))
        assert torch.allclose(a, b, atol=1e-5), arch       # node order must not matter
        assert (a >= 0).all()


def test_gnn_reads_edges_deepsets_does_not():
    from torch_geometric.data import Batch

    from models.gnn.net import Normalizer, make_net

    graphs = synthetic_graphs(10)
    norm = Normalizer(graphs)
    g = graphs[0]
    no_edges = TeamGraph(**{**g.__dict__, "edge_index": np.zeros((2, 0), dtype=np.int64),
                            "edge_attr": np.zeros((0, len(EDGE_FEATURES)), dtype=np.float32)})
    torch.manual_seed(0)
    gnn, ds = make_net("gnn", hidden=16, layers=2, dropout=0.0).eval(), make_net("deepsets", hidden=16, dropout=0.0).eval()
    full, empty = Batch.from_data_list([norm.to_data(g)]), Batch.from_data_list([norm.to_data(no_edges)])
    assert not torch.allclose(gnn(full), gnn(empty))
    assert torch.allclose(ds(full), ds(empty))


def test_training_beats_the_mean_and_is_reproducible():
    from models.gnn.train import DEFAULTS, pick_device, run_neural

    graphs = synthetic_graphs()
    cfg = {**DEFAULTS, "hidden": 32, "max_epochs": 40, "patience": 10}
    quiet = lambda _m: None  # noqa: E731
    r1, _ = run_neural(graphs, ("gnn",), [0], pick_device("cpu"), cfg, quiet)
    r2, _ = run_neural(graphs, ("gnn",), [0], pick_device("cpu"), cfg, quiet)
    np.testing.assert_allclose(r1[0].predictions["test"], r2[0].predictions["test"], rtol=1e-5)

    y = np.array([g.target for g in graphs if g.split == "test"])
    mu = np.mean([g.target for g in graphs if g.split == "train"])
    assert np.abs(r1[0].predictions["test"] - y).mean() < np.abs(mu - y).mean()


def test_cli_end_to_end(tmp_path):
    from models.gnn.__main__ import main
    from models.gnn.net import PassGNN

    ds_path = tmp_path / "graphs.pkl"
    dataset.save(synthetic_graphs(80), ds_path, {"xg_model": "test"})
    report, model = tmp_path / "report", tmp_path / "gnn.pt"
    rc = main(["evaluate", "--dataset", str(ds_path), "--report-dir", str(report), "--model-out", str(model),
               "--seeds", "2", "--max-epochs", "5", "--device", "cpu"])
    assert rc == 0
    text = (report / "README.md").read_text(encoding="utf-8")
    assert "DeepSets" in text and "gnn - deepsets" in text
    assert (report / "test_mae.png").stat().st_size > 10_000

    saved = torch.load(model, weights_only=False)
    assert len(saved["state_dicts"]) == 2
    net = PassGNN(hidden=saved["cfg"]["hidden"], layers=saved["cfg"]["layers"], dropout=saved["cfg"]["dropout"])
    net.load_state_dict(saved["state_dicts"][0])  # the saved weights rebuild the network
