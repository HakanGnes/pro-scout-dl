"""The pre-registered edge-feature experiment, end to end on synthetic data."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from test_gnn import synthetic_graphs  # noqa: E402

from features.pass_networks import dataset  # noqa: E402
from features.pass_networks.graph import EDGE_FEATURES_RICH  # noqa: E402


def _with_rich_edges(graphs, seed=0):
    rng = np.random.default_rng(seed)
    for g in graphs:
        extra = rng.random((g.edge_attr.shape[0], len(EDGE_FEATURES_RICH) - g.edge_attr.shape[1])).astype(np.float32)
        g.edge_attr_rich = np.hstack([g.edge_attr, extra])
    return graphs


@pytest.fixture
def phase1(tmp_path):
    from models.gnn.__main__ import main as gnn_main

    ds = tmp_path / "graphs.pkl"
    dataset.save(_with_rich_edges(synthetic_graphs(80)), ds, {"xg_model": "test"})
    model = tmp_path / "pass_gnn.pt"
    assert gnn_main(["evaluate", "--dataset", str(ds), "--report-dir", str(tmp_path / "r"), "--model-out", str(model),
                     "--seeds", "2", "--max-epochs", "3", "--device", "cpu"]) == 0
    return tmp_path, ds, model


def _args(tmp_path, ds, model):
    return ["--dataset", str(ds), "--phase1-model", str(model), "--phase1-report", str(model.with_suffix(".json")),
            "--model-out", str(tmp_path / "v2.pt"), "--report", str(tmp_path / "prereg.md"),
            "--device", "cpu", "--max-epochs", "3"]


def test_experiment_runs_and_reports(phase1):
    import torch

    from models.gnn.exp_edge_features import main

    tmp_path, ds, model = phase1
    assert main(_args(tmp_path, ds, model)) == 0
    text = (tmp_path / "prereg.md").read_text(encoding="utf-8")
    assert "GNN-v2 − DeepSets absolute error on the league test weeks" in text
    assert "Success criterion (upper end below 0)" in text
    saved = torch.load(tmp_path / "v2.pt", weights_only=False)
    assert len(saved["archs"]["gnn"]) == 5 and len(saved["normalizer"]["e_mean"]) == len(EDGE_FEATURES_RICH)

    from models.gnn.ensemble import Ensemble
    graphs, _ = dataset.load(ds)
    assert Ensemble.load(tmp_path / "v2.pt").predict(graphs[:5]).shape == (5, 5)  # the saved v2 reloads


def test_experiment_stops_if_phase1_does_not_reproduce(phase1):
    from models.gnn.exp_edge_features import main

    tmp_path, ds, model = phase1
    report = model.with_suffix(".json")
    data = json.loads(report.read_text(encoding="utf-8"))
    data["test"]["deepsets"]["mae"] += 0.05
    report.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not reproduce"):
        main(_args(tmp_path, ds, model))
    assert not (tmp_path / "prereg.md").exists()


def test_old_datasets_are_refused(tmp_path, phase1):
    from models.gnn.exp_edge_features import main

    tmp_path, ds, model = phase1
    graphs, meta = dataset.load(ds)
    for g in graphs:
        g.edge_attr_rich = None
    dataset.save(graphs, ds, meta)
    with pytest.raises(SystemExit, match="Rebuild"):
        main(_args(tmp_path, ds, model))
