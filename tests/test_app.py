"""The Streamlit page, driven headlessly. Skipped without Streamlit or PyTorch."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from test_sim import graph, node  # noqa: E402

from features.pass_networks import dataset  # noqa: E402
from features.pass_networks.graph import EDGE_FEATURES  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_cache():
    """The page caches the loaded simulator for the whole process; each test starts clean."""
    import streamlit as st

    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    from models.gnn.__main__ import main as gnn_main
    from sim import bundle as B

    rng = np.random.default_rng(0)
    squads = {t: list(range(t * 100, t * 100 + 14)) for t in (1, 2, 3, 4)}
    roles = ["GK"] + ["DF"] * 4 + ["MD"] * 5 + ["FW"] * 4
    graphs, m = [], 0
    for week in range(1, 39):
        for a, b in ((1, 2), (3, 4)) if week % 2 else ((1, 3), (2, 4)):
            m += 1
            for t in (a, b):
                g = graph(m, t, {p: node(roles[k], box=float(rng.integers(0, 4))) for k, p in enumerate(squads[t])},
                          week=week)
                g.edge_index = np.array([[k, (k + 1) % 14] for k in range(14)], dtype=np.int64).T
                g.edge_attr = np.ones((14, len(EDGE_FEATURES)), dtype=np.float32)
                graphs.append(g)
    dataset.assign_splits(graphs)
    paths = {"PROSCOUT_DATASET": tmp_path / "g.pkl", "PROSCOUT_BUNDLE": tmp_path / "b.pkl",
             "PROSCOUT_MODEL": tmp_path / "m.pt"}
    dataset.save(graphs, paths["PROSCOUT_DATASET"], {"xg_model": "test"})
    assert gnn_main(["evaluate", "--dataset", str(paths["PROSCOUT_DATASET"]), "--report-dir", str(tmp_path / "r"),
                     "--model-out", str(paths["PROSCOUT_MODEL"]), "--seeds", "1", "--max-epochs", "2",
                     "--device", "cpu"]) == 0
    names = {p: {"name": f"Player {p}"} for sq in squads.values() for p in sq}
    B.save(B.build_bundle(graphs, names, {t: {"name": f"Club {t}"} for t in squads}), paths["PROSCOUT_BUNDLE"])
    for k, v in paths.items():
        monkeypatch.setenv(k, str(v))
    return paths


def test_page_compares_and_ranks(artifacts):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../app/swap.py", default_timeout=120).run()
    assert not at.exception
    assert [s.label for s in at.selectbox][:3] == ["Competition", "Team", "Player to replace"]
    labels = [m.label for m in at.metric]
    assert "Change in npxG per match" in labels and "Same player (sanity check)" in labels
    at.radio[0].set_value("Rank candidates").run()
    at.button[0].click().run()
    assert not at.exception
    table = at.dataframe[0].value
    assert {"player", "delta", "se_matches", "sd_models"} <= set(table.columns)
    assert table["delta"].is_monotonic_decreasing


def test_page_explains_missing_files(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PROSCOUT_DATASET", str(tmp_path / "missing.pkl"))
    at = AppTest.from_file("../app/swap.py", default_timeout=60).run()
    assert not at.exception
    assert "python -m features.pass_networks build" in at.error[0].value
