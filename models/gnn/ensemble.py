"""Load the saved neural models and predict with all of them at once."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from features.pass_networks.graph import TeamGraph

from .net import Normalizer, make_net


class Ensemble:
    """Every saved (architecture, seed) network; predictions are (n_models, n_graphs)."""

    def __init__(self, nets: list[tuple[str, torch.nn.Module]], normalizer: Normalizer, device: torch.device,
                 meta: dict | None = None):
        self.nets = nets
        self.normalizer = normalizer
        self.device = device
        self.meta = meta or {}

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.nets]

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu", archs: tuple[str, ...] = ("deepsets", "gnn")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("format") != 2:
            raise SystemExit(f"{path} is from an older version that saved only the GNN. "
                             "Re-run: python -m models.gnn evaluate")
        cfg, device = saved["cfg"], torch.device(device)
        normalizer = Normalizer.from_state(saved["normalizer"])
        nets = []
        for arch in archs:
            for seed, state in enumerate(saved["archs"].get(arch, [])):
                net = make_net(arch, hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"],
                               edge_dim=normalizer.edge_dim)
                net.load_state_dict(state)
                nets.append((f"{arch}-{seed}", net.to(device).eval()))
        if not nets:
            raise SystemExit(f"No networks for {archs} in {path}")
        meta = {k: saved.get(k) for k in ("chosen_on_validation", "dataset_meta", "cfg")}
        return cls(nets, normalizer, device, meta)

    @torch.no_grad()
    def predict(self, graphs: list[TeamGraph], batch_size: int = 512) -> np.ndarray:
        if not graphs:
            return np.zeros((len(self.nets), 0))
        loader = DataLoader([self.normalizer.to_data(g) for g in graphs], batch_size=batch_size)
        out = np.zeros((len(self.nets), len(graphs)), dtype=np.float64)
        start = 0
        for batch in loader:
            batch = batch.to(self.device)
            n = batch.num_graphs
            for k, (_, net) in enumerate(self.nets):
                out[k, start:start + n] = net(batch).squeeze(1).cpu().numpy()
            start += n
        return out
