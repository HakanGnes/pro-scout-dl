"""Neural models on passing networks (PyTorch Geometric).

Two architectures see exactly the same node features:
  * DeepSets — every player encoded on their own, then summed and averaged. No edges.
  * PassGNN  — the same encoder followed by message passing over completed passes
               (GINE convolutions that also read the edge features).
If PassGNN beats DeepSets, who passes to whom carries information beyond
the players' individual numbers. That comparison is the point of this phase.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

from features.pass_networks.graph import EDGE_FEATURES, NODE_FEATURES, TeamGraph

ARCHITECTURES = ("deepsets", "gnn")


class Normalizer:
    """Per-feature standardisation fitted on the training graphs only."""

    def __init__(self, graphs: list[TeamGraph]):
        x = np.concatenate([g.x for g in graphs])
        e = np.concatenate([g.edge_attr for g in graphs])
        self.x_mean, self.x_std = x.mean(0), x.std(0) + 1e-6
        self.e_mean, self.e_std = e.mean(0), e.std(0) + 1e-6

    def state(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("x_mean", "x_std", "e_mean", "e_std")}

    @classmethod
    def from_state(cls, state: dict) -> "Normalizer":
        obj = cls.__new__(cls)
        for k, v in state.items():
            setattr(obj, k, np.asarray(v, dtype=np.float32))
        return obj

    def to_data(self, g: TeamGraph) -> Data:
        x = (g.x - self.x_mean) / self.x_std
        e = (g.edge_attr - self.e_mean) / self.e_std
        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(g.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(e, dtype=torch.float32),
            y=torch.tensor([[g.target if g.target is not None else 0.0]], dtype=torch.float32),
        )


def _mlp(i: int, h: int, o: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o))


class DeepSets(nn.Module):
    def __init__(self, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = _mlp(len(NODE_FEATURES), hidden, hidden, dropout)
        self.head = _mlp(2 * hidden, hidden, 1, dropout)

    def forward(self, data) -> torch.Tensor:
        h = torch.relu(self.encoder(data.x))
        pooled = torch.cat([global_add_pool(h, data.batch) / 11.0, global_mean_pool(h, data.batch)], dim=1)
        return nn.functional.softplus(self.head(pooled))


class PassGNN(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = _mlp(len(NODE_FEATURES), hidden, hidden, dropout)
        self.convs = nn.ModuleList(
            GINEConv(_mlp(hidden, hidden, hidden, dropout), edge_dim=len(EDGE_FEATURES), train_eps=True)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = dropout
        self.head = _mlp(2 * hidden, hidden, 1, dropout)

    def forward(self, data) -> torch.Tensor:
        h = torch.relu(self.encoder(data.x))
        for conv, norm in zip(self.convs, self.norms):
            # Passes are summed, so a busy edge counts more; residual keeps each player's own signal.
            h = h + nn.functional.dropout(torch.relu(norm(conv(h, data.edge_index, data.edge_attr))),
                                          p=self.dropout, training=self.training)
        pooled = torch.cat([global_add_pool(h, data.batch) / 11.0, global_mean_pool(h, data.batch)], dim=1)
        return nn.functional.softplus(self.head(pooled))


def make_net(arch: str, **kw) -> nn.Module:
    if arch == "deepsets":
        return DeepSets(**{k: v for k, v in kw.items() if k != "layers"})
    if arch == "gnn":
        return PassGNN(**kw)
    raise ValueError(f"unknown architecture {arch!r}")
