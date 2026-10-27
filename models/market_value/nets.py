"""Models 4-5: LSTM and Transformer over the 36-month history, plus a static-input MLP."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from features.market_value.dataset import CH, FEET, MONTHS, POSITIONS, SEQ_CHANNELS, SUB_POSITIONS, Samples
from features.market_value.tabular import LEAGUES

ARCHITECTURES = ("lstm", "transformer")
CFG = {"hidden": 64, "layers": 2, "heads": 4, "ff": 128, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
       "batch_size": 256, "max_epochs": 100, "patience": 10, "clip": 1.0}
NUMERIC_STATIC = ("age", "height_cm", "league_strength", "momentum")
ONE_HOT = {"position": POSITIONS, "sub_position": SUB_POSITIONS, "foot": FEET, "league": LEAGUES}
RAW_SEQ = ("has_value", "revalued", "transfer")  # 0/1 flags are not standardised


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Normalizer:
    """Standardisation and one-hot encoding fitted on the training samples only."""

    def __init__(self, train: Samples):
        x = train.seq.reshape(-1, len(SEQ_CHANNELS)).astype(np.float64)
        self.seq_mean, self.seq_std = x.mean(0), x.std(0) + 1e-6
        for ch in RAW_SEQ:
            self.seq_mean[CH[ch]], self.seq_std[CH[ch]] = 0.0, 1.0
        st = train.static
        self.fill = {c: float(np.nanmean(st[c].to_numpy(dtype=float))) for c in NUMERIC_STATIC}
        filled = np.column_stack([st[c].fillna(self.fill[c]).to_numpy(dtype=float) for c in NUMERIC_STATIC])
        self.st_mean, self.st_std = filled.mean(0), filled.std(0) + 1e-6

    @property
    def static_dim(self) -> int:
        return len(NUMERIC_STATIC) + 2 + sum(len(v) for v in ONE_HOT.values())  # + height missing + anchor_january

    def tensors(self, s: Samples) -> dict[str, torch.Tensor]:
        seq = (s.seq - self.seq_mean) / self.seq_std
        mask = s.seq[:, :, CH["has_value"]] > 0  # True = month is seen (after the first valuation)
        seq = np.where(mask[:, :, None], seq, 0.0)
        st = s.static
        num = np.column_stack([st[c].fillna(self.fill[c]).to_numpy(dtype=float) for c in NUMERIC_STATIC])
        parts = [(num - self.st_mean) / self.st_std, st["height_cm"].isna().to_numpy(dtype=float)[:, None],
                 st["anchor_january"].to_numpy(dtype=float)[:, None]]
        for col, cats in ONE_HOT.items():
            idx = {c: i for i, c in enumerate(cats)}
            oh = np.zeros((len(st), len(cats)))
            oh[np.arange(len(st)), [idx.get(v, idx.get("Missing", 0)) for v in st[col]]] = 1.0
            parts.append(oh)
        return {"seq": torch.tensor(seq, dtype=torch.float32), "mask": torch.tensor(mask),
                "static": torch.tensor(np.hstack(parts), dtype=torch.float32),
                "y": torch.tensor(s.y, dtype=torch.float32)}

    def state(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}

    @classmethod
    def from_state(cls, state: dict) -> "Normalizer":
        obj = cls.__new__(cls)
        for k, v in state.items():
            setattr(obj, k, np.asarray(v) if isinstance(v, list) else v)
        return obj


def _mlp(i: int, h: int, o: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o))


class LSTMNet(nn.Module):
    def __init__(self, static_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        self.lstm = nn.LSTM(len(SEQ_CHANNELS), hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.static = _mlp(static_dim, hidden, hidden, dropout)
        self.head = _mlp(2 * hidden, hidden, 1, dropout)

    def forward(self, seq, mask, static):
        out, _ = self.lstm(seq)
        return self.head(torch.cat([out[:, -1], torch.relu(self.static(static))], dim=1)).squeeze(1)


class TransformerNet(nn.Module):
    def __init__(self, static_dim: int, hidden: int = 64, layers: int = 2, heads: int = 4, ff: int = 128,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.inp = nn.Linear(len(SEQ_CHANNELS), hidden)
        self.pos = nn.Parameter(torch.zeros(1, MONTHS, hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.static = _mlp(static_dim, hidden, hidden, dropout)
        self.head = _mlp(3 * hidden, hidden, 1, dropout)

    def forward(self, seq, mask, static):
        h = self.encoder(self.inp(seq) + self.pos, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(torch.cat([h[:, -1], mean, torch.relu(self.static(static))], dim=1)).squeeze(1)


def make_net(arch: str, static_dim: int, cfg: dict = CFG) -> nn.Module:
    kw = {k: cfg[k] for k in ("hidden", "layers", "heads", "ff", "dropout")}
    if arch == "lstm":
        return LSTMNet(static_dim, **kw)
    if arch == "transformer":
        return TransformerNet(static_dim, **kw)
    raise ValueError(f"unknown architecture {arch!r}")


@dataclass
class Run:
    arch: str
    seed: int
    best_epoch: int
    val_mae: float
    seconds: float
    state_dict: dict = field(repr=False, default_factory=dict)


@torch.no_grad()
def predict(net: nn.Module, t: dict, device, batch: int = 2048) -> np.ndarray:
    net.eval()
    out = []
    for i in range(0, len(t["y"]), batch):
        sl = slice(i, i + batch)
        out.append(net(t["seq"][sl].to(device), t["mask"][sl].to(device), t["static"][sl].to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def fit(arch: str, train: dict, val: dict, static_dim: int, seed: int, device, cfg: dict = CFG, log=print) -> Run:
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    net = make_net(arch, static_dim, cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    y_val = val["y"].numpy().astype(np.float64)
    best, best_epoch, best_state, waited = np.inf, 0, None, 0
    n = len(train["y"])
    for epoch in range(1, cfg["max_epochs"] + 1):
        net.train()
        for idx in torch.randperm(n, generator=gen).split(cfg["batch_size"]):
            pred = net(train["seq"][idx].to(device), train["mask"][idx].to(device), train["static"][idx].to(device))
            loss = torch.nn.functional.l1_loss(pred, train["y"][idx].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"])
            opt.step()
        val_mae = float(np.mean(np.abs(predict(net, val, device) - y_val)))
        if val_mae < best - 1e-6:
            best, best_epoch, waited = val_mae, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break
    run = Run(arch, seed, best_epoch, best, time.perf_counter() - started, best_state)
    log(f"    {arch:11s} seed {seed}: best epoch {best_epoch:3d}, val MAE {best:.4f}, {run.seconds:.0f}s")
    return run


def load_net(run_state: dict, arch: str, static_dim: int, cfg: dict, device) -> nn.Module:
    net = make_net(arch, static_dim, cfg)
    net.load_state_dict(run_state)
    return net.to(device).eval()
