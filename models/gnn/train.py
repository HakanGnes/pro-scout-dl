"""Training loop, multi-seed evaluation and paired bootstrap comparisons."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from features.pass_networks.graph import TeamGraph

from .baselines import metrics
from .net import Normalizer, make_net

DEFAULTS = {"hidden": 64, "layers": 3, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
            "batch_size": 64, "max_epochs": 300, "patience": 30}


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class RunResult:
    arch: str
    seed: int
    best_epoch: int
    val_mse: float
    seconds: float
    state_dict: dict
    predictions: dict[str, np.ndarray] = field(default_factory=dict)


@torch.no_grad()
def predict(net, loader, device) -> np.ndarray:
    net.eval()
    out = [net(batch.to(device)).squeeze(1).cpu().numpy() for batch in loader]
    return np.concatenate(out) if out else np.zeros(0)


def fit(arch: str, train_data, val_data, eval_data: dict, seed: int, device, cfg: dict, log=print) -> RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    edge_dim = train_data[0].edge_attr.shape[1]
    net = make_net(arch, hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"], edge_dim=edge_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True, generator=gen)
    val_loader = DataLoader(val_data, batch_size=256)
    y_val = np.array([d.y.item() for d in val_data])

    best, best_epoch, best_state, waited = float("inf"), 0, None, 0
    started = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        net.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(batch), batch.y)
            loss.backward()
            opt.step()
        val_mse = float(((predict(net, val_loader, device) - y_val) ** 2).mean())
        if val_mse < best - 1e-5:
            best, best_epoch, best_state, waited = val_mse, epoch, copy.deepcopy(net.state_dict()), 0
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break
    net.load_state_dict(best_state)
    result = RunResult(arch, seed, best_epoch, best, time.perf_counter() - started,
                       {k: v.cpu() for k, v in best_state.items()})
    for split, data in eval_data.items():
        result.predictions[split] = predict(net, DataLoader(data, batch_size=256), device)
    log(f"    {arch:8s} seed {seed}: best epoch {best_epoch:3d}, val MSE {best:.4f}, {result.seconds:.0f}s")
    return result


def run_neural(graphs: list[TeamGraph], archs, seeds, device, cfg: dict, log=print, edge_key: str = "edge_attr"):
    """Train every (architecture, seed); return results and the fitted normaliser."""
    by_split = {s: [g for g in graphs if g.split == s] for s in ("train", "val", "test", "test_intl")}
    norm = Normalizer(by_split["train"], edge_key)
    data = {s: [norm.to_data(g) for g in gs] for s, gs in by_split.items()}
    eval_data = {s: data[s] for s in ("val", "test", "test_intl")}
    results = []
    for arch in archs:
        for seed in seeds:
            results.append(fit(arch, data["train"], data["val"], eval_data, seed, device, cfg, log))
    return results, norm


def paired_bootstrap(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, match_ids: np.ndarray,
                     n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean |error| of a minus b, with a 95% interval from resampling whole matches."""
    d = np.abs(pa - y) - np.abs(pb - y)
    codes, uniq = pd.factorize(match_ids)
    sums, counts = np.bincount(codes, weights=d), np.bincount(codes)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(uniq), len(uniq))
        boots[i] = sums[idx].sum() / counts[idx].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def bootstrap_mae(y: np.ndarray, p: np.ndarray, match_ids: np.ndarray, **kw) -> tuple[float, float, float]:
    """MAE of p with a 95% interval from resampling whole matches."""
    return paired_bootstrap(y, p, y, match_ids, **kw)  # |p - y| - |y - y| = |p - y|


def summarise(results: list[RunResult], targets: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    """Per-architecture metrics: mean ± sd over seeds, and the seed ensemble."""
    rows, ensembles = [], {}
    for arch in sorted({r.arch for r in results}):
        runs = [r for r in results if r.arch == arch]
        ensembles[arch] = {s: np.mean([r.predictions[s] for r in runs], axis=0) for s in targets}
        for split, y in targets.items():
            per_seed = pd.DataFrame([metrics(y, r.predictions[split]) for r in runs])
            ens = metrics(y, ensembles[arch][split])
            rows.append({"model": arch, "split": split,
                         "mae_mean": per_seed["mae"].mean(), "mae_sd": per_seed["mae"].std(ddof=1),
                         "r2_mean": per_seed["r2"].mean(), "r2_sd": per_seed["r2"].std(ddof=1),
                         "ens_mae": ens["mae"], "ens_r2": ens["r2"], "ens_spearman": ens["spearman"],
                         "best_epoch_mean": float(np.mean([r.best_epoch for r in runs]))})
    return pd.DataFrame(rows), ensembles
