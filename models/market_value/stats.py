"""Metrics and player-clustered bootstrap intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Errors on the log scale plus the same predictions read as values."""
    e = p - y
    ape = np.abs(np.expm1(e))  # |V_hat - V| / V
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "median_ape": float(np.median(ape)),
        "within_20pct": float(np.mean(ape <= 0.20)),
        "bias": float(np.mean(e)),
        "n": int(len(y)),
    }


def cluster_bootstrap(d: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of d with a 95% interval from resampling whole clusters (here: players)."""
    codes, uniq = pd.factorize(clusters)
    sums, counts = np.bincount(codes, weights=d), np.bincount(codes)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        boots[i] = (w @ sums) / (w @ counts)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def paired(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, players: np.ndarray, n_boot: int = 2000,
           seed: int = 0) -> tuple[float, float, float]:
    """MAE of a minus MAE of b, with a player-clustered 95% interval."""
    return cluster_bootstrap(np.abs(pa - y) - np.abs(pb - y), players, n_boot, seed)
