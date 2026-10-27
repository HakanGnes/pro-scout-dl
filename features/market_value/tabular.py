"""Fixed tabular summaries of the monthly history, for the gradient-boosting model.

Computed only from Samples.seq and Samples.static, so the GBM sees no information the sequence
models do not. The list is fixed by the pre-registration; do not change it after training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import CH, FEET, MONTHS, POSITIONS, SUB_POSITIONS, Samples

COUNT_CHANNELS = ("league_min90", "league_apps", "goals", "assists", "cards", "europe_min90", "cup_min90")
CATEGORICAL = {"position": POSITIONS, "sub_position": SUB_POSITIONS, "foot": FEET}
LEAGUES = ("BE1", "DK1", "ES1", "FR1", "GB1", "GR1", "IT1", "L1", "NL1", "PO1", "RU1", "SC1", "TR1", "UKR1", "Missing")


def _window(seq: np.ndarray, ch: str, last: int, skip: int = 0) -> np.ndarray:
    """Sum of a channel over `last` months, ending `skip` months before t."""
    end = MONTHS - skip
    return seq[:, end - last:end, CH[ch]].sum(axis=1)


def _value_change(seq: np.ndarray, months: int) -> np.ndarray:
    """log V(t) - log V(t - months); NaN if there was no value yet. Month index j holds the value at its end."""
    j = MONTHS - 1 - months
    lv, has = seq[:, :, CH["log_value"]], seq[:, j, CH["has_value"]] > 0
    return np.where(has, lv[:, -1] - lv[:, j], np.nan)


def _months_since(flags: np.ndarray) -> np.ndarray:
    """Months since the last month with a flag (0 = the month before t); NaN if none in the window."""
    any_ = flags.any(axis=1)
    last = MONTHS - 1 - np.argmax(flags[:, ::-1], axis=1)
    return np.where(any_, MONTHS - 1 - last, np.nan)


def tabular(samples: Samples) -> pd.DataFrame:
    seq = samples.seq.astype(np.float64)
    st = samples.static
    f: dict[str, np.ndarray] = {}

    f["log_value"] = seq[:, -1, CH["log_value"]]
    for k in (6, 12, 24, MONTHS - 1):
        f[f"value_change_{k}m"] = _value_change(seq, k)
    revalued = seq[:, :, CH["revalued"]] > 0
    f["months_since_revaluation"] = _months_since(revalued)
    f["revaluations_12m"] = revalued[:, -12:].sum(axis=1)
    f["revaluations_36m"] = revalued.sum(axis=1)
    has = seq[:, :, CH["has_value"]] > 0
    f["months_with_value"] = has.sum(axis=1)
    peak = np.where(has, seq[:, :, CH["log_value"]], -np.inf).max(axis=1)
    f["below_peak_36m"] = peak - f["log_value"]

    for ch in COUNT_CHANNELS:
        f[f"{ch}_6m"] = _window(seq, ch, 6)
        f[f"{ch}_12m"] = _window(seq, ch, 12)
        f[f"{ch}_prev12m"] = _window(seq, ch, 12, skip=12)
    minutes90_12 = sum(f[f"{c}_12m"] for c in ("league_min90", "europe_min90", "cup_min90"))
    for ch in ("goals", "assists", "cards"):
        f[f"{ch}_per90_12m"] = np.where(minutes90_12 >= 1, f[f"{ch}_12m"] / np.maximum(minutes90_12, 1), np.nan)
    f["minutes90_12m"] = minutes90_12

    transfers = seq[:, :, CH["transfer"]] > 0
    f["months_since_transfer"] = _months_since(transfers)
    f["transfers_36m"] = transfers.sum(axis=1)
    fees = seq[:, :, CH["transfer_fee_log"]]
    last_t = MONTHS - 1 - np.argmax(transfers[:, ::-1], axis=1)
    f["last_transfer_fee_log"] = np.where(transfers.any(axis=1), fees[np.arange(len(fees)), last_t], np.nan)
    f["max_transfer_fee_log_36m"] = fees.max(axis=1)

    w = seq[:, :, CH["league_min90"]]
    s = seq[:, :, CH["league_strength"]]
    for name, sl in (("league_strength_12m", slice(-12, None)), ("league_strength_prev12m", slice(-24, -12))):
        ww = w[:, sl]
        f[name] = np.where(ww.sum(axis=1) > 0, (ww * s[:, sl]).sum(axis=1) / np.maximum(ww.sum(axis=1), 1e-9), np.nan)

    f["age"] = st["age"].to_numpy(dtype=float)
    f["height_cm"] = st["height_cm"].to_numpy(dtype=float)
    f["league_strength"] = st["league_strength"].to_numpy(dtype=float)
    f["anchor_january"] = st["anchor_january"].to_numpy(dtype=float)
    f["momentum"] = st["momentum"].to_numpy(dtype=float)
    out = pd.DataFrame(f)
    for col, cats in {**CATEGORICAL, "league": LEAGUES}.items():
        out[col] = pd.Categorical(st[col], categories=cats)
    return out
