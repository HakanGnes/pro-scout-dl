"""The market-value page's data: every test prediction of the registered run, and the headline metrics
recomputed from them (they must match reports/market_value/README.md).

    python -m models.market_value export-web        -> web/public/data/market_value.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from etl.config import REPO_ROOT

from .stats import metrics as study_metrics

WEB_FILE = REPO_ROOT / "web" / "public" / "data" / "market_value.json"
NAMES = {"persistence": "Persistence", "age_curve": "Age curve", "lgbm": "LightGBM", "lstm": "LSTM",
         "transformer": "Transformer"}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """The study's metrics (stats.metrics), under the names the page uses."""
    m = study_metrics(y, p)
    return {"mae": m["mae"], "median_pct": m["median_ape"], "within_20": m["within_20pct"], "n": m["n"]}


def export(predictions: Path, out: Path = WEB_FILE) -> dict:
    d = pd.read_csv(predictions)
    y = d["y"].to_numpy()
    overall = {m: metrics(y, d[f"pred_{m}"].to_numpy()) for m in NAMES}
    by_anchor = {a: {m: metrics(g["y"].to_numpy(), g[f"pred_{m}"].to_numpy())["mae"] for m in NAMES}
                 for a, g in d.groupby("anchor")}
    # one row per sample: player, anchor, value then, value a year later, each model's predicted value (euros)
    rows = []
    for r in d.itertuples(index=False):
        rows.append([int(r.player_id), r.name, r.anchor, float(r.value_now), float(r.value_target),
                     *[round(float(r.value_now * np.exp(getattr(r, f"pred_{m}"))), -3) for m in NAMES]])
    rec = {"models": list(NAMES), "model_names": NAMES, "overall": overall, "by_anchor": by_anchor,
           "fields": ["player_id", "name", "anchor", "value_now", "value_next", *[f"pred_{m}" for m in NAMES]],
           "rows": rows, "players": int(d["player_id"].nunique())}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return rec
