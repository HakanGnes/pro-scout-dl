"""The web pages' data files (models/*/web.py): what goes in and what is left out."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def test_market_value_export_matches_study_metrics(tmp_path):
    from models.market_value.stats import metrics
    from models.market_value.web import export

    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(0, 0.5, n)
    d = pd.DataFrame({"player_id": rng.integers(1, 40, n), "name": "P", "anchor": rng.choice(["2023-01-01", "2024-07-01"], n),
                      "value_now": 1e6, "value_target": 1e6 * np.exp(y), "y": y})
    for m in ("persistence", "age_curve", "lgbm", "lstm", "transformer"):
        d[f"pred_{m}"] = 0.0 if m == "persistence" else y + rng.normal(0, 0.3, n)
    d.to_csv(tmp_path / "p.csv.gz", index=False)
    rec = export(tmp_path / "p.csv.gz", tmp_path / "mv.json")
    ref = metrics(y, d["pred_lstm"].to_numpy())
    assert rec["overall"]["lstm"]["mae"] == ref["mae"] and rec["overall"]["lstm"]["within_20"] == ref["within_20pct"]
    row = rec["rows"][0]
    assert row[5] == 1e6                                          # persistence: value unchanged
    assert row[8] == round(1e6 * np.exp(d["pred_lstm"].iloc[0]), -3)
    assert json.loads((tmp_path / "mv.json").read_text())["players"] == d["player_id"].nunique()


def test_rumours_export_leaves_headlines_out(tmp_path):
    from models.rumours.web import export

    pd.DataFrame({"player_id": [1], "club_id": [2], "player": ["A"], "club": ["K"], "current_club": [None],
                  "articles": [3], "papers": [2], "first_seen": ["2026-03-01"], "last_seen": ["2026-03-05"],
                  "latest_p": [0.2], "highest_p": [0.3], "happened": [np.nan], "outcome": ["open"]}).to_csv(tmp_path / "s_pairs.csv", index=False)
    (tmp_path / "s_check.json").write_text(json.dumps({"reliability": [{"band": "(-0.001, 0.1]", "claims": 1, "predicted": 0.05, "happened": 0.0}]}))
    pd.DataFrame({"label": [0, 1, 0, 1], "p_metadata": [0.05, 0.6, 0.15, 0.25],
                  "text": ["secret headline"] * 4}).to_csv(tmp_path / "pred.csv.gz", index=False)
    (tmp_path / "res.json").write_text(json.dumps({"test": {"metadata": {"n": 4, "rate": 0.5, "auc": 0.75, "mean_p": 0.26}},
                                                   "primary": [0.01, -0.01, 0.03], "commit": "abc"}))
    rec = export(tmp_path / "s_pairs.csv", tmp_path / "s_check.json", tmp_path / "pred.csv.gz", tmp_path / "res.json", tmp_path / "r.json")
    text = (tmp_path / "r.json").read_text()
    assert "secret headline" not in text and "title" not in text
    assert rec["pairs"][0][2] is None and rec["pairs"][0][9] is None
    assert rec["check_2026"]["reliability"][0]["band"] == "0–10%"
    assert [b["band"] for b in rec["test_2025"]["reliability"]] == ["0–10%", "10–20%", "20–30%", "50–100%"]
