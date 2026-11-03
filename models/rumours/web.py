"""The rumours page's data: the 2026 player-club pairs with their probability and outcome, the 2026 check,
and the registered 2025 test record. Headlines are not included: the archived articles are the papers' text and
are not redistributed (README, Data and attribution); a pair is listed with how often and how widely it ran.

    python -m models.rumours export-web        -> web/public/data/rumours.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from etl.config import REPO_ROOT

WEB_FILE = REPO_ROOT / "web" / "public" / "data" / "rumours.json"
BANDS = (0, 0.1, 0.2, 0.3, 0.5, 1.0)


def _clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _band(lo: float, hi: float) -> str:
    return f"{max(lo, 0) * 100:.0f}–{hi:.0%}"


def reliability(y: pd.Series, p: pd.Series) -> list[dict]:
    b = pd.cut(p, list(BANDS), include_lowest=True)
    t = pd.DataFrame({"y": y, "p": p, "b": b}).groupby("b", observed=True).agg(
        claims=("y", "size"), predicted=("p", "mean"), happened=("y", "mean"))
    return [{"band": _band(iv.left, iv.right), "claims": int(r.claims),
             "predicted": float(r.predicted), "happened": float(r.happened)} for iv, r in t.iterrows()]


def export(pairs: Path, check: Path, predictions: Path, results: Path, out: Path = WEB_FILE) -> dict:
    pt = pd.read_csv(pairs, parse_dates=["first_seen", "last_seen"])
    chk = json.loads(check.read_text(encoding="utf-8"))
    pred = pd.read_csv(predictions)
    res = json.loads(results.read_text(encoding="utf-8"))
    p = pred["p_metadata"] if "p_metadata" in pred else pred["metadata"]
    test = res["test"]["metadata"]
    for r in chk.get("reliability", []):           # bands as the page writes them
        lo, hi = (float(x) for x in r["band"].strip("(]").split(","))
        r["band"] = _band(lo, hi)
    rows = [[r.player, r.club, _clean(r.current_club), int(r.articles), int(r.papers), f"{r.first_seen:%Y-%m-%d}",
             f"{r.last_seen:%Y-%m-%d}", round(float(r.latest_p), 4), round(float(r.highest_p), 4),
             None if pd.isna(r.happened) else int(r.happened), r.outcome] for r in pt.itertuples(index=False)]
    rec = {
        "fields": ["player", "club", "current_club", "articles", "papers", "first_seen", "last_seen", "latest_p",
                   "highest_p", "happened", "outcome"],
        "pairs": rows, "check_2026": chk,
        "test_2025": {"claims": int(test["n"]), "rate": float(test["rate"]), "auc": float(test["auc"]),
                      "mean_p": float(test["mean_p"]), "reliability": reliability(pred["label"], p)},
        "study": {"primary": res["primary"], "auc": {m: float(v["auc"]) for m, v in res["test"].items()},
                  "commit": res.get("commit")},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return rec
