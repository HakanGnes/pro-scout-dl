"""The rumour model for use: fit it once as registered and save it, then score headlines.

`fit` repeats the registered recipe exactly (models trained on the training period, recalibration fitted on
the validation period), so the model in the app is the model whose test results are in reports/rumours/.
The corrected run found that the headline's wording adds nothing measurable (reports/rumours/README.md), so as
registered the model in use is the metadata model alone, recalibrated on validation; `text=True` fits the
metadata + BERTurk fusion instead. `Scorer` links a headline to players and clubs, keeps the claims the rules
call rumours, builds the same inputs, and returns the probability that the move happens within 180 days.

Only headlines dated up to the end of the Transfermarkt data (its last valuation, June 2026) are scored: after
it, a player's club and value are unknown, so a claim could not be built correctly.
"""
from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from etl.news.archive import fold
from features.rumours.entities import Gazetteer
from features.rumours.labels import claims as make_claims

from . import bert
from .dataset import Names, masked_text, metadata
from .models import Calibrator, metadata_gbm, metadata_predict

BUNDLE = "bundle.pkl"
HISTORY_COLS = ["url", "site", "date", "player_id", "club_id"]
# Use only (not part of the registered study): a player is someone with a Transfermarkt value in the 2 years
# before the headline, so coaches and retired players named in a headline are not taken for players.
ACTIVE_DAYS = 730
# Checking predictions (never a model input): after the Transfermarkt data ends, a move counts as reported done
# once this many archived headlines from this many papers say so in completed-move words. "Anlaşma sağlandı"
# (agreed) is not enough: it is often followed by nothing.
REPORTED_MIN_ARTICLES, REPORTED_MIN_PAPERS = 3, 2
OFFICIAL = re.compile(r"\b(?:resmen|resmi(?:les|len)|imzay[i]? att|imzalad|imza toren|kadrosuna katt|yeni transfer|"
                      r"kap(?:a)? bildir|transferini (?:resmen )?acikla|formasini giydi|renklerine bagla)")
def headline_slug(title: str) -> str:
    """A headline written the way the study's URL slugs are: "Galatasaray'dan Icardi'ye teklif!" ->
    "galatasaraydan icardiye teklif"."""
    t = re.sub(r"[’'`´]", "", fold(title or ""))
    return " ".join(re.split(r"[^a-z0-9]+", t)).strip()


def featurize(c: pd.DataFrame, history: pd.DataFrame, g: Gazetteer, names: Names, players: pd.DataFrame,
              valuations: pd.DataFrame) -> pd.DataFrame:
    """Masked text and metadata for claims `c`; news volume is counted over `history` plus `c`."""
    c = c.reset_index(drop=True).copy()
    c["text"] = masked_text(c, names)
    meta = metadata(c, pd.concat([history[HISTORY_COLS], c[HISTORY_COLS]], ignore_index=True), g, players,
                    valuations)
    return pd.concat([c, meta.drop(columns=["site"])], axis=1)


# ---------------------------------------------------------------- fit and save


def fit(data: pd.DataFrame, all_claims: pd.DataFrame, g: Gazetteer, players: pd.DataFrame,
        valuations: pd.DataFrame, out: Path, seeds: list[int], meta_seeds: list[int], club_names: dict | None = None,
        device: str = "auto", model_name: str = bert.MODEL_NAME, hparams: dict | None = None, log=print,
        text: bool = False) -> dict:
    """Train as registered on data (the study table) and save everything the Scorer needs to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    tr, va = (data[data["split"] == s].reset_index(drop=True) for s in ("train", "val"))
    boosters: list = []
    pm, info = metadata_gbm(tr, va, {"val": va}, seeds=meta_seeds, keep=boosters)
    for i, b in enumerate(boosters):
        b.save_model(str(out / f"metadata_{i}.txt"))
    pb = np.zeros(len(va))
    runs = []
    for s in (seeds if text else []):
        p, run = bert.fine_tune(tr["text"].tolist(), tr["label"].to_numpy(float), va["text"].tolist(),
                                va["label"].to_numpy(), {"val": va["text"].tolist()}, seed=s, model_name=model_name,
                                device=device, hparams=hparams, log=log, save_to=out / f"bert_seed{s}")
        pb += p["val"] / len(seeds)
        runs.append(run)
    scores = [pm["val"], pb] if text else [pm["val"]]
    fusion = Calibrator().fit(scores, va["label"].to_numpy())
    keep_players = players[["player_id", "name", "date_of_birth", "position"]].copy()
    keep_vals = valuations[["player_id", "date", "market_value_in_eur"]].copy()
    bundle = {"gazetteer": g, "history": all_claims[HISTORY_COLS].copy(), "players": keep_players,
              "valuations": keep_vals, "categories": info["categories"], "fusion": fusion,
              "n_metadata": len(boosters), "bert_dirs": [f"bert_seed{s}" for s in seeds] if text else [],
              "mode": "fusion" if text else "metadata",
              "club_names": club_names or {}}
    with open(out / BUNDLE, "wb") as f:
        pickle.dump(bundle, f)
    from sklearn.metrics import roc_auc_score
    summary = {"fitted": time.strftime("%Y-%m-%d %H:%M"), "train": len(tr), "val": len(va),
               "mode": "fusion" if text else "metadata",
               "val_auc": {"metadata": roc_auc_score(va["label"], pm["val"]),
                           **({"bert": roc_auc_score(va["label"], pb),
                               "fusion": roc_auc_score(va["label"], fusion(scores))} if text else {})},
               "calibration_coef": fusion.coef(), "bert_runs": runs,
               "metadata_rounds": info["best_rounds"]}
    (out / "fit.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


# ---------------------------------------------------------------- score


@dataclass
class Scorer:
    g: Gazetteer
    names: Names
    history: pd.DataFrame
    players: pd.DataFrame
    valuations: pd.DataFrame
    categories: dict
    boosters: list
    berts: list
    fusion: Calibrator
    club_names: dict

    @property
    def snapshot(self) -> pd.Timestamp:
        """The day the Transfermarkt data ends (its last valuation)."""
        return pd.Timestamp(self.valuations["date"].max())

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "Scorer":
        import lightgbm as lgb
        path = Path(path)
        if not (path / BUNDLE).exists():
            raise SystemExit(f"No saved rumour model in {path}. Run: python -m models.rumours fit")
        with open(path / BUNDLE, "rb") as f:
            b = pickle.load(f)
        boosters = [lgb.Booster(model_file=str(path / f"metadata_{i}.txt")) for i in range(b["n_metadata"])]
        berts = [bert.load(path / d, device) for d in b["bert_dirs"]]
        return cls(b["gazetteer"], Names.from_gazetteer(b["gazetteer"], b["players"]), b["history"], b["players"],
                   b["valuations"], b["categories"], boosters, berts, b["fusion"], b.get("club_names", {}))

    def active(self, player_ids: pd.Series, dates: pd.Series) -> np.ndarray:
        """Whether each player had a market value in the ACTIVE_DAYS before the date (or before the snapshot)."""
        if not hasattr(self, "_last_value"):
            self._last_value = self.valuations.groupby("player_id")["date"].max()
        last = player_ids.map(self._last_value)
        ref = dates.clip(upper=self.snapshot)
        return (last.notna() & (last >= ref - pd.Timedelta(days=ACTIVE_DAYS))).to_numpy()

    def claims(self, headlines: pd.DataFrame) -> pd.DataFrame:
        """headlines: url, site, date and a title (or an archive slug). One row per (headline, player, club)
        claim, every stance, dated up to the snapshot, about players active then."""
        slug = headlines["slug"] if "slug" in headlines else headlines["title"].map(headline_slug)
        dates = pd.to_datetime(headlines["date"])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        arts = headlines.assign(slug=slug, date=dates.dt.normalize())
        arts = arts[arts["date"] <= self.snapshot]
        c = make_claims(arts[["url", "site", "date", "slug"]], self.g)
        return c[self.active(c["player_id"], c["date"])].reset_index(drop=True) if len(c) else c

    def score(self, headlines: pd.DataFrame) -> pd.DataFrame:
        """Claims of the headlines with `p` (fusion probability) for rumours; NaN for other stances."""
        c = self.claims(headlines)
        if c.empty:
            return pd.DataFrame(columns=["url", "site", "date", "slug", "player_id", "club_id", "current_club_id",
                                         "stance", "label", "move_date", "p", "p_metadata", "p_bert", "text"])
        rumours = c[c["stance"] == "rumour"]
        c["p"] = np.nan
        c["p_metadata"] = np.nan
        c["p_bert"] = np.nan
        if len(rumours):
            # news volume counts every claim of this batch too, whatever its stance, as in training
            history = pd.concat([self.history, c[HISTORY_COLS]], ignore_index=True)
            f = featurize(rumours, history, self.g, self.names, self.players, self.valuations)
            pm = metadata_predict(self.boosters, self.categories, f)
            if self.berts:
                pb = np.mean([bert.predict(m, t, f["text"].tolist(), d, bert.HPARAMS["max_len"])
                              for m, t, d in self.berts], axis=0)
                c.loc[rumours.index, "p"] = self.fusion([pm, pb])
                c.loc[rumours.index, "p_bert"] = pb
            else:
                c.loc[rumours.index, "p"] = self.fusion([pm])
            c.loc[rumours.index, "p_metadata"] = pm
            c.loc[rumours.index, "text"] = f["text"].to_numpy()
        return c

    def player_name(self, pid: int) -> str:
        if not hasattr(self, "_pname"):
            self._pname = dict(zip(self.players["player_id"].astype(int), self.players["name"]))
        return self._pname.get(int(pid), str(pid))

    def club_name(self, cid) -> str:
        if cid is not None and not (isinstance(cid, float) and np.isnan(cid)) and int(cid) in self.club_names:
            return self.club_names[int(cid)]
        if not hasattr(self, "_cname"):
            best: dict[int, tuple] = {}
            for alias, i in self.g.club_alias.items():
                name = " ".join(alias)
                if i not in best or len(name) > len(best[i]):
                    best[i] = name
            self._cname = {i: n.title() for i, n in best.items()}
        if cid is None or (isinstance(cid, float) and np.isnan(cid)):
            return "–"
        return self._cname.get(int(cid), str(cid))


def reported_moves(claims: pd.DataFrame, after: pd.Timestamp) -> pd.DataFrame:
    """Moves the archive reports as done after `after`: (player, club) in announcement headlines with
    completed-move words, from enough articles and papers. `date` is the day the last needed article appeared.
    Used only to check predictions, never as an input."""
    a = claims[(claims["stance"] == "announcement") & (claims["date"] > after)]
    a = a[a["slug"].str.contains(OFFICIAL)].drop_duplicates(["url", "player_id", "club_id"]).sort_values("date")
    rows = []
    for (pid, cid), d in a.groupby(["player_id", "club_id"]):
        if d["url"].nunique() >= REPORTED_MIN_ARTICLES and d["site"].nunique() >= REPORTED_MIN_PAPERS:
            rows.append({"player_id": int(pid), "club_id": int(cid), "date": d["date"].iloc[REPORTED_MIN_ARTICLES - 1],
                         "articles": d["url"].nunique(), "papers": d["site"].nunique()})
    return pd.DataFrame(rows, columns=["player_id", "club_id", "date", "articles", "papers"])


def outcome(c: pd.DataFrame, snapshot: pd.Timestamp, reported: pd.DataFrame | None = None,
            archive_end: pd.Timestamp | None = None, horizon_days: int = 180) -> pd.DataFrame:
    """What happened within 180 days of each claim, as far as can be told: `happened` (1/0/NaN when the window
    is still open) and a readable `outcome`. Transfermarkt's record first; after its end, moves the archive
    reports as done (`reported`). Windows that run past the last day anything is known stay open."""
    dates = pd.to_datetime(c["date"])
    end = dates + pd.Timedelta(days=horizon_days)
    known_until = archive_end if (reported is not None and archive_end is not None) else snapshot
    tm = c["label"] == 1
    news = pd.Series(False, index=c.index)
    news_day = pd.Series(pd.NaT, index=c.index, dtype="datetime64[ns]")
    if reported is not None and len(reported):
        key = c["player_id"].astype(int).astype(str) + "-" + c["club_id"].astype(int).astype(str)
        rk = dict(zip(reported["player_id"].astype(str) + "-" + reported["club_id"].astype(str), reported["date"]))
        news_day = pd.to_datetime(key.map(rk))
        news = news_day.notna() & (news_day <= end) & ~tm
    happened = pd.Series(np.nan, index=c.index)
    happened[tm | news] = 1.0
    happened[~(tm | news) & (end <= known_until)] = 0.0
    text = pd.Series(f"open: nothing reported by {known_until:%d %b %Y}", index=c.index)
    text[happened == 0] = "did not happen"
    text[tm] = "happened (Transfermarkt, " + pd.to_datetime(c.loc[tm, "move_date"]).dt.strftime("%d %b %Y") + ")"
    text[news] = "happened (reported done in news, " + news_day[news].dt.strftime("%d %b %Y") + ")"
    return pd.DataFrame({"happened": happened, "outcome": text}, index=c.index)


def pairs(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, club) rumour: how often and how widely it was reported, the latest and highest
    probability, and the outcome."""
    r = scored[scored["stance"] == "rumour"].sort_values("date")
    g = r.groupby(["player_id", "club_id"], sort=False)
    out = g.agg(player=("player", "last"), club=("club", "last"), current_club=("current_club", "last"),
                articles=("url", "nunique"), papers=("site", "nunique"), first_seen=("date", "min"),
                last_seen=("date", "max"), latest_p=("p", "last"), highest_p=("p", "max"),
                happened=("happened", "max"), outcome=("outcome", "last")).reset_index()
    return out.sort_values("last_seen", ascending=False).reset_index(drop=True)


def reliability(y: pd.Series, p: pd.Series, bands=(0, 0.1, 0.2, 0.3, 0.5, 1.0)) -> pd.DataFrame:
    known = y.notna()
    b = pd.cut(p[known], list(bands), include_lowest=True)
    t = pd.DataFrame({"y": y[known], "p": p[known], "band": b}).groupby("band", observed=True).agg(
        claims=("y", "size"), predicted=("p", "mean"), happened=("y", "mean"))
    t.index = t.index.astype(str)
    return t
