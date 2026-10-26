"""xG models: preparation, leave-one-competition-out evaluation, final fit.

Penalties are not modelled: every penalty gets the conversion rate of the
training penalties. All other shots (open play and direct free kicks) go
through the model.

The model used downstream is chosen by a rule fixed before looking at the
results: lowest pooled out-of-fold log loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ASSIST_TYPES = ["pass", "dribble", "cross", "aerial", "loose_ball", "through_ball", "rebound", "set_piece", "other"]
SET_PIECE_TYPES = ["none", "corner", "free_kick", "throw_in"]

NUMERIC = ["distance_m", "angle_rad", "log_distance", "x", "y_offset_m", "seconds_since_prev", "minute", "score_diff"]
BINARY = ["is_header", "is_counter", "same_player_prev", "is_free_kick"]
CATEGORICAL = ["assist_type", "set_piece"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
GEOMETRY = ["distance_m", "angle_rad", "log_distance", "is_header"]

MODEL_NAMES = ("baseline", "lr_geometry", "lr_full", "lgbm")


# ---------------------------------------------------------------- data


def prepare(shots: pd.DataFrame) -> pd.DataFrame:
    """Model matrix from shot records (one row per non-penalty shot)."""
    X = pd.DataFrame(index=shots.index)
    X["distance_m"] = shots["distance_m"].astype(float)
    X["angle_rad"] = shots["angle_rad"].astype(float)
    X["log_distance"] = np.log1p(X["distance_m"])
    X["x"] = shots["x"].astype(float)
    X["y_offset_m"] = (shots["y"].astype(float) - 50.0).abs() * 0.68
    X["seconds_since_prev"] = shots["seconds_since_prev"].astype(float)
    X["minute"] = shots["minute"].astype(float)
    X["score_diff"] = shots["score_diff"].astype(float)
    for col in ("is_header", "is_counter", "same_player_prev"):
        X[col] = shots[col].astype(bool).astype(int)
    X["is_free_kick"] = (shots["shot_type"] == "free_kick").astype(int)
    X["assist_type"] = pd.Categorical(shots["assist_type"], categories=ASSIST_TYPES)
    X["set_piece"] = pd.Categorical(shots["set_piece"], categories=SET_PIECE_TYPES)
    return X


def split_penalties(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_pen = shots["shot_type"] == "penalty"
    return shots[~is_pen], shots[is_pen]


# ---------------------------------------------------------------- models


class ConstantModel:
    """Predicts the training goal rate for every shot."""

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


def make_model(name: str, seed: int = 0):
    if name == "baseline":
        return ConstantModel()
    if name == "lr_geometry":
        return Pipeline([
            ("cols", ColumnTransformer([("num", StandardScaler(), GEOMETRY)])),
            ("lr", LogisticRegression(max_iter=1000)),
        ])
    if name == "lr_full":
        return Pipeline([
            ("cols", ColumnTransformer([
                ("num", StandardScaler(), NUMERIC),
                ("bin", "passthrough", BINARY),
                ("cat", OneHotEncoder(categories=[ASSIST_TYPES, SET_PIECE_TYPES], handle_unknown="ignore"),
                 CATEGORICAL),
            ])),
            ("lr", LogisticRegression(max_iter=2000, C=1.0)),
        ])
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, verbose=-1,
        )
    raise ValueError(f"unknown model {name!r}")


def _fit(model, X: pd.DataFrame, y: np.ndarray):
    return model.fit(X[FEATURES], y)


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURES])[:, 1]


# ---------------------------------------------------------------- metrics


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Mean |observed - predicted| over equal-count bins, weighted by bin size."""
    order = np.argsort(p)
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) * abs(y[chunk].mean() - p[chunk].mean())
    return total / len(y)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "goals": int(y.sum()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else math.nan,
        "ece": float(expected_calibration_error(y, p)),
        "xg_over_goals": float(p.sum() / y.sum()) if y.sum() else math.nan,
    }


# ---------------------------------------------------------------- evaluation


@dataclass
class Evaluation:
    oof: pd.DataFrame                                  # one row per shot: competition, is_goal, shot_type, p_<model>
    pooled: pd.DataFrame                               # model -> metrics on all non-penalty shots
    per_fold: pd.DataFrame                             # (model, competition) -> metrics
    penalty_rate: dict[str, float] = field(default_factory=dict)
    best: str = ""


def evaluate(shots: pd.DataFrame, models: tuple[str, ...] = MODEL_NAMES, seed: int = 0) -> Evaluation:
    """Leave-one-competition-out: train on all other competitions, predict the held-out one."""
    shots = shots.reset_index(drop=True)
    comps = sorted(shots["competition"].unique())
    oof = shots[["event_id", "competition", "shot_type", "is_goal"]].copy()
    for name in models:
        oof[f"p_{name}"] = np.nan
    penalty_rate: dict[str, float] = {}

    for comp in comps:
        train, test = shots[shots["competition"] != comp], shots[shots["competition"] == comp]
        tr_np, tr_pen = split_penalties(train)
        te_np, te_pen = split_penalties(test)
        X_tr, y_tr = prepare(tr_np), tr_np["is_goal"].astype(int).to_numpy()
        X_te = prepare(te_np)
        pen = float(tr_pen["is_goal"].mean()) if len(tr_pen) else 0.75
        penalty_rate[comp] = pen
        for name in models:
            model = _fit(make_model(name, seed), X_tr, y_tr)
            oof.loc[te_np.index, f"p_{name}"] = _predict(model, X_te)
            oof.loc[te_pen.index, f"p_{name}"] = pen

    pooled, folds = [], []
    np_mask = oof["shot_type"] != "penalty"
    y_all = oof.loc[np_mask, "is_goal"].astype(int).to_numpy()
    for name in models:
        pooled.append({"model": name, **metrics(y_all, oof.loc[np_mask, f"p_{name}"].to_numpy())})
        for comp in comps:
            m = np_mask & (oof["competition"] == comp)
            folds.append({"model": name, "competition": comp,
                          **metrics(oof.loc[m, "is_goal"].astype(int).to_numpy(), oof.loc[m, f"p_{name}"].to_numpy())})
    pooled_df = pd.DataFrame(pooled).set_index("model")
    best = str(pooled_df["log_loss"].idxmin())
    return Evaluation(oof, pooled_df, pd.DataFrame(folds), penalty_rate, best)


def bootstrap_log_loss_diff(ev: Evaluation, match_ids: pd.Series, a: str, b: str,
                            n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean per-shot log-loss difference a - b with a 95% interval, resampling whole matches.

    `match_ids` is aligned with `ev.oof` (same row order). Negative = `a` is better.
    """
    keep = (ev.oof["shot_type"] != "penalty").to_numpy()
    y = ev.oof["is_goal"].astype(int).to_numpy()[keep]

    def per_shot(name: str) -> np.ndarray:
        p = np.clip(ev.oof[f"p_{name}"].to_numpy()[keep], 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d = per_shot(a) - per_shot(b)
    codes, uniq = pd.factorize(match_ids.to_numpy()[keep])
    sums, counts = np.bincount(codes, weights=d), np.bincount(codes)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(uniq), len(uniq))
        boots[i] = sums[idx].sum() / counts[idx].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# ---------------------------------------------------------------- final model


@dataclass
class XGModel:
    name: str
    model: object
    penalty_xg: float
    trained_at: str
    n_train: int
    cv: dict = field(default_factory=dict)

    def predict(self, shots: pd.DataFrame) -> np.ndarray:
        """xG for every shot, penalties included."""
        out = np.full(len(shots), self.penalty_xg, dtype=float)
        is_pen = (shots["shot_type"] == "penalty").to_numpy()
        if (~is_pen).any():
            out[~is_pen] = _predict(self.model, prepare(shots[~is_pen]))
        return out


def fit_final(shots: pd.DataFrame, name: str, seed: int = 0, cv: dict | None = None) -> XGModel:
    non_pen, pens = split_penalties(shots)
    model = _fit(make_model(name, seed), prepare(non_pen), non_pen["is_goal"].astype(int).to_numpy())
    return XGModel(
        name=name,
        model=model,
        penalty_xg=float(pens["is_goal"].mean()) if len(pens) else 0.75,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_train=int(len(shots)),
        cv=cv or {},
    )


def feature_importance(xg: XGModel) -> pd.Series | None:
    """Gain importance for tree models, None otherwise."""
    booster = getattr(xg.model, "booster_", None)
    if booster is None:
        return None
    gain = booster.feature_importance(importance_type="gain")
    return pd.Series(gain / gain.sum(), index=booster.feature_name()).sort_values(ascending=False)
