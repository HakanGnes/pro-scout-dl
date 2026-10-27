"""Models 1-3: persistence, age curve, LightGBM."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from features.market_value.dataset import Samples
from features.market_value.tabular import tabular

MIN_CELL = 20  # age-curve cells with fewer training samples fall back to the age-only median, then the global one

LGBM_PARAMS = {
    "objective": "l1", "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 50,
    "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1, "n_estimators": 5000, "verbose": -1,
}
EARLY_STOPPING = 200


def persistence(s: Samples) -> np.ndarray:
    return np.zeros(len(s))


class AgeCurve:
    def fit(self, s: Samples) -> "AgeCurve":
        df = pd.DataFrame({"age": np.floor(s.meta["age"]).astype(int), "pos": s.meta["position"], "y": s.y})
        cell = df.groupby(["age", "pos"])["y"].agg(["median", "size"])
        self.cell = cell.loc[cell["size"] >= MIN_CELL, "median"].to_dict()
        by_age = df.groupby("age")["y"].agg(["median", "size"])
        self.by_age = by_age.loc[by_age["size"] >= MIN_CELL, "median"].to_dict()
        self.overall = float(df["y"].median())
        return self

    def predict(self, s: Samples) -> np.ndarray:
        ages = np.floor(s.meta["age"]).astype(int).to_numpy()
        return np.array([self.cell.get((a, p), self.by_age.get(a, self.overall))
                         for a, p in zip(ages, s.meta["position"])])


def fit_lgbm(train: Samples, val: Samples, seeds: list[int], log=print):
    import lightgbm as lgb

    xt, xv = tabular(train), tabular(val)
    models = []
    for seed in seeds:
        started = time.perf_counter()
        m = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=seed)
        with warnings.catch_warnings():  # newer LightGBM renames eval_set; both spellings work
            warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")
            m.fit(xt, train.y, eval_set=[(xv, val.y)], eval_metric="l1",
                  callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        models.append(m)
        log(f"    lgbm seed {seed}: best round {m.best_iteration_:4d}, val MAE {m.best_score_['valid_0']['l1']:.4f}, "
            f"{time.perf_counter() - started:.0f}s")
    return models


def predict_lgbm(models, s: Samples) -> np.ndarray:
    x = tabular(s)
    return np.mean([m.predict(x, num_iteration=m.best_iteration_) for m in models], axis=0)


def lgbm_importance(models) -> pd.Series:
    imp = np.mean([m.booster_.feature_importance("gain") for m in models], axis=0)
    return pd.Series(imp / imp.sum(), index=models[0].booster_.feature_name()).sort_values(ascending=False)
