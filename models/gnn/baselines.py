"""Non-graph baselines and shared regression metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features.pass_networks.graph import BASELINE_FEATURES, VOLUME_FEATURES


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(y)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "spearman": float(spearmanr(y, p)[0]) if len(y) > 2 and np.ptp(p) > 1e-12 else float("nan"),
        "bias": float(err.mean()),
    }


def _xy(frame: pd.DataFrame, cols) -> tuple[np.ndarray, np.ndarray]:
    return frame[list(cols)].to_numpy(dtype=float), frame["target"].to_numpy(dtype=float)


def fit_predict_mean(train: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    mu = float(train["target"].mean())
    return {k: np.full(len(f), mu) for k, f in frames.items()}


def fit_predict_volume(train: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Ridge regression on log pass volumes: how far does 'more passes, more chances' go?"""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    X, y = _xy(train, VOLUME_FEATURES)
    model = make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    return {k: np.clip(model.predict(_xy(f, VOLUME_FEATURES)[0]), 0, None) for k, f in frames.items()}


def fit_predict_lgbm(train: pd.DataFrame, val: pd.DataFrame, frames: dict[str, pd.DataFrame],
                     seed: int = 0) -> tuple[dict[str, np.ndarray], object]:
    """LightGBM on hand-crafted graph statistics; early stopping on the validation weeks."""
    import lightgbm as lgb

    X, y = _xy(train, BASELINE_FEATURES)
    Xv, yv = _xy(val, BASELINE_FEATURES)
    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, verbose=-1,
    )
    model.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return {k: np.clip(model.predict(_xy(f, BASELINE_FEATURES)[0]), 0, None) for k, f in frames.items()}, model
