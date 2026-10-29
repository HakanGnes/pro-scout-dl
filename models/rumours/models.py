"""The non-neural models of the rumour study and the shared calibration / fusion step."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion

from .dataset import META_CATEGORICAL, META_NUMERIC

EPS = 1e-6
TFIDF_C = (0.1, 0.3, 1.0, 3.0, 10.0)
LGBM_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=50,
                   colsample_bytree=0.8, subsample=0.8, subsample_freq=1, n_estimators=3000, verbose=-1)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def site_rates(train: pd.DataFrame, frame: pd.DataFrame, prior: float = 50.0) -> np.ndarray:
    """Each paper's training hit rate, shrunk towards the overall rate with `prior` pseudo-claims."""
    base = train["label"].mean()
    g = train.groupby("site")["label"].agg(["sum", "count"])
    rate = (g["sum"] + prior * base) / (g["count"] + prior)
    return frame["site"].map(rate).fillna(base).to_numpy(float)


def _meta_matrix(frame: pd.DataFrame, cats: dict[str, list]) -> pd.DataFrame:
    x = frame[META_NUMERIC].astype(float).copy()
    for c in META_CATEGORICAL:
        x[c] = pd.Categorical(frame[c], categories=cats[c])
    return x


def metadata_gbm(train: pd.DataFrame, val: pd.DataFrame, frames: dict[str, pd.DataFrame],
                 seeds: list[int]) -> tuple[dict[str, np.ndarray], dict]:
    """LightGBM on the metadata inputs; early stopping on validation AUC; seed predictions averaged."""
    import lightgbm as lgb

    cats = {c: sorted(train[c].astype(str).unique()) for c in META_CATEGORICAL}
    xt, xv = _meta_matrix(train, cats), _meta_matrix(val, cats)
    out = {k: np.zeros(len(f)) for k, f in frames.items()}
    rounds = []
    for s in seeds:
        m = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(xt, train["label"], eval_set=[(xv, val["label"])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False)])
        rounds.append(int(m.best_iteration_ or m.n_estimators))
        for k, f in frames.items():
            out[k] += m.predict_proba(_meta_matrix(f, cats))[:, 1] / len(seeds)
    imp = dict(zip(xt.columns, m.booster_.feature_importance("gain").round(1).tolist()))
    return out, {"best_rounds": rounds, "gain_last_seed": imp}


def tfidf_lr(train: pd.DataFrame, val: pd.DataFrame, frames: dict[str, pd.DataFrame],
             text: str = "text") -> tuple[dict[str, np.ndarray], dict]:
    """Word 1-2 grams and character 3-5 grams, logistic regression; C picked on validation AUC."""
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    xt = vec.fit_transform(train[text])
    xv = vec.transform(val[text])
    scores = {}
    for c in TFIDF_C:
        lr = LogisticRegression(C=c, max_iter=5000)
        lr.fit(xt, train["label"])
        scores[c] = roc_auc_score(val["label"], lr.predict_proba(xv)[:, 1])
    best = max(scores, key=scores.get)
    lr = LogisticRegression(C=best, max_iter=5000).fit(xt, train["label"])
    out = {k: lr.predict_proba(vec.transform(f[text]))[:, 1] for k, f in frames.items()}
    return out, {"C": best, "val_auc_by_C": {str(k): round(v, 4) for k, v in scores.items()}}


class Calibrator:
    """Logistic recalibration of one or more model scores, fitted on the validation period.

    One score: Platt scaling of a model's probability. Two scores: the fusion model."""

    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e6, max_iter=1000)

    def fit(self, scores: list[np.ndarray], y: np.ndarray) -> "Calibrator":
        self.lr.fit(np.column_stack([logit(s) for s in scores]), y)
        return self

    def __call__(self, scores: list[np.ndarray]) -> np.ndarray:
        return self.lr.predict_proba(np.column_stack([logit(s) for s in scores]))[:, 1]

    def coef(self) -> list[float]:
        return [round(float(v), 4) for v in [*self.lr.coef_[0], self.lr.intercept_[0]]]
