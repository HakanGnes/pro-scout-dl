"""Test-period metrics, pair-clustered bootstrap intervals and the written report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    both = len(set(y)) > 1
    return {"auc": float(roc_auc_score(y, p)) if both else float("nan"),
            "avg_precision": float(average_precision_score(y, p)) if both else float("nan"),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6),
                                                                                  labels=[0, 1])),
            "mean_p": float(np.mean(p)), "rate": float(np.mean(y)), "n": int(len(y))}


def _cluster_index(clusters: np.ndarray):
    codes, uniq = pd.factorize(clusters)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    return members


def auc_differences(y: np.ndarray, preds: dict[str, np.ndarray], pairs: list[tuple[str, str]],
                    clusters: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, tuple]:
    """AUC(a) - AUC(b) with 95% intervals from resampling whole clusters (player-club pairs)."""
    members = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    draws = {f"{a} - {b}": [] for a, b in pairs}
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        yb = y[idx]
        if len(set(yb)) < 2:
            continue
        auc = {m: roc_auc_score(yb, preds[m][idx]) for m in {x for p in pairs for x in p}}
        for a, b in pairs:
            draws[f"{a} - {b}"].append(auc[a] - auc[b])
    out = {}
    for a, b in pairs:
        d = roc_auc_score(y, preds[a]) - roc_auc_score(y, preds[b])
        lo, hi = np.percentile(draws[f"{a} - {b}"], [2.5, 97.5])
        out[f"{a} - {b}"] = (float(d), float(lo), float(hi))
    return out


def by_group(frame: pd.DataFrame, preds: dict[str, np.ndarray], col: str) -> pd.DataFrame:
    rows = []
    for key, idx in frame.groupby(col).indices.items():
        y = frame["label"].to_numpy()[idx]
        row = {col: key, "n": len(idx), "rate": round(float(y.mean()), 3)}
        for m, p in preds.items():
            row[m] = round(roc_auc_score(y, p[idx]), 3) if len(set(y)) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    return pd.DataFrame({"mean_p": p, "rate": y, "bin": q}).groupby("bin").agg(
        n=("rate", "size"), mean_p=("mean_p", "mean"), rate=("rate", "mean")).round(3)


def _fmt(v: tuple) -> str:
    return f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"


def write(report_dir: Path, results: dict, groups: dict[str, pd.DataFrame], calib: pd.DataFrame) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    for name, df in groups.items():
        df.to_csv(report_dir / f"auc_by_{name}.csv", index=False)
    calib.to_csv(report_dir / "reliability_fusion.csv")
    t = results["test"]
    lines = [
        "# Transfer rumour credibility: results",
        "",
        f"Pre-registration: [`{results['prereg']}`](../../{results['prereg']}). Run at commit "
        f"`{results.get('commit', '?')}` on {results['finished']}.",
        "",
        f"Test period 2025: {t['base']['n']:,} rumour claims, {results['test_pairs']:,} player-club pairs, "
        f"{t['base']['rate']:.1%} came true within 180 days.",
        "",
        "## Primary outcome",
        "",
        f"AUC(fusion) - AUC(metadata) = {_fmt(results['primary'])} (95% interval, {results['n_boot']:,} bootstrap resamples of "
        f"player-club pairs). **{'Supported' if results['primary'][1] > 0 else 'Not supported'}.**",
        "",
        "## All models on the test period (probabilities recalibrated on validation)",
        "",
        "| model | AUC | avg precision | Brier | log loss | mean p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for m, r in t.items():
        lines.append(f"| {m} | {r['auc']:.4f} | {r['avg_precision']:.4f} | {r['brier']:.4f} | {r['log_loss']:.4f} | "
                     f"{r['mean_p']:.3f} |")
    lines += ["", "## Secondary comparisons (AUC difference, 95% interval)", ""]
    lines += [f"- {k}: {_fmt(v)}" for k, v in results["secondary"].items()]
    lines += ["", "Validation AUC, fitted settings and per-paper / per-month tables: `results.json`, `auc_by_*.csv`.", ""]
    path = report_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
