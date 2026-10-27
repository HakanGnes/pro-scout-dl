"""Subgroup tables, per-anchor errors and the Markdown report for the Phase 3 study."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.market_value.dataset import Samples

LABEL = {"persistence": "Persistence", "age_curve": "Age curve", "lgbm": "LightGBM", "lstm": "LSTM",
         "transformer": "Transformer"}
# Categorical slots 1-5 of the default validated palette (light surface); line ends are labelled
# directly and every number is also in a table, as the sub-3:1 slots require.
COLORS = {"persistence": "#2a78d6", "age_curve": "#eb6834", "lgbm": "#1baf7a", "lstm": "#eda100",
          "transformer": "#e87ba4"}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _bands(test: Samples) -> dict[str, pd.Series]:
    m = test.meta
    return {
        "value at t": pd.cut(m["value_now"], [0, 1e6, 5e6, 20e6, np.inf], right=False,
                             labels=["< €1m", "€1–5m", "€5–20m", "≥ €20m"]),
        "age at t": pd.cut(m["age"], [0, 22, 26, 30, 99], right=False, labels=["≤ 21", "22–25", "26–29", "≥ 30"]),
        "position": m["position"],
        "league": m["league"],
        "value changed": pd.Series(np.where(test.y == 0, "unchanged", "changed"), index=m.index),
    }


def subgroups(test: Samples, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    err = {k: np.abs(p - test.y) for k, p in preds.items()}
    for kind, labels in _bands(test).items():
        for level in (labels.cat.categories if hasattr(labels, "cat") else sorted(labels.dropna().unique())):
            mask = (labels == level).to_numpy()
            if mask.sum() == 0:
                continue
            rows.append({"group": kind, "level": str(level), "n": int(mask.sum()),
                         **{LABEL[k]: float(e[mask].mean()) for k, e in err.items()}})
    return pd.DataFrame(rows)


def by_anchor(test: Samples, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    a = test.meta["anchor"].dt.date.to_numpy()
    rows = []
    for anchor in sorted(set(a)):
        mask = a == anchor
        rows.append({"anchor": anchor.isoformat(), "n": int(mask.sum()),
                     **{LABEL[k]: float(np.abs(p[mask] - test.y[mask]).mean()) for k, p in preds.items()}})
    return pd.DataFrame(rows)


def _plot(path: Path, results: dict, anchors: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(LABEL)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)

    # Left: test MAE with player-clustered 95% intervals (one series, so no legend).
    ci = results["test_mae_ci"]
    ys = np.arange(len(models))[::-1]
    for yv, m in zip(ys, models):
        mean, lo, hi = ci[m]
        ax1.plot([lo, hi], [yv, yv], color="#2a78d6", lw=2, solid_capstyle="round")
        ax1.plot(mean, yv, "o", ms=8, color="#2a78d6", mec=SURFACE, mew=2)
        ax1.text(hi + 0.004, yv, f"{mean:.3f}", va="center", fontsize=9, color=INK)
    ax1.set_yticks(ys, [LABEL[m] for m in models])
    ax1.set_xlabel("Test MAE of log value (95% interval)", color=INK, fontsize=9)
    ax1.grid(axis="x", color=GRID, lw=0.8)
    ax1.set_axisbelow(True)
    ax1.set_title("Error one year ahead", loc="left", fontsize=11, color=INK)
    lo_all = min(v[1] for v in ci.values())
    hi_all = max(v[2] for v in ci.values())
    ax1.set_xlim(lo_all - 0.02, hi_all + 0.05)

    # Right: test MAE per anchor date, one line per model, labelled at the line end.
    x = np.arange(len(anchors))
    for m in models:
        ax2.plot(x, anchors[LABEL[m]], color=COLORS[m], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5)
    ends = sorted(((anchors[LABEL[m]].iloc[-1], m) for m in models))
    span = max(anchors[[LABEL[m] for m in models]].max().max() - anchors[[LABEL[m] for m in models]].min().min(), 1e-6)
    placed = []
    for v, m in ends:  # nudge labels apart so they do not collide
        pos = max(v, placed[-1] + span * 0.07) if placed else v
        placed.append(pos)
        ax2.text(x[-1] + 0.12, pos, LABEL[m], va="center", fontsize=9, color=INK)
    ax2.set_xticks(x, [a[:7] for a in anchors["anchor"]])
    ax2.set_xlim(-0.2, x[-1] + 1.1)
    ax2.set_ylabel("Test MAE", color=INK, fontsize=9)
    ax2.grid(axis="y", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.set_title("By prediction date", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _fmt(v: tuple) -> str:
    return f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"


def write(report_dir: Path, results: dict, groups: pd.DataFrame, anchors: pd.DataFrame,
          importance: pd.Series) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    _plot(report_dir / "test_mae.png", results, anchors)
    prim = results["primary"]["transformer_minus_lgbm"]
    ok = results["primary"]["success"]
    m = results["metrics"]
    models = list(LABEL)

    def table(split: str) -> list[str]:
        out = ["| model | MAE | RMSE | median abs. % error | within ±20% | bias |", "| --- | --- | --- | --- | --- | --- |"]
        for k in models:
            r = m[k][split]
            out.append(f"| {LABEL[k]} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['median_ape']:.1%} | "
                       f"{r['within_20pct']:.1%} | {r['bias']:+.3f} |")
        return out

    def frame(df: pd.DataFrame, cols: list[str]) -> list[str]:
        head = "| " + " | ".join(cols) + " |"
        out = [head, "|" + " --- |" * len(cols)]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else f"{r[c]:,}" if isinstance(r[c], int)
                                          else str(r[c]) for c in cols) + " |")
        return out

    n_test = m["persistence"]["test"]["n"]
    lines = [
        "# Market value one year ahead: GBM vs. LSTM vs. Transformer",
        "",
        f"Pre-registered study: [`{results['preregistration']}`](../../{results['preregistration']}). One run, "
        f"{results['seconds'] / 60:.1f} minutes, seeds {results['seeds']}, device `{results.get('device', '?')}`.",
        "",
        "## Primary outcome",
        "",
        f"Transformer − LightGBM test MAE: **{_fmt(prim)}** (95% interval from {2000:,} resamples of players).",
        "",
        f"Success criterion (upper end below 0): **{'met' if ok else 'not met'}**. "
        + ("The Transformer reading the raw monthly history beats gradient boosting on hand-made summaries."
           if ok else "The Transformer does not beat LightGBM; LightGBM is the model used downstream."),
        "",
        "## Secondary outcomes (not used to decide)",
        "",
        *[f"- {k}: {_fmt(v)}" for k, v in results["secondary"].items()],
        "",
        f"## Test anchors ({n_test:,} samples)",
        "",
        "MAE is on the log scale: 0.30 means the predicted value is typically about 35% off "
        "(exp(0.30) − 1). Median absolute % error and the share within ±20% read the same predictions as euros.",
        "",
        *table("test"),
        "",
        "![Test MAE](test_mae.png)",
        "",
        "## By prediction date",
        "",
        *frame(anchors, ["anchor", "n", *[LABEL[k] for k in models]]),
        "",
        "## By group (test MAE)",
        "",
        *frame(groups, ["group", "level", "n", *[LABEL[k] for k in models]]),
        "",
        "## Validation anchors",
        "",
        *table("val"),
        "",
        "## Training details",
        "",
        f"- LightGBM best rounds: {results['lgbm_rounds']}.",
        *[f"- {LABEL[a]}: " + ", ".join(f"seed {r['seed']} epoch {r['best_epoch']} (val MAE {r['val_mae']:.4f}, "
                                        f"{r['seconds']}s)" for r in rs) for a, rs in results["runs"].items()],
        "- LightGBM feature importance (share of gain, top 12): "
        + ", ".join(f"{k} {v:.1%}" for k, v in importance.head(12).items()) + ".",
        "",
    ]
    path = report_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
