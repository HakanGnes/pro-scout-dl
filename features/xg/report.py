"""Evaluation report: Markdown + static figures under reports/xg/."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import Evaluation, XGModel, feature_importance

# Reference palette (light surface); categorical slots in fixed order.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"lr_geometry": "#2a78d6", "lr_full": "#eb6834", "lgbm": "#1baf7a"}
MARKERS = {"lr_geometry": "o", "lr_full": "s", "lgbm": "D"}
LABELS = {
    "baseline": "Constant rate",
    "lr_geometry": "Logistic: distance + angle",
    "lr_full": "Logistic: all features",
    "lgbm": "LightGBM: all features",
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8,
    })


def calibration_figure(ev: Evaluation, path: Path, bins: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    oof = ev.oof[ev.oof["shot_type"] != "penalty"]
    y = oof["is_goal"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 0.7], [0, 0.7], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.66, 0.62, "perfect calibration", color=MUTED, fontsize=8.5, rotation=38, ha="right")
    for name in ("lr_geometry", "lr_full", "lgbm"):
        p = oof[f"p_{name}"].to_numpy()
        order = np.argsort(p)
        chunks = np.array_split(order, bins)
        px = [p[c].mean() for c in chunks]
        py = [y[c].mean() for c in chunks]
        ax.plot(px, py, color=SERIES[name], lw=2, marker=MARKERS[name], ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[name], zorder=3)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("Predicted xG (mean of bin)")
    ax.set_ylabel("Observed goal rate")
    ax.set_title("Out-of-fold calibration, non-penalty shots", loc="left")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_2)
    fig.text(0.01, 0.005, f"{bins} equal-count bins of {len(y) // bins:,} shots; leave-one-competition-out predictions",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def xg_map_figure(xg: XGModel, path: Path) -> None:
    """Model xG over the attacking half for a footed open-play shot after a simple pass."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from .shots import shot_geometry

    _style(plt)
    xs, ys = np.linspace(60, 99.5, 80), np.linspace(0.5, 99.5, 100)
    gx, gy = np.meshgrid(xs, ys)
    geo = np.array([shot_geometry(x, y) for x, y in zip(gx.ravel(), gy.ravel())])
    grid = pd.DataFrame({
        "shot_type": "open_play", "distance_m": geo[:, 0], "angle_rad": geo[:, 1],
        "x": gx.ravel(), "y": gy.ravel(), "seconds_since_prev": 1.6, "minute": 45.0, "score_diff": 0,
        "is_header": False, "is_counter": False, "same_player_prev": False,
        "assist_type": "pass", "set_piece": "none",
    })
    z = xg.predict(grid).reshape(gx.shape)

    # Landscape half pitch, goal at the top: horizontal = across the pitch (m), vertical = towards goal (m).
    across, depth = gy * 0.68, gx * 1.05
    levels = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    cmap = ListedColormap(SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.grid(False)
    cs = ax.contourf(across, depth, z, levels=levels, cmap=cmap, norm=BoundaryNorm(levels, cmap.N))
    line = dict(color=SURFACE, lw=1.4)
    for half_width, box_depth in ((20.16, 16.5), (9.16, 5.5)):  # penalty area, goal area
        ax.plot([34 - half_width, 34 - half_width, 34 + half_width, 34 + half_width],
                [105, 105 - box_depth, 105 - box_depth, 105], **line)
    ax.plot([0, 68], [105, 105], **line)
    ax.plot([34 - 3.66, 34 + 3.66], [105, 105], color=INK, lw=4, solid_capstyle="butt")
    ax.scatter([34], [94], s=12, color=SURFACE, zorder=3)
    ax.set_xlim(0, 68)
    ax.set_ylim(63, 106)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    cbar = fig.colorbar(cs, ax=ax, fraction=0.03, pad=0.02, ticks=levels[:-1])
    cbar.set_label("xG", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_title("Where a footed shot after a pass is worth what", loc="left")
    fig.text(0.01, 0.01, f"Model: {LABELS.get(xg.name, xg.name)}. Last 42 m of the pitch, goal at the top; "
             "other features at typical values.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def importance_figure(imp: pd.Series, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    imp = imp[imp > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values * 100, color=SERIES["lr_geometry"], height=0.62)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(imp.values * 100):
        ax.text(v + 0.4, i, f"{v:.1f}%", va="center", color=INK_2, fontsize=8.5)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title("What the LightGBM model relies on", loc="left")
    ax.set_xlim(0, imp.max() * 100 * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _md_table(df: pd.DataFrame, floatfmt: dict[str, str]) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                cells.append("–")
            elif c in floatfmt and isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(format(v, floatfmt[c]))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(ev: Evaluation, final: XGModel, shots: pd.DataFrame, out_dir: Path,
                 bootstrap: dict[str, tuple[float, float, float]] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_figure(ev, out_dir / "calibration.png")
    xg_map_figure(final, out_dir / "xg_map.png")
    imp = feature_importance(final)
    if imp is not None:
        importance_figure(imp, out_dir / "feature_importance.png")

    pooled = ev.pooled.reset_index()
    pooled.insert(1, "description", pooled["model"].map(LABELS))
    pooled = pooled[["model", "description", "log_loss", "brier", "roc_auc", "ece", "xg_over_goals"]]
    # A constant prediction has no ranking ability; its pooled AUC is an artefact of per-fold constants.
    pooled.loc[pooled["model"] == "baseline", "roc_auc"] = np.nan
    fmt = {"log_loss": ".4f", "brier": ".4f", "roc_auc": ".3f", "ece": ".4f", "xg_over_goals": ".3f"}

    folds = ev.per_fold.pivot(index="competition", columns="model", values="log_loss")
    folds = folds[[m for m in ("baseline", "lr_geometry", "lr_full", "lgbm") if m in folds]].reset_index()
    goals = ev.per_fold[ev.per_fold["model"] == final.name][["competition", "n", "goals", "xg_over_goals"]]
    folds = folds.merge(goals, on="competition")

    counts = shots.groupby("shot_type")["is_goal"].agg(["size", "sum", "mean"]).reset_index()
    counts.columns = ["shot_type", "shots", "goals", "goal_rate"]

    lines = [
        "# Expected goals (xG) — evaluation",
        "",
        f"Model in use: **{LABELS.get(final.name, final.name)}** (`{final.name}`), trained {final.trained_at} "
        f"on {final.n_train:,} shots. Penalties get a fixed xG of {final.penalty_xg:.3f}, "
        "the conversion rate of the training penalties.",
        "",
        "## Data",
        "",
        "Wyscout public dataset, 2017/18: five leagues, World Cup 2018, Euro 2016. "
        f"{len(shots):,} shots, {int(shots['is_goal'].sum()):,} goals. "
        f"{int(shots.get('coords_mirrored', pd.Series(dtype=bool)).sum())} shots recorded on the wrong half "
        "were mirrored back (see `features/xg/shots.py`).",
        "",
        _md_table(counts, {"goal_rate": ".3f"}),
        "",
        "Features use only what is known before the strike: shot location (distance, angle, lateral offset), "
        "header or foot, how the chance was created (pass, cross, through ball, dribble, rebound, loose ball, "
        "aerial duel), set-piece origin, counter-attack, time since the team's previous action, minute and "
        "score difference. Wyscout tags that describe the outcome are excluded: `opportunity` (201) sits on "
        "99% of goals, plus accuracy (1801/1802), goal-mouth position (1201–1223) and `blocked` (2101).",
        "",
        "## Validation",
        "",
        "Leave one competition out: each of the seven competitions is predicted by models trained on the other "
        "six. The model is chosen by a rule fixed in advance: lowest pooled out-of-fold log loss. "
        "Metrics below cover non-penalty shots.",
        "",
        _md_table(pooled, fmt),
        "",
        "- `log_loss`, `brier`: lower is better. `roc_auc`: ranking quality. `ece`: mean gap between predicted "
        "and observed rates over 10 equal-count bins. `xg_over_goals`: total xG / total goals (1.0 = unbiased).",
        "- The constant model is the floor: any useful model must beat its log loss.",
    ]
    if bootstrap:
        lines += ["", "Paired difference in per-shot log loss, 95% interval from 2,000 bootstrap resamples of matches:", ""]
        for pair, (mean, lo, hi) in bootstrap.items():
            lines.append(f"- {pair}: {mean:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    lines += [
        "",
        "### By held-out competition (log loss; goals vs xG for the model in use)",
        "",
        _md_table(folds, {c: ".4f" for c in ("baseline", "lr_geometry", "lr_full", "lgbm")} | {"xg_over_goals": ".3f"}),
        "",
        "## Figures",
        "",
        "![Calibration](calibration.png)",
        "",
        "![xG map](xg_map.png)",
    ]
    if imp is not None:
        lines += ["", "![Feature importance](feature_importance.png)"]
    lines += [
        "",
        "## Limitations",
        "",
        "- Event data has no defender or goalkeeper positions, so a shot under pressure and an open shot from "
        "the same spot get the same xG. Freeze-frame data (e.g. StatsBomb) would address this.",
        "- One season per competition. Leave-one-competition-out tests transfer across leagues, not across time.",
        "- Own goals carry no xG: team xG will sit slightly below goals by construction.",
        "",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
