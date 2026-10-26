"""Report for the passing-network models: Markdown + one figure under reports/pass_network_gnn/."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ACCENT, QUIET = "#2a78d6", "#86b6ef"

LABELS = {
    "mean": "Training mean",
    "volume": "Ridge on pass volume",
    "lgbm_stats": "LightGBM on 19 graph statistics",
    "deepsets": "DeepSets: players, no passes between them",
    "gnn": "GNN: players + who passes to whom",
}
ORDER = ("mean", "volume", "lgbm_stats", "deepsets", "gnn")
SHORT = {"volume": "Ridge: pass volume", "lgbm_stats": "LightGBM: graph stats",
         "deepsets": "DeepSets: players only", "gnn": "GNN: players + passes"}


def _fmt(v, spec):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return format(v, spec)


def mae_figure(ci: pd.DataFrame, chosen: str, path: Path) -> None:
    """Test MAE per model with 95% bootstrap intervals; the chosen model highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    })
    ci = ci.set_index("model")
    mean_mae = ci.loc["mean", "mae"] if "mean" in ci.index else None
    # The training mean sits far to the right and would squash the comparison; it is quoted in the note.
    ci = ci.loc[[m for m in ORDER if m in ci.index and m != "mean"][::-1]]
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(ci) + 1.7))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for i, (name, r) in enumerate(ci.iterrows()):
        color = ACCENT if name == chosen else QUIET
        ax.plot([r["lo"], r["hi"]], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.plot([r["mae"]], [i], "o", ms=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(r["hi"] + 0.002, i, f"{r['mae']:.3f}", va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(ci)), [SHORT.get(m, m) for m in ci.index], color=INK_2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean absolute error, team npxG per match (lower is better)")
    lo, hi = ci["lo"].min(), ci["hi"].max()
    span = hi - lo
    ax.set_xlim(lo - 0.15 * span, hi + 0.25 * span)
    fig.text(0.01, 0.97, "Test weeks: how close each model gets to a team's xG", fontsize=12,
             fontweight="bold", va="top")
    note = "Dots: error on the last 30% of gameweeks; lines: 95% interval, resampling matches."
    note += f"\nHighlighted: chosen on validation weeks." + (
        f" Predicting the training mean: {mean_mae:.3f}." if mean_mae is not None else "")
    fig.text(0.01, 0.01, note, color=MUTED, fontsize=7.5, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(out_dir: Path, *, counts: dict, table: pd.DataFrame, neural: pd.DataFrame,
                 ci: pd.DataFrame, pairs: dict, chosen: str, cfg: dict, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mae_figure(ci, chosen, out_dir / "test_mae.png")

    def rows(split: str) -> list[str]:
        t = table[table["split"] == split].set_index("model")
        out = ["| model | description | MAE | RMSE | R² | Spearman |", "| --- | --- | --- | --- | --- | --- |"]
        for m in ORDER:
            if m in t.index:
                r = t.loc[m]
                name = f"**{m}**" if m == chosen else m
                out.append(f"| {name} | {LABELS[m]} | {_fmt(r['mae'], '.3f')} | {_fmt(r['rmse'], '.3f')} | "
                           f"{_fmt(r['r2'], '.3f')} | {_fmt(r['spearman'], '.3f')} |")
        return out

    n = neural[neural["split"] == "test"].set_index("model")
    lines = [
        "# Passing networks → team xG",
        "",
        "Can a team's passing network in a match tell us how many chances it created? Each team in each "
        "match is a graph: players are nodes, completed passes between them are directed, weighted edges. "
        "The target is the team's **non-penalty xG** in that match, from the project's xG model "
        f"(`{meta.get('xg_model', '?')}`). Shots never enter the features.",
        "",
        "## Data and split",
        "",
        f"{sum(counts.values()):,} team-match graphs. The split is by time and fixed in advance: in each league the "
        "first 70% of gameweeks are for fitting (the last 5 of those for early stopping) and the last 30% for "
        "testing; World Cup 2018 and Euro 2016 are a separate test set the models never train on.",
        "",
        "| split | graphs |",
        "| --- | --- |",
        *[f"| {k} | {v:,} |" for k, v in counts.items()],
        "",
        "## Models",
        "",
        "Five models of increasing structure, all fitted on the same training weeks. DeepSets and the GNN see "
        "the same player features; only the GNN also sees who passes to whom. If the GNN beats DeepSets, the "
        "network structure carries information beyond each player's own numbers.",
        "",
        f"Neural models: {len(meta.get('seeds', []))} seeds each, hidden size {cfg['hidden']}, "
        f"{cfg['layers']} GINE layers (GNN), Adam lr {cfg['lr']}, weight decay {cfg['weight_decay']}, "
        f"early stopping on validation MSE (patience {cfg['patience']}). Their rows below are the "
        "average of the seeds' predictions.",
        "",
        f"The model carried into Phase 2 is chosen on the **validation weeks** (lowest MAE): **{chosen}**.",
        "",
        "## Results: league test weeks",
        "",
        *rows("test"),
        "",
        *(["## Results: World Cup + Euro (never trained on)", "", *rows("test_intl"), ""]
          if counts.get("test_intl") else []),
        "## Is the difference real?",
        "",
        "Paired difference in absolute error on the league test weeks (negative = first model better), "
        "95% interval from 2,000 bootstrap resamples of matches:",
        "",
        *[f"- {k}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for k, (m, lo, hi) in pairs.items()],
        "",
        "Seed-to-seed spread on the test weeks (single models, not the ensemble):",
        "",
        "| model | MAE mean ± sd | R² mean ± sd | mean best epoch |",
        "| --- | --- | --- | --- |",
        *[f"| {m} | {r['mae_mean']:.3f} ± {r['mae_sd']:.3f} | {r['r2_mean']:.3f} ± {r['r2_sd']:.3f} | "
          f"{r['best_epoch_mean']:.0f} |" for m, r in n.iterrows()],
        "",
        "![Test MAE by model](test_mae.png)",
        "",
        "## Limitations",
        "",
        "- A match's xG is noisy: much of it depends on a few chances, so even a perfect model of the passing "
        "network explains only part of it.",
        "- The receiver of a pass is inferred from the next event (found for 99% of accurate passes).",
        "- One season per league; the xG model used for the target was fitted on all shots of that season.",
        "- The graph describes the same match whose xG it predicts; it explains a performance, it does not "
        "forecast the next one. Phase 2 uses it for counterfactual player swaps.",
        "",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
