"""Pre-registered experiment: does a GNN with richer edge features beat DeepSets?

    python -m models.gnn.exp_edge_features

Protocol: docs/preregistration/2026-10-26-gnn-edge-features.md (registered before this ran).

1. Load the Phase 1 networks (models/artifacts/pass_gnn.pt) and check that they reproduce the
   Phase 1 test MAEs; DeepSets' Phase 1 predictions are the comparison.
2. Train GNN-v2: the Phase 1 GNN with the same configuration and seeds 0–4, reading the six
   rich edge features instead of two. One run, nothing tuned.
3. Primary outcome: GNN-v2 − DeepSets absolute error on the league test weeks, 95% interval from
   2,000 bootstrap resamples of matches (seed 0). Success: the upper end is below 0.
4. Secondary (reported, not decisive): the same on World Cup + Euro; GNN-v2 − Phase 1 GNN.

Writes reports/pass_network_gnn/prereg_edge_features.md and saves GNN-v2 next to the Phase 1 model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

from etl.config import REPO_ROOT

DATASET = REPO_ROOT / "data" / "processed" / "pass_networks.pkl"
PHASE1_MODEL = REPO_ROOT / "models" / "artifacts" / "pass_gnn.pt"
PHASE1_REPORT = REPO_ROOT / "models" / "artifacts" / "pass_gnn.json"
OUT_MODEL = REPO_ROOT / "models" / "artifacts" / "pass_gnn_edge_v2.pt"
REPORT = REPO_ROOT / "reports" / "pass_network_gnn" / "prereg_edge_features.md"
PREREG = "docs/preregistration/2026-10-26-gnn-edge-features.md"

SEEDS = [0, 1, 2, 3, 4]
N_BOOT, BOOT_SEED = 2000, 0
REPRODUCE_TOL = 1e-3  # Phase 1 MAEs are stored to 4 decimals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m models.gnn.exp_edge_features", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--phase1-model", type=Path, default=PHASE1_MODEL)
    parser.add_argument("--phase1-report", type=Path, default=PHASE1_REPORT)
    parser.add_argument("--model-out", type=Path, default=OUT_MODEL)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=None, help="tests only; the registered run uses the default")
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    from features.pass_networks import dataset
    from features.pass_networks.graph import EDGE_FEATURES_RICH

    from .baselines import metrics
    from .ensemble import Ensemble
    from .train import DEFAULTS, paired_bootstrap, pick_device, run_neural

    started = time.perf_counter()
    graphs, meta = dataset.load(args.dataset)
    if any(g.edge_attr_rich is None for g in graphs):
        raise SystemExit("This dataset predates the rich edge features. Rebuild it: python -m features.pass_networks build")
    for p, cmd in ((args.phase1_model, "python -m models.gnn evaluate"), (args.phase1_report, "python -m models.gnn evaluate")):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run: {cmd}")

    by_split = {s: [g for g in graphs if g.split == s] for s in ("test", "test_intl")}
    y = {s: np.array([g.target for g in gs]) for s, gs in by_split.items() if gs}
    ids = {s: np.array([g.match_id for g in by_split[s]]) for s in y}

    # 1. Phase 1 networks must reproduce the Phase 1 report before anything is compared with them.
    phase1 = Ensemble.load(args.phase1_model, pick_device(args.device))
    p1_preds = {s: phase1.predict(by_split[s]) for s in y}
    names = np.array(phase1.names)
    ens = {arch: {s: p1_preds[s][np.char.startswith(names, arch)].mean(0) for s in y} for arch in ("deepsets", "gnn")}
    reported = json.loads(args.phase1_report.read_text(encoding="utf-8"))["test"]
    for arch in ("deepsets", "gnn"):
        got = metrics(y["test"], ens[arch]["test"])["mae"]
        want = reported[arch]["mae"]
        print(f"Phase 1 {arch}: test MAE {got:.4f} (reported {want:.4f})")
        if abs(got - want) > REPRODUCE_TOL:
            raise SystemExit(f"Phase 1 {arch} does not reproduce its reported test MAE; the dataset or model changed. "
                             "Stopping before the experiment.")

    # 2. GNN-v2: same configuration and seeds, rich edges.
    cfg = dict(DEFAULTS)
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    print(f"Training GNN-v2 ({len(EDGE_FEATURES_RICH)} edge features, seeds {SEEDS})...")
    results, norm = run_neural(graphs, ("gnn",), SEEDS, pick_device(args.device), cfg, edge_key="edge_attr_rich")
    v2 = {s: np.mean([r.predictions[s] for r in results], axis=0) for s in y}

    # 3–4. Outcomes, exactly as registered.
    primary = paired_bootstrap(y["test"], v2["test"], ens["deepsets"]["test"], ids["test"], N_BOOT, BOOT_SEED)
    success = primary[2] < 0
    secondary = {"GNN-v2 − DeepSets, World Cup + Euro":
                 paired_bootstrap(y["test_intl"], v2["test_intl"], ens["deepsets"]["test_intl"], ids["test_intl"],
                                  N_BOOT, BOOT_SEED) if "test_intl" in y else None,
                 "GNN-v2 − Phase 1 GNN, league test weeks":
                 paired_bootstrap(y["test"], v2["test"], ens["gnn"]["test"], ids["test"], N_BOOT, BOOT_SEED)}
    table = {name: {s: metrics(y[s], p[s]) for s in y}
             for name, p in (("DeepSets (Phase 1)", ens["deepsets"]), ("GNN (Phase 1)", ens["gnn"]), ("GNN-v2", v2))}

    print(f"Primary: GNN-v2 − DeepSets MAE {primary[0]:+.4f}, 95% CI [{primary[1]:+.4f}, {primary[2]:+.4f}] -> "
          f"{'SUCCESS' if success else 'not supported'}")
    for k, v in secondary.items():
        if v:
            print(f"  {k}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": 2, "cfg": cfg, "normalizer": norm.state(),
                "archs": {"gnn": [r.state_dict for r in results]}, "experiment": PREREG}, args.model_out)

    _write_report(args.report, primary, success, secondary, table, results, meta)
    print(f"Wrote {args.report} ({time.perf_counter() - started:.0f}s)")
    return 0


def _write_report(path: Path, primary, success, secondary, table, results, meta) -> None:
    def row(name, m):
        return f"| {name} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.3f} | {m['spearman']:.3f} |"

    lines = [
        "# Pre-registered experiment: richer edge features",
        "",
        f"Protocol: [`{PREREG}`](../../{PREREG}), registered before this run. Run on {date.today().isoformat()}; "
        f"target xG model `{meta.get('xg_model', '?')}`.",
        "",
        "## Primary outcome",
        "",
        f"GNN-v2 − DeepSets absolute error on the league test weeks: **{primary[0]:+.4f}**, 95% interval "
        f"[{primary[1]:+.4f}, {primary[2]:+.4f}] (2,000 bootstrap resamples of matches).",
        "",
        f"Success criterion (upper end below 0): **{'met' if success else 'not met'}**. "
        + ("The richer edges let the GNN beat DeepSets." if success else
           "Richer edge features do not make the GNN beat DeepSets; the Phase 1 conclusion stands."),
        "",
        "## Secondary outcomes (not used to decide)",
        "",
        *[f"- {k}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in secondary.items() if v],
        "",
        "## Test metrics",
        "",
        "| model | MAE | RMSE | R² | Spearman |",
        "| --- | --- | --- | --- | --- |",
        *[row(name, ms["test"]) for name, ms in table.items()],
        "",
    ]
    if any("test_intl" in ms for ms in table.values()):
        lines += ["World Cup + Euro:", "", "| model | MAE | RMSE | R² | Spearman |", "| --- | --- | --- | --- | --- |",
                  *[row(name, ms["test_intl"]) for name, ms in table.items() if "test_intl" in ms], ""]
    lines += [
        "GNN-v2 seeds: " + ", ".join(f"seed {r.seed} best epoch {r.best_epoch} (val MSE {r.val_mse:.4f})" for r in results)
        + ".",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
