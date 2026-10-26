"""Command line: python -m models.gnn evaluate

Loads data/processed/pass_networks.pkl (python -m features.pass_networks build),
fits the baselines and the neural models on the training weeks, reports on the
test weeks and the international tournaments, and saves the chosen network.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from etl.config import REPO_ROOT

DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "pass_networks.pkl"
DEFAULT_REPORT = REPO_ROOT / "reports" / "pass_network_gnn"
DEFAULT_MODEL = REPO_ROOT / "models" / "artifacts" / "pass_gnn.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m models.gnn", description="Passing-network models")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="baselines + DeepSets + GNN, report, save the chosen model")
    ev.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ev.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    ev.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    ev.add_argument("--seeds", type=int, default=5)
    ev.add_argument("--max-epochs", type=int, default=None)
    ev.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args(argv)

    import pandas as pd
    import torch

    from features.pass_networks import dataset

    from . import baselines as B
    from .report import write_report
    from .train import DEFAULTS, bootstrap_mae, paired_bootstrap, pick_device, run_neural, summarise

    if not args.dataset.exists():
        raise SystemExit(f"No dataset at {args.dataset}. Run: python -m features.pass_networks build")
    graphs, meta = dataset.load(args.dataset)
    cfg = dict(DEFAULTS)
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    seeds = list(range(args.seeds))
    device = pick_device(args.device)
    started = time.perf_counter()

    frame = dataset.stats_frame(graphs)
    frames = {s: frame[frame["split"] == s].reset_index(drop=True) for s in dataset.SPLITS}
    counts = {s: len(f) for s, f in frames.items()}
    print(f"{len(graphs):,} graphs; " + ", ".join(f"{k}={v:,}" for k, v in counts.items()) + f"; device {device}")
    eval_splits = tuple(s for s in ("val", "test", "test_intl") if counts[s])
    targets = {s: frames[s]["target"].to_numpy() for s in eval_splits}

    print("Baselines...")
    preds = {
        "mean": B.fit_predict_mean(frames["train"], {s: frames[s] for s in eval_splits}),
        "volume": B.fit_predict_volume(frames["train"], {s: frames[s] for s in eval_splits}),
    }
    preds["lgbm_stats"], _ = B.fit_predict_lgbm(frames["train"], frames["val"], {s: frames[s] for s in eval_splits})

    print(f"Neural models ({len(seeds)} seeds each)...")
    results, norm = run_neural(graphs, ("deepsets", "gnn"), seeds, device, cfg)
    neural, ensembles = summarise(results, targets)
    preds.update(ensembles)

    rows = [{"model": m, "split": s, **B.metrics(targets[s], p[s])} for m, p in preds.items() for s in eval_splits]
    table = pd.DataFrame(rows)
    val = table[table["split"] == "val"].set_index("model")
    chosen = str(val.drop(index="mean")["mae"].idxmin())
    print(table.pivot(index="model", columns="split", values=["mae", "r2"]).round(3).to_string())
    print(f"Chosen on validation MAE: {chosen}")

    # 95% intervals for each model's test MAE and for the key paired differences.
    test_ids = frames["test"]["match_id"].to_numpy()
    y = targets["test"]
    ci = pd.DataFrame([dict(zip(("model", "mae", "lo", "hi"), (m, *bootstrap_mae(y, p["test"], test_ids))))
                       for m, p in preds.items()])
    pairs = {f"{a} - {b}": paired_bootstrap(y, preds[a]["test"], preds[b]["test"], test_ids)
             for a, b in (("gnn", "deepsets"), ("gnn", "lgbm_stats"), ("deepsets", "lgbm_stats"))}
    for k, (m, lo, hi) in pairs.items():
        print(f"  {k}: {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    # Save every seed of the GNN (Phase 2 needs the graph model) plus what it takes to rebuild it.
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": "gnn", "cfg": cfg, "normalizer": norm.state(),
        "state_dicts": [r.state_dict for r in results if r.arch == "gnn"],
        "chosen_on_validation": chosen, "dataset_meta": meta,
    }, args.model_out)
    args.model_out.with_suffix(".json").write_text(json.dumps({
        "chosen_on_validation": chosen, "cfg": cfg, "seeds": seeds,
        "test": table[table["split"] == "test"].set_index("model")[["mae", "r2"]].round(4).to_dict("index"),
    }, indent=2), encoding="utf-8")
    print(f"Saved {args.model_out}")

    report = write_report(args.report_dir, counts=counts, table=table, neural=neural, ci=ci, pairs=pairs,
                          chosen=chosen, cfg=cfg, meta={**meta, "seeds": seeds})
    print(f"Wrote report to {report} ({time.perf_counter() - started:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
