"""Command line: python -m models.market_value run

The pre-registered Phase 3 study (docs/preregistration/2026-10-27-market-value.md), in one run:
persistence, age curve, LightGBM, LSTM and Transformer (5 seeds each for the trained models), then the
registered outcomes on the test anchors. Writes reports/market_value/ and saves the models.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from etl.config import REPO_ROOT

DATASET = REPO_ROOT / "data" / "processed" / "market_value.pkl"
ARTIFACTS = REPO_ROOT / "models" / "artifacts" / "market_value"
REPORT_DIR = REPO_ROOT / "reports" / "market_value"
PREDICTIONS = REPO_ROOT / "data" / "processed" / "market_value_predictions.csv.gz"
PREREG = "docs/preregistration/2026-10-27-market-value.md"

SEEDS = [0, 1, 2, 3, 4]
N_BOOT, BOOT_SEED = 2000, 0
# The snapshot the study was registered on; a different snapshot would be a different study.
REGISTERED_FILES = {
    "player_valuations.csv.gz": "608e32e5ad2231074f3505c469ad727be66d7e6efda6af918afd85379439631d",
    "players.csv.gz": "d22e407981d5b51a79bf8ff59835729f3526f7dc3495a6d8ed2f852ed2e86403",
    "appearances.csv.gz": "e096c2b158fc3d52c4550526c4d4850dd62f31007652f4bcae112eadb1646802",
    "transfers.csv.gz": "90326983daf7e6ac7aabdfe62b90936d9bf1dd2171e53dec75c3751eb1620a83",
    "competitions.csv.gz": "8924ddfbc0e9989f4a42a3c32ebb6faa671614625086d7fa84353f955352ea88",
}
MODELS = ("persistence", "age_curve", "lgbm", "lstm", "transformer")


def check_snapshot(info: dict, strict: bool = True) -> None:
    files = info.get("files") or {}
    wrong = {n: h for n, h in REGISTERED_FILES.items() if files.get(n) != h}
    if wrong and strict:
        raise SystemExit("The dataset was not built from the registered snapshot "
                         f"(differs: {', '.join(sorted(wrong))}). Rebuild: python -m features.market_value build")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m models.market_value", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="train and evaluate every model as registered")
    r.add_argument("--dataset", type=Path, default=DATASET)
    r.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    r.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    r.add_argument("--predictions", type=Path, default=PREDICTIONS)
    r.add_argument("--device", default="auto")
    r.add_argument("--max-epochs", type=int, help="tests only; the registered run uses the default")
    r.add_argument("--seeds", type=int, nargs="+", help="tests only; the registered run uses 0-4")
    r.add_argument("--allow-other-snapshot", action="store_true", help="tests only")
    args = parser.parse_args(argv)

    import numpy as np
    import pandas as pd
    import torch

    from features.market_value.dataset import load

    from . import baselines, nets, report
    from .stats import cluster_bootstrap, metrics, paired

    started = time.perf_counter()
    samples = load(args.dataset)
    check_snapshot(samples.info, strict=not args.allow_other_snapshot)
    seeds = args.seeds or SEEDS
    cfg = dict(nets.CFG, **({"max_epochs": args.max_epochs} if args.max_epochs else {}))
    device = nets.pick_device(args.device)
    split = {s: samples.split(s) for s in ("train", "val", "test")}
    for s, d in split.items():
        print(f"{s:<5} {len(d):>7,} samples, {d.meta['player_id'].nunique():>6,} players")
    print(f"Device: {device}. Seeds: {seeds}.")

    preds: dict[str, dict[str, np.ndarray]] = {}
    preds["persistence"] = {s: baselines.persistence(d) for s, d in split.items()}
    age = baselines.AgeCurve().fit(split["train"])
    preds["age_curve"] = {s: age.predict(d) for s, d in split.items()}

    print("LightGBM...")
    gbms = baselines.fit_lgbm(split["train"], split["val"], seeds)
    preds["lgbm"] = {s: baselines.predict_lgbm(gbms, d) for s, d in split.items()}

    norm = nets.Normalizer(split["train"])
    tensors = {s: norm.tensors(d) for s, d in split.items()}
    runs: dict[str, list] = {}
    for arch in nets.ARCHITECTURES:
        print(f"{arch}...")
        runs[arch] = [nets.fit(arch, tensors["train"], tensors["val"], norm.static_dim, seed, device, cfg)
                      for seed in seeds]
        loaded = [nets.load_net(r.state_dict, arch, norm.static_dim, cfg, device) for r in runs[arch]]
        preds[arch] = {s: np.mean([nets.predict(n, tensors[s], device) for n in loaded], axis=0) for s in split}

    # ---------------------------------------------------------------- registered outcomes
    test = split["test"]
    y, players = test.y, test.meta["player_id"].to_numpy()
    boot = {"n_boot": N_BOOT, "seed": BOOT_SEED}
    primary = paired(y, preds["transformer"]["test"], preds["lgbm"]["test"], players, **boot)
    success = primary[2] < 0
    secondary = {
        "LSTM − LightGBM": paired(y, preds["lstm"]["test"], preds["lgbm"]["test"], players, **boot),
        "Transformer − LSTM": paired(y, preds["transformer"]["test"], preds["lstm"]["test"], players, **boot),
        **{f"{report.LABEL[m]} − persistence": paired(y, preds[m]["test"], preds["persistence"]["test"], players, **boot)
           for m in MODELS if m != "persistence"},
    }
    table = {m: {s: metrics(split[s].y, preds[m][s]) for s in split} for m in MODELS}
    mae_ci = {m: cluster_bootstrap(np.abs(preds[m]["test"] - y), players, **boot) for m in MODELS}
    groups = report.subgroups(test, {m: preds[m]["test"] for m in MODELS})
    by_anchor = report.by_anchor(test, {m: preds[m]["test"] for m in MODELS})

    print(f"Primary: Transformer − LightGBM MAE {primary[0]:+.4f}, 95% CI [{primary[1]:+.4f}, {primary[2]:+.4f}] -> "
          f"{'SUCCESS' if success else 'not supported'}")
    for k, v in secondary.items():
        print(f"  {k}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")
    print(pd.DataFrame({report.LABEL[m]: table[m]["test"] for m in MODELS}).T.round(4).to_string())

    # ---------------------------------------------------------------- save
    args.artifacts.mkdir(parents=True, exist_ok=True)
    for seed, m in zip(seeds, gbms):
        m.booster_.save_model(str(args.artifacts / f"lgbm_seed{seed}.txt"), num_iteration=m.best_iteration_)
    torch.save({"format": 1, "cfg": cfg, "normalizer": norm.state(), "static_dim": norm.static_dim,
                "archs": {a: [r.state_dict for r in rs] for a, rs in runs.items()}, "preregistration": PREREG},
               args.artifacts / "nets.pt")
    (args.artifacts / "age_curve.json").write_text(json.dumps(
        {"cells": {f"{a}|{p}": v for (a, p), v in age.cell.items()}, "by_age": {str(a): v for a, v in age.by_age.items()},
         "overall": age.overall}, indent=1), encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    out = test.meta[["player_id", "name", "anchor", "value_now", "value_target"]].assign(y=y)
    for m in MODELS:
        out[f"pred_{m}"] = preds[m]["test"]
    out.to_csv(args.predictions, index=False)

    results = {
        "preregistration": PREREG, "primary": {"transformer_minus_lgbm": primary, "success": bool(success)},
        "secondary": secondary, "metrics": table, "test_mae_ci": mae_ci,
        "lgbm_rounds": [int(m.best_iteration_) for m in gbms],
        "runs": {a: [{"seed": r.seed, "best_epoch": r.best_epoch, "val_mae": r.val_mae, "seconds": round(r.seconds)}
                     for r in rs] for a, rs in runs.items()},
        "dataset": {"files": samples.info.get("files"), "anchors": samples.info.get("anchors")},
        "seeds": seeds, "device": str(device), "cfg": cfg, "seconds": round(time.perf_counter() - started),
    }
    (args.artifacts / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    report.write(args.report_dir, results, groups, by_anchor, baselines.lgbm_importance(gbms))
    print(f"Wrote {args.report_dir / 'README.md'} ({time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
