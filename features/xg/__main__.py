"""Command line: python -m features.xg <command>

  train   extract shots, evaluate all models (leave one competition out), fit the
          chosen model on every shot, save it and write the report
  score   predict xG for every shot in the database and store it in `shots`

Both read events from PostgreSQL by default; `train --source raw` reads the raw
Wyscout JSON instead (no database needed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from etl.config import DEFAULT_WYSCOUT_DIR, REPO_ROOT

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "artifacts" / "xg.joblib"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "xg"


def _load_shots(source: str, raw_dir: Path, database_url: str | None):
    import pandas as pd

    from .shots import collect_shots, iter_matches_db, iter_matches_raw

    started = time.perf_counter()
    if source == "raw":
        records = collect_shots(iter_matches_raw(raw_dir))
    else:
        from etl.db import connect
        with connect(database_url) as conn:
            records = collect_shots(iter_matches_db(conn))
    shots = pd.DataFrame(records)
    if shots.empty:
        raise SystemExit("No shots found. Load the data first: python -m etl load-wyscout")
    print(f"{len(shots):,} shots, {int(shots['is_goal'].sum()):,} goals from {shots['competition'].nunique()} "
          f"competitions ({time.perf_counter() - started:.0f}s)")
    return shots


def cmd_train(args) -> int:
    import joblib

    from .model import bootstrap_log_loss_diff, evaluate, fit_final
    from .report import write_report

    shots = _load_shots(args.source, args.raw_dir, args.database_url)

    print("Evaluating (leave one competition out)...")
    ev = evaluate(shots, seed=args.seed)
    print(ev.pooled[["log_loss", "brier", "roc_auc", "ece", "xg_over_goals"]].round(4).to_string())
    print(f"Chosen by lowest out-of-fold log loss: {ev.best}")

    match_ids = shots.reset_index(drop=True)["match_id"]
    others = [m for m in ("lr_full", "lr_geometry") if m != ev.best]
    boot = {f"{ev.best} - {o}": bootstrap_log_loss_diff(ev, match_ids, ev.best, o) for o in others}
    for pair, (mean, lo, hi) in boot.items():
        print(f"  {pair}: {mean:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")

    final = fit_final(shots, ev.best, seed=args.seed, cv=ev.pooled.loc[ev.best].to_dict())
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, args.model_path)
    meta = {"model": final.name, "trained_at": final.trained_at, "n_train": final.n_train,
            "penalty_xg": final.penalty_xg, "cv": final.cv}
    args.model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved model to {args.model_path}")

    report = write_report(ev, final, shots, args.report_dir, boot)
    print(f"Wrote report to {report}")
    return 0


def cmd_score(args) -> int:
    import joblib

    from etl.db import connect, init_db

    from .store import write_shots

    if not args.model_path.exists():
        raise SystemExit(f"No model at {args.model_path}. Run: python -m features.xg train")
    model = joblib.load(args.model_path)
    shots = _load_shots("db", args.raw_dir, args.database_url)
    shots["xg"] = model.predict(shots)
    shots["xg_model"] = f"{model.name}@{model.trained_at}"
    with connect(args.database_url) as conn:
        init_db(conn)
        n = write_shots(conn, shots)
    print(f"Stored xG for {n:,} shots (model {model.name}); "
          f"total xG {shots['xg'].sum():,.1f} vs {int(shots['is_goal'].sum()):,} goals.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.xg", description="Expected goals model")
    parser.add_argument("--database-url", help="overrides DATABASE_URL")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_WYSCOUT_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="evaluate, fit and save the xG model, write the report")
    tr.add_argument("--source", choices=("db", "raw"), default="db")
    tr.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    tr.add_argument("--seed", type=int, default=0)

    sub.add_parser("score", help="store xG for every shot in the database")

    args = parser.parse_args(argv)
    return cmd_train(args) if args.command == "train" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
