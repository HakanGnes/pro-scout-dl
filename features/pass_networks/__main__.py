"""Command line: python -m features.pass_networks build [--source db|raw]

Builds one passing network per team per match, attaches the team's non-penalty
xG (from the saved xG model) as the target, assigns the time-based split and
saves everything to data/processed/pass_networks.pkl.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from etl.config import DEFAULT_WYSCOUT_DIR, REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "pass_networks.pkl"
DEFAULT_XG_MODEL = REPO_ROOT / "models" / "artifacts" / "xg.joblib"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.pass_networks", description="Passing-network dataset")
    parser.add_argument("--database-url", help="overrides DATABASE_URL")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build graphs + targets + split")
    b.add_argument("--source", choices=("db", "raw"), default="db")
    b.add_argument("--raw-dir", type=Path, default=DEFAULT_WYSCOUT_DIR)
    b.add_argument("--xg-model", type=Path, default=DEFAULT_XG_MODEL)
    b.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    import joblib
    import numpy as np

    from features.events import iter_matches_db, iter_matches_raw, load_contexts_db, load_contexts_raw

    from . import dataset

    if not args.xg_model.exists():
        raise SystemExit(f"No xG model at {args.xg_model}. Run: python -m features.xg train")
    xg = joblib.load(args.xg_model)

    started = time.perf_counter()
    if args.source == "raw":
        contexts = load_contexts_raw(args.raw_dir)
        graphs, shots = dataset.build(iter_matches_raw(args.raw_dir), contexts)
    else:
        from etl.db import connect
        with connect(args.database_url) as conn:
            contexts = load_contexts_db(conn)
            graphs, shots = dataset.build(iter_matches_db(conn), contexts)
    dataset.attach_targets(graphs, shots, xg)
    counts = dataset.assign_splits(graphs)

    nodes = np.array([len(g.player_ids) for g in graphs])
    edges = np.array([g.edge_index.shape[1] for g in graphs])
    targets = np.array([g.target for g in graphs])
    print(f"{len(graphs):,} graphs from {len(graphs) // 2:,} matches in {time.perf_counter() - started:.0f}s")
    print(f"  nodes per graph: median {np.median(nodes):.0f} (min {nodes.min()}, max {nodes.max()}); "
          f"edges: median {np.median(edges):.0f} (min {edges.min()}, max {edges.max()})")
    print(f"  target npxG: mean {targets.mean():.3f}, sd {targets.std():.3f}, min {targets.min():.3f}, max {targets.max():.3f}")
    print("  split: " + ", ".join(f"{s}={counts.get(s, 0):,}" for s in dataset.SPLITS))

    dataset.save(graphs, args.out, meta={"source": args.source, "xg_model": f"{xg.name}@{xg.trained_at}"})
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
