"""Command line: python -m features.market_value build [--source db|raw]

Builds one sample per (player, anchor date) with the 36-month history, static inputs and the
12-month log value change as target, and saves them to data/processed/market_value.pkl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from etl.config import DEFAULT_TRANSFERMARKT_DIR, REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "market_value.pkl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.market_value", description="Market-value dataset")
    parser.add_argument("--database-url", help="overrides DATABASE_URL")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build samples, monthly histories and targets")
    b.add_argument("--source", choices=("db", "raw"), default="db")
    b.add_argument("--raw-dir", type=Path, default=DEFAULT_TRANSFERMARKT_DIR)
    b.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    import numpy as np

    from etl.transfermarkt.tables import read_all

    from . import dataset

    started = time.perf_counter()
    if args.source == "raw":
        tables = read_all(args.raw_dir)
        manifest = args.raw_dir / "manifest.json"
        files = json.loads(manifest.read_text(encoding="utf-8"))["files"] if manifest.exists() else []
    else:
        from etl.db import connect
        with connect(args.database_url) as conn:
            if conn.execute("SELECT count(*) FROM tm_player_valuations").fetchone()[0] == 0:
                raise SystemExit("No Transfermarkt data in the database. Run: python -m etl load-transfermarkt")
            tables = read_all(conn)
            files = [{"name": n, "sha256": h} for n, h in conn.execute(
                "SELECT file_name, sha256 FROM etl_loads WHERE source = 'transfermarkt' ORDER BY 1").fetchall()]
    print(f"Read {sum(len(t) for t in tables.values()):,} rows from {args.source} in {time.perf_counter() - started:.0f}s")

    samples = dataset.build(tables)
    dataset.save(samples, args.out, {"source": args.source, "files": {f["name"]: f["sha256"] for f in files}})

    for split in ("train", "val", "test"):
        s = samples.split(split)
        print(f"  {split:<5} {len(s):>7,} samples, {s.meta['player_id'].nunique():>6,} players, "
              f"target mean {s.y.mean():+.3f} sd {s.y.std():.3f}, unchanged {np.mean(s.y == 0):.1%}")
    print(f"Wrote {args.out} ({len(samples):,} samples, {time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
