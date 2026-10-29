"""Command line: python -m features.rumours claims

Links every archived transfer headline to Transfermarkt players and clubs, labels each (player, club)
claim with whether the move happened within a year, and saves articles and claims to data/processed/.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from etl.config import DEFAULT_NEWS_DIR, DEFAULT_TRANSFERMARKT_DIR, REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "rumours"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.rumours", description="Transfer rumour claims")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("claims", help="link headlines to players and clubs and label the claims")
    c.add_argument("--history-dir", type=Path, default=DEFAULT_NEWS_DIR / "history")
    c.add_argument("--transfermarkt-dir", type=Path, default=DEFAULT_TRANSFERMARKT_DIR)
    c.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from .build import build

    started = time.perf_counter()
    arts, claims = build(args.history_dir, args.transfermarkt_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    arts.to_pickle(args.out / "articles.pkl")
    claims.to_pickle(args.out / "claims.pkl")
    print(f"{len(arts):,} articles, {claims['url'].nunique():,} with a claim, {len(claims):,} claims "
          f"({time.perf_counter() - started:.0f}s)")
    if len(claims):
        by = claims.assign(year=claims["date"].dt.year).groupby(["year", "stance"])["label"].agg(["size", "mean"])
        print(by.unstack("stance").to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"Wrote {args.out / 'articles.pkl'} and {args.out / 'claims.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
