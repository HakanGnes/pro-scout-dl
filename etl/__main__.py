"""Command line: python -m etl <command>

  init-db            create tables and indexes (safe to re-run)
  download-wyscout   download the Wyscout public dataset from figshare
  load-wyscout       load the raw files into PostgreSQL (skips unchanged files)
  quality            run data-quality checks; exit code 1 on any FAIL
  reset-db --yes     drop every project table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_WYSCOUT_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m etl", description="Pro-Scout DL data pipeline")
    parser.add_argument("--database-url", help="overrides DATABASE_URL")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create tables and indexes")

    dl = sub.add_parser("download-wyscout", help="download the Wyscout dataset from figshare")
    dl.add_argument("--raw-dir", type=Path, default=DEFAULT_WYSCOUT_DIR)
    dl.add_argument("--only", nargs="+", metavar="NAME",
                    help="download only files whose name contains one of these words, e.g. --only events matches")

    ld = sub.add_parser("load-wyscout", help="load the raw files into the database")
    ld.add_argument("--raw-dir", type=Path, default=DEFAULT_WYSCOUT_DIR)
    ld.add_argument("--force", action="store_true", help="reload files even if unchanged")

    sub.add_parser("quality", help="run data-quality checks")

    rs = sub.add_parser("reset-db", help="drop every project table")
    rs.add_argument("--yes", action="store_true", help="confirm; without it nothing is dropped")

    args = parser.parse_args(argv)

    if args.command == "download-wyscout":
        from .wyscout.download import download_wyscout
        download_wyscout(args.raw_dir, args.only)
        return 0

    from .db import connect, init_db, reset_db

    with connect(args.database_url) as conn:
        if args.command == "init-db":
            init_db(conn)
            print("Schema is up to date.")
        elif args.command == "load-wyscout":
            from .wyscout.load import load_wyscout
            init_db(conn)
            results = load_wyscout(conn, args.raw_dir, force=args.force)
            loaded = [r for r in results if r.status == "loaded"]
            print(f"Done: {len(loaded)} file(s) loaded, "
                  f"{sum(r.status == 'skipped' for r in results)} unchanged, "
                  f"{sum(r.rows for r in loaded):,} rows, {sum(r.seconds for r in loaded):.0f}s.")
        elif args.command == "quality":
            from .quality import format_report, has_failures, run_checks
            init_db(conn)  # an empty schema reports "no data" instead of crashing
            checks = run_checks(conn)
            print(format_report(checks))
            return 1 if has_failures(checks) else 0
        elif args.command == "reset-db":
            if not args.yes:
                print("Refusing to drop tables without --yes.", file=sys.stderr)
                return 2
            reset_db(conn)
            print("All project tables dropped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
