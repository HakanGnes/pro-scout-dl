"""Command line: python -m etl <command>

  init-db            create tables and indexes (safe to re-run)
  download-wyscout   download the Wyscout public dataset from figshare
  load-wyscout       load the raw files into PostgreSQL (skips unchanged files)
  download-transfermarkt  download the Transfermarkt open dataset (Phase 3)
  load-transfermarkt      load the Transfermarkt files into PostgreSQL (skips unchanged files)
  probe-news         find which candidate news feeds work; writes etl/news/feeds.json
  collect-news       fetch the feeds once; archive new items (and insert them if the database is up)
  load-news          load the whole news archive into PostgreSQL
  probe-archive      sample Internet Archive URLs of news sites: are historical headlines usable?
  history-index      walk the sites' Internet Archive index; keep transfer-looking article URLs (resumable)
  history-titles     fetch real titles of those URLs from the archive (resumable)
  history-status     progress of the two history steps
  quality            run data-quality checks; exit code 1 on any FAIL
  reset-db --yes     drop every project table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_NEWS_DIR, DEFAULT_TRANSFERMARKT_DIR, DEFAULT_WYSCOUT_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m etl", description="Pro-Scout DL data pipeline")
    parser.add_argument("--database-url", help="overrides DATABASE_URL")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create tables and indexes")

    dl = sub.add_parser("download-wyscout", help="download the Wyscout dataset from figshare")
    dl.add_argument("--raw-dir", type=Path, default=DEFAULT_WYSCOUT_DIR)
    dl.add_argument("--only", nargs="+", metavar="NAME",
                    help="download only files whose name contains one of these words, e.g. --only events matches")

    dt = sub.add_parser("download-transfermarkt", help="download the Transfermarkt open dataset")
    dt.add_argument("--raw-dir", type=Path, default=DEFAULT_TRANSFERMARKT_DIR)
    dt.add_argument("--force", action="store_true", help="download again even if the files exist")

    ld = sub.add_parser("load-wyscout", help="load the raw files into the database")
    ld.add_argument("--raw-dir", type=Path, default=DEFAULT_WYSCOUT_DIR)
    ld.add_argument("--force", action="store_true", help="reload files even if unchanged")

    lt = sub.add_parser("load-transfermarkt", help="load the Transfermarkt files into the database")
    lt.add_argument("--raw-dir", type=Path, default=DEFAULT_TRANSFERMARKT_DIR)
    lt.add_argument("--force", action="store_true", help="reload files even if unchanged")

    sub.add_parser("probe-news", help="try candidate news feeds, write etl/news/feeds.json")
    cn = sub.add_parser("collect-news", help="fetch the news feeds once")
    cn.add_argument("--raw-dir", type=Path, default=DEFAULT_NEWS_DIR)
    cn.add_argument("--no-db", action="store_true", help="only write the archive")
    cn.add_argument("--log-file", type=Path, help="append output to this file (for the scheduled task)")
    ln = sub.add_parser("load-news", help="load the news archive into the database")
    ln.add_argument("--raw-dir", type=Path, default=DEFAULT_NEWS_DIR)

    pa = sub.add_parser("probe-archive", help="sample historical headlines from the Internet Archive")
    pa.add_argument("--raw-dir", type=Path, default=DEFAULT_NEWS_DIR)

    hi = sub.add_parser("history-index", help="walk the Internet Archive index of the news sites (resumable)")
    hi.add_argument("--raw-dir", type=Path, default=DEFAULT_NEWS_DIR)
    hi.add_argument("--pause", type=float, default=1.5, help="seconds between requests")
    ht = sub.add_parser("history-titles", help="fetch archived titles of the candidate URLs (resumable)")
    ht.add_argument("--raw-dir", type=Path, default=DEFAULT_NEWS_DIR)
    ht.add_argument("--transfermarkt-dir", type=Path, default=DEFAULT_TRANSFERMARKT_DIR)
    ht.add_argument("--pause", type=float, default=1.5, help="seconds between requests")
    ht.add_argument("--limit", type=int, help="stop after this many pages")
    hs = sub.add_parser("history-status", help="progress of history-index and history-titles")
    hs.add_argument("--raw-dir", type=Path, default=DEFAULT_NEWS_DIR)

    sub.add_parser("quality", help="run data-quality checks")

    rs = sub.add_parser("reset-db", help="drop every project table")
    rs.add_argument("--yes", action="store_true", help="confirm; without it nothing is dropped")

    args = parser.parse_args(argv)

    if args.command == "download-wyscout":
        from .wyscout.download import download_wyscout
        download_wyscout(args.raw_dir, args.only)
        return 0
    if args.command == "download-transfermarkt":
        from .transfermarkt.download import download_transfermarkt
        download_transfermarkt(args.raw_dir, args.force)
        return 0

    if args.command == "probe-news":
        from .news.collect import probe
        return 0 if probe() else 1
    if args.command == "probe-archive":
        import json as _json

        from .news.archive import probe as probe_archive
        report = probe_archive()
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        (args.raw_dir / "archive_probe.json").write_text(_json.dumps(report, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
        return 0
    if args.command in ("history-index", "history-titles", "history-status"):
        from .news import history
        root = args.raw_dir / "history"
        if args.command == "history-index":
            history.crawl_index(root, pause=args.pause)
            history.status(root)
        elif args.command == "history-titles":
            match = None
            if (args.transfermarkt_dir / "players.csv.gz").exists():
                from .transfermarkt.tables import read_raw
                players = read_raw(args.transfermarkt_dir, "players")["name"].dropna()
                clubs = read_raw(args.transfermarkt_dir, "clubs")["name"].dropna()
                match = history.entity_matcher(((n.split()[0], n.split()[-1]) for n in players if " " in n), clubs)
            history.fetch_titles(root, prioritise=match, limit=args.limit, pause=args.pause)
            history.status(root)
        else:
            history.status(root)
        return 0
    if args.command == "collect-news":
        from .news.collect import collect
        if args.log_file:  # pythonw has no console: send everything to the log file
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            sys.stdout = sys.stderr = args.log_file.open("a", encoding="utf-8", buffering=1)
        conn = None
        if not args.no_db:
            from .db import connect, init_db
            try:
                conn = connect(args.database_url)
                init_db(conn)
            except Exception as e:  # noqa: BLE001 - the archive is enough; load-news catches up later
                print(f"Database not reachable ({type(e).__name__}); archiving only. Run `python -m etl load-news` later.")
                conn = None
        try:
            result = collect(args.raw_dir, conn=conn)
        finally:
            if conn is not None:
                conn.close()
        return 0 if not result["errors"] or result["items"] else 1

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
        elif args.command == "load-transfermarkt":
            from .transfermarkt.load import load_transfermarkt
            init_db(conn)
            results = load_transfermarkt(conn, args.raw_dir, force=args.force)
            loaded = [r for r in results if r.status == "loaded"]
            print(f"Done: {len(loaded)} file(s) loaded, {sum(r.status == 'skipped' for r in results)} unchanged, "
                  f"{sum(r.rows for r in loaded):,} rows, {sum(r.seconds for r in loaded):.0f}s.")
        elif args.command == "load-news":
            from .news.collect import load_archive
            init_db(conn)
            load_archive(conn, args.raw_dir)
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
