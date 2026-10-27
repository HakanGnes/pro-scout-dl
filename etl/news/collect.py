"""Probe, collect and load news feeds.

Collection writes every *new* item to an append-only archive, data/raw/news/YYYY/MM/YYYY-MM-DD.jsonl.gz
(the source of truth, works without the database), and remembers what it has seen in
data/raw/news/seen.txt. If the database is reachable the new items also go into `news_items`;
otherwise `load-news` loads the archive later (idempotent).
"""
from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .feeds import CANDIDATES, EXCLUDE, FEEDS_FILE, load_feeds
from .rss import discover, fetch, parse_feed

Log = Callable[[str], None]
COLS = ("item_id", "feed", "url", "title", "summary", "published_at", "guid", "categories", "fetched_at")


# ---------------------------------------------------------------- probe


def probe(candidates: dict[str, dict] = CANDIDATES, out: Path = FEEDS_FILE, log: Log = print,
          fetcher=fetch) -> dict[str, list[str]]:
    """Try every candidate URL plus feeds advertised on each homepage; keep those that parse with items."""
    working: dict[str, list[str]] = {}
    report = []
    for name, c in candidates.items():
        urls = list(c.get("urls", []))
        if c.get("home"):
            try:
                body, final = fetcher(c["home"])
                urls += [u for u in discover(body, final) if u not in urls]
            except Exception as e:  # noqa: BLE001 - report and move on
                report.append((name, c["home"], f"homepage: {type(e).__name__}: {e}"[:90]))
        for url in urls:
            try:
                body, final = fetcher(url)
                items = parse_feed(body, name, final)
            except Exception as e:  # noqa: BLE001
                report.append((name, url, f"{type(e).__name__}: {e}"[:90]))
                continue
            if not items:
                report.append((name, url, "parsed, 0 items"))
                continue
            dated = sum(i.published_at is not None for i in items)
            report.append((name, url, f"OK {len(items)} items ({dated} dated), e.g. {items[0].title[:50]!r}"))
            if url in EXCLUDE or final in EXCLUDE:
                report.append((name, url, f"excluded: {EXCLUDE.get(final) or EXCLUDE[url]}"))
                continue
            if final not in working.setdefault(name, []):
                working[name].append(final)
    for name, url, status in report:
        log(f"  {name:<16} {status:<60} {url}")
    out.write_text(json.dumps({"probed_at": _now(), "feeds": working}, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    log(f"{sum(len(v) for v in working.values())} working feed(s) from {len(working)} of {len(candidates)} sources "
        f"-> {out}")
    return working


# ---------------------------------------------------------------- collect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _seen(path: Path) -> set[str]:
    return set(path.read_text(encoding="utf-8").split()) if path.exists() else set()


def collect(raw_dir: Path, feeds: dict[str, list[str]] | None = None, conn=None, log: Log = print,
            fetcher=fetch) -> dict:
    """Fetch every feed once; archive and (if conn) insert the items not seen before."""
    feeds = feeds if feeds is not None else load_feeds()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    seen_path = raw_dir / "seen.txt"
    seen = _seen(seen_path)
    fetched_at = _now()
    new, errors, total = [], {}, 0
    for name, urls in feeds.items():
        for url in urls:
            try:
                body, final = fetcher(url)
                items = parse_feed(body, name, final)
            except Exception as e:  # noqa: BLE001 - one broken feed must not stop the others
                errors[url] = f"{type(e).__name__}: {e}"[:200]
                continue
            total += len(items)
            for it in items:
                if it.item_id not in seen:
                    seen.add(it.item_id)
                    new.append({**it.to_dict(), "fetched_at": fetched_at})
    if new:
        day = datetime.now(timezone.utc)
        path = raw_dir / f"{day:%Y}" / f"{day:%m}" / f"{day:%Y-%m-%d}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "at", encoding="utf-8") as f:   # each run appends one gzip member
            for row in new:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with seen_path.open("a", encoding="utf-8") as f:
            f.write("".join(r["item_id"] + "\n" for r in new))
    inserted = insert(conn, new) if conn is not None and new else 0
    log(f"{fetched_at}  {total} items in {sum(map(len, feeds.values())) - len(errors)} feed(s), {len(new)} new"
        + (f", {inserted} inserted" if conn is not None else ", database not used")
        + (f"; {len(errors)} feed error(s): " + "; ".join(f"{u} ({e})" for u, e in errors.items()) if errors else ""))
    _append_log(raw_dir, {"at": fetched_at, "items": total, "new": len(new), "errors": errors})
    return {"items": total, "new": len(new), "inserted": inserted, "errors": errors}


def _append_log(raw_dir: Path, row: dict) -> None:
    with (raw_dir / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- database


def insert(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO news_items ({', '.join(COLS)}) VALUES ({', '.join(['%s'] * len(COLS))}) "
            "ON CONFLICT (item_id) DO NOTHING",
            [tuple(r.get(c) for c in COLS) for r in rows])
        return max(cur.rowcount, 0)


def iter_archive(raw_dir: Path):
    for path in sorted(Path(raw_dir).rglob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def load_archive(conn, raw_dir: Path, log: Log = print) -> int:
    """Load every archived item into news_items (existing ones are kept)."""
    started = time.perf_counter()
    before = conn.execute("SELECT count(*) FROM news_items").fetchone()[0]
    batch = []
    for row in iter_archive(raw_dir):
        batch.append(row)
        if len(batch) >= 5000:
            insert(conn, batch)
            batch = []
    insert(conn, batch)
    after = conn.execute("SELECT count(*) FROM news_items").fetchone()[0]
    log(f"news_items: {after:,} rows ({after - before:,} new) in {time.perf_counter() - started:.1f}s")
    return after - before
