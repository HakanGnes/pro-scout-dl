"""Historical transfer headlines from the Internet Archive (2018-2026).

Two resumable steps, both safe to stop with Ctrl+C and start again:

  history-index   walk every block of each site's CDX index (paged queries) and keep the article URLs
                  whose slug looks like a transfer story -> data/raw/news/history/<site>/index.jsonl.gz
  history-titles  fetch the archived page of each candidate once and keep its real <title> and
                  description -> data/raw/news/history/titles.jsonl.gz. Candidates that name a club or a
                  player known to Transfermarkt go first.

Requests are paced (default 1.5 s apart) and a 429 makes the crawler wait a minute: the Internet
Archive is a shared, free service.
"""
from __future__ import annotations

import gzip
import json
import re
import threading
import time
import urllib.error
from collections.abc import Callable, Iterable
from pathlib import Path

from .archive import SITES, WAYBACK, _query, fold, slug_text
from .collect import read_jsonl_gz, repair
from .rss import clean_text, fetch

Log = Callable[[str], None]

# Stems (ASCII-folded, '-' between words) that mark a transfer story. Narrower than the probe's list:
# 'istiyor', 'resmi' or 'gorus' alone matched too much that was not about transfers.
TRANSFER_STEMS = (
    "transfer", "imza", "anlas", "teklif", "kiral", "bonservis", "sozlesme", "ilgilen", "masada", "pazarlik",
    "talip", "radar", "takas", "opsiyon", "satin-al", "renklerine-bagla", "kadrosuna-katt", "resmen", "veda-et",
    "ayrildi", "ayrilik", "yeni-adres", "serbest-kal", "bedelsiz", "geri-don", "gorusme",
)
OTHER_SPORTS = ("basketbol", "voleybol", "euroleague", "eurocup", "nba", "hentbol", "tenis", "formula", "f1-",
                "motogp", "golf", "boks", "gures", "atletizm", "yuzme", "esports", "e-spor", "kadin-futbol")
_TITLE = re.compile(rb"<title[^>]*>(.*?)</title>", re.I | re.S)
_DESC = re.compile(rb"<meta[^>]+(?:name|property)=[\"'](?:og:)?description[\"'][^>]*>", re.I)
_CONTENT = re.compile(rb"content=[\"'](.*?)[\"']", re.I | re.S)
_CHARSET = re.compile(rb"charset=[\"']?([A-Za-z0-9_\-]+)", re.I)


def is_transfer_slug(text: str) -> bool:
    t = "-" + text.replace(" ", "-") + "-"
    return any(s in t for s in TRANSFER_STEMS) and not any(o in t for o in OTHER_SPORTS)


def page_text(body: bytes) -> tuple[str | None, str | None]:
    """(title, description) of an HTML page, decoded properly."""
    head = body[:300_000]
    m = _CHARSET.search(head[:4096])
    enc = m.group(1).decode("ascii", "ignore") if m else None

    def dec(b: bytes) -> str:
        for e in ([enc] if enc else []) + ["utf-8", "cp1254"]:
            try:
                return b.decode(e)
            except (LookupError, UnicodeDecodeError):
                continue
        return b.decode("utf-8", errors="replace")

    t = _TITLE.search(head)
    d = _DESC.search(head)
    c = _CONTENT.search(d.group(0)) if d else None
    return (clean_text(dec(t.group(1))) or None if t else None, clean_text(dec(c.group(1))) or None if c else None)


# ---------------------------------------------------------------- step 1: index


def _state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)


def _paced(fetcher, url: str, pause: float, log: Log, **kw):
    """fetch with pacing; on 429 wait a minute (up to 5 times)."""
    for attempt in range(6):
        try:
            out = fetcher(url, **kw)
            time.sleep(pause)
            return out
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            log("    429 Too Many Requests: waiting 60 s")
            time.sleep(60)
    raise AssertionError("unreachable")


def _fetch_block(prefix: str, page: int, pause: float, log: Log, fetcher) -> list:
    body, _ = _paced(fetcher, _query(prefix) + f"&fl=timestamp,original&output=json&pageSize=1&page={page}",
                     pause, log, timeout=120, retries=3, wait=10.0)
    return json.loads(body or b"[]")[1:]


def _keep(site: str, rows: list) -> list[dict]:
    return [{"site": site, "url": u, "ts": ts, "slug": text} for ts, u in rows
            if (text := slug_text(u)) and is_transfer_slug(text)]


def _append(out: Path, rows: list[dict]) -> None:
    if rows:
        with gzip.open(out, "at", encoding="utf-8") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _crawl_site(site: str, prefix: str, root: Path, state: dict, save: Callable[[], None], pause: float, log: Log,
                fetcher, max_pages: int | None) -> int:
    from .archive import num_pages

    st = state[site]
    if st["pages"] is None:
        st["pages"] = num_pages(prefix, fetcher)
        save()
    out = root / site / "index.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    repair(out, log)
    # blocks that failed in an earlier run (e.g. a 503) get one more try
    for err in list(st["errors"]):
        try:
            new = _keep(site, _fetch_block(prefix, err["page"], pause, log, fetcher))
        except Exception:  # noqa: BLE001 - still failing; leave it recorded
            continue
        _append(out, new)
        st["kept"] += len(new)
        st["errors"].remove(err)
        save()
        log(f"  {site:<16} block {err['page']:,} retried: ok")
    last = min(st["pages"], max_pages) if max_pages else st["pages"]
    started, first = time.perf_counter(), st["done"]
    for page in range(first, last):
        try:
            new = _keep(site, _fetch_block(prefix, page, pause, log, fetcher))
        except Exception as e:  # noqa: BLE001 - recorded; the next run retries it
            st["errors"].append({"page": page, "error": f"{type(e).__name__}: {e}"[:160]})
            new = []
        _append(out, new)
        st["done"], st["kept"] = page + 1, st["kept"] + len(new)
        save()
        if (page + 1) % 25 == 0 or page + 1 == last:
            per = (time.perf_counter() - started) / (page + 1 - first)
            log(f"  {site:<16} block {page + 1:,}/{st['pages']:,}  kept {st['kept']:,}  "
                f"errors {len(st['errors'])}  (~{per * (last - page - 1) / 60:.0f} min left for this site)")
    log(f"  {site:<16} done ({st['done']:,}/{st['pages']:,} blocks, {st['kept']:,} kept, {len(st['errors'])} errors)")
    return st["kept"]


def crawl_index(root: Path, sites: dict[str, str] = SITES, pause: float = 1.5, workers: int = 3, log: Log = print,
                fetcher=fetch, max_pages: int | None = None) -> dict[str, int]:
    """Walk every index block of every site, `workers` sites at a time; append transfer-looking article
    URLs. Progress is saved after every block, so a stopped run resumes where it was."""
    from concurrent.futures import ThreadPoolExecutor

    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "index_state.json"
    state = _state(state_path)
    for site in sites:
        state.setdefault(site, {"pages": None, "done": 0, "kept": 0, "errors": []})
    lock = threading.Lock()

    def save() -> None:
        with lock:
            _save_state(state_path, state)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {site: pool.submit(_crawl_site, site, prefix, root, state, save, pause, log, fetcher, max_pages)
                   for site, prefix in sites.items()}
    kept = {}
    for site, fut in futures.items():
        try:
            kept[site] = fut.result()
        except Exception as e:  # noqa: BLE001 - e.g. the block count could not be read; the next run retries
            log(f"  {site:<16} stopped: {type(e).__name__}: {e}")
            kept[site] = state[site]["kept"]
    return kept


def iter_jsonl(paths: Iterable[Path]):
    for p in paths:
        if p.exists():
            yield from read_jsonl_gz(p)[0]


def url_key(url: str) -> str:
    """The same article under http/https, www, :80, /amp/ or a tracking query is one article."""
    u = url.split("#")[0].split("?")[0]
    u = re.sub(r"^https?://(www\.)?", "", u.strip().lower()).replace(":80/", "/").replace("/amp/", "/")
    return u.rstrip("/")


def candidates(root: Path) -> list[dict]:
    """One row per article (its earliest capture, whose exact URL and time the archive can serve)."""
    best: dict[str, dict] = {}
    for r in iter_jsonl(sorted(root.glob("*/index.jsonl.gz"))):
        k = url_key(r["url"])
        if k not in best or r["ts"] < best[k]["ts"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: (r["ts"], r["url"]))


# ---------------------------------------------------------------- step 2: titles


def entity_matcher(players: Iterable[tuple[str, str]], clubs: Iterable[str]) -> Callable[[str], bool]:
    """True if a slug names a known club (all its words) or a player (first and last name)."""
    club_tokens = set()
    for name in clubs:
        words = [w for w in re.split(r"[^a-z0-9]+", fold(name)) if len(w) >= 4 and w not in ("club", "sports", "futbol")]
        if words:
            club_tokens.add(words[0])
    by_last: dict[str, set[str]] = {}
    for first, last in players:
        f, la = fold(first or "").split(), fold(last or "").split()
        if f and la and len(la[-1]) >= 3:
            by_last.setdefault(la[-1], set()).add(f[0])

    def match(slug: str) -> bool:
        toks = set(slug.split())
        if toks & club_tokens:
            return True
        return any(t in by_last and by_last[t] & toks for t in toks)
    return match


def fetch_titles(root: Path, prioritise: Callable[[str], bool] | None = None, limit: int | None = None,
                 pause: float = 1.5, log: Log = print, fetcher=fetch) -> int:
    """Fetch the archived page of each candidate not done yet; append title + description."""
    out = root / "titles.jsonl.gz"
    repair(out, log)
    done = {url_key(r["url"]) for r in iter_jsonl([out])}
    todo = [c for c in candidates(root) if url_key(c["url"]) not in done]
    if prioritise:
        todo.sort(key=lambda c: not prioritise(c["slug"]))   # stable: matched first, then by date
    if limit:
        todo = todo[:limit]
    log(f"{len(done):,} titles already fetched, {len(todo):,} to go"
        + (f" ({sum(prioritise(c['slug']) for c in todo):,} name a known club or player)" if prioritise else ""))
    started, n = time.perf_counter(), 0
    for c in todo:
        row = {"url": c["url"], "site": c["site"], "ts": c["ts"], "slug": c["slug"]}
        try:
            body, _ = _paced(fetcher, f"{WAYBACK}/{c['ts']}id_/{c['url']}", pause, log, timeout=60, retries=2, wait=10.0)
            row["title"], row["description"] = page_text(body)
        except Exception as e:  # noqa: BLE001 - recorded, so the URL is not fetched again
            row["error"] = f"{type(e).__name__}: {e}"[:160]
        with gzip.open(out, "at", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        if n % 100 == 0 or n == len(todo):
            per = (time.perf_counter() - started) / n
            log(f"  {n:,}/{len(todo):,} titles  ({per:.1f}s each, ~{per * (len(todo) - n) / 3600:.1f} h left)")
    return n


def status(root: Path, log: Log = print) -> dict:
    state = _state(root / "index_state.json")
    for site, st in state.items():
        log(f"  {site:<16} index {st['done']:,}/{st['pages']:,} blocks, {st['kept']:,} kept, {len(st['errors'])} errors")
    cands = candidates(root)
    titles = list(iter_jsonl([root / "titles.jsonl.gz"]))
    ok = sum(1 for t in titles if t.get("title"))
    years: dict[str, int] = {}
    for c in cands:
        years[c["ts"][:4]] = years.get(c["ts"][:4], 0) + 1
    log(f"{len(cands):,} candidate URLs ({', '.join(f'{y}: {n:,}' for y, n in sorted(years.items()))}); "
        f"titles fetched {len(titles):,} ({ok:,} with a title)")
    return {"candidates": len(cands), "titles": len(titles), "with_title": ok, "by_year": years}
