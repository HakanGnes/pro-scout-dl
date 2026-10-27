"""Historical transfer headlines from the Internet Archive (feasibility probe).

The Wayback Machine's CDX API lists every archived URL of a site with its capture time. Turkish
sports sites put the headline in the URL slug ("...-icardi-ile-anlasti-123456"), so archived URLs
alone can yield dated, if ASCII-folded, headlines. `probe-archive` samples a few months per site and
reports how many article URLs and transfer-looking slugs there are, plus a few real <title>s from the
archived pages, so we can decide whether the historical route is worth building.

Method: random blocks of each site's CDX index (paged queries), scaled up by the number of blocks.
"""
from __future__ import annotations

import json
import random
import re
import time
import unicodedata
from collections.abc import Callable
from urllib.parse import quote, urlsplit

from .rss import clean_text, fetch

CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK = "https://web.archive.org/web"

# site -> URL prefix whose captures to list (sections that hold football news)
SITES: dict[str, str] = {
    "fotomac": "fotomac.com.tr/",
    "fanatik": "fanatik.com.tr/",
    "aspor": "aspor.com.tr/",
    "sporx": "sporx.com/",
    "ntvspor": "ntvspor.net/",
    "hurriyet_spor": "hurriyet.com.tr/sporarena/",
    "milliyet_skorer": "milliyet.com.tr/skorer/",
    "sabah_spor": "sabah.com.tr/spor/",
}
# The CDX index is sorted by URL, not by date, so a date-limited query on a big site still scans
# the whole site and times out. Paged queries read one fixed block of the index each; a random sample
# of blocks estimates the whole site and its spread over the years.
PAGES_PER_SITE = 8
FROM, TO = "2018", "2026"

# ASCII-folded stems of words that mark a transfer story in a slug
TRANSFER_WORDS = ("transfer", "imza", "anlas", "teklif", "resmi", "kiral", "bonservis", "gorus", "ilgilen",
                  "istiyor", "masada", "pazarlik", "sozlesme", "ayrilik", "veda", "yeni-adres", "kapiy")
_SLUG = re.compile(r"[^/]*[a-z]-[a-z][^/]*$")
_NUM = re.compile(r"-?\d{5,}$")
_TITLE = re.compile(rb"<title[^>]*>(.*?)</title>", re.I | re.S)

Log = Callable[[str], None]


def fold(s: str) -> str:
    """Lower-case ASCII, the way Turkish sites build slugs (ı->i, ş->s, ğ->g, ...)."""
    s = s.replace("ı", "i").replace("İ", "i")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def slug_text(url: str) -> str | None:
    """Readable words of an article URL's last path segment, or None if it is not an article slug."""
    path = urlsplit(url if "://" in url else "http://" + url).path.rstrip("/")
    last = path.rsplit("/", 1)[-1]
    last = re.sub(r"\.(html?|aspx?|php)$", "", last.lower())
    if not _SLUG.search(last):
        return None
    words = _NUM.sub("", last).replace("_", "-").strip("-")
    return words.replace("-", " ") if words.count("-") >= 3 else None


def is_transfer(text: str) -> bool:
    t = text.replace(" ", "-")
    return any(w in t for w in TRANSFER_WORDS)


def _query(prefix: str) -> str:
    return (f"{CDX}?url={quote(prefix)}&matchType=prefix&from={FROM}&to={TO}"
            "&filter=statuscode:200&filter=mimetype:text/html")


def num_pages(prefix: str, fetcher=fetch) -> int:
    body, _ = fetcher(_query(prefix) + "&showNumPages=true&pageSize=1", timeout=120, retries=3, wait=10.0)
    return int(body.strip() or 0)


def cdx_page(prefix: str, page: int, fetcher=fetch) -> list[tuple[str, str]]:
    """(timestamp, original URL) of one index block."""
    body, _ = fetcher(_query(prefix) + f"&fl=timestamp,original&output=json&pageSize=1&page={page}",
                      timeout=120, retries=3, wait=10.0)
    rows = json.loads(body or b"[]")
    return [(r[0], r[1]) for r in rows[1:]] if rows else []


def archived_title(ts: str, url: str, fetcher=fetch) -> str | None:
    body, _ = fetcher(f"{WAYBACK}/{ts}id_/{url}", timeout=60)
    m = _TITLE.search(body[:200_000])
    return clean_text(m.group(1).decode("utf-8", errors="replace")) if m else None


def probe(sites: dict[str, str] = SITES, pages_per_site: int = PAGES_PER_SITE, titles_per_site: int = 3,
          pause: float = 1.0, seed: int = 0, log: Log = print, fetcher=fetch) -> dict:
    rng = random.Random(seed)
    report: dict[str, dict] = {}
    for name, prefix in sites.items():
        r = {"pages": 0, "sampled": 0, "articles": 0, "transfer": 0, "by_year": {}, "estimate": 0,
             "examples": [], "titles": [], "errors": []}
        try:
            r["pages"] = num_pages(prefix, fetcher)
        except Exception as e:  # noqa: BLE001 - report and go on to the next site
            r["errors"].append(f"pages: {type(e).__name__}: {e}"[:120])
        first_seen: dict[str, str] = {}
        transfer_urls: dict[str, str] = {}
        for page in sorted(rng.sample(range(r["pages"]), min(pages_per_site, r["pages"]))):
            try:
                rows = cdx_page(prefix, page, fetcher)
            except Exception as e:  # noqa: BLE001
                r["errors"].append(f"page {page}: {type(e).__name__}: {e}"[:120])
                continue
            r["sampled"] += 1
            for ts, u in rows:
                text = slug_text(u)
                if text and (u not in first_seen or ts < first_seen[u]):
                    first_seen[u] = ts
                    if is_transfer(text):
                        transfer_urls[u] = text
            time.sleep(pause)
        r["articles"], r["transfer"] = len(first_seen), len(transfer_urls)
        for u in transfer_urls:
            y = first_seen[u][:4]
            r["by_year"][y] = r["by_year"].get(y, 0) + 1
        r["by_year"] = dict(sorted(r["by_year"].items()))
        r["estimate"] = round(r["transfer"] / r["sampled"] * r["pages"]) if r["sampled"] else 0
        r["examples"] = list(transfer_urls.values())[:3]
        for u in list(transfer_urls)[:titles_per_site]:
            try:
                r["titles"].append(archived_title(first_seen[u], u, fetcher))
            except Exception as e:  # noqa: BLE001
                r["titles"].append(f"({type(e).__name__})")
            time.sleep(pause)
        report[name] = r
        years = ", ".join(f"{y}: {n}" for y, n in r["by_year"].items())
        log(f"  {name:<16} {r['pages']:>6,} index blocks, sampled {r['sampled']}: {r['transfer']:,} transfer / "
            f"{r['articles']:,} article URLs -> ~{r['estimate']:,} transfer URLs in {FROM}-{TO}"
            + (f"  [{years}]" if years else "") + (f"  errors: {'; '.join(r['errors'])}" if r["errors"] else ""))
        for ex in r["examples"]:
            log(f"      slug : {ex}")
        for t in r["titles"]:
            log(f"      title: {t}")
    total = sum(r["estimate"] for r in report.values())
    log(f"Estimated {total:,} transfer-looking article URLs across {len(sites)} sites, {FROM}-{TO} "
        f"(from {sum(r['sampled'] for r in report.values())} sampled index blocks).")
    return report
