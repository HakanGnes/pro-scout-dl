"""From the archive index to labelled claims: one row per (article, player, club) the headline links.

The article date is, in order of preference: the publication time the archived page states about
itself (fetched by `python -m etl history-titles`), the date in the URL (fotomac, aspor, sabah), or the
day of the first archive capture. The capture can come months after publication, so the first two are
used whenever they exist.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from etl.news.history import candidates, iter_jsonl, url_key
from etl.transfermarkt.tables import read_raw

from .entities import Gazetteer
from .labels import claims

# Study periods (pre-registered in docs/preregistration/2026-10-29-rumour-credibility.md). The gaps keep
# a claim's 180-day label window from reaching into the next period.
SPLITS = {"train": ("2018-01-01", "2022-12-31"), "val": ("2023-07-01", "2024-06-30"), "test": ("2025-01-01", "2025-12-31")}
TEST_START = pd.Timestamp(SPLITS["test"][0])

DATE_IN_URL = re.compile(r"/(20\d\d)/(\d\d)/(\d\d)/")
# article ids the sites append to slugs: "2169107", "sxhbq709319sxq", "5cf403f6237d8a159cbc784e"
ID_TOKEN = re.compile(r"^\d{5,}$|^(?=[a-z0-9]*\d)[a-z0-9]{8,}$")


def clean_slug(slug: str) -> str:
    return " ".join(t for t in slug.split() if not ID_TOKEN.match(t))


# Articles whose archived titles (and publication times) are fetched: rumour claims captured in this range.
TITLE_RANGE = ("2018-01-01", "2026-06-30")


def _day(value: str | None) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(value[:10]) if value else None
    except ValueError:
        return None


def article_date(url: str, ts: str, published: str | None = None) -> tuple[pd.Timestamp, str]:
    """(date, where it came from): the page's own publication time, the URL date or the capture day."""
    captured = pd.Timestamp(ts[:8])
    p = _day(published)
    if p is not None and p <= captured + pd.Timedelta(days=1):
        return min(p, captured), "published"
    m = DATE_IN_URL.search(url)
    if m:
        try:
            d = pd.Timestamp(f"{m[1]}-{m[2]}-{m[3]}")
            if d <= captured:
                return d, "url"
        except ValueError:
            pass
    return captured, "capture"


def articles(history_dir: Path) -> pd.DataFrame:
    """One row per article: the same slug on the same site under several URLs counts once."""
    published = {url_key(t["url"]): t.get("published") for t in iter_jsonl([Path(history_dir) / "titles.jsonl.gz"])}
    rows, seen = [], set()
    for r in candidates(history_dir):               # sorted by capture time: the first copy is kept
        if (r["site"], r["slug"]) in seen:
            continue
        seen.add((r["site"], r["slug"]))
        date, source = article_date(r["url"], r["ts"], published.get(url_key(r["url"])))
        rows.append({"url": r["url"], "site": r["site"], "date": date, "date_source": source,
                     "captured": pd.Timestamp(r["ts"][:8]), "slug": clean_slug(r["slug"])})
    return pd.DataFrame(rows, columns=["url", "site", "date", "date_source", "captured", "slug"])


def gazetteer(transfermarkt_dir: Path, slugs: list[str]) -> Gazetteer:
    t = {name: read_raw(transfermarkt_dir, name) for name in ("players", "clubs", "player_valuations")}
    # the club names as they were at each move are not in the loaded schema, so read the file in full
    transfers = pd.read_csv(Path(transfermarkt_dir) / "transfers.csv.gz", low_memory=False)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"])
    transfers = transfers.dropna(subset=["from_club_id", "to_club_id"])
    return Gazetteer.build(t["players"], t["clubs"], transfers, t["player_valuations"], slugs=slugs)


def split_of(dates: pd.Series) -> pd.Series:
    out = pd.Series(None, index=dates.index, dtype="object")
    for name, (lo, hi) in SPLITS.items():
        out[(dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))] = name
    return out


def study_urls(claims: pd.DataFrame, arts: pd.DataFrame) -> list[str]:
    """Rumour articles captured in TITLE_RANGE: their publication dates decide which period they fall in."""
    urls = set(claims.loc[claims["stance"] == "rumour", "url"])
    a = arts[arts["url"].isin(urls)]
    a = a[(a["captured"] >= pd.Timestamp(TITLE_RANGE[0])) & (a["captured"] <= pd.Timestamp(TITLE_RANGE[1]))]
    return sorted(a["url"].unique())


def build(history_dir: Path, transfermarkt_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (articles, claims)."""
    arts = articles(history_dir)
    g = gazetteer(transfermarkt_dir, arts["slug"].tolist())
    return arts, claims(arts, g)
