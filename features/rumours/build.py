"""From the archive index to labelled claims: one row per (article, player, club) the headline links.

The article date is the date in the URL when the site puts one there (fotomac, aspor, sabah), otherwise
the day of the first archive capture, which is on or shortly after publication. Real publication times
come later from the archived pages, for claim articles only.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from etl.news.history import candidates
from etl.transfermarkt.tables import read_raw

from .entities import Gazetteer
from .labels import claims

DATE_IN_URL = re.compile(r"/(20\d\d)/(\d\d)/(\d\d)/")
# article ids the sites append to slugs: "2169107", "sxhbq709319sxq", "5cf403f6237d8a159cbc784e"
ID_TOKEN = re.compile(r"^\d{5,}$|^(?=[a-z0-9]*\d)[a-z0-9]{8,}$")


def clean_slug(slug: str) -> str:
    return " ".join(t for t in slug.split() if not ID_TOKEN.match(t))


def article_date(url: str, ts: str) -> pd.Timestamp:
    captured = pd.Timestamp(ts[:8])
    m = DATE_IN_URL.search(url)
    if m:
        try:
            d = pd.Timestamp(f"{m[1]}-{m[2]}-{m[3]}")
            if d <= captured:
                return d
        except ValueError:
            pass
    return captured


def articles(history_dir: Path) -> pd.DataFrame:
    rows = [{"url": r["url"], "site": r["site"], "date": article_date(r["url"], r["ts"]),
             "slug": clean_slug(r["slug"])} for r in candidates(history_dir)]
    return pd.DataFrame(rows, columns=["url", "site", "date", "slug"])


def gazetteer(transfermarkt_dir: Path, slugs: list[str]) -> Gazetteer:
    t = {name: read_raw(transfermarkt_dir, name) for name in ("players", "clubs", "player_valuations")}
    # the club names as they were at each move are not in the loaded schema, so read the file in full
    transfers = pd.read_csv(Path(transfermarkt_dir) / "transfers.csv.gz", low_memory=False)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"])
    transfers = transfers.dropna(subset=["from_club_id", "to_club_id"])
    return Gazetteer.build(t["players"], t["clubs"], transfers, t["player_valuations"], slugs=slugs)


def build(history_dir: Path, transfermarkt_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (articles, claims)."""
    arts = articles(history_dir)
    g = gazetteer(transfermarkt_dir, arts["slug"].tolist())
    return arts, claims(arts, g)
