"""Turn linked headlines into (player, club, date) claims and label them with what happened.

A claim links a player to a club that is not his club at the time. It is labelled 1 if Transfermarkt
records him moving to that club within HORIZON days after the headline, 0 otherwise. Transfermarkt
dates are often the official start of a contract (1 July, say), later than the day a move is agreed, so
a headline that announces a done deal also comes out as 1. The stance of the headline (rumour or
announcement) is therefore kept with every claim, and rumour credibility is measured on rumours only.
"""
from __future__ import annotations

import re

import pandas as pd

from .entities import Gazetteer

HORIZON = pd.Timedelta(days=365)
GRACE = pd.Timedelta(days=30)     # a move recorded up to 30 days before the headline still counts

# Stance from the folded headline. Announcements are checked first: "resmen imzayi atti" is not a rumour.
ANNOUNCE = re.compile(
    r"\b(resmen|resmi(?:len)?(?! teklif)|imzay[i]? att|imzalad|imza(?:yi)? att|imza toren|transfer etti|transferini acikla|"
    r"kiralad|kadrosuna katt|renklerine bagla|kap(?:a)? bildir|borsaya bildir|kamuyu aydinlatma|yeni transferi|"
    r"transfer oldu|ile anlast[i]|anlasma sagla|anlasmaya var|anlasma tamam|transferi tamamland|bitti|bitirdi|"
    r"resmilesti|(?<!basini )duyurdu|yillik anlasma|yillik sozlesme|sozlesme imzala|mutlu son|yeni adresi|kabul etti)")
RUMOUR = re.compile(
    r"\b(istiyor|istedi|istemis|ilgileniyor|ilgilen|teklif|gorusme|pazarlik|talip|radar|gundem|masada|devrede|"
    r"devreye|iddia|yakin|gidiyor|geliyor|gelecek|bekleniyor|olabilir|mi\b|hedef|listesinde|takip|izliyor|"
    r"harekete gec|kapisini|atak|atag|bomba|yaris|kapis|harekat|hamle|operasyon|israr|aday|kanca|rota\b|rotasi|"
    r"surpriz|cevab|itiraf|rakip|karsi karsiya|vites|kesif|gelisme|bitiriyor|sona geldi|durdu|yolcusu|"
    r"transfer zamani)")


def stance(slug: str) -> str:
    if ANNOUNCE.search(slug):
        return "announcement"
    if RUMOUR.search(slug):
        return "rumour"
    return "other"


def claims(articles: pd.DataFrame, g: Gazetteer) -> pd.DataFrame:
    """One row per (article, player, club) where the club is not the player's club at the time."""
    rows = []
    for a in articles.itertuples(index=False):
        if pd.isna(a.date):
            continue
        players, clubs = g.link(a.slug, a.date)
        if not players or not clubs:
            continue
        st = stance(a.slug)
        for p in {m.id for m in players}:
            current = g.club_at(p, a.date - pd.Timedelta(days=1))
            moves = g.moves(p)
            for c in {m.id for m in clubs}:
                if c == current:
                    continue
                earlier = [d for d, _, to in moves if to == c and d < a.date - GRACE]
                window = [d for d, _, to in moves if to == c and a.date - GRACE <= d <= a.date + HORIZON]
                rows.append({"url": a.url, "site": a.site, "date": a.date, "slug": a.slug, "player_id": p,
                             "club_id": c, "current_club_id": current, "stance": st,
                             "label": int(bool(window)), "move_date": min(window) if window else pd.NaT,
                             "was_there_before": bool(earlier)})
    return pd.DataFrame(rows)
