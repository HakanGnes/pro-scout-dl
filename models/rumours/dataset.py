"""The study table: one row per rumour claim in a study period, with masked text and metadata inputs.

Text: the headline slug with the claimed player, the claimed club, the player's current club and any
other linked player or club replaced by placeholders (case endings kept: "galatasaraydan" ->
"hedefkulupdan"). The models then learn how papers talk, not which names came true in the past.

Metadata: everything about a claim except its wording, all from records dated before the article:
paper, month, how much the same pair and player were in the news, the player's last market value and
age, and whether the clubs play in Türkiye.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.rumours.build import split_of
from features.rumours.entities import Gazetteer, stems, tokens

PLAYER, TARGET, CURRENT, OTHER_PLAYER, OTHER_CLUB = "oyuncu", "hedefkulup", "mevcutkulup", "baskaoyuncu", "baskakulup"
META_NUMERIC = ["month", "pair_30d", "pair_sites_30d", "pair_365d", "pair_age_days", "player_30d", "player_clubs_30d",
                "log_value", "age", "target_tr", "current_tr", "current_known"]
META_CATEGORICAL = ["site", "position"]


@dataclass
class Names:
    """The token sequences that name each player and club."""
    player: dict[int, list[tuple[str, ...]]]
    club: dict[int, list[tuple[str, ...]]]

    @classmethod
    def from_gazetteer(cls, g: Gazetteer, players: pd.DataFrame) -> "Names":
        club: dict[int, list[tuple[str, ...]]] = defaultdict(list)
        for alias, cid in g.club_alias.items():
            club[int(cid)].append(alias)
        player: dict[int, list[tuple[str, ...]]] = {}
        for pid, name in zip(players["player_id"].astype(int), players["name"].fillna("")):
            t = tuple(tokens(name))
            if t:
                # full name, last name, and first names of 4+ letters ("omer faruk beyaz" for Ömer Beyaz)
                player[pid] = [t] + ([(t[-1],)] + [(f,) for f in t[:-1] if len(f) >= 4] if len(t) > 1 else [])
        return cls(player, dict(club))


def mask(slug: str, spans: list[tuple[list[tuple[str, ...]], str]]) -> str:
    """Replace every occurrence of the given names (longest first) by their placeholder, keeping endings."""
    toks = slug.split()
    names = sorted(((alias, ph) for aliases, ph in spans for alias in aliases), key=lambda x: -len(x[0]))
    out, i = [], 0
    while i < len(toks):
        hit = None
        for alias, ph in names:
            n = len(alias)
            if i + n > len(toks) or tuple(toks[i:i + n - 1]) != alias[:-1]:
                continue
            last = toks[i + n - 1]
            for s in stems(last):
                if s == alias[-1]:
                    hit = (n, ph + last[len(s):])
                    break
            if hit:
                break
        if hit:
            out.append(hit[1])
            i += hit[0]
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def masked_text(claims: pd.DataFrame, names: Names) -> pd.Series:
    by_url = claims.groupby("url")
    players_in = by_url["player_id"].agg(lambda s: set(s.astype(int))).to_dict()
    clubs_in = by_url["club_id"].agg(lambda s: set(s.astype(int))).to_dict()
    out = []
    for r in claims.itertuples(index=False):
        p, c = int(r.player_id), int(r.club_id)
        cur = None if pd.isna(r.current_club_id) else int(r.current_club_id)
        spans = [(names.player.get(p, []), PLAYER), (names.club.get(c, []), TARGET)]
        if cur is not None and cur != c:
            spans.append((names.club.get(cur, []), CURRENT))
        for q in players_in[r.url] - {p}:
            spans.append((names.player.get(q, []), OTHER_PLAYER))
        for k in clubs_in[r.url] - {c, cur}:
            spans.append((names.club.get(k, []), OTHER_CLUB))
        out.append(mask(r.slug, spans))
    return pd.Series(out, index=claims.index)


def _count_before(dates: np.ndarray, keys: np.ndarray, at: np.ndarray, at_keys: np.ndarray,
                  days: int) -> np.ndarray:
    """For each (key, date) in `at`: number of rows with the same key dated in [date - days, date)."""
    order = np.lexsort((dates, keys))
    k, d = keys[order], dates[order]
    starts = {}
    uniq, first = np.unique(k, return_index=True)
    bounds = list(first) + [len(k)]
    for j, u in enumerate(uniq):
        starts[u] = (bounds[j], bounds[j + 1])
    res = np.zeros(len(at), dtype=float)
    delta = np.timedelta64(days, "D")
    for i, (key, t) in enumerate(zip(at_keys, at)):
        lo, hi = starts.get(key, (0, 0))
        seg = d[lo:hi]
        res[i] = np.searchsorted(seg, t, "left") - np.searchsorted(seg, t - delta, "left")
    return res


def metadata(study: pd.DataFrame, all_claims: pd.DataFrame, g: Gazetteer, players: pd.DataFrame,
             valuations: pd.DataFrame) -> pd.DataFrame:
    """Inputs that need no text, each from records dated strictly before the article's day."""
    # news volume: distinct articles (any stance, any period) about the pair / the player
    arts = all_claims.drop_duplicates(["url", "player_id", "club_id"])
    pair_all = arts["player_id"].astype(str) + "-" + arts["club_id"].astype(str)
    pair_at = study["player_id"].astype(str) + "-" + study["club_id"].astype(str)
    d_all, d_at = arts["date"].to_numpy("datetime64[D]"), study["date"].to_numpy("datetime64[D]")
    m = pd.DataFrame(index=study.index)
    m["site"] = study["site"]
    m["month"] = study["date"].dt.month
    m["pair_30d"] = _count_before(d_all, pair_all.to_numpy(), d_at, pair_at.to_numpy(), 30)
    m["pair_365d"] = _count_before(d_all, pair_all.to_numpy(), d_at, pair_at.to_numpy(), 365)
    site_pair = arts.drop_duplicates(["site", "player_id", "club_id", "date"])
    sp_key = (site_pair["player_id"].astype(str) + "-" + site_pair["club_id"].astype(str)).to_numpy()
    # distinct papers in 30 days: count first mentions per paper within the window (approximate by paper-days)
    m["pair_sites_30d"] = 0.0
    for site in site_pair["site"].unique():
        s = site_pair["site"].to_numpy() == site
        m["pair_sites_30d"] += _count_before(site_pair["date"].to_numpy("datetime64[D]")[s], sp_key[s], d_at,
                                             pair_at.to_numpy(), 30) > 0
    mentions: dict[str, list] = defaultdict(list)
    for key, t in zip(pair_all.to_numpy(), d_all):
        mentions[key].append(t)
    first = {k: np.sort(np.array(v, dtype="datetime64[D]")) for k, v in mentions.items()}
    age_days = []   # days since the pair's first mention in the year before (0: new pair)
    for key, t in zip(pair_at, d_at):
        seg = first.get(key, np.array([], dtype="datetime64[D]"))
        prior = seg[(seg < t) & (seg >= t - np.timedelta64(365, "D"))]
        age_days.append(float((t - prior[0]) / np.timedelta64(1, "D")) if len(prior) else 0.0)
    m["pair_age_days"] = age_days
    p_all, p_at = arts["player_id"].astype(int).to_numpy(), study["player_id"].astype(int).to_numpy()
    m["player_30d"] = _count_before(d_all, p_all, d_at, p_at, 30)
    pc = arts.drop_duplicates(["player_id", "club_id", "date"]).sort_values("date")
    clubs_30 = []
    grouped = {pid: (grp["date"].to_numpy("datetime64[D]"), grp["club_id"].to_numpy())
               for pid, grp in pc.groupby("player_id")}
    for pid, t in zip(p_at, d_at):
        ds, cs = grouped.get(pid, (np.array([], dtype="datetime64[D]"), np.array([])))
        sel = (ds < t) & (ds >= t - np.timedelta64(30, "D"))
        clubs_30.append(float(len(set(cs[sel]))))
    m["player_clubs_30d"] = clubs_30

    # the player on that day: last market value before it, age, position
    v = valuations.dropna(subset=["market_value_in_eur"]).sort_values("date")
    vv = {pid: (grp["date"].to_numpy("datetime64[D]"), np.log1p(grp["market_value_in_eur"].to_numpy(float)))
          for pid, grp in v.groupby("player_id")}
    logv = []
    for pid, t in zip(p_at, d_at):
        ds, vals = vv.get(pid, (np.array([], dtype="datetime64[D]"), np.array([])))
        i = np.searchsorted(ds, t, "left") - 1
        logv.append(vals[i] if i >= 0 else np.nan)
    m["log_value"] = logv
    info = players.set_index(players["player_id"].astype(int))
    dob = info["date_of_birth"].reindex(p_at).to_numpy("datetime64[D]")
    m["age"] = (d_at - dob).astype("timedelta64[D]").astype(float) / 365.25
    m.loc[np.isnat(dob), "age"] = np.nan
    m["position"] = info["position"].reindex(p_at).fillna("unknown").to_numpy()
    m["target_tr"] = study["club_id"].astype(int).isin(g.turkish_clubs).astype(float).to_numpy()
    cur = study["current_club_id"]
    m["current_known"] = cur.notna().astype(float).to_numpy()
    m["current_tr"] = cur.fillna(-1).astype(int).isin(g.turkish_clubs).astype(float).to_numpy()
    return m


def build(claims: pd.DataFrame, g: Gazetteer, players: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    """Rumour claims in the study periods, with split, cluster id, masked text and metadata."""
    if claims.empty:
        raise SystemExit("No claims: no headline names a known player and club.")
    c = claims.copy()
    c["split"] = split_of(c["date"])
    study = c[(c["stance"] == "rumour") & c["split"].notna()].copy()
    names = Names.from_gazetteer(g, players)
    study["text"] = masked_text(study, names)
    study["pair"] = study["player_id"].astype(int).astype(str) + "-" + study["club_id"].astype(int).astype(str)
    meta = metadata(study, claims, g, players, valuations)
    out = pd.concat([study[["url", "site", "date", "slug", "player_id", "club_id", "current_club_id", "label",
                            "split", "pair", "text"]], meta.drop(columns=["site"])], axis=1)
    return out.reset_index(drop=True)
