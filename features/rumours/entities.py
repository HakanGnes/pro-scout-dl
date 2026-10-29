"""Find Transfermarkt players and clubs in ASCII-folded Turkish headline slugs.

Headlines glue Turkish case endings to names ("galatasaraya", "icardinin", "trabzonspordan"), so a
token matches a name when the name is the token minus at most SUFFIX_MAX trailing letters.

Players are matched with first and last name together ("mauro icardi") or, when the slug gives only
the last name ("galatasaray icardi icin"), only if exactly one player with that last name is tied to a
club named in the slug: playing for it at the time, or moving to or from it within a year.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from etl.news.archive import fold

# Turkish case and possessive endings as they appear in folded slugs ("galatasaraya", "icardinin",
# "trabzonspordan", "fenerbahceyle"). Only these may follow a name; "transfer" is not "trans" + "fer".
ENDINGS = ("", "a", "e", "i", "u", "ya", "ye", "yi", "yu", "in", "un", "nin", "nun", "da", "de", "ta", "te",
           "dan", "den", "tan", "ten", "na", "ne", "ni", "nu", "nda", "nde", "ndan", "nden", "la", "le", "yla",
           "yle", "si", "su", "sin", "sun", "nin", "li", "lu", "lar", "ler", "lari", "leri", "yi", "ye")
YOUTH = re.compile(r"\b(u\d\d|yth|youth|reserves|ii|b|b team|academy|akademi|juniors?)\b")
DROP_WORDS = {"fc", "sk", "fk", "cf", "ac", "as", "sc", "afc", "sv", "jk", "kulubu", "kulubü", "spor kulubu", "club",
              "de", "the", "calcio", "1907", "1903", "1905", "1967", "a.s.", "a.s"}
NO_NAME = {"without club", "retired", "unknown", "career break", "own youth", "ban", "unattached"}
# What Turkish headlines call the big clubs (folded). Transfermarkt ids.
NICKNAMES: dict[str, int] = {
    "fb": 36, "kanarya": 36, "sari lacivertli": 36, "sari lacivertliler": 36, "fenerbahce": 36,
    "gs": 141, "cimbom": 141, "sari kirmizili": 141, "sari kirmizililar": 141, "galatasaray": 141,
    "bjk": 114, "kara kartal": 114, "siyah beyazli": 114, "siyah beyazlilar": 114, "besiktas": 114,
    "firtina": 449, "bordo mavili": 449, "bordo mavililer": 449, "trabzonspor": 449,
    "basaksehir": 6890,
    "manchester united": 985, "man united": 985, "manchester city": 281, "man city": 281,
    "bayern munih": 27, "bayern": 27, "barcelona": 131, "barca": 131, "real madrid": 418, "inter": 46,
    "milan": 5, "juventus": 506, "psg": 583, "paris saint germain": 583, "atletico madrid": 13,
    "tottenham": 148, "arsenal": 11, "chelsea": 631, "liverpool": 31, "dortmund": 16, "borussia dortmund": 16,
    "napoli": 6195, "roma": 12, "lazio": 398, "ajax": 610, "benfica": 294, "porto": 720, "sporting": 336,
    "al hilal": 1114, "al nassr": 18544, "al ittihad": 8023, "al ahli": 18487,
}


def tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", fold(text)) if t]


def stems(token: str) -> list[str]:
    """The token itself and the token with one known Turkish ending removed (at least 3 letters left)."""
    out = [token]
    for e in ENDINGS:
        if e and token.endswith(e) and len(token) - len(e) >= 3:
            out.append(token[: -len(e)])
    return out


@dataclass
class Mention:
    kind: str            # "player" | "club"
    id: int
    start: int           # token index in the slug
    how: str             # "full", "last", "alias", "nickname"


@dataclass
class Gazetteer:
    club_alias: dict[tuple[str, ...], int]
    last_names: dict[str, list[int]]              # last-name token -> player ids
    first_names: dict[int, set[str]]              # player id -> first-name tokens
    player_value: dict[int, float]
    transfers: pd.DataFrame                        # player_id, date, from_club_id, to_club_id
    turkish_clubs: set[int]
    common: set[str] = field(default_factory=set)  # tokens too frequent to be a last name on their own
    max_alias_len: int = 4

    # ------------------------------------------------------------ building

    @classmethod
    def build(cls, players: pd.DataFrame, clubs: pd.DataFrame, transfers: pd.DataFrame,
              valuations: pd.DataFrame | None = None, slugs: list[str] | None = None) -> "Gazetteer":
        # clubs: every name a club appears under, most-used club wins an ambiguous alias
        used = Counter(transfers["to_club_id"]) + Counter(transfers["from_club_id"])
        names = pd.concat([
            transfers[["from_club_id", "from_club_name"]].set_axis(["id", "name"], axis=1),
            transfers[["to_club_id", "to_club_name"]].set_axis(["id", "name"], axis=1),
            clubs[["club_id", "name"]].set_axis(["id", "name"], axis=1),
        ]).dropna().drop_duplicates()
        candidates: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for cid, name in zip(names["id"].astype(int), names["name"]):
            f = fold(name)
            if f in NO_NAME or YOUTH.search(f):
                continue
            toks = tuple(t for t in tokens(f) if t not in DROP_WORDS)
            if not toks:
                continue
            candidates[toks].append(cid)
            if len(toks) > 1 and toks[0].endswith("spor") and len(toks[0]) >= 7:
                candidates[(toks[0],)].append(cid)          # "kayserispor erciyes" -> "kayserispor"
        common: set[str] = set()
        if slugs:
            df = Counter(t for s in slugs for t in set(s.split()))
            common = {t for t, n in df.items() if n >= 0.002 * len(slugs)}   # in >= 0.2% of headlines
        # a one-word club name that is also an everyday word in headlines ("sona", "yeni") is not used
        alias = {k: max(set(v), key=lambda i: used.get(i, 0)) for k, v in candidates.items()
                 if not (len(k) == 1 and (k[0] in common or len(k[0]) < 4))}
        for nick, cid in NICKNAMES.items():
            alias[tuple(nick.split())] = cid                 # curated names override

        last_names: dict[str, list[int]] = defaultdict(list)
        first_names: dict[int, set[str]] = {}
        for pid, name in zip(players["player_id"].astype(int), players["name"].fillna("")):
            toks = tokens(name)
            if not toks:
                continue
            last = toks[-1]
            if len(last) >= 3:
                last_names[last].append(pid)
            first_names[pid] = set(toks[:-1])
        value = {}
        if valuations is not None and len(valuations):
            value = valuations.groupby("player_id")["market_value_in_eur"].max().astype(float).to_dict()
        tr = transfers[["player_id", "transfer_date", "from_club_id", "to_club_id"]].rename(
            columns={"transfer_date": "date"}).sort_values(["player_id", "date"]).reset_index(drop=True)
        turkish = set(clubs.loc[clubs["domestic_competition_id"] == "TR1", "club_id"].astype(int))
        return cls(dict(alias), dict(last_names), first_names, value, tr, turkish, common,
                   max(len(k) for k in alias))

    # ------------------------------------------------------------ queries

    def club_at(self, player_id: int, date: pd.Timestamp) -> int | None:
        tr = self._by_player().get(player_id)
        if tr is None:
            return None
        dates, to = tr
        i = np.searchsorted(dates, np.datetime64(date), side="right") - 1
        return int(to[i]) if i >= 0 else None

    def moves(self, player_id: int) -> list[tuple[pd.Timestamp, int, int]]:
        if not hasattr(self, "_moves"):
            t = self.transfers
            self._moves: dict[int, list] = {}
            for pid, d, fr, to in zip(t["player_id"].astype(int), t["date"], t["from_club_id"].astype(int),
                                      t["to_club_id"].astype(int)):
                self._moves.setdefault(pid, []).append((d, fr, to))
        return self._moves.get(player_id, [])

    def _by_player(self):
        if not hasattr(self, "_cache"):
            g = self.transfers.groupby("player_id")
            self._cache = {pid: (d["date"].to_numpy(dtype="datetime64[ns]"), d["to_club_id"].to_numpy())
                           for pid, d in g}
        return self._cache

    def clubs_in(self, toks: list[str]) -> list[Mention]:
        found, i = [], 0
        while i < len(toks):
            hit = None
            for n in range(min(self.max_alias_len, len(toks) - i), 0, -1):
                head = tuple(toks[i:i + n - 1])
                for last in stems(toks[i + n - 1]):
                    cid = self.club_alias.get(head + (last,))
                    if cid is not None and (n > 1 or len(last) >= 4):
                        hit = Mention("club", cid, i, "alias")
                        break
                if hit:
                    i += n
                    break
            if hit:
                found.append(hit)
            else:
                i += 1
        return found

    def players_in(self, toks: list[str], clubs: list[int], date: pd.Timestamp | None) -> list[Mention]:
        found, tokset = [], set(toks)
        for i, t in enumerate(toks):
            for last in stems(t):
                ids = self.last_names.get(last)
                if not ids:
                    continue
                prev = toks[i - 1] if i else ""
                full = [p for p in ids if prev in self.first_names.get(p, ()) or
                        any(f in tokset and len(f) >= 4 for f in self.first_names.get(p, ()))]
                if full:
                    pid = max(full, key=lambda p: self.player_value.get(p, 0.0))
                    found.append(Mention("player", pid, i, "full"))
                    break
                if last in self.common or len(last) < 4:
                    continue
                tied = [p for p in ids if self._tied(p, clubs, date)]
                if len(tied) == 1:
                    found.append(Mention("player", tied[0], i, "last"))
                    break
        return found

    def _tied(self, pid: int, clubs: list[int], date: pd.Timestamp | None) -> bool:
        if date is None:
            return False
        cur = self.club_at(pid, date)
        if cur is not None and cur in clubs:
            return True
        win = pd.Timedelta(days=365)
        return any(abs(d - date) <= win and (to in clubs or fr in clubs) for d, fr, to in self.moves(pid))

    def link(self, slug: str, date: pd.Timestamp | None) -> tuple[list[Mention], list[Mention]]:
        """Players named in full first; clubs in the remaining tokens; then last-name-only players."""
        toks = slug.split()
        full = [m for m in self.players_in(toks, [], None) if m.how == "full"]
        taken = {m.start for m in full} | {m.start - 1 for m in full}     # last name and the first name before it
        masked = [t if i not in taken else "_" for i, t in enumerate(toks)]
        clubs = self.clubs_in(masked)
        club_tokens = {m.start for m in clubs}
        rest = [m for m in self.players_in(masked, [c.id for c in clubs], date)
                if m.how == "last" and m.start not in club_tokens]
        return sorted(full + rest, key=lambda m: m.start), clubs
