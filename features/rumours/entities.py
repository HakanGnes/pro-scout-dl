"""Find Transfermarkt players and clubs in ASCII-folded Turkish headline slugs.

Headlines glue Turkish case endings to names ("galatasaraya", "icardinin", "trabzonspordan"), so a
token matches a name when the name is the token minus at most SUFFIX_MAX trailing letters.

Players are matched with first and last name together ("mauro icardi") or, when the slug gives only
the last name ("galatasaray icardi icin"), only if exactly one player with that last name is tied to a
club named in the slug, from what was known on the headline's day: playing for it then, or having moved
to or from it in the year before. Failing that, a last name is accepted for the most valuable player
active then (a market value in the two years before) who carries it, if he was worth at least EUR 1m and
at least five times any other active player with that name (Dušan, not Vanja, Vlahović).
Only records dated before the headline are used, so whether the rumour later came true cannot decide
whether it is linked.
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
    how: str             # "full", "last", "last_left", "last_unique", "alias"
    span: int = 1        # number of tokens the mention covers


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
    # player id -> (valuation dates, running maximum of the value): what a player was worth up to a day
    value_history: dict = field(default_factory=dict)

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
        value, history = {}, {}
        if valuations is not None and len(valuations):
            v = valuations.dropna(subset=["market_value_in_eur"]).sort_values(["player_id", "date"])
            value = v.groupby("player_id")["market_value_in_eur"].max().astype(float).to_dict()
            for pid, d in v.groupby("player_id"):
                history[int(pid)] = (d["date"].to_numpy("datetime64[ns]"),
                                     np.maximum.accumulate(d["market_value_in_eur"].to_numpy(float)))
        tr = transfers[["player_id", "transfer_date", "from_club_id", "to_club_id"]].rename(
            columns={"transfer_date": "date"}).sort_values(["player_id", "date"]).reset_index(drop=True)
        turkish = set(clubs.loc[clubs["domestic_competition_id"] == "TR1", "club_id"].astype(int))
        return cls(dict(alias), dict(last_names), first_names, value, tr, turkish, common,
                   max(len(k) for k in alias), history)

    # ------------------------------------------------------------ queries

    ACTIVE_DAYS = 730
    # Rule 4 of docs/preregistration/2026-11-02-rumour-credibility-corrected.md: on only if >= 85% of 80
    # hand-labelled headlines it links name the right player. Measured 2026-11-02: 52 of 80 (65%), so off.
    UNIQUE_SURNAME = False
    MIN_UNIQUE_VALUE = 1_000_000
    DOMINANCE = 5.0

    def value_at(self, player_id: int, date: pd.Timestamp | None) -> float:
        """Highest market value up to the day (0 if none); without a day, the highest ever."""
        h = getattr(self, "value_history", {}).get(player_id)
        if date is None or h is None:
            return self.player_value.get(player_id, 0.0) if date is None else 0.0
        i = np.searchsorted(h[0], np.datetime64(date), side="right") - 1
        return float(h[1][i]) if i >= 0 else 0.0

    def active_at(self, player_id: int, date: pd.Timestamp) -> bool:
        """A market value dated in the ACTIVE_DAYS before the day."""
        h = getattr(self, "value_history", {}).get(player_id)
        if h is None:
            return False
        d = np.datetime64(date)
        i = np.searchsorted(h[0], d, side="right") - 1
        return i >= 0 and h[0][i] >= d - np.timedelta64(self.ACTIVE_DAYS, "D")

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
                        hit = Mention("club", cid, i, "alias", n)
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
        found = []
        for i, t in enumerate(toks):
            for last in stems(t):
                ids = self.last_names.get(last)
                if not ids:
                    continue
                prev = toks[i - 1] if i else ""
                others = set(toks[:i] + toks[i + 1:])        # a first name must be another word of the headline
                full = [p for p in ids if prev in self.first_names.get(p, ()) or
                        any(f in others and len(f) >= 4 for f in self.first_names.get(p, ()))]
                if full:
                    pid = max(full, key=lambda p: self.value_at(p, date))
                    found.append(Mention("player", pid, i, "full"))
                    break
                if last in self.common or len(last) < 4 or date is None:
                    continue
                ties = {p: self._tie(p, clubs, date) for p in ids}
                tied = [p for p, k in ties.items() if k]
                if len(tied) == 1:
                    found.append(Mention("player", tied[0], i, "last" if ties[tied[0]] != "left" else "last_left"))
                    break
                if not tied and self.UNIQUE_SURNAME and len(last) >= 5:
                    # players known by one name only ("Marcelo", "Rodrigo") are left out: the word is too often
                    # someone else's first name
                    vals = sorted(((self.value_at(p, date), p) for p in ids
                                   if self.first_names.get(p) and self.active_at(p, date)), reverse=True)
                    if vals and vals[0][0] >= self.MIN_UNIQUE_VALUE and (
                            len(vals) == 1 or vals[0][0] >= self.DOMINANCE * vals[1][0]):
                        found.append(Mention("player", vals[0][1], i, "last_unique"))
                        break
        return found

    def _tie(self, pid: int, clubs: list[int], date: pd.Timestamp) -> str | None:
        """How a player is tied to a club named in the headline, from records before the day: "current"
        (plays there), "joined" (moved there in the year before) or "left" (moved away in the year before)."""
        cur = self.club_at(pid, date - pd.Timedelta(days=1))
        if cur is not None and cur in clubs:
            return "current"
        year = date - pd.Timedelta(days=365)
        past = [(fr, to) for d, fr, to in self.moves(pid) if year <= d < date]
        if any(to in clubs for _, to in past):
            return "joined"
        if any(fr in clubs for fr, _ in past):
            return "left"
        return None

    def link(self, slug: str, date: pd.Timestamp | None) -> tuple[list[Mention], list[Mention]]:
        """Players named in full first; clubs in the remaining tokens; then last-name-only players."""
        toks = slug.split()
        full = [m for m in self.players_in(toks, [], date) if m.how == "full"]
        taken = {m.start for m in full} | {m.start - 1 for m in full}     # last name and the first name before it
        masked = [t if i not in taken else "_" for i, t in enumerate(toks)]
        clubs = self.clubs_in(masked)
        club_tokens = {m.start + k for m in clubs for k in range(m.span)}   # "germain" in "paris saint germain"
        rest = [m for m in self.players_in(masked, [c.id for c in clubs], date)
                if m.how != "full" and m.start not in club_tokens]
        players = full + rest
        if len(players) > 1:
            # a player tied only by having left a named club is background ("Tadić detayı") when the
            # headline names someone else
            players = [m for m in players if m.how != "last_left"]
        return sorted(players, key=lambda m: m.start), clubs
