"""Everything the simulator and the app need besides the graphs and the networks:
player profiles plus readable names for players, teams and competitions.

Saved to data/processed/sim_bundle.pkl by `python -m sim build`.
"""
from __future__ import annotations

import json
import pickle
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from features.pass_networks.graph import TeamGraph

from .profiles import Profile, build_profiles


@dataclass
class Bundle:
    profiles: dict[int, Profile]
    players: dict[int, dict] = field(default_factory=dict)    # id -> {name, role}
    teams: dict[int, dict] = field(default_factory=dict)      # id -> {name, competition}
    meta: dict = field(default_factory=dict)

    def player_name(self, pid: int) -> str:
        return self.players.get(pid, {}).get("name") or f"player {pid}"

    def team_name(self, tid: int) -> str:
        return self.teams.get(tid, {}).get("name") or f"team {tid}"

    def find(self, query: str, kind: str = "player") -> list[int]:
        """Ids whose name contains the query, ignoring case and accents ('ozturk' finds 'Öztürk')."""
        table = self.players if kind == "player" else self.teams
        q = _fold(query)
        exact = [i for i, d in table.items() if _fold(d.get("name", "")) == q]
        return exact or sorted(i for i, d in table.items() if q in _fold(d.get("name", "")))


def _fold(text: str) -> str:
    text = text.replace("ı", "i").replace("İ", "I")  # Turkish dotless i has no decomposition
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).casefold().strip()


def names_raw(raw_dir: Path) -> tuple[dict[int, dict], dict[int, dict]]:
    from etl.wyscout.parse import parse_player, parse_team

    raw_dir = Path(raw_dir)
    players, teams = {}, {}
    p_file = next(iter(sorted(raw_dir.rglob("players.json"))), None)
    t_file = next(iter(sorted(raw_dir.rglob("teams.json"))), None)
    if p_file:
        for d in json.loads(p_file.read_text(encoding="utf-8")):
            r = parse_player(d)
            if r[0] is not None:
                players[r[0]] = {"name": r[1] or " ".join(x for x in (r[2], r[4]) if x), "role": r[8]}
    if t_file:
        for d in json.loads(t_file.read_text(encoding="utf-8")):
            r = parse_team(d)
            if r[0] is not None:
                teams[r[0]] = {"name": r[1]}
    return players, teams


def names_db(conn) -> tuple[dict[int, dict], dict[int, dict]]:
    players = {pid: {"name": name or " ".join(x for x in (first, last) if x), "role": role}
               for pid, name, first, last, role in conn.execute(
                   "SELECT player_id, short_name, first_name, last_name, role_code FROM players")}
    teams = {tid: {"name": name} for tid, name in conn.execute("SELECT team_id, name FROM teams")}
    return players, teams


def build_bundle(graphs: list[TeamGraph], players: dict, teams: dict, meta: dict | None = None) -> Bundle:
    for g in graphs:  # the competition a club plays its league games in
        if not g.is_international and g.team_id in teams:
            teams[g.team_id]["competition"] = g.competition
    return Bundle(build_profiles(graphs), players, teams, meta or {})


def save(bundle: Bundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> Bundle:
    with path.open("rb") as f:
        return pickle.load(f)
