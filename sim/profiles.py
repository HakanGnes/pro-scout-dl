"""Player profiles: what a player typically produces in a full match, from his league games.

A profile pools every league appearance of at least MIN_SHARE of a match:
  * count features (passes made/received, final-third and box entries, progressive passes,
    touches) become rates per full match: total count / total share of matches played;
  * pass accuracy is pooled over all passes (total accurate / total attempted);
  * the position is the minutes-weighted mean of the per-match median positions;
  * the role is the player's listed role.

Putting a profile into a match slot scales the rates back to the slot's minutes, so a
substitute who came on for 20 minutes is replaced by 20 minutes of the candidate.
International tournaments are left out: national-team roles differ from club roles.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from features.pass_networks.graph import NODE_FEATURES, ROLES, TeamGraph

F = {name: i for i, name in enumerate(NODE_FEATURES)}
COUNT_FEATURES = ("log_passes", "log_received", "log_final_third", "log_box", "log_progressive", "log_touches")
MIN_SHARE = 0.10  # appearances shorter than ~9 minutes say little about a player's rates


@dataclass
class Profile:
    player_id: int
    role: str | None
    matches: int
    match_share: float                      # sum of match shares played (≈ full matches)
    rates: dict[str, float]                 # count feature -> per full match
    pass_accuracy: float
    median_x: float
    median_y: float
    team_ids: set[int] = field(default_factory=set)

    @property
    def minutes(self) -> float:
        return self.match_share * 90.0

    def node_features(self, minutes_share: float, is_starter: bool) -> np.ndarray:
        """This player's typical node vector for a slot of the given length."""
        x = np.zeros(len(NODE_FEATURES), dtype=np.float32)
        if self.role in ROLES:
            x[ROLES.index(self.role)] = 1.0
        x[F["minutes_share"]] = minutes_share
        x[F["is_starter"]] = float(is_starter)
        for feat in COUNT_FEATURES:
            x[F[feat]] = math.log1p(self.rates[feat] * minutes_share)
        x[F["pass_accuracy"]] = self.pass_accuracy
        x[F["median_x"]] = self.median_x
        x[F["median_y"]] = self.median_y
        return x


def _role(row: np.ndarray) -> str | None:
    hot = row[[F[f"role_{r.lower()}"] for r in ROLES]]
    return ROLES[int(hot.argmax())] if hot.max() > 0 else None


def build_profiles(graphs: list[TeamGraph], include_international: bool = False) -> dict[int, Profile]:
    acc = defaultdict(lambda: defaultdict(float))
    teams: dict[int, set[int]] = defaultdict(set)
    roles: dict[int, str | None] = {}
    for g in graphs:
        if g.is_international and not include_international:
            continue
        for k, pid in enumerate(g.player_ids):
            row = g.x[k]
            share = float(row[F["minutes_share"]])
            if share < MIN_SHARE:
                continue
            a = acc[pid]
            a["matches"] += 1
            a["share"] += share
            for feat in COUNT_FEATURES:
                a[feat] += math.expm1(float(row[F[feat]]))
            passes = math.expm1(float(row[F["log_passes"]]))
            a["accurate"] += float(row[F["pass_accuracy"]]) * passes
            a["x"] += float(row[F["median_x"]]) * share
            a["y"] += float(row[F["median_y"]]) * share
            teams[pid].add(g.team_id)
            roles.setdefault(pid, _role(row))
    profiles = {}
    for pid, a in acc.items():
        passes = a["log_passes"]
        profiles[pid] = Profile(
            player_id=pid,
            role=roles.get(pid),
            matches=int(a["matches"]),
            match_share=a["share"],
            rates={feat: a[feat] / a["share"] for feat in COUNT_FEATURES},
            pass_accuracy=a["accurate"] / passes if passes else 0.0,
            median_x=a["x"] / a["share"],
            median_y=a["y"] / a["share"],
            team_ids=teams[pid],
        )
    return profiles
