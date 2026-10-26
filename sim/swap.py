"""Swap a player for a candidate in every match he played and measure the change in team npxG.

For each league match of `team_id` in which `player_out` was on the pitch, his node is
replaced by `player_in`'s profile scaled to the same minutes (same starter flag, same
passing links, since Phase 1 found that the links carry no measurable xG information).
Every saved network predicts the team's non-penalty xG before and after; the result is the
change per match, averaged over matches and networks.

Uncertainty is reported two ways:
  * `se_matches` — standard error of the mean change across the matches (how consistent it is);
  * `sd_models`  — spread of the season-average change across the networks (how much the
    models agree).
This is a simulation under the model's assumptions, not a validated forecast.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from features.pass_networks.graph import NODE_FEATURES, TeamGraph

from .profiles import Profile

F = {name: i for i, name in enumerate(NODE_FEATURES)}


@dataclass
class SwapResult:
    team_id: int
    player_out: int
    player_in: int
    matches: int
    skipped: int                    # matches where the candidate was already in the team's graph
    per_match: pd.DataFrame         # match_id, gameweek, minutes_share, actual, before, after, delta
    delta: float                    # mean change in npxG per match
    se_matches: float
    sd_models: float
    before: float                   # mean predicted npxG per match before the swap
    actual: float                   # mean actual npxG in those matches

    @property
    def delta_per_90(self) -> float:
        """Change scaled to a full match of the swapped slot (comparable across starters and subs)."""
        share = self.per_match["minutes_share"].mean() if len(self.per_match) else 0.0
        return self.delta / share if share else float("nan")


def swap_graph(g: TeamGraph, player_out: int, profile: Profile) -> TeamGraph:
    """A copy of `g` with `player_out`'s node replaced by `profile` at the same minutes."""
    k = g.player_ids.index(player_out)
    x = g.x.copy()
    x[k] = profile.node_features(float(g.x[k, F["minutes_share"]]), bool(g.x[k, F["is_starter"]] > 0.5))
    ids = list(g.player_ids)
    ids[k] = profile.player_id
    return replace(g, x=x, player_ids=ids)


class Simulator:
    def __init__(self, graphs: list[TeamGraph], profiles: dict[int, Profile], ensemble):
        self.profiles = profiles
        self.ensemble = ensemble
        self.by_team: dict[int, list[TeamGraph]] = defaultdict(list)
        for g in graphs:
            if not g.is_international:
                self.by_team[g.team_id].append(g)
        for gs in self.by_team.values():
            gs.sort(key=lambda g: (g.gameweek or 0, g.match_id))
        self._cache: dict[int, np.ndarray] = {}

    def team_players(self, team_id: int) -> pd.DataFrame:
        """Players who appeared for the team: matches and share of the season's minutes."""
        rows = defaultdict(lambda: [0, 0.0])
        for g in self.by_team.get(team_id, []):
            for k, pid in enumerate(g.player_ids):
                share = float(g.x[k, F["minutes_share"]])
                if share > 0:
                    rows[pid][0] += 1
                    rows[pid][1] += share
        return pd.DataFrame([(p, m, s) for p, (m, s) in rows.items()],
                            columns=["player_id", "matches", "match_share"]).sort_values("match_share", ascending=False)

    def _matches_with(self, team_id: int, player_id: int) -> list[TeamGraph]:
        return [g for g in self.by_team.get(team_id, [])
                if player_id in g.player_ids and g.x[g.player_ids.index(player_id), F["minutes_share"]] > 0]

    def _before(self, graphs: list[TeamGraph]) -> np.ndarray:
        """Predictions on the unchanged graphs, cached per graph."""
        missing = [g for g in graphs if id(g) not in self._cache]
        if missing:
            preds = self.ensemble.predict(missing)
            for j, g in enumerate(missing):
                self._cache[id(g)] = preds[:, j]
        return np.stack([self._cache[id(g)] for g in graphs], axis=1) if graphs else np.zeros((0, 0))

    def swap(self, team_id: int, player_out: int, player_in: int) -> SwapResult:
        return self.rank(team_id, player_out, [player_in])[1][player_in]

    def rank(self, team_id: int, player_out: int, candidates: list[int]) -> tuple[pd.DataFrame, dict[int, SwapResult]]:
        """Swap results for many candidates at once (one batched prediction)."""
        games = self._matches_with(team_id, player_out)
        if not games:
            raise ValueError(f"player {player_out} has no league minutes for team {team_id}")
        before = self._before(games)                                   # (n_models, n_games)
        plan, swapped = [], []
        for cand in candidates:
            profile = self.profiles.get(cand)
            if profile is None:
                raise ValueError(f"no profile for player {cand} (no league appearance of 9 minutes or more)")
            keep = [j for j, g in enumerate(games) if cand == player_out or cand not in g.player_ids]
            plan.append((cand, keep))
            swapped.extend(swap_graph(games[j], player_out, profile) for j in keep)
        after_all = self.ensemble.predict(swapped)

        results, rows, start = {}, [], 0
        for cand, keep in plan:
            after = after_all[:, start:start + len(keep)]
            start += len(keep)
            delta = after - before[:, keep]                            # (n_models, n_kept)
            per_match = pd.DataFrame({
                "match_id": [games[j].match_id for j in keep],
                "gameweek": [games[j].gameweek for j in keep],
                "minutes_share": [float(games[j].x[games[j].player_ids.index(player_out), F["minutes_share"]]) for j in keep],
                "actual": [games[j].target for j in keep],
                "before": before[:, keep].mean(0),
                "after": after.mean(0),
                "delta": delta.mean(0),
            })
            n = len(keep)
            res = SwapResult(
                team_id=team_id, player_out=player_out, player_in=cand, matches=n, skipped=len(games) - n,
                per_match=per_match,
                delta=float(delta.mean()) if n else float("nan"),
                se_matches=float(per_match["delta"].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                sd_models=float(delta.mean(1).std(ddof=1)) if n and delta.shape[0] > 1 else float("nan"),
                before=float(per_match["before"].mean()) if n else float("nan"),
                actual=float(per_match["actual"].mean()) if n else float("nan"),
            )
            results[cand] = res
            rows.append({"player_in": cand, "matches": n, "skipped": res.skipped, "delta": res.delta,
                         "delta_per_90": res.delta_per_90, "se_matches": res.se_matches,
                         "sd_models": res.sd_models})
        table = pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)
        return table, results


def candidates_by_role(profiles: dict[int, Profile], role: str | None, min_minutes: float = 900.0,
                       exclude_team: int | None = None) -> list[int]:
    return [p.player_id for p in profiles.values()
            if (role is None or p.role == role) and p.minutes >= min_minutes
            and (exclude_team is None or exclude_team not in p.team_ids)]
