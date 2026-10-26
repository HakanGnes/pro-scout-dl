"""Sanity checks for the swap simulator, with pass criteria fixed before running them.

1. Self-swap: replacing a player by his own season profile should change little.
   Pass: mean absolute season-level change < SELF_SWAP_MAX npxG per match.
2. Forward -> goalkeeper: replacing each team's most-used forward by the profile of the
   most-used goalkeeper from another team should lower the team's xG.
   Pass: the change is negative for at least GK_SWAP_MIN_SHARE of teams.
3. Candidate ranking: for one team, ranking every regular forward should favour forwards who
   get the ball into the box. Pass: Spearman(candidate box-entry rate, change) > 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .bundle import Bundle
from .swap import Simulator, candidates_by_role

SELF_SWAP_MAX = 0.05
GK_SWAP_MIN_SHARE = 0.90
SELF_SWAP_PLAYERS_PER_TEAM = 3
SELF_SWAP_MIN_MATCHES = 15
RANK_MIN_MINUTES = 1800.0


@dataclass
class CheckResult:
    name: str
    passed: bool
    summary: str
    table: pd.DataFrame


def self_swap(sim: Simulator, bundle: Bundle) -> CheckResult:
    rows = []
    for team_id in sim.by_team:
        players = sim.team_players(team_id)
        players = players[players["matches"] >= SELF_SWAP_MIN_MATCHES].head(SELF_SWAP_PLAYERS_PER_TEAM)
        for pid in players["player_id"]:
            if pid not in sim.profiles:
                continue
            r = sim.swap(team_id, int(pid), int(pid))
            rows.append({"team": bundle.team_name(team_id), "player": bundle.player_name(int(pid)),
                         "matches": r.matches, "delta": r.delta, "abs_delta": abs(r.delta)})
    t = pd.DataFrame(rows)
    if t.empty:
        return CheckResult("Self-swap", False, f"no player with ≥ {SELF_SWAP_MIN_MATCHES} matches", t)
    mean_abs = float(t["abs_delta"].mean())
    summary = (f"{len(t)} players: mean change {t['delta'].mean():+.4f}, mean |change| {mean_abs:.4f}, "
               f"90th percentile |change| {t['abs_delta'].quantile(0.9):.4f} npxG per match "
               f"(pass if mean |change| < {SELF_SWAP_MAX})")
    return CheckResult("Self-swap", mean_abs < SELF_SWAP_MAX, summary,
                       t.sort_values("abs_delta", ascending=False).head(10))


def forward_to_goalkeeper(sim: Simulator, bundle: Bundle) -> CheckResult:
    gks = sorted((p for p in sim.profiles.values() if p.role == "GK"), key=lambda p: -p.minutes)
    rows = []
    for team_id in sim.by_team:
        players = sim.team_players(team_id)
        fwd = [int(p) for p in players["player_id"] if sim.profiles.get(int(p)) and sim.profiles[int(p)].role == "FW"]
        # The most-used keeper who never played for this team (otherwise he is already on the pitch).
        keeper = next((p for p in gks if team_id not in p.team_ids), None)
        if not fwd or keeper is None:
            continue
        r = sim.swap(team_id, fwd[0], keeper.player_id)
        rows.append({"team": bundle.team_name(team_id), "forward": bundle.player_name(fwd[0]),
                     "keeper": bundle.player_name(keeper.player_id), "matches": r.matches,
                     "before": r.before, "delta": r.delta})
    t = pd.DataFrame(rows)
    if t.empty:
        return CheckResult("Forward → goalkeeper", False, "no team with a forward and an outside keeper", t)
    share_neg = float((t["delta"] < 0).mean())
    summary = (f"{len(t)} teams, each forward replaced by the most-used keeper from another team: change "
               f"negative for {share_neg:.0%} of teams, median change {t['delta'].median():+.3f} npxG per match "
               f"(pass if ≥ {GK_SWAP_MIN_SHARE:.0%})")
    return CheckResult("Forward → goalkeeper", share_neg >= GK_SWAP_MIN_SHARE, summary,
                       t.sort_values("delta", ascending=False).head(10))


def candidate_ranking(sim: Simulator, bundle: Bundle, team_id: int | None = None) -> CheckResult:
    if team_id is None:  # the team with the most predicted chances: the hardest place to improve
        team_id = max(sim.by_team, key=lambda t: np.mean([g.target for g in sim.by_team[t]]))
    players = sim.team_players(team_id)
    fwd = next((int(p) for p in players["player_id"]
                if sim.profiles.get(int(p)) and sim.profiles[int(p)].role == "FW"), None)
    cands = candidates_by_role(sim.profiles, "FW", RANK_MIN_MINUTES, exclude_team=team_id)
    if fwd is None or len(cands) < 3:
        return CheckResult("Candidate ranking", False, "not enough forwards to rank", pd.DataFrame())
    table, _ = sim.rank(team_id, fwd, cands)
    table["box_rate"] = [sim.profiles[c].rates["log_box"] for c in table["player_in"]]
    rho = float(spearmanr(table["box_rate"], table["delta"])[0])
    table.insert(1, "name", [bundle.player_name(c) for c in table["player_in"]])
    summary = (f"{bundle.team_name(team_id)}: replacing {bundle.player_name(fwd)} with each of {len(table)} "
               f"forwards (≥ {RANK_MIN_MINUTES:.0f} league minutes); Spearman(box-entry rate, change) = {rho:+.2f} "
               "(pass if > 0)")
    shown = pd.concat([table.head(5), table.tail(5)])
    return CheckResult("Candidate ranking", rho > 0, summary, shown)


def run_all(sim: Simulator, bundle: Bundle) -> list[CheckResult]:
    return [self_swap(sim, bundle), forward_to_goalkeeper(sim, bundle), candidate_ranking(sim, bundle)]


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:+.3f}" if isinstance(v, float) and c in ("delta", "delta_per_90")
                                      else f"{v:.3f}" if isinstance(v, float) else str(v)
                                      for c, v in zip(cols, r.values)) + " |")
    return "\n".join(out)


def write_report(results: list[CheckResult], path, meta: dict) -> None:
    lines = [
        "# Player-swap simulator — sanity checks",
        "",
        "The simulator replaces a player by another player's season profile in every league match he "
        "played and lets the passing-network models (DeepSets + GNN, "
        f"{meta.get('n_models', '?')} networks) predict the team's non-penalty xG before and after. "
        "These checks have pass criteria fixed in `sim/checks.py` before they were run.",
        "",
        "| check | result | summary |",
        "| --- | --- | --- |",
        *[f"| {r.name} | {'pass' if r.passed else 'FAIL'} | {r.summary} |" for r in results],
        "",
    ]
    for r in results:
        lines += [f"## {r.name}", "", r.summary, "", _md(r.table), ""]
    lines += [
        "## What the numbers mean",
        "",
        "- `delta`: change in the team's predicted non-penalty xG per match, averaged over the matches the "
        "replaced player played and over all networks.",
        "- A candidate's profile comes from his own team; a player who thrives in a dominant side carries "
        "that context with him. Treat results as a simulation under the model's assumptions.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
