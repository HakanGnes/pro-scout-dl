"""Assemble the passing-network dataset: graphs, xG targets, a time-based split.

Split (fixed before any model is trained):
  * leagues: within each competition, the first 70% of gameweeks are for fitting
    (the last 5 of those are the validation set used for early stopping) and the
    remaining 30% are the test set;
  * World Cup and Euro: a separate test set the models never train on.
Both teams of a match always land in the same split.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from features.events import Event, MatchContext
from features.xg.shots import extract_shots, shots_to_records

from .graph import TeamGraph, build_team_graphs

FIT_SHARE = 0.70
VAL_GAMEWEEKS = 5
SPLITS = ("train", "val", "test", "test_intl")


def build(matches: Iterator[tuple[str, list[Event]]], contexts: dict[int, MatchContext],
          log=print) -> tuple[list[TeamGraph], pd.DataFrame]:
    """Graphs and shot records in one pass over the events."""
    graphs: list[TeamGraph] = []
    shots: list[dict] = []
    skipped = 0
    for k, (_label, events) in enumerate(matches, 1):
        ctx = contexts.get(events[0].match_id)
        if ctx is None or len(ctx.sides) != 2:
            skipped += 1
            continue
        graphs.extend(build_team_graphs(events, ctx))
        shots.extend(shots_to_records(extract_shots(events)))
        if k % 500 == 0:
            log(f"  {k:,} matches")
    if skipped:
        log(f"  {skipped} matches without a usable context skipped")
    return graphs, pd.DataFrame(shots)


def attach_targets(graphs: list[TeamGraph], shots: pd.DataFrame, xg) -> None:
    """Target = the team's non-penalty xG in the match, from the given xG model."""
    shots = shots.copy()
    shots["xg"] = xg.predict(shots)
    npxg = shots[shots["shot_type"] != "penalty"].groupby(["match_id", "team_id"])["xg"].sum()
    lookup = npxg.to_dict()
    for g in graphs:
        g.target = float(lookup.get((g.match_id, g.team_id), 0.0))


def assign_splits(graphs: list[TeamGraph]) -> Counter:
    last_week: dict[str, int] = defaultdict(int)
    for g in graphs:
        if not g.is_international and g.gameweek:
            last_week[g.competition] = max(last_week[g.competition], g.gameweek)
    for g in graphs:
        if g.is_international or not g.gameweek:
            g.split = "test_intl"
            continue
        cutoff = math.ceil(FIT_SHARE * last_week[g.competition])
        if g.gameweek > cutoff:
            g.split = "test"
        elif g.gameweek > cutoff - VAL_GAMEWEEKS:
            g.split = "val"
        else:
            g.split = "train"
    return Counter(g.split for g in graphs)


def save(graphs: list[TeamGraph], path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"graphs": graphs, "meta": meta or {}}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> tuple[list[TeamGraph], dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["graphs"], data["meta"]


def stats_frame(graphs: list[TeamGraph]) -> pd.DataFrame:
    """One row per graph: ids, split, target and the baseline features."""
    rows = []
    for g in graphs:
        rows.append({"match_id": g.match_id, "team_id": g.team_id, "competition": g.competition,
                     "gameweek": g.gameweek, "split": g.split, "target": g.target, **g.stats})
    return pd.DataFrame(rows)
