"""Build one passing network per team per match.

Nodes are the players who were on the pitch; a directed edge i -> j counts
completed passes from i to j. Wyscout does not record who received a pass, so
the receiver is inferred: the next event (within RECEIVER_LOOKAHEAD events, same
period) by a different player of the same team. On the 2017/18 data this finds a
receiver for 99.0% of accurate passes.

Shots are deliberately left out of every feature (touches, positions): the target
of the models is the team's xG, which is computed from those shots.

Positions are medians, not means, so the rare possessions Wyscout records on the
wrong half (14 of 19,316 corners) do not move a player across the pitch.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from features.events import Event, MatchContext, sort_key

PASS, FREE_KICK, SHOT = 8, 3, 10
FREE_KICK_SHOTS = {33, 35}  # free-kick shot, penalty
TAG_ACCURATE = 1801

RECEIVER_LOOKAHEAD = 3
PERIOD_START_MIN = {"1H": 0, "2H": 45, "E1": 90, "E2": 105}

# Zones in Wyscout percent coordinates (105 x 68 m pitch).
FINAL_THIRD_X = 200 / 3
BOX_X = (105 - 16.5) / 105 * 100
BOX_Y = ((34 - 20.16) / 68 * 100, (34 + 20.16) / 68 * 100)
PROGRESSIVE_M = 10.0

ROLES = ("GK", "DF", "MD", "FW")
NODE_FEATURES = (
    "role_gk", "role_df", "role_md", "role_fw",
    "minutes_share", "is_starter",
    "log_passes", "log_received", "pass_accuracy",
    "median_x", "median_y",
    "log_final_third", "log_box", "log_progressive", "log_touches",
)
EDGE_FEATURES = ("log_passes", "mean_progress_10m")


@dataclass
class TeamGraph:
    match_id: int
    team_id: int
    competition: str
    gameweek: int | None
    is_international: bool
    side: str | None
    player_ids: list[int]
    x: np.ndarray            # (n_nodes, len(NODE_FEATURES)) float32
    edge_index: np.ndarray   # (2, n_edges) int64, rows = (source, target) node positions
    edge_attr: np.ndarray    # (n_edges, len(EDGE_FEATURES)) float32
    stats: dict[str, float] = field(default_factory=dict)  # graph-level features for the baseline
    target: float | None = None   # team non-penalty xG in the match
    split: str | None = None


def _is_shot(e: Event) -> bool:
    return e.type_id == SHOT or (e.type_id == FREE_KICK and e.subtype_id in FREE_KICK_SHOTS)


def _in_box(x: float, y: float) -> bool:
    return x >= BOX_X and BOX_Y[0] <= y <= BOX_Y[1]


def match_end_minute(events: list[Event]) -> float:
    last = max((PERIOD_START_MIN.get(e.period, 0) + e.event_sec / 60 for e in events if e.period != "P"), default=90.0)
    return max(90.0, last)


def minutes_played(ctx: MatchContext, team_id: int, end: float) -> dict[int, tuple[float, bool]]:
    """player_id -> (minutes on the pitch, started)."""
    start: dict[int, float] = {}
    stop: dict[int, float] = {}
    started: dict[int, bool] = {}
    for team, pid, is_starter, _ in ctx.lineups:
        if team == team_id and is_starter:
            start[pid], started[pid] = 0.0, True
    for team, p_in, p_out, minute in ctx.substitutions:
        if team != team_id or minute is None:
            continue
        minute = min(float(minute), end)
        start.setdefault(p_in, minute)
        started.setdefault(p_in, False)
        stop[p_out] = minute
    for team, pid, _, red in ctx.lineups:
        if team == team_id and red and pid in start:
            stop[pid] = min(stop.get(pid, end), float(red))
    return {pid: (max(0.0, min(stop.get(pid, end), end) - s), started[pid]) for pid, s in start.items()}


def _receiver(events: list[Event], i: int) -> Event | None:
    p = events[i]
    for j in range(i + 1, min(i + 1 + RECEIVER_LOOKAHEAD, len(events))):
        n = events[j]
        if n.period != p.period:
            return None
        if n.team_id == p.team_id:
            if n.player_id == p.player_id:
                return None  # the passer touched it again first
            if n.player_id:
                return n
    return None


def build_team_graphs(events: list[Event], ctx: MatchContext) -> list[TeamGraph]:
    """One graph per team of the match (shoot-outs ignored)."""
    events = sorted((e for e in events if e.period != "P"), key=sort_key)
    end = match_end_minute(events)
    graphs = []
    for team_id in sorted(ctx.sides):
        graphs.append(_team_graph(events, ctx, team_id, end))
    return graphs


def _team_graph(events: list[Event], ctx: MatchContext, team_id: int, end: float) -> TeamGraph:
    acc = defaultdict(lambda: defaultdict(float))
    xs: dict[int, list[float]] = defaultdict(list)
    ys: dict[int, list[float]] = defaultdict(list)
    edges: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])

    for i, e in enumerate(events):
        if e.team_id != team_id or not e.player_id or _is_shot(e):
            continue
        a = acc[e.player_id]
        a["touches"] += 1
        if e.x is not None and e.y is not None:
            xs[e.player_id].append(float(e.x))
            ys[e.player_id].append(float(e.y))
        if e.type_id != PASS:
            continue
        a["passes"] += 1
        if TAG_ACCURATE not in e.tags:
            continue
        a["accurate"] += 1
        sx, sy = float(e.x or 0), float(e.y or 0)
        ex = min(100.0, max(0.0, float(e.end_x if e.end_x is not None else sx)))
        ey = min(100.0, max(0.0, float(e.end_y if e.end_y is not None else sy)))
        progress_m = (ex - sx) * 1.05
        if sx < FINAL_THIRD_X <= ex:
            a["final_third"] += 1
        if _in_box(ex, ey) and not _in_box(sx, sy):
            a["box"] += 1
        if progress_m >= PROGRESSIVE_M:
            a["progressive"] += 1
        rec = _receiver(events, i)
        if rec is not None:
            acc[rec.player_id]["received"] += 1
            edge = edges[(e.player_id, rec.player_id)]
            edge[0] += 1
            edge[1] += progress_m

    minutes = minutes_played(ctx, team_id, end)
    players = sorted({p for p, (m, _) in minutes.items() if m > 0} | set(acc))
    pos = {p: k for k, p in enumerate(players)}

    team_acc = sum(acc[p]["accurate"] for p in players) / max(1.0, sum(acc[p]["passes"] for p in players))
    med_x = {p: float(np.median(xs[p])) / 100 for p in players if xs[p]}
    med_y = {p: float(np.median(ys[p])) / 100 for p in players if ys[p]}
    fill_x = float(np.mean(list(med_x.values()))) if med_x else 0.5
    fill_y = float(np.mean(list(med_y.values()))) if med_y else 0.5

    x = np.zeros((len(players), len(NODE_FEATURES)), dtype=np.float32)
    for p in players:
        a, k = acc[p], pos[p]
        role = ctx.roles.get(p)
        if role in ROLES:
            x[k, ROLES.index(role)] = 1.0
        mins, started = minutes.get(p, (0.0, False))
        x[k, 4] = mins / end
        x[k, 5] = float(started)
        x[k, 6] = math.log1p(a["passes"])
        x[k, 7] = math.log1p(a["received"])
        x[k, 8] = a["accurate"] / a["passes"] if a["passes"] else team_acc
        x[k, 9] = med_x.get(p, fill_x)
        x[k, 10] = med_y.get(p, fill_y)
        x[k, 11] = math.log1p(a["final_third"])
        x[k, 12] = math.log1p(a["box"])
        x[k, 13] = math.log1p(a["progressive"])
        x[k, 14] = math.log1p(a["touches"])

    pairs = sorted((s, t) for s, t in edges if s in pos and t in pos)
    edge_index = np.array([[pos[s] for s, _ in pairs], [pos[t] for _, t in pairs]], dtype=np.int64).reshape(2, -1)
    edge_attr = np.array(
        [[math.log1p(edges[k][0]), edges[k][1] / edges[k][0] / 10.0] for k in pairs], dtype=np.float32
    ).reshape(-1, len(EDGE_FEATURES))

    graph = TeamGraph(
        match_id=ctx.match_id, team_id=team_id, competition=ctx.competition, gameweek=ctx.gameweek,
        is_international=ctx.is_international, side=ctx.sides.get(team_id),
        player_ids=players, x=x, edge_index=edge_index, edge_attr=edge_attr,
    )
    graph.stats = graph_stats(players, acc, {k: edges[k] for k in pairs}, med_x, team_acc)
    return graph


def graph_stats(players, acc, edges, med_x, team_acc) -> dict[str, float]:
    """Hand-crafted team-level features from the same information the GNN sees."""
    import networkx as nx

    tot = {k: sum(acc[p][k] for p in players) for k in ("passes", "accurate", "final_third", "box", "progressive", "touches")}
    n = len(players)
    weights = np.array([v[0] for v in edges.values()], dtype=float)
    completed = weights.sum()
    out_strength = defaultdict(float)
    for (s, _), v in edges.items():
        out_strength[s] += v[0]
    shares = np.sort(np.array(list(out_strength.values())) / completed)[::-1] if completed else np.zeros(1)
    p = weights / completed if completed else np.zeros(1)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / math.log(len(p))) if len(p) > 1 else 0.0
    reciprocal = sum(1 for (s, t) in edges if (t, s) in edges)

    g = nx.Graph()
    g.add_nodes_from(players)
    for (s, t), v in edges.items():
        w = g[s][t]["weight"] + v[0] if g.has_edge(s, t) else v[0]
        g.add_edge(s, t, weight=w, distance=1.0 / w)
    betweenness = nx.betweenness_centrality(g, weight="distance") if n > 2 else {}
    clustering = nx.clustering(g, weight="weight") if n > 2 else {}
    touches = np.array([acc[q]["touches"] for q in players if q in med_x], dtype=float)
    xs = np.array([med_x[q] for q in players if q in med_x], dtype=float)

    return {
        "n_players": float(n),
        "passes": tot["passes"],
        "accurate_passes": tot["accurate"],
        "pass_accuracy": team_acc,
        "final_third_entries": tot["final_third"],
        "box_entries": tot["box"],
        "progressive_passes": tot["progressive"],
        "touches": tot["touches"],
        "completed_to_receiver": float(completed),
        "n_edges": float(len(edges)),
        "density": len(edges) / (n * (n - 1)) if n > 1 else 0.0,
        "reciprocity": reciprocal / len(edges) if edges else 0.0,
        "top_passer_share": float(shares[0]) if len(shares) else 0.0,
        "top3_passer_share": float(shares[:3].sum()) if len(shares) else 0.0,
        "edge_entropy": entropy,
        "mean_progress_m": float(sum(v[1] for v in edges.values()) / completed) if completed else 0.0,
        "team_median_x": float((xs * touches).sum() / touches.sum()) if touches.sum() else 0.5,
        "max_betweenness": float(max(betweenness.values())) if betweenness else 0.0,
        "mean_clustering": float(np.mean(list(clustering.values()))) if clustering else 0.0,
    }


BASELINE_FEATURES = (
    "n_players", "passes", "accurate_passes", "pass_accuracy", "final_third_entries", "box_entries",
    "progressive_passes", "touches", "completed_to_receiver", "n_edges", "density", "reciprocity",
    "top_passer_share", "top3_passer_share", "edge_entropy", "mean_progress_m", "team_median_x",
    "max_betweenness", "mean_clustering",
)
VOLUME_FEATURES = ("passes", "final_third_entries", "box_entries", "progressive_passes")
