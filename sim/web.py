"""Export what the tactics board needs to run in the browser: the ten networks' weights, every league team's
match graphs, player profiles and names, and each team's season passing network.

    python -m sim export-web            -> web/public/data/

Numbers that the networks read are stored as base64 float32 / int32, so the browser sees exactly the values
PyTorch sees. `golden()` writes reference predictions that the TypeScript port is tested against.
"""
from __future__ import annotations

import base64
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from features.pass_networks.graph import EDGE_FEATURES, NODE_FEATURES, ROLES, TeamGraph

from .profiles import COUNT_FEATURES, MIN_SHARE
from .swap import F, Simulator

WEB_DATA = Path(__file__).resolve().parents[1] / "web" / "public" / "data"
FORMAT = 1


def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def export_models(ensemble) -> dict:
    """Weights of every network, by parameter name, plus the input normalisation."""
    norm = ensemble.normalizer
    nets = []
    for name, net in ensemble.nets:
        arch = name.split("-")[0]
        params = {}
        for key, t in net.state_dict().items():
            a = t.detach().cpu().numpy().astype(np.float32)
            params[key] = {"shape": list(a.shape), "data": b64(a, np.float32)}
        nets.append({"name": name, "arch": arch, "params": params})
    return {"format": FORMAT, "node_features": list(NODE_FEATURES), "edge_features": list(EDGE_FEATURES),
            "x_mean": b64(norm.x_mean, np.float32), "x_std": b64(norm.x_std, np.float32),
            "e_mean": b64(norm.e_mean, np.float32), "e_std": b64(norm.e_std, np.float32),
            "layers": int(ensemble.meta.get("cfg", {}).get("layers", 3)), "nets": nets}


def graph_record(g: TeamGraph) -> dict:
    return {"match_id": int(g.match_id), "gameweek": int(g.gameweek or 0), "side": g.side,
            "target": None if g.target is None else float(g.target), "player_ids": [int(p) for p in g.player_ids],
            "x": b64(g.x, np.float32), "edge_index": b64(g.edge_index, np.int32), "edge_attr": b64(g.edge_attr, np.float32),
            "n_nodes": len(g.player_ids), "n_edges": int(g.edge_index.shape[1])}


def team_record(sim: Simulator, bundle, team_id: int) -> dict:
    games = sim.by_team[team_id]
    squad = defaultdict(lambda: {"matches": 0, "starts": 0, "share": 0.0, "x": 0.0, "y": 0.0})
    links: dict[tuple[int, int], float] = defaultdict(float)
    for g in games:
        for k, pid in enumerate(g.player_ids):
            share = float(g.x[k, F["minutes_share"]])
            if share <= 0:
                continue
            s = squad[pid]
            s["matches"] += 1
            s["starts"] += int(g.x[k, F["is_starter"]] > 0.5)
            s["share"] += share
            s["x"] += float(g.x[k, F["median_x"]]) * share
            s["y"] += float(g.x[k, F["median_y"]]) * share
        for (a, b), e in zip(g.edge_index.T, g.edge_attr):
            pa, pb = g.player_ids[int(a)], g.player_ids[int(b)]
            links[(min(pa, pb), max(pa, pb))] += math.expm1(float(e[0]))     # completed passes, both ways
    rows = []
    for pid, s in squad.items():
        p = sim.profiles.get(pid)
        rows.append({"player_id": int(pid), "name": bundle.player_name(pid), "role": p.role if p else None,
                     "matches": s["matches"], "starts": s["starts"], "minutes": round(s["share"] * 90),
                     "x": round(s["x"] / s["share"], 4), "y": round(s["y"] / s["share"], 4)})
    rows.sort(key=lambda r: (-r["starts"], -r["minutes"]))
    keepers = [r for r in rows if r["role"] == "GK"]
    outfield = [r for r in rows if r["role"] != "GK"]
    xi = ([keepers[0]["player_id"]] if keepers else []) + [r["player_id"] for r in outfield[:11 - min(1, len(keepers))]]
    n = len(games)
    pass_links = [[a, b, round(v / n, 2)] for (a, b), v in links.items() if a in xi and b in xi]
    return {"team_id": int(team_id), "name": bundle.team_name(team_id),
            "competition": bundle.teams.get(team_id, {}).get("competition"), "squad": rows, "xi": xi,
            "links_per_match": pass_links, "matches": [graph_record(g) for g in games]}


def profile_record(p, bundle) -> list:
    """[id, name, role, minutes, matches, team ids, six per-match rates, pass accuracy, x, y]"""
    return [int(p.player_id), bundle.player_name(p.player_id), p.role, round(p.minutes, 1), p.matches,
            sorted(int(t) for t in p.team_ids), *[float(p.rates[f]) for f in COUNT_FEATURES],
            float(p.pass_accuracy), float(p.median_x), float(p.median_y)]


def export(sim: Simulator, bundle, out: Path = WEB_DATA, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "teams").mkdir(exist_ok=True)
    models = export_models(sim.ensemble)
    (out / "models.json").write_text(json.dumps(models), encoding="utf-8")
    teams = []
    for tid in sorted(sim.by_team, key=bundle.team_name):
        rec = team_record(sim, bundle, tid)
        (out / "teams" / f"{tid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        teams.append({"team_id": int(tid), "name": rec["name"], "competition": rec["competition"],
                      "matches": len(rec["matches"])})
    profiles = [profile_record(p, bundle) for p in sim.profiles.values()]
    (out / "players.json").write_text(json.dumps({
        "fields": ["player_id", "name", "role", "minutes", "matches", "team_ids", *COUNT_FEATURES,
                   "pass_accuracy", "median_x", "median_y"],
        "roles": list(ROLES), "min_share": MIN_SHARE, "players": profiles}, ensure_ascii=False), encoding="utf-8")
    index = {"format": FORMAT, "teams": teams, "n_models": len(sim.ensemble.nets), "models": sim.ensemble.names,
             "source": "Wyscout public dataset, 2017/18 (Pappalardo et al. 2019, CC BY 4.0)"}
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*.json"))
    log(f"{len(teams)} teams, {len(profiles):,} player profiles, {len(sim.ensemble.nets)} networks -> {out} "
        f"({size / 1e6:.1f} MB)")
    return index


def golden(sim: Simulator, bundle, path: Path, cases: list[tuple[int, dict[int, int]]]) -> None:
    """Reference values for the TypeScript port: each network's prediction on a few graphs, and swap results."""
    team0 = cases[0][0]
    games = sim.by_team[team0][:3]
    preds = sim.ensemble.predict(games)
    out = {"graphs": [{"team_id": int(team0), "match_id": int(g.match_id), "pred": preds[:, j].tolist()}
                      for j, g in enumerate(games)], "swaps": []}
    for team, swaps in cases:
        r = sim.swap_many(team, swaps)
        out["swaps"].append({"team_id": int(team), "swaps": {str(k): int(v) for k, v in swaps.items()},
                             "matches": r.matches, "skipped": r.skipped, "team_matches": r.team_matches,
                             "delta": r.delta, "se_matches": r.se_matches, "sd_models": r.sd_models,
                             "before": r.before, "season_delta": r.season_delta,
                             "per_match_delta": r.per_match["delta"].tolist()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")



# ---------------------------------------------------------------- best replacements (precomputed)

RANK_TOP = 15


def team_rankings(sim: Simulator, team_id: int, xi: list[int], min_minutes: float = 900.0, top: int = RANK_TOP) -> dict:
    """For each player of the board's XI: the best candidates of his role from other teams (league minutes
    ≥ min_minutes), ranked by the change in npxG per match he played, as `python -m sim swap --rank`."""
    from .swap import candidates_by_role

    out = {}
    for pid in xi:
        p = sim.profiles.get(pid)
        if p is None:
            continue
        cands = candidates_by_role(sim.profiles, p.role, min_minutes, exclude_team=team_id)
        table, _ = sim.rank(team_id, pid, cands)
        table = table[table["matches"] > 0].head(top)
        out[str(pid)] = {"role": p.role, "candidates": len(cands),
                         "best": [[int(r.player_in), round(float(r.delta), 5), round(float(r.se_matches), 5),
                                   round(float(r.sd_models), 5), int(r.matches)] for r in table.itertuples()]}
    return {"team_id": int(team_id), "min_minutes": min_minutes,
            "fields": ["player_in", "delta", "se_matches", "sd_models", "matches"], "players": out}


def _rank_worker(job):
    import torch

    torch.set_num_threads(1)
    dataset, bundle_path, model, out, teams = job
    from .__main__ import load_simulator

    sim, bundle = load_simulator(dataset, bundle_path, model)
    for tid in teams:
        path = out / "rankings" / f"{tid}.json"
        if path.exists():
            continue
        xi = json.loads((out / "teams" / f"{tid}.json").read_text(encoding="utf-8"))["xi"]
        rec = team_rankings(sim, tid, xi)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        tmp.replace(path)
        print(f"  {bundle.team_name(tid)} done", flush=True)
    return len(teams)


def rankings(dataset: Path, bundle_path: Path, model: Path, out: Path = WEB_DATA, workers: int = 4, log=print) -> None:
    """Rank candidates for every team's XI, `workers` teams at a time. Teams already done are skipped, so an
    interrupted run can be resumed."""
    import multiprocessing as mp

    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    (out / "rankings").mkdir(exist_ok=True)
    todo = [t["team_id"] for t in index["teams"] if not (out / "rankings" / f"{t['team_id']}.json").exists()]
    log(f"{len(index['teams']) - len(todo)} teams already ranked, {len(todo)} to go, {workers} workers")
    if not todo:
        return
    chunks = [todo[i::workers] for i in range(workers) if todo[i::workers]]
    jobs = [(dataset, bundle_path, model, out, c) for c in chunks]
    if len(jobs) == 1:
        _rank_worker(jobs[0])
        return
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        pool.map(_rank_worker, jobs)
