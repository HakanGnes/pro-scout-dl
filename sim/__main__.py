"""Command line: python -m sim <command>

  build   player profiles + names -> data/processed/sim_bundle.pkl
  check   sanity checks with fixed pass criteria -> reports/swap_sim/README.md
  swap    one swap, or a ranking of candidates:
          python -m sim swap --team "Arsenal" --out "A. Lacazette" --in "H. Kane"
          python -m sim swap --team "Arsenal" --out "A. Lacazette" --rank --top 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from etl.config import DEFAULT_WYSCOUT_DIR, REPO_ROOT

DATASET = REPO_ROOT / "data" / "processed" / "pass_networks.pkl"
BUNDLE = REPO_ROOT / "data" / "processed" / "sim_bundle.pkl"
MODEL = REPO_ROOT / "models" / "artifacts" / "pass_gnn.pt"
REPORT = REPO_ROOT / "reports" / "swap_sim" / "README.md"


def _resolve(bundle, query: str, kind: str, restrict: set[int] | None = None) -> int:
    if query.isdigit():
        return int(query)
    hits = bundle.find(query, kind)
    if restrict is not None:
        hits = [h for h in hits if h in restrict]
    if not hits:
        raise SystemExit(f"No {kind} matches {query!r}")
    if len(hits) > 1:
        names = ", ".join(f"{(bundle.player_name if kind == 'player' else bundle.team_name)(h)} ({h})" for h in hits[:8])
        raise SystemExit(f"{query!r} matches several {kind}s: {names}. Use the id.")
    return hits[0]


def load_simulator(dataset_path: Path, bundle_path: Path, model_path: Path, device: str = "cpu"):
    from features.pass_networks import dataset
    from models.gnn.ensemble import Ensemble

    from . import bundle as B
    from .swap import Simulator

    for p, cmd in ((dataset_path, "python -m features.pass_networks build"), (bundle_path, "python -m sim build"),
                   (model_path, "python -m models.gnn evaluate")):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run: {cmd}")
    graphs, _ = dataset.load(dataset_path)
    bundle = B.load(bundle_path)
    ensemble = Ensemble.load(model_path, device)
    return Simulator(graphs, bundle.profiles, ensemble), bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim", description="Player-swap simulation")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--device", default="cpu")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="player profiles and names")
    b.add_argument("--source", choices=("db", "raw"), default="db")
    b.add_argument("--raw-dir", type=Path, default=DEFAULT_WYSCOUT_DIR)
    b.add_argument("--database-url")

    c = sub.add_parser("check", help="sanity checks -> report")
    c.add_argument("--report", type=Path, default=REPORT)

    s = sub.add_parser("swap", help="simulate one swap or rank candidates")
    s.add_argument("--team", required=True)
    s.add_argument("--out", required=True, dest="player_out")
    s.add_argument("--in", dest="player_in")
    s.add_argument("--rank", action="store_true", help="rank all regular players of the same role")
    s.add_argument("--min-minutes", type=float, default=900.0)
    s.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    if args.command == "build":
        from features.pass_networks import dataset

        from . import bundle as B

        if not args.dataset.exists():
            raise SystemExit(f"Missing {args.dataset}. Run: python -m features.pass_networks build")
        graphs, meta = dataset.load(args.dataset)
        if args.source == "raw":
            players, teams = B.names_raw(args.raw_dir)
        else:
            from etl.db import connect
            with connect(args.database_url) as conn:
                players, teams = B.names_db(conn)
        bundle = B.build_bundle(graphs, players, teams, meta)
        B.save(bundle, args.bundle)
        roles = {}
        for p in bundle.profiles.values():
            roles[p.role] = roles.get(p.role, 0) + 1
        regular = sum(1 for p in bundle.profiles.values() if p.minutes >= 900)
        print(f"{len(bundle.profiles):,} player profiles ({regular:,} with ≥ 900 league minutes); by role: "
              + ", ".join(f"{k}={v}" for k, v in sorted(roles.items(), key=lambda kv: str(kv[0]))))
        print(f"Saved {args.bundle}")
        return 0

    sim, bundle = load_simulator(args.dataset, args.bundle, args.model, args.device)

    if args.command == "check":
        from .checks import run_all, write_report

        results = run_all(sim, bundle)
        for r in results:
            print(f"[{'pass' if r.passed else 'FAIL'}] {r.name}: {r.summary}")
        write_report(results, args.report, {"n_models": len(sim.ensemble.nets)})
        print(f"Wrote {args.report}")
        return 0 if all(r.passed for r in results) else 1

    team = _resolve(bundle, args.team, "team", set(sim.by_team))
    roster = set(int(p) for p in sim.team_players(team)["player_id"])
    out = _resolve(bundle, args.player_out, "player", roster)
    print(f"{bundle.team_name(team)}: replacing {bundle.player_name(out)} "
          f"({sim.profiles[out].role if out in sim.profiles else '?'})")
    if args.rank:
        from .swap import candidates_by_role

        role = sim.profiles[out].role if out in sim.profiles else None
        cands = candidates_by_role(sim.profiles, role, args.min_minutes, exclude_team=team)
        table, _ = sim.rank(team, out, cands)
        table.insert(1, "name", [bundle.player_name(c) for c in table["player_in"]])
        print(f"{len(table)} candidates ({role}, ≥ {args.min_minutes:.0f} league minutes); top {args.top}:")
        print(table.head(args.top).round(3).to_string(index=False))
        return 0
    if not args.player_in:
        raise SystemExit("Give --in PLAYER or --rank")
    cand = _resolve(bundle, args.player_in, "player", set(sim.profiles))
    r = sim.swap(team, out, cand)
    print(f"  with {bundle.player_name(cand)}: {r.delta:+.3f} npxG per match over {r.matches} matches "
          f"(± {r.se_matches:.3f} across matches, ± {r.sd_models:.3f} across {len(sim.ensemble.nets)} networks); "
          f"before {r.before:.2f}, actual {r.actual:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
