"""Streamlit page: swap a player and see how a team's xG would change.

    streamlit run app/swap.py

Needs data/processed/pass_networks.pkl, data/processed/sim_bundle.pkl and
models/artifacts/pass_gnn.pt (see the README). Paths can be overridden with the
PROSCOUT_DATASET, PROSCOUT_BUNDLE and PROSCOUT_MODEL environment variables.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.__main__ import BUNDLE, DATASET, MODEL, load_simulator  # noqa: E402
from sim.swap import candidates_by_role  # noqa: E402

st.set_page_config(page_title="Pro-Scout · Player swap", layout="wide")


@st.cache_resource(show_spinner="Loading graphs and networks…")
def simulator():
    return load_simulator(Path(os.environ.get("PROSCOUT_DATASET", DATASET)),
                          Path(os.environ.get("PROSCOUT_BUNDLE", BUNDLE)),
                          Path(os.environ.get("PROSCOUT_MODEL", MODEL)))


def player_label(bundle, sim, pid: int) -> str:
    p = sim.profiles.get(pid)
    role = p.role if p else "?"
    minutes = f"{p.minutes:,.0f}′" if p else "–"
    return f"{bundle.player_name(pid)} · {role} · {minutes}"


try:
    sim, bundle = simulator()
except SystemExit as exc:  # a missing file: show the command that creates it
    st.error(str(exc))
    st.stop()

st.title("Player swap: how would the team's xG change?")
st.caption("Replace a player by another player's season profile in every league match he played; "
           f"{len(sim.ensemble.nets)} passing-network models (DeepSets + GNN) predict the team's "
           "non-penalty xG before and after. A simulation under the models' assumptions, not a forecast.")

# ---------------------------------------------------------------- selection

competitions = sorted({bundle.teams.get(t, {}).get("competition", "?") for t in sim.by_team})
c1, c2, c3 = st.columns(3)
competition = c1.selectbox("Competition", competitions)
teams = sorted((t for t in sim.by_team if bundle.teams.get(t, {}).get("competition", "?") == competition),
               key=bundle.team_name)
team = c2.selectbox("Team", teams, format_func=bundle.team_name)
roster = sim.team_players(team)
roster = roster[roster["player_id"].map(lambda p: int(p) in sim.profiles)]
player_out = c3.selectbox("Player to replace", [int(p) for p in roster["player_id"]],
                          format_func=lambda p: player_label(bundle, sim, p))
out_role = sim.profiles[player_out].role

mode = st.radio("What to do", ["Compare one candidate", "Rank candidates"], horizontal=True)
min_minutes = st.slider("Candidates' minimum league minutes", 0, 3000, 900, step=90)

# ---------------------------------------------------------------- one candidate

if mode == "Compare one candidate":
    pool = sorted(candidates_by_role(sim.profiles, None, min_minutes, exclude_team=team),
                  key=lambda p: (sim.profiles[p].role != out_role, bundle.player_name(p)))
    candidate = st.selectbox("Candidate (same role listed first)", pool,
                             format_func=lambda p: player_label(bundle, sim, p))
    result = sim.swap(team, player_out, candidate)
    self_check = sim.swap(team, player_out, player_out)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Change in npxG per match", f"{result.delta:+.3f}",
              help="Mean over the matches the replaced player played and over all networks.")
    m2.metric("± across matches", f"{result.se_matches:.3f}", help="Standard error of the mean over matches.")
    m3.metric("± across networks", f"{result.sd_models:.3f}", help="Spread of the season average between networks.")
    m4.metric("Same player (sanity check)", f"{self_check.delta:+.3f}",
              help="Replacing him by his own season profile; should be close to zero.")

    st.write(f"Over **{result.matches}** matches, {bundle.team_name(team)} averaged **{result.actual:.2f}** npxG "
             f"(models: {result.before:.2f}). With {bundle.player_name(candidate)} instead of "
             f"{bundle.player_name(player_out)}: **{result.before + result.delta:.2f}**."
             + (f" {result.skipped} matches skipped: the candidate was already in that line-up." if result.skipped else ""))

    pm = result.per_match.sort_values("gameweek")
    chart = pd.DataFrame({"Before (models)": pm["before"].to_numpy(), "After swap": pm["after"].to_numpy()},
                         index=pm["gameweek"].to_numpy())
    st.line_chart(chart, x_label="Gameweek", y_label="Team npxG")
    with st.expander("Match by match"):
        st.dataframe(pm.rename(columns={"minutes_share": "share of match played", "actual": "actual npxG"}).round(3),
                     hide_index=True)

# ---------------------------------------------------------------- ranking

else:
    role = st.selectbox("Role", ["GK", "DF", "MD", "FW", "any"], index=["GK", "DF", "MD", "FW"].index(out_role)
                        if out_role in ("GK", "DF", "MD", "FW") else 4)
    top = st.slider("Show the top", 5, 50, 15)
    pool = candidates_by_role(sim.profiles, None if role == "any" else role, min_minutes, exclude_team=team)
    st.write(f"{len(pool):,} candidates.")
    if st.button("Rank", type="primary"):
        with st.spinner("Simulating every candidate…"):
            table, _ = sim.rank(team, player_out, pool)
        table.insert(1, "player", [bundle.player_name(p) for p in table["player_in"]])
        table.insert(2, "team(s)", [", ".join(sorted(bundle.team_name(t) for t in sim.profiles[p].team_ids))
                                    for p in table["player_in"]])
        st.dataframe(table.head(top).drop(columns=["player_in"]).round(3), hide_index=True)
        st.caption("A candidate's profile comes from his own team: a player in a dominant side carries that "
                   "context with him.")
