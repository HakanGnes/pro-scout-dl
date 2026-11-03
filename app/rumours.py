"""Streamlit page: how likely is a Turkish transfer rumour to come true?

    streamlit run app/rumours.py

Needs the saved model (python -m models.rumours fit). The "2026 rumours" tab reads
data/processed/rumours/scored_2026.csv (python -m models.rumours score-archive); the "2025 record" tab reads
the registered run's test predictions. Paths can be overridden with PROSCOUT_RUMOUR_MODEL,
PROSCOUT_RUMOUR_SCORED and PROSCOUT_RUMOUR_PREDICTIONS.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rumours.__main__ import ARTIFACTS, PREDICTIONS, SCORED  # noqa: E402

st.set_page_config(page_title="Pro-Scout · Transfer rumours", layout="wide")

PAPERS = ["fotomac", "fanatik", "aspor", "sporx", "hurriyet_spor", "milliyet_skorer", "ntvspor", "sabah_spor"]
MODEL = Path(os.environ.get("PROSCOUT_RUMOUR_MODEL", ARTIFACTS))
SCORED_FILE = Path(os.environ.get("PROSCOUT_RUMOUR_SCORED", SCORED))
PRED_FILE = Path(os.environ.get("PROSCOUT_RUMOUR_PREDICTIONS", PREDICTIONS))


@st.cache_resource(show_spinner="Loading the rumour model…")
def scorer():
    from models.rumours.live import Scorer
    return Scorer.load(MODEL, os.environ.get("PROSCOUT_DEVICE", "auto"))


def pct(p) -> str:
    return "–" if pd.isna(p) else f"{p:.0%}"


st.title("Transfer rumours: how likely is the move?")
st.caption("A rumour links a player to a club he is not at. The model gives the probability that Transfermarkt "
           "records the move within 180 days, from what is known without reading the headline's wording: how much "
           "and how widely the pair is in the news, how long it has been, the player's value and age. A registered "
           "test found that adding BERTurk's reading of the wording gives no measurable gain, so it is not used. "
           "On 2025 rumours the model ranked a true rumour above a false one 77% of the time (AUC 0.77). "
           "Player clubs and values come from Transfermarkt data that ends on 12 June 2026, so only headlines up "
           "to that day are scored; later moves are unknown to it. People without a market value in the two years "
           "before the headline (coaches, retired players) are left out.")

recent, typed, record = st.tabs(["2026 rumours", "Try a headline", "2025 record"])

# ---------------------------------------------------------------- 2026

with recent:
    if not SCORED_FILE.exists():
        st.info("Not scored yet. Run `python -m models.rumours score-archive`.")
    else:
        import json
        scored = pd.read_csv(SCORED_FILE, parse_dates=["date"])
        pair_file = SCORED_FILE.with_name(SCORED_FILE.stem + "_pairs.csv")
        check_file = SCORED_FILE.with_name(SCORED_FILE.stem + "_check.json")
        check = json.loads(check_file.read_text(encoding="utf-8")) if check_file.exists() else {}
        st.write(f"Rumours published from {scored['date'].min():%d %B} to {scored['date'].max():%d %B %Y}: after "
                 "the model's test year and before the Transfermarkt data ends, so no model has seen them. "
                 "*Outcome*: the move as Transfermarkt records it, or after its end as the archived news reports it "
                 "done (at least 3 headlines from 2 papers saying \"resmen\", \"imzaladı\", \"yeni transferi\"). "
                 "Outcomes are only used to check the model, never as its input.")
        if "auc_model" in check:
            lo, hi = check["auc_model_95"]
            closed = (f"the {check['claims_closed']:,} claims up to {pd.Timestamp(check['closed_until']):%d %B}, "
                      "whose 180-day window has closed" if "closed_until" in check else f"{check['claims']:,} claims")
            st.write(f"**How reliable were these probabilities?** Checked on {closed}: the outcome is known for "
                     f"{check['known']:,}, and {check['happened']:,} happened. AUC {check['auc_model']:.2f} "
                     f"(95% interval {lo:.2f}–{hi:.2f}); on the 2025 test it was 0.77. The two are not "
                     "measured alike: after 12 June only moves the papers report as done are seen, so a move that "
                     "happened can count here as not happened. Read each row as: of the claims given about this "
                     "probability, this share happened.")
            rel = pd.DataFrame(check["reliability"]).set_index("band")
            st.dataframe(rel.rename(columns={"predicted": "mean predicted", "happened": "share that happened"})
                         .style.format({"mean predicted": "{:.0%}", "share that happened": "{:.0%}"}), width="stretch")
            st.caption("A probability is not a verdict: of rumours given 15%, about 15 in 100 come true, and "
                       "the model cannot say which ones. Heavily reported sagas that end elsewhere (many papers, "
                       "many headlines) are its known weak spot.")
        per_pair = st.toggle("One row per player–club pair", value=True) and pair_file.exists()
        c1, c2, c3 = st.columns(3)
        club = c1.text_input("Club or player contains", "")
        min_p = c2.slider("Minimum probability", 0.0, 1.0, 0.0, 0.05)
        papers = c3.multiselect("Papers", sorted(scored["site"].dropna().unique()), disabled=per_pair)
        if per_pair:
            rows = pd.read_csv(pair_file, parse_dates=["first_seen", "last_seen"])
            pcol = "latest_p"
        else:
            rows = scored[scored["stance"] == "rumour"].copy()
            pcol = "p"
            if papers:
                rows = rows[rows["site"].isin(papers)]
        if club:
            k = club.lower()
            rows = rows[rows["club"].str.lower().str.contains(k, na=False)
                        | rows["player"].str.lower().str.contains(k, na=False)]
        rows = rows[rows[pcol] >= min_p]
        happened = rows["happened"] == 1
        if per_pair:
            st.write(f"{len(rows):,} player–club pairs, most recently reported first; {happened.sum():,} happened.")
            view = rows.assign(latest=rows["latest_p"].map(pct), highest=rows["highest_p"].map(pct),
                               first_seen=rows["first_seen"].dt.strftime("%Y-%m-%d"),
                               last_seen=rows["last_seen"].dt.strftime("%Y-%m-%d"))
            st.dataframe(view[["player", "club", "current_club", "articles", "papers", "first_seen", "last_seen",
                               "latest", "highest", "outcome"]], hide_index=True, width="stretch")
        else:
            st.write(f"{len(rows):,} rumour headlines, newest first; {happened.sum():,} about moves that happened.")
            view = rows.assign(probability=rows["p"].map(pct), date=rows["date"].dt.strftime("%Y-%m-%d"))
            st.dataframe(view[["date", "site", "player", "club", "current_club", "probability", "outcome", "title"]],
                         hide_index=True, width="stretch")

# ---------------------------------------------------------------- typed

with typed:
    st.write("Type a headline as a paper would write it. The model finds the player and the club, decides whether "
             "it is a rumour, and scores each player–club pair it finds.")
    title = st.text_input("Headline", "Fenerbahçe'den Victor Osimhen için yeni teklif", key="headline")
    c1, c2 = st.columns(2)
    paper = c1.selectbox("Paper", PAPERS)
    last_day = pd.Timestamp(os.environ.get("PROSCOUT_RUMOUR_DATA_END", "2026-06-12")).date()
    day = c2.date_input("Date (up to the end of the data)", last_day, max_value=last_day)
    if st.button("Score", type="primary") and title.strip():
        try:
            s = scorer()
        except SystemExit as exc:
            st.error(str(exc))
            st.stop()
        from etl.news.history import is_transfer_slug
        from models.rumours.live import headline_slug
        if not is_transfer_slug(headline_slug(title)):
            st.info("This headline has none of the transfer words the study's headlines were selected by "
                    "(transfer, teklif, imza, ...); the score is less reliable for it.")
        out = s.score(pd.DataFrame({"url": ["typed"], "site": [paper], "date": [pd.Timestamp(day)], "title": [title]}))
        if out.empty:
            st.warning("No active player together with a club he is not at was found in this headline.")
        for row in out.itertuples():
            st.subheader(f"{s.player_name(row.player_id)} → {s.club_name(row.club_id)}")
            if row.stance != "rumour":
                st.write(f"The rules read this as **{row.stance}**, not a rumour, so it is not scored "
                         "(the model was trained on rumours only).")
                continue
            st.metric("Probability", pct(row.p))
            st.caption(f"Current club (Transfermarkt): {s.club_name(row.current_club_id)}. The headline as the "
                       f"model handles it, names replaced: `{row.text}`. The wording itself does not change the "
                       "probability; the paper, the date and the news around the pair do.")

# ---------------------------------------------------------------- record

with record:
    if not PRED_FILE.exists():
        st.info("Run `python -m models.rumours run` to create the test predictions.")
    else:
        pred = pd.read_csv(PRED_FILE, parse_dates=["date"])
        p = pred["p_metadata"] if "p_metadata" in pred else pred["metadata"]
        st.write(f"The registered test (corrected run): {len(pred):,} rumour claims published in 2025, "
                 f"{pred['label'].mean():.0%} came true within 180 days. Each probability is the metadata model's, "
                 "the model in use, made before the outcome was known to it.")
        bins = pd.cut(p, [0, 0.1, 0.2, 0.3, 0.5, 1.0], include_lowest=True)
        table = pd.DataFrame({"p": p, "label": pred["label"], "bin": bins}).groupby("bin", observed=True).agg(
            claims=("label", "size"), predicted=("p", "mean"), came_true=("label", "mean"))
        table.index = table.index.astype(str)
        st.write("**When the model says X%, how often did it happen?**")
        st.dataframe(table.style.format({"predicted": "{:.0%}", "came_true": "{:.0%}"}), width="stretch")
        st.bar_chart(table[["predicted", "came_true"]])
        st.write("**Highest and lowest scored 2025 rumours** (text with names masked, as the model saw it)")
        show = pred.assign(probability=p.map(pct), outcome=np.where(pred["label"] == 1, "happened", "did not"),
                           date=pred["date"].dt.strftime("%Y-%m-%d"))
        cols = ["date", "site", "probability", "outcome", "text"]
        a, b = st.columns(2)
        a.dataframe(show.loc[p.sort_values(ascending=False).index[:15], cols], hide_index=True)
        b.dataframe(show.loc[p.sort_values().index[:15], cols], hide_index=True)
