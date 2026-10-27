"""Build the market-value samples: one per (player, anchor date).

Everything for an anchor t is computed from records dated before t (the point-in-time rule in the
pre-registration). The output holds, for every sample:

  seq     (36, len(SEQ_CHANNELS)) raw monthly history, oldest month first, last month = the month before t
  static  dict of static inputs (age, position, league, ...)
  y       log V(t+12) - log V(t)
  meta    player, anchor, split, values, dates

Tabular summaries for the GBM are derived from seq + static in features.market_value.tabular, so all
models see the same information.
"""
from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from etl.transfermarkt.quality import COVERED_LEAGUES

MONTHS = 36
ACTIVE_DAYS = 365      # league appearance and a valuation in the year before t
HORIZON_DAYS = 365     # target: last valuation in [t, t + 365 days)
AGE_RANGE = (16, 40)
HEIGHT_RANGE = (150, 215)

SEQ_CHANNELS = (
    "log_value",         # as of month end, carried forward; 0 before the first valuation
    "has_value",
    "revalued",          # a valuation dated in this month
    "log_value_rel",     # log_value - log V(t); 0 before the first valuation
    "league_min90",
    "league_apps",
    "goals",             # all competitions
    "assists",
    "cards",             # yellow + red
    "europe_min90",
    "cup_min90",
    "transfer",          # a transfer dated in this month
    "transfer_fee_log",  # log(1 + fee in EUR m); 0 if unknown or free
    "league_strength",   # strength (at t) of the league played most this month; 0 if none
)
CH = {name: i for i, name in enumerate(SEQ_CHANNELS)}

POSITIONS = ("Goalkeeper", "Defender", "Midfield", "Attack", "Missing")
SUB_POSITIONS = ("Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield", "Central Midfield",
                 "Attacking Midfield", "Left Midfield", "Right Midfield", "Left Winger", "Right Winger",
                 "Second Striker", "Centre-Forward", "Missing")
FEET = ("right", "left", "both", "Missing")

TRAIN_ANCHORS = [date(y, m, 1) for y in range(2013, 2021) for m in (1, 7) if date(2013, 7, 1) <= date(y, m, 1) <= date(2020, 7, 1)]
VAL_ANCHORS = [date(2021, 7, 1), date(2022, 1, 1)]
TEST_ANCHORS = [date(2023, 1, 1), date(2023, 7, 1), date(2024, 1, 1), date(2024, 7, 1), date(2025, 1, 1)]
SPLIT_OF = {**{a: "train" for a in TRAIN_ANCHORS}, **{a: "val" for a in VAL_ANCHORS}, **{a: "test" for a in TEST_ANCHORS}}

Log = Callable[[str], None]


@dataclass
class Samples:
    seq: np.ndarray                      # (N, MONTHS, C) float32
    static: pd.DataFrame                 # N rows
    y: np.ndarray                        # (N,) float64
    meta: pd.DataFrame                   # N rows
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Samples":
        idx = np.flatnonzero(mask)
        return Samples(self.seq[idx], self.static.iloc[idx].reset_index(drop=True), self.y[idx],
                       self.meta.iloc[idx].reset_index(drop=True), self.info)

    def split(self, name: str) -> "Samples":
        return self.subset(self.meta["split"].to_numpy() == name)


# ---------------------------------------------------------------- monthly long tables


def _month(d: pd.Series) -> np.ndarray:
    return (d.dt.year * 12 + d.dt.month - 1).to_numpy()


def _anchor_month(t: date) -> int:
    return t.year * 12 + t.month - 1


def competition_group(competitions: pd.DataFrame) -> Callable[[str], str]:
    """league (the 14 covered first divisions), europe (UEFA), national (national teams) or cup (the rest).

    National-team appearances exist only from December 2025 (after every anchor), so they are kept
    apart from cups but not used as an input.
    """
    sub = dict(zip(competitions["competition_id"], competitions["sub_type"].fillna("")))
    typ = dict(zip(competitions["competition_id"], competitions["type"].fillna("")))

    def group(cid: str) -> str:
        if cid in COVERED_LEAGUES:
            return "league"
        if sub.get(cid, "").startswith("uefa_"):
            return "europe"
        if typ.get(cid) == "national_team_competition":
            return "national"
        return "cup"
    return group


def monthly_appearances(appearances: pd.DataFrame, competitions: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, month): minutes per competition group, apps, goals, assists, cards, main league."""
    group = competition_group(competitions)
    a = appearances.copy()
    groups = {cid: group(cid) for cid in a["competition_id"].unique()}
    a["group"] = a["competition_id"].map(groups)
    a["m"] = _month(a["date"])
    a["cards"] = a["yellow_cards"] + a["red_cards"]
    for g in ("league", "europe", "cup", "national"):
        a[f"{g}_min"] = np.where(a["group"] == g, a["minutes_played"], 0)
    a["league_apps"] = (a["group"] == "league").astype(int)
    agg = a.groupby(["player_id", "m"], sort=True)[
        ["league_min", "league_apps", "goals", "assists", "cards", "europe_min", "cup_min", "national_min"]].sum()
    lg = a[a["group"] == "league"].groupby(["player_id", "m", "competition_id"])["minutes_played"].sum()
    lg = lg.reset_index().sort_values(["player_id", "m", "minutes_played", "competition_id"],
                                      ascending=[True, True, False, True])
    main = lg.drop_duplicates(["player_id", "m"]).set_index(["player_id", "m"])["competition_id"]
    out = agg.join(main.rename("league")).reset_index()
    return out.astype({c: "float64" for c in agg.columns})


# ---------------------------------------------------------------- per-anchor construction


def _latest_before(df: pd.DataFrame, date_col: str, t: pd.Timestamp) -> pd.DataFrame:
    before = df[df[date_col] < t]
    return before.groupby("player_id", sort=False).tail(1)


def _league_strength(app_window: pd.DataFrame, current_value: pd.Series) -> dict[str, float]:
    """Median log value (at t) of players with league minutes in the window, per league."""
    played = app_window[(app_window["league_min"] > 0)][["player_id", "league"]].drop_duplicates()
    played = played.assign(lv=np.log(played["player_id"].map(current_value)))
    return played.dropna().groupby("league")["lv"].median().to_dict()


def build_anchor(t: date, tables: dict[str, pd.DataFrame], monthly: pd.DataFrame,
                 log: Log = print) -> tuple[Samples, dict]:
    T = pd.Timestamp(t)
    M = _anchor_month(t)
    vals = tables["player_valuations"]
    vals = vals[vals["market_value_in_eur"] > 0]
    players = tables["players"].set_index("player_id")

    # --- population
    app_year = monthly[(monthly["m"] >= M - 12) & (monthly["m"] < M)]
    active = set(app_year.loc[app_year["league_apps"] > 0, "player_id"])
    cur = _latest_before(vals, "date", T)
    cur = cur[(cur["date"] >= T - pd.Timedelta(days=ACTIVE_DAYS)) & cur["player_id"].isin(active)]
    cur = cur[cur["player_id"].isin(players.index)]
    dob = players.loc[cur["player_id"], "date_of_birth"].to_numpy()
    age = (T - pd.to_datetime(dob)).days / 365.25
    cur = cur[(~np.isnan(age)) & (age >= AGE_RANGE[0]) & (age <= AGE_RANGE[1])]
    n_population = len(cur)

    # --- target
    window = vals[(vals["date"] >= T) & (vals["date"] < T + pd.Timedelta(days=HORIZON_DAYS))]
    fut = window.groupby("player_id", sort=False).tail(1).set_index("player_id")
    cur = cur[cur["player_id"].isin(fut.index)].reset_index(drop=True)
    pid = cur["player_id"].to_numpy()
    n = len(pid)
    v_now = cur["market_value_in_eur"].to_numpy(dtype=np.float64)
    v_fut = fut.loc[pid, "market_value_in_eur"].to_numpy(dtype=np.float64)
    y = np.log(v_fut) - np.log(v_now)

    # --- monthly tensor
    seq = np.zeros((n, MONTHS, len(SEQ_CHANNELS)), dtype=np.float32)
    row_of = pd.Series(np.arange(n), index=pid)
    start = M - MONTHS

    current_value = pd.Series(v_now, index=pid)
    strength = _league_strength(app_year, current_value)

    mw = monthly[(monthly["m"] >= start) & (monthly["m"] < M) & monthly["player_id"].isin(row_of.index)]
    r, k = row_of.loc[mw["player_id"]].to_numpy(), (mw["m"] - start).to_numpy()
    for col, ch, scale in (("league_min", "league_min90", 90.0), ("league_apps", "league_apps", 1.0),
                           ("goals", "goals", 1.0), ("assists", "assists", 1.0), ("cards", "cards", 1.0),
                           ("europe_min", "europe_min90", 90.0), ("cup_min", "cup_min90", 90.0)):
        seq[r, k, CH[ch]] = mw[col].to_numpy() / scale
    seq[r, k, CH["league_strength"]] = mw["league"].map(strength).fillna(0.0).to_numpy()

    # valuations: last value in each month, carried forward from the last value before the window
    pv = vals[vals["player_id"].isin(row_of.index) & (vals["date"] < T)]
    pv = pv.assign(m=_month(pv["date"]))
    before = pv[pv["m"] < start].groupby("player_id").tail(1)
    inside = pv[pv["m"] >= start].groupby(["player_id", "m"]).tail(1)
    lv = np.full((n, MONTHS), np.nan)
    lv[row_of.loc[inside["player_id"]].to_numpy(), (inside["m"] - start).to_numpy()] = np.log(
        inside["market_value_in_eur"].to_numpy(dtype=np.float64))
    seq[row_of.loc[inside["player_id"]].to_numpy(), (inside["m"] - start).to_numpy(), CH["revalued"]] = 1.0
    carry = np.full(n, np.nan)
    carry[row_of.loc[before["player_id"]].to_numpy()] = np.log(before["market_value_in_eur"].to_numpy(dtype=np.float64))
    for j in range(MONTHS):
        carry = np.where(np.isnan(lv[:, j]), carry, lv[:, j])
        lv[:, j] = carry
    has = ~np.isnan(lv)
    seq[:, :, CH["has_value"]] = has
    seq[:, :, CH["log_value"]] = np.where(has, lv, 0.0)
    seq[:, :, CH["log_value_rel"]] = np.where(has, lv - np.log(v_now)[:, None], 0.0)
    assert np.allclose(lv[:, -1], np.log(v_now)), "last month must hold the current value"

    tr = tables["transfers"]
    tr = tr[tr["player_id"].isin(row_of.index) & (tr["transfer_date"] < T)]
    tr = tr.assign(m=_month(tr["transfer_date"]))
    tr = tr[tr["m"] >= start]
    if len(tr):
        rr, kk = row_of.loc[tr["player_id"]].to_numpy(), (tr["m"] - start).to_numpy()
        seq[rr, kk, CH["transfer"]] = 1.0
        fee = np.log1p(tr["transfer_fee"].fillna(0.0).clip(lower=0).to_numpy() / 1e6)
        np.maximum.at(seq[:, :, CH["transfer_fee_log"]], (rr, kk), fee.astype(np.float32))

    # --- static
    p = players.loc[pid]
    height = p["height_in_cm"].astype("float64")
    height = height.where(height.between(*HEIGHT_RANGE))
    last_league = (mw[mw["league"].notna()].sort_values("m").groupby("player_id").tail(1)
                   .set_index("player_id")["league"])
    a_dob = (T - pd.to_datetime(p["date_of_birth"].to_numpy())).days / 365.25
    # market momentum: mean 12-month log change of current values (both known at t), clipped to +-2
    year_ago = _latest_before(vals, "date", T - pd.Timedelta(days=365)).set_index("player_id")["market_value_in_eur"]
    pop_now = _latest_before(vals, "date", T)
    pop_now = pop_now[pop_now["player_id"].isin(active) & (pop_now["date"] >= T - pd.Timedelta(days=ACTIVE_DAYS))]
    ya = pop_now["player_id"].map(year_ago)
    change = np.log(pop_now["market_value_in_eur"].to_numpy(dtype=float)) - np.log(ya.to_numpy(dtype=float))
    momentum = float(np.nanmean(np.clip(change, -2, 2)))

    league_now = pd.Series(pid).map(last_league).fillna("Missing").to_numpy()
    static = pd.DataFrame({
        "age": a_dob,
        "position": p["position"].fillna("Missing").where(p["position"].isin(POSITIONS), "Missing").to_numpy(),
        "sub_position": p["sub_position"].fillna("Missing").where(p["sub_position"].isin(SUB_POSITIONS), "Missing").to_numpy(),
        "foot": p["foot"].fillna("Missing").where(p["foot"].isin(FEET), "Missing").to_numpy(),
        "height_cm": height.to_numpy(),
        "league": league_now,
        "league_strength": pd.Series(league_now).map(strength).to_numpy(dtype=float),
        "anchor_january": float(t.month == 1),
        "momentum": momentum,
    })
    meta = pd.DataFrame({
        "player_id": pid, "name": p["name"].to_numpy(), "anchor": pd.Timestamp(t), "split": SPLIT_OF.get(t, "unused"),
        "value_now": v_now, "value_now_date": cur["date"].to_numpy(),
        "value_target": v_fut, "value_target_date": fut.loc[pid, "date"].to_numpy(),
        "age": a_dob, "position": static["position"].to_numpy(), "league": league_now,
    })
    stats = {"anchor": t.isoformat(), "population": n_population, "samples": n,
             "dropped_no_target": n_population - n, "momentum": round(momentum, 4)}
    return Samples(seq, static, y, meta), stats


def concat(parts: list[Samples], info: dict) -> Samples:
    return Samples(np.concatenate([s.seq for s in parts]), pd.concat([s.static for s in parts], ignore_index=True),
                   np.concatenate([s.y for s in parts]), pd.concat([s.meta for s in parts], ignore_index=True), info)


def _prepare(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Plain int64 ids (they are never missing in these columns) for fast indexing."""
    out = dict(tables)
    for name, cols in (("players", ["player_id"]), ("player_valuations", ["player_id", "market_value_in_eur"]),
                       ("appearances", ["player_id"]), ("transfers", ["player_id"])):
        out[name] = tables[name].astype({c: "int64" for c in cols})
    return out


def build(tables: dict[str, pd.DataFrame], anchors: list[date] | None = None, log: Log = print) -> Samples:
    anchors = anchors or sorted(SPLIT_OF)
    tables = _prepare(tables)
    log("Aggregating appearances by month...")
    monthly = monthly_appearances(tables["appearances"], tables["competitions"])
    parts, stats = [], []
    for t in anchors:
        s, st = build_anchor(t, tables, monthly, log)
        parts.append(s)
        stats.append(st)
        log(f"  {t}  {SPLIT_OF.get(t, 'unused'):<5}  population {st['population']:>5,}  samples {st['samples']:>5,}  "
            f"no target {st['dropped_no_target']:>4,}  momentum {st['momentum']:+.3f}")
    return concat(parts, {"anchors": stats, "seq_channels": list(SEQ_CHANNELS), "months": MONTHS})


def save(samples: Samples, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples.info = {**samples.info, **meta}
    with path.open("wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> Samples:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m features.market_value build")
    with path.open("rb") as f:
        return pickle.load(f)
