from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etl.transfermarkt.tables import TABLES, read_all
from features.market_value import dataset as ds
from features.market_value.dataset import CH, MONTHS
from features.market_value.tabular import tabular

T = date(2019, 7, 1)  # a training anchor used in the hand-checked tests


# ---------------------------------------------------------------- synthetic snapshot


def _write(raw: Path, name: str, df: pd.DataFrame) -> None:
    with gzip.open(raw / f"{name}.csv.gz", "wt", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)


def base_tables() -> dict[str, pd.DataFrame]:
    comps = pd.DataFrame({
        "competition_id": ["GB1", "ES1", "CL", "FAC", "FIWC"], "competition_code": "x",
        "name": ["premier-league", "laliga", "ucl", "fa-cup", "world-cup"],
        "sub_type": ["first_tier", "first_tier", "uefa_champions_league", "domestic_cup", "world_cup"],
        "type": ["domestic_league", "domestic_league", "international_cup", "domestic_cup", "national_team_competition"],
        "country_id": 1, "country_name": ["England", "Spain", None, "England", None],
        "domestic_league_code": None, "confederation": None, "total_clubs": None, "url": "u",
    })
    clubs = pd.DataFrame({"club_id": [1, 2, 3], "name": ["A", "B", "C"], "domestic_competition_id": ["GB1", "GB1", "ES1"]})
    players = pd.DataFrame({
        "player_id": [10, 11, 12, 13], "name": ["Ten", "Eleven", "Twelve", "Thirteen"],
        "date_of_birth": ["1995-01-01 00:00:00", "1997-06-15 00:00:00", "2005-01-01 00:00:00", None],
        "position": ["Attack", "Defender", "Midfield", "Attack"],
        "sub_position": ["Centre-Forward", "Centre-Back", "Central Midfield", "Left Winger"],
        "foot": ["right", None, "left", "both"], "height_in_cm": [185.0, 17.0, 175.0, 180.0],
        "country_of_citizenship": "X", "current_club_id": 1, "market_value_in_eur": 1.0,
    })
    vals = pd.DataFrame([
        # player 10: valued every June/December; changes in the target year
        (10, "2016-06-10", 1_000_000), (10, "2017-12-10", 2_000_000), (10, "2018-12-10", 4_000_000),
        (10, "2019-06-10", 5_000_000), (10, "2019-12-10", 8_000_000), (10, "2020-06-10", 10_000_000),
        # player 11: no valuation in [t, t+365) -> dropped
        (11, "2019-02-01", 3_000_000), (11, "2021-01-01", 3_000_000),
        # player 12: 14 at t -> outside the age range
        (12, "2019-03-01", 100_000), (12, "2019-12-01", 200_000),
        # player 13: no date of birth
        (13, "2019-03-01", 100_000), (13, "2019-12-01", 200_000),
    ], columns=["player_id", "date", "market_value_in_eur"]).assign(current_club_id=1, current_club_name="A",
                                                                    player_club_domestic_competition_id="GB1")
    apps = pd.DataFrame([
        # (game, player, club, date, competition, yellow, red, goals, assists, minutes)
        (1, 10, 1, "2019-03-02", "GB1", 1, 0, 2, 1, 90),
        (2, 10, 1, "2019-03-20", "GB1", 0, 0, 0, 0, 45),
        (3, 10, 1, "2019-03-10", "CL", 0, 0, 1, 0, 90),
        (4, 10, 1, "2019-05-01", "FAC", 0, 1, 0, 0, 120),
        (5, 10, 1, "2018-09-01", "ES1", 0, 0, 0, 0, 90),
        (6, 11, 2, "2019-04-01", "GB1", 0, 0, 0, 0, 90),
        (7, 12, 2, "2019-04-01", "GB1", 0, 0, 0, 0, 90),
        (8, 13, 2, "2019-04-01", "GB1", 0, 0, 0, 0, 90),
    ], columns=["game_id", "player_id", "player_club_id", "date", "competition_id", "yellow_cards", "red_cards",
                "goals", "assists", "minutes_played"])
    apps = apps.assign(appearance_id=apps["game_id"].astype(str) + "_" + apps["player_id"].astype(str),
                       player_current_club_id=1, player_name="x")
    transfers = pd.DataFrame([
        (10, "2018-08-15", 3, 1, 20_000_000.0, 3e6),     # Spain -> England for EUR 20m
        (10, "2019-07-01", 1, 2, None, 5e6),             # dated on t: not known before t
        (10, "2025-01-01", 2, 3, 0.0, 1e6),              # future
    ], columns=["player_id", "transfer_date", "from_club_id", "to_club_id", "transfer_fee", "market_value_in_eur"])
    transfers = transfers.assign(transfer_season="x", from_club_name="x", to_club_name="y", player_name="x")
    return {"competitions": comps, "clubs": clubs, "players": players, "player_valuations": vals,
            "appearances": apps, "transfers": transfers}


@pytest.fixture
def raw_dir(tmp_path) -> Path:
    for name, df in base_tables().items():
        _write(tmp_path, name, df)
    return tmp_path


def anchor(raw: Path, t: date = T) -> ds.Samples:
    tables = ds._prepare(read_all(raw))
    monthly = ds.monthly_appearances(tables["appearances"], tables["competitions"])
    return ds.build_anchor(t, tables, monthly, log=lambda _m: None)[0]


# ---------------------------------------------------------------- tables


def test_read_raw_types(raw_dir):
    t = read_all(raw_dir)
    assert set(t) == set(TABLES)
    assert str(t["players"]["date_of_birth"].dtype) == "datetime64[ns]"
    assert str(t["player_valuations"]["market_value_in_eur"].dtype) == "Int64"
    assert t["transfers"]["transfer_fee"].isna().sum() == 1


def test_load_and_read_db_agree(conn, raw_dir):
    from etl.transfermarkt.load import load_transfermarkt
    from etl.transfermarkt.quality import tm_checks

    results = load_transfermarkt(conn, raw_dir, log=lambda _m: None)
    assert [r.status for r in results] == ["loaded"] * len(TABLES)
    assert all(r.status == "skipped" for r in load_transfermarkt(conn, raw_dir, log=lambda _m: None))
    raw, db = read_all(raw_dir), read_all(conn)
    for name in TABLES:
        pd.testing.assert_frame_equal(raw[name], db[name])
    checks = {c.name: c for c in tm_checks(conn)}
    assert checks["tm heights plausible"].status == "WARN"           # the 17 cm height
    assert checks["tm minutes in range"].status == "OK"
    assert checks["tm league coverage"].status == "WARN"             # a tiny sample covers few seasons


def test_quality_without_transfermarkt(conn):
    from etl.transfermarkt.quality import tm_checks

    assert tm_checks(conn)[-1].status == "WARN"  # not loaded: a warning, not a crash


# ---------------------------------------------------------------- samples


def test_population_and_target(raw_dir):
    s = anchor(raw_dir)
    assert s.meta["player_id"].tolist() == [10]       # 11 no target, 12 too young, 13 no birth date
    assert s.meta.loc[0, "value_now"] == 5_000_000     # last value before t
    assert s.meta.loc[0, "value_target"] == 10_000_000  # last value in [t, t + 365 days)
    assert s.y[0] == pytest.approx(np.log(2.0))
    assert s.meta.loc[0, "split"] == "train"


def test_monthly_history(raw_dir):
    s = anchor(raw_dir)
    seq = s.seq[0]
    k = lambda y, m: (y * 12 + m - 1) - (T.year * 12 + T.month - 1 - MONTHS)  # noqa: E731 - index of a month
    march = seq[k(2019, 3)]
    assert march[CH["league_min90"]] == pytest.approx(135 / 90)
    assert march[CH["league_apps"]] == 2 and march[CH["goals"]] == 3 and march[CH["assists"]] == 1
    assert march[CH["europe_min90"]] == pytest.approx(1.0) and march[CH["cards"]] == 1
    assert seq[k(2019, 5), CH["cup_min90"]] == pytest.approx(120 / 90) and seq[k(2019, 5), CH["cards"]] == 1
    # valuations carried forward from before the window; revaluation flags in their months
    assert seq[0, CH["log_value"]] == pytest.approx(np.log(1e6)) and seq[0, CH["has_value"]] == 1
    assert seq[k(2017, 12), CH["revalued"]] == 1 and seq[k(2017, 11), CH["revalued"]] == 0
    assert seq[k(2018, 12), CH["log_value"]] == pytest.approx(np.log(4e6))
    assert seq[-1, CH["log_value_rel"]] == pytest.approx(0.0)
    # the EUR 20m transfer is seen; the one dated on t and the future one are not
    assert seq[k(2018, 8), CH["transfer"]] == 1
    assert seq[k(2018, 8), CH["transfer_fee_log"]] == pytest.approx(np.log1p(20.0))
    assert seq[:, CH["transfer"]].sum() == 1
    # league strength is set where league minutes exist
    assert seq[k(2019, 3), CH["league_strength"]] == pytest.approx(np.median(np.log([5e6])))
    assert s.static.loc[0, "league"] == "GB1" and s.static.loc[0, "anchor_january"] == 0.0


def test_point_in_time(raw_dir, tmp_path_factory):
    """Records dated on or after t must not change a sample's inputs."""
    before = anchor(raw_dir)
    t = base_tables()
    extra = tmp_path_factory.mktemp("future")
    t["player_valuations"] = pd.concat([t["player_valuations"], pd.DataFrame(
        [(10, "2019-07-01", 99_000_000, 1, "A", "GB1"), (10, "2019-08-01", 50_000_000, 1, "A", "GB1")],
        columns=t["player_valuations"].columns)])
    t["appearances"] = pd.concat([t["appearances"], t["appearances"].iloc[[0]].assign(
        game_id=99, appearance_id="99_10", date="2019-07-01", minutes_played=90, goals=5)])
    t["transfers"] = pd.concat([t["transfers"], t["transfers"].iloc[[0]].assign(transfer_date="2019-07-02")])
    for name, df in t.items():
        _write(extra, name, df)
    after = anchor(extra)
    np.testing.assert_array_equal(before.seq, after.seq)
    pd.testing.assert_frame_equal(before.static, after.static)
    assert after.meta.loc[0, "value_target"] == 10_000_000  # the target still uses the last value in the year


def test_tabular_summaries(raw_dir):
    s = anchor(raw_dir)
    x = tabular(s).iloc[0]
    assert x["log_value"] == pytest.approx(np.log(5e6))
    assert x["value_change_12m"] == pytest.approx(np.log(5e6) - np.log(2e6))   # value at end of June 2018
    assert x["value_change_6m"] == pytest.approx(np.log(5e6) - np.log(4e6))    # value at end of December 2018
    assert x["months_since_revaluation"] == 0 and x["revaluations_12m"] == 2
    assert x["league_min90_12m"] == pytest.approx((135 + 90) / 90)
    assert x["months_since_transfer"] == 10 and x["transfers_36m"] == 1
    assert x["goals_per90_12m"] == pytest.approx(3 / ((135 + 90 + 90 + 120) / 90))
    assert str(x["league"]) == "GB1"


def test_splits_do_not_overlap():
    def first(split):
        return min(a for a, s in ds.SPLIT_OF.items() if s == split)

    def target_end(split):
        last = max(a for a, s in ds.SPLIT_OF.items() if s == split)
        return date(last.year + 1, last.month, 1)

    assert len(ds.TRAIN_ANCHORS) == 15 and len(ds.TEST_ANCHORS) == 5
    assert target_end("train") <= first("val") and target_end("val") <= first("test")


# ---------------------------------------------------------------- models and statistics


def test_cluster_bootstrap():
    from models.market_value.stats import cluster_bootstrap, metrics, paired

    d = np.full(30, -0.1)
    assert cluster_bootstrap(d, np.repeat(np.arange(10), 3)) == pytest.approx((-0.1, -0.1, -0.1))
    y = np.zeros(40)
    mean, lo, hi = paired(y, np.full(40, 0.1), np.full(40, 0.3), np.arange(40))
    assert mean == pytest.approx(-0.2) and hi < 0
    r = metrics(np.log([1.0, 2.0]), np.log([1.1, 2.0]))
    assert r["median_ape"] == pytest.approx(0.05) and r["within_20pct"] == 1.0


def test_age_curve_fallbacks():
    from models.market_value.baselines import MIN_CELL, AgeCurve

    n = MIN_CELL
    meta = pd.DataFrame({"age": [24.5] * n + [30.2] * 3, "position": ["Attack"] * n + ["Defender"] * 3})
    s = ds.Samples(np.zeros((n + 3, MONTHS, len(ds.SEQ_CHANNELS)), np.float32), pd.DataFrame(index=meta.index),
                   np.r_[np.full(n, 0.2), np.full(3, -0.5)], meta)
    curve = AgeCurve().fit(s)
    q = ds.Samples(s.seq[:3], pd.DataFrame(index=range(3)), np.zeros(3),
                   pd.DataFrame({"age": [24.9, 24.1, 30.0], "position": ["Attack", "Goalkeeper", "Defender"]}))
    # (24, Attack) has its own cell; (24, Goalkeeper) falls back to age 24; age 30 has 3 samples -> overall
    assert curve.predict(q).tolist() == pytest.approx([0.2, 0.2, curve.overall])


def synthetic_snapshot(raw: Path, n_players: int = 80, seed: int = 0) -> None:
    """A full 2012-2026 snapshot with yearly value trends, enough to fill every split."""
    rng = np.random.default_rng(seed)
    t = base_tables()
    pids = np.arange(100, 100 + n_players)
    t["players"] = pd.DataFrame({
        "player_id": pids, "name": [f"P{i}" for i in pids],
        "date_of_birth": [f"{rng.integers(1988, 2000)}-0{rng.integers(1, 9)}-15" for _ in pids],
        "position": rng.choice(["Attack", "Midfield", "Defender", "Goalkeeper"], n_players),
        "sub_position": "Centre-Back", "foot": "right", "height_in_cm": 180.0, "country_of_citizenship": "X",
        "current_club_id": 1, "market_value_in_eur": 1.0})
    vrows, arows = [], []
    for p in pids:
        lv, slope = rng.uniform(13, 17), rng.normal(0, 0.15)
        for year in range(2011, 2027):
            for month in (6, 12):
                if (year, month) > (2026, 6):
                    continue
                lv += slope / 2 + rng.normal(0, 0.05)
                vrows.append((p, f"{year}-{month:02d}-10", int(np.exp(lv) // 1000 * 1000 + 1000), 1, "A", "GB1"))
        for k, d in enumerate(pd.date_range("2012-08-01", "2026-05-01", freq="MS")):
            if d.month in (6, 7):
                continue
            arows.append((int(p) * 1000 + k, p, 1, d.date().isoformat(), "GB1", 0, 0, int(rng.random() < 0.2), 0,
                          int(rng.integers(0, 91))))
    t["player_valuations"] = pd.DataFrame(vrows, columns=t["player_valuations"].columns)
    apps = pd.DataFrame(arows, columns=["game_id", "player_id", "player_club_id", "date", "competition_id",
                                        "yellow_cards", "red_cards", "goals", "assists", "minutes_played"])
    t["appearances"] = apps.assign(appearance_id=apps["game_id"].astype(str) + "_" + apps["player_id"].astype(str),
                                   player_current_club_id=1, player_name="x")
    t["transfers"] = t["transfers"].iloc[:0]
    for name, df in t.items():
        _write(raw, name, df)
    (raw / "manifest.json").write_text(json.dumps({"files": []}), encoding="utf-8")


def test_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    from features.market_value.__main__ import main as build_main
    from models.market_value.__main__ import main as run_main

    raw = tmp_path / "raw"
    raw.mkdir()
    synthetic_snapshot(raw)
    data = tmp_path / "mv.pkl"
    assert build_main(["build", "--source", "raw", "--raw-dir", str(raw), "--out", str(data)]) == 0
    s = ds.load(data)
    assert {"train", "val", "test"} <= set(s.meta["split"])

    with pytest.raises(SystemExit, match="registered snapshot"):
        run_main(["run", "--dataset", str(data), "--artifacts", str(tmp_path / "a")])
    out = tmp_path / "report"
    assert run_main(["run", "--dataset", str(data), "--artifacts", str(tmp_path / "a"), "--report-dir", str(out),
                     "--predictions", str(tmp_path / "p.csv.gz"), "--max-epochs", "2", "--seeds", "0",
                     "--allow-other-snapshot", "--device", "cpu"]) == 0
    text = (out / "README.md").read_text(encoding="utf-8")
    assert "Primary outcome" in text and "Transformer − LightGBM" in text
    assert (out / "test_mae.png").stat().st_size > 10_000
    results = json.loads((tmp_path / "a" / "results.json").read_text(encoding="utf-8"))
    assert set(results["metrics"]) == {"persistence", "age_curve", "lgbm", "lstm", "transformer"}
    assert results["metrics"]["persistence"]["test"]["mae"] == pytest.approx(np.abs(ds.load(data).split("test").y).mean())
    preds = pd.read_csv(tmp_path / "p.csv.gz")
    assert len(preds) == results["metrics"]["lgbm"]["test"]["n"]
