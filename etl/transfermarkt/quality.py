"""Data-quality checks for the tm_* tables (same statuses as etl.quality)."""
from __future__ import annotations

import psycopg

from ..quality import FAIL, INFO, OK, WARN, Check

# The 14 first divisions whose appearances the dataset covers from 2012/13.
COVERED_LEAGUES = ("BE1", "DK1", "ES1", "FR1", "GB1", "GR1", "IT1", "L1", "NL1", "PO1", "RU1", "SC1", "TR1", "UKR1")
FIRST_SEASON, LAST_FULL_SEASON = 2012, 2025
HEIGHT_RANGE = (150, 215)
MAX_MINUTES = 150  # 90 + stoppage + extra time


def _one(conn, sql: str, params: tuple = ()) -> tuple:
    return conn.execute(sql, params).fetchone()


def tm_checks(conn: psycopg.Connection) -> list[Check]:
    tables = ("tm_competitions", "tm_clubs", "tm_players", "tm_player_valuations", "tm_appearances", "tm_transfers")
    counts = {t: _one(conn, f"SELECT count(*) FROM {t}")[0] for t in tables}
    checks = [Check("tm row counts", INFO, ", ".join(f"{t[3:]}={n:,}" for t, n in counts.items()))]
    if counts["tm_player_valuations"] == 0 or counts["tm_appearances"] == 0:
        checks.append(Check("tm data present", WARN, "Transfermarkt not loaded; run `python -m etl load-transfermarkt`"))
        return checks

    lo, hi, n_players = _one(conn, "SELECT min(date), max(date), count(DISTINCT player_id) FROM tm_player_valuations")
    checks.append(Check("tm valuation dates", INFO, f"{lo} to {hi}, {n_players:,} players"))

    bad = _one(conn, "SELECT count(*) FROM tm_player_valuations WHERE market_value_in_eur <= 0")[0]
    checks.append(Check("tm valuations positive", WARN if bad else OK,
                        f"{bad:,} valuation(s) of 0 EUR (dropped from features)" if bad else "all > 0"))

    gap = _one(conn, """
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY g) FROM (
            SELECT date - lag(date) OVER (PARTITION BY player_id ORDER BY date) AS g FROM tm_player_valuations
        ) x WHERE g IS NOT NULL""")[0]
    checks.append(Check("tm revaluation cadence", INFO, f"median {gap:.0f} days between a player's valuations"))

    orphan_v = _one(conn, "SELECT count(DISTINCT player_id) FROM tm_player_valuations v "
                          "WHERE NOT EXISTS (SELECT 1 FROM tm_players p WHERE p.player_id = v.player_id)")[0]
    orphan_a = _one(conn, "SELECT count(*) FROM tm_appearances a "
                          "WHERE NOT EXISTS (SELECT 1 FROM tm_players p WHERE p.player_id = a.player_id)")[0]
    checks.append(Check("tm players known", WARN if orphan_v or orphan_a else OK,
                        f"{orphan_v:,} valued player(s) and {orphan_a:,} appearance(s) without a players row"))

    no_dob, n = _one(conn, "SELECT count(*) FILTER (WHERE date_of_birth IS NULL), count(*) FROM tm_players")
    checks.append(Check("tm birth dates", INFO, f"{no_dob:,} of {n:,} players without a date of birth (not sampled)"))

    h_lo, h_hi = HEIGHT_RANGE
    bad_h = _one(conn, "SELECT count(*) FROM tm_players WHERE height_in_cm NOT BETWEEN %s AND %s", HEIGHT_RANGE)[0]
    checks.append(Check("tm heights plausible", WARN if bad_h else OK,
                        f"{bad_h:,} height(s) outside {h_lo}-{h_hi} cm (treated as missing)" if bad_h else "all plausible"))

    bad_m = _one(conn, "SELECT count(*) FROM tm_appearances WHERE minutes_played NOT BETWEEN 0 AND %s", (MAX_MINUTES,))[0]
    checks.append(Check("tm minutes in range", FAIL if bad_m else OK,
                        f"{bad_m:,} appearance(s) outside 0-{MAX_MINUTES} minutes" if bad_m else f"all within 0-{MAX_MINUTES}"))

    future = _one(conn, "SELECT count(*), max(transfer_date) FROM tm_transfers "
                        "WHERE transfer_date > (SELECT max(date) FROM tm_appearances)")
    checks.append(Check("tm future-dated transfers", INFO,
                        f"{future[0]:,} transfer(s) dated after the last appearance (up to {future[1]}); "
                        "ignored until their date" if future[0] else "none"))

    rows = conn.execute("""
        SELECT competition_id, count(DISTINCT (extract(year FROM date) - (extract(month FROM date) < 7)::int))
        FROM tm_appearances
        WHERE competition_id = ANY(%s)
          AND date >= make_date(%s, 7, 1) AND date < make_date(%s, 7, 1)
        GROUP BY competition_id""", (list(COVERED_LEAGUES), FIRST_SEASON, LAST_FULL_SEASON + 1)).fetchall()
    seasons = LAST_FULL_SEASON - FIRST_SEASON + 1
    have = dict(rows)
    short = {lg: have.get(lg, 0) for lg in COVERED_LEAGUES if have.get(lg, 0) < seasons}
    checks.append(Check("tm league coverage", WARN if short else OK,
                        f"all {len(COVERED_LEAGUES)} leagues have appearances in each of {seasons} seasons "
                        f"{FIRST_SEASON}/{FIRST_SEASON % 100 + 1:02d}-{LAST_FULL_SEASON}/{LAST_FULL_SEASON % 100 + 1:02d}"
                        if not short else "seasons with appearances: " + ", ".join(f"{k}={v}" for k, v in short.items())))
    return checks

