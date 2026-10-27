"""The Transfermarkt tables Phase 3 uses: which columns, which types, and two readers.

`read_raw` (the downloaded csv.gz files) and `read_db` (the tm_* tables) return identical
DataFrames, so features can be built from either and a test can check that they agree.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# column -> kind: "int" (nullable Int64), "str", "date", "num" (float64)
SPEC: dict[str, dict[str, str]] = {
    "competitions": {"competition_id": "str", "name": "str", "type": "str", "sub_type": "str", "country_name": "str"},
    "clubs": {"club_id": "int", "name": "str", "domestic_competition_id": "str"},
    "players": {"player_id": "int", "name": "str", "date_of_birth": "date", "position": "str", "sub_position": "str",
                "foot": "str", "height_in_cm": "int", "country_of_citizenship": "str"},
    "player_valuations": {"player_id": "int", "date": "date", "market_value_in_eur": "int", "current_club_id": "int"},
    "appearances": {"appearance_id": "str", "game_id": "int", "player_id": "int", "player_club_id": "int", "date": "date",
                    "competition_id": "str", "yellow_cards": "int", "red_cards": "int", "goals": "int",
                    "assists": "int", "minutes_played": "int"},
    "transfers": {"player_id": "int", "transfer_date": "date", "from_club_id": "int", "to_club_id": "int",
                  "transfer_fee": "num", "market_value_in_eur": "num"},
}
# Load order and a stable row order (so raw and database frames line up).
TABLES = tuple(SPEC)
ORDER = {
    "competitions": ["competition_id"], "clubs": ["club_id"], "players": ["player_id"],
    "player_valuations": ["player_id", "date"], "appearances": ["appearance_id"],
    "transfers": ["player_id", "transfer_date", "from_club_id", "to_club_id"],
}


def db_table(table: str) -> str:
    return f"tm_{table}"


def _cast(df: pd.DataFrame, table: str) -> pd.DataFrame:
    spec = SPEC[table]
    missing = set(spec) - set(df.columns)
    if missing:
        raise ValueError(f"{table}: columns missing from the source: {sorted(missing)}")
    out = pd.DataFrame(index=df.index)
    for col, kind in spec.items():
        s = df[col]
        if kind == "int":
            out[col] = pd.to_numeric(s, errors="raise").round().astype("Int64")
        elif kind == "num":
            out[col] = pd.to_numeric(s, errors="raise").astype("float64")
        elif kind == "date":
            out[col] = pd.to_datetime(s, errors="raise").dt.normalize().astype("datetime64[ns]")
        else:
            out[col] = s.astype("object").where(s.notna(), None)
    return out.sort_values(ORDER[table], kind="stable").reset_index(drop=True)


def read_raw(raw_dir: Path, table: str) -> pd.DataFrame:
    path = Path(raw_dir) / f"{table}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m etl download-transfermarkt`")
    return _cast(pd.read_csv(path, usecols=list(SPEC[table]), low_memory=False), table)


def read_db(conn, table: str) -> pd.DataFrame:
    cols = list(SPEC[table])
    cur = conn.execute(f"SELECT {', '.join(cols)} FROM {db_table(table)}")
    return _cast(pd.DataFrame(cur.fetchall(), columns=cols), table)


def read_all(source, tables: tuple[str, ...] = TABLES) -> dict[str, pd.DataFrame]:
    """`source` is a raw directory (Path/str) or an open database connection."""
    if isinstance(source, (str, Path)):
        return {t: read_raw(Path(source), t) for t in tables}
    return {t: read_db(source, t) for t in tables}
