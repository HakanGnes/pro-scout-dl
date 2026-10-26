# Pro-Scout DL

Deep learning for football scouting: model a team's passing network as a graph and
estimate how its attacking output would change if one player were replaced by another.
Later phases add market-value projection (sequence models) and transfer-rumour
credibility (Turkish/English NLP).

Football is a network game. Individual statistics miss how a player's output depends
on the teammates and the tactical structure around them; graph neural networks can
capture that.

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Infrastructure: PostgreSQL, Wyscout ETL, data-quality checks | done |
| 1 | xG model, passing networks, first GNN vs. a gradient-boosting baseline | in progress: xG done |
| 2 | Player-swap simulation + Streamlit page | planned |
| 3 | Market value: GBM vs. LSTM vs. Transformer | planned |
| 4 | Rumour credibility with BERTurk + fusion score | planned |
| 5 | Tactics board, deployment, write-ups | planned |

Every deep model is compared with a simple baseline, and the result is reported even
when the baseline wins.

## Quick start (Windows, PowerShell)

Requirements: Python 3.10+, Docker Desktop, a few GB of free disk for the raw files and the database.

```powershell
git clone https://github.com/HakanGnes/pro-scout-dl.git
cd pro-scout-dl
Copy-Item .env.example .env

# 1. Database
docker compose up -d
docker compose ps            # wait until the db service is "healthy"

# 2. Python environment
py -m venv .venv
.venv\Scripts\Activate.ps1   # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev,ml]"

# 3. Data
python -m etl download-wyscout    # from figshare, every file verified by MD5
python -m etl load-wyscout        # ~5 minutes; re-runs skip unchanged files
python -m etl quality             # exit code 1 if any check FAILs

# 4. Expected goals
python -m features.xg train       # evaluate, fit, save the model, write reports/xg/
python -m features.xg score       # store xG for every shot in the database
```

On macOS/Linux use `cp .env.example .env`, `python3 -m venv .venv` and
`source .venv/bin/activate`; everything else is the same.

If port 5432 is already taken by a local PostgreSQL, set `POSTGRES_PORT=5433` in `.env`
and change the port in `DATABASE_URL` to match.

## Commands

| Command | What it does |
| --- | --- |
| `python -m etl init-db` | Create tables and indexes (idempotent) |
| `python -m etl download-wyscout [--only events matches]` | Download and extract the dataset into `data/raw/wyscout/`, write `manifest.json` with licenses and checksums |
| `python -m etl load-wyscout [--force]` | Load every raw file in its own transaction; unchanged files (same SHA-256) are skipped |
| `python -m etl quality` | Run the data-quality report |
| `python -m etl reset-db --yes` | Drop every project table |
| `python -m features.xg train [--source raw]` | Extract shots, compare four xG models (leave one competition out), fit the best on all shots, save it to `models/artifacts/`, write `reports/xg/` |
| `python -m features.xg score` | Predict xG for every shot and replace the `shots` table |

## Database

```
competitions ─┐
teams ────────┼── matches ──┬── match_teams ──┬── lineups
players       │             │                 └── substitutions
tags          │             └── events ── event_tags
etl_loads (one row per loaded file + its SHA-256)
```

- Wyscout ids are kept as primary keys.
- Coordinates are Wyscout's 0–100 percentages, always from the acting team's point of view
  (x towards the opponent's goal, y from the top touchline).
- Re-loading a file replaces the children of the matches it contains, so a corrected
  file never leaves stale rows behind.

The full schema with comments is in [`db/schema.sql`](db/schema.sql).

### Data-quality report

`python -m etl quality` checks, among other things, that every match has one home and
one away team, coordinates are in range, every event's team plays in its match, and
which player ids are missing from `players.json`. It also **rebuilds every team's score
from goal-tagged events** (shots, free-kick shots and penalties, plus opponents' own
goals, excluding shoot-outs) and compares it with the official score: a strong signal that
events and tags were loaded correctly.

## Expected goals (Phase 1, step 1)

Wyscout ships no xG, so the project builds its own: the target for the passing-network GNN
(a team's xG in a match, view `team_match_xg`) comes from it.

- **Data:** 45,869 shots and 5,052 goals. 25 shots that Wyscout recorded on the wrong half are mirrored back.
- **Features:** only pre-shot information — location, header or foot, how the chance was created,
  set-piece origin, counter-attack, tempo, minute, score. Outcome tags are excluded: Wyscout's
  `opportunity` tag alone sits on 99% of goals and would leak the label.
- **Validation:** leave one competition out; the model is picked by a rule fixed in advance
  (lowest out-of-fold log loss).

| Model | Log loss | Brier | ROC AUC | Calibration error |
| --- | --- | --- | --- | --- |
| Constant rate | 0.3300 | 0.0918 | – | 0.0063 |
| Logistic: distance + angle | 0.2805 | 0.0798 | 0.772 | 0.0057 |
| Logistic: all features | 0.2753 | 0.0786 | 0.785 | 0.0053 |
| **LightGBM: all features** (in use) | **0.2750** | **0.0785** | **0.785** | **0.0039** |

Context features improve clearly on location alone (95% interval of the log-loss gain excludes zero).
LightGBM and the full logistic model are statistically tied; LightGBM is kept for its better calibration,
which matters most when xG is summed per team. Full report: [`reports/xg/README.md`](reports/xg/README.md).

## Tests

```powershell
pytest -q                                   # parser tests only
$env:TEST_DATABASE_URL = "postgresql://proscout:proscout@localhost:5432/proscout_test"
pytest -q                                   # plus end-to-end load tests
```

The database tests run against `proscout_test` (created automatically by
`docker compose` on first start) and **drop every table in it**; they refuse to run against a
database whose name does not contain `test`. They use a small fictional sample in
`tests/fixtures/wyscout_sample/` that reproduces the real dataset's quirks. CI runs the same
suite against a PostgreSQL service container.

## Repository layout

```
pro-scout-dl/
├── db/              schema.sql, init/ (creates the test database)
├── etl/             command line, database helpers, quality checks
│   └── wyscout/     parse.py (pure functions), load.py, download.py
├── tests/           parser and end-to-end tests, fictional sample data
├── data/            raw/ and processed/ (git-ignored)
├── features/xg/     Phase 1: shot features, xG models, report, scoring
├── models/          Phase 1+: GNN, valuation, NLP (artifacts/ is git-ignored)
├── reports/xg/      xG evaluation: metrics, calibration, xG map
├── sim/             Phase 2: player-swap simulation
├── app/             Streamlit
└── notebooks/
```

## Data and attribution

Event data: the Wyscout public dataset, figshare collection
[doi:10.6084/m9.figshare.c.4415000](https://doi.org/10.6084/m9.figshare.c.4415000) —
every match of the 2017/18 season in England, France, Germany, Italy and Spain, the 2018
World Cup and Euro 2016. The data is not redistributed in this repository; `download-wyscout`
fetches it from the source and records each file's license in `data/raw/wyscout/manifest.json`.
Use it for non-commercial research and cite:

- Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P., Pedreschi, D., Giannotti, F.
  (2019). A public data set of spatio-temporal match events in soccer competitions.
  *Scientific Data* 6, 236. https://doi.org/10.1038/s41597-019-0247-7
- Pappalardo, L., Cintia, P., Ferragina, P., Massucco, E., Pedreschi, D., Giannotti, F. (2019).
  PlayeRank: Data-driven performance evaluation and player ranking in soccer via a machine
  learning approach. *ACM TIST* 10(5), 59. https://doi.org/10.1145/3343172
