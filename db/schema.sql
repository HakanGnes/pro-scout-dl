-- Pro-Scout DL — core schema (Phase 0: Wyscout public event dataset)
--
-- Applied by `python -m etl init-db`. Every statement is idempotent, so the
-- file can be re-run safely after additions.
--
-- Conventions
--   * Wyscout ids are kept as primary keys (wyId -> <entity>_id).
--   * Pitch coordinates are Wyscout's 0–100 percentages of pitch length (x,
--     own goal -> opponent goal) and width (y, from the top touchline), always
--     from the perspective of the team performing the event.
--   * Player foreign keys are intentionally NOT enforced: a few events and
--     lineups reference players missing from players.json. `python -m etl
--     quality` reports them instead of the load failing.

-- ---------------------------------------------------------------- reference

CREATE TABLE IF NOT EXISTS competitions (
    competition_id  INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    area_name       TEXT,
    format          TEXT,               -- e.g. 'Domestic league', 'International cup'
    type            TEXT                -- 'club' | 'international'
);

CREATE TABLE IF NOT EXISTS teams (
    team_id         INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    official_name   TEXT,
    city            TEXT,
    area_name       TEXT,
    type            TEXT                -- 'club' | 'national'
);

CREATE TABLE IF NOT EXISTS players (
    player_id                INTEGER PRIMARY KEY,
    short_name               TEXT,
    first_name               TEXT,
    middle_name              TEXT,
    last_name                TEXT,
    birth_date               DATE,
    birth_area               TEXT,
    passport_area            TEXT,
    role_code                TEXT,      -- 'GK' | 'DF' | 'MD' | 'FW'
    role_name                TEXT,
    height_cm                SMALLINT,
    weight_kg                SMALLINT,
    foot                     TEXT,
    current_team_id          INTEGER,
    current_national_team_id INTEGER
);

CREATE TABLE IF NOT EXISTS tags (
    tag_id       INTEGER PRIMARY KEY,
    label        TEXT NOT NULL,         -- e.g. 'goal', 'accurate', 'key_pass'
    description  TEXT
);

-- ---------------------------------------------------------------- matches

CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY,
    competition_id  INTEGER NOT NULL REFERENCES competitions,
    season_id       INTEGER,
    round_id        INTEGER,
    gameweek        SMALLINT,
    kickoff_utc     TIMESTAMPTZ,
    status          TEXT,
    duration        TEXT,               -- 'Regular' | 'ExtraTime' | 'Penalties'
    winner_team_id  INTEGER,            -- NULL = draw
    venue           TEXT,
    label           TEXT
);
CREATE INDEX IF NOT EXISTS matches_competition_idx ON matches (competition_id, season_id);

CREATE TABLE IF NOT EXISTS match_teams (
    match_id   INTEGER  NOT NULL REFERENCES matches ON DELETE CASCADE,
    team_id    INTEGER  NOT NULL REFERENCES teams,
    side       TEXT     CHECK (side IN ('home', 'away')),
    score      SMALLINT,                -- final score incl. extra time, excl. shoot-out
    score_ht   SMALLINT,
    score_et   SMALLINT,
    score_p    SMALLINT,                -- penalty shoot-out
    coach_id   INTEGER,
    PRIMARY KEY (match_id, team_id)
);
CREATE INDEX IF NOT EXISTS match_teams_team_idx ON match_teams (team_id);

CREATE TABLE IF NOT EXISTS lineups (
    match_id            INTEGER  NOT NULL,
    team_id             INTEGER  NOT NULL,
    player_id           INTEGER  NOT NULL,
    is_starter          BOOLEAN  NOT NULL,
    goals               SMALLINT NOT NULL DEFAULT 0,
    own_goals           SMALLINT NOT NULL DEFAULT 0,
    yellow_card_minute  SMALLINT,       -- NULL = no card
    red_card_minute     SMALLINT,
    PRIMARY KEY (match_id, player_id),
    FOREIGN KEY (match_id, team_id) REFERENCES match_teams ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS lineups_player_idx ON lineups (player_id);

CREATE TABLE IF NOT EXISTS substitutions (
    match_id    INTEGER  NOT NULL,
    team_id     INTEGER  NOT NULL,
    player_in   INTEGER  NOT NULL,
    player_out  INTEGER  NOT NULL,
    minute      SMALLINT,
    PRIMARY KEY (match_id, player_in, player_out),
    FOREIGN KEY (match_id, team_id) REFERENCES match_teams ON DELETE CASCADE
);

-- ---------------------------------------------------------------- events

CREATE TABLE IF NOT EXISTS events (
    event_id      BIGINT           PRIMARY KEY,
    match_id      INTEGER          NOT NULL REFERENCES matches ON DELETE CASCADE,
    team_id       INTEGER          NOT NULL,
    player_id     INTEGER,                      -- NULL when Wyscout gives 0 (e.g. ball out)
    period        TEXT             NOT NULL,    -- '1H' | '2H' | 'E1' | 'E2' | 'P'
    event_sec     DOUBLE PRECISION NOT NULL,    -- seconds since the start of the period
    type_id       SMALLINT,                     -- Wyscout eventId    (8 = Pass, 10 = Shot, ...)
    type_name     TEXT,
    subtype_id    SMALLINT,                     -- Wyscout subEventId (85 = Simple pass, ...)
    subtype_name  TEXT,
    start_x       SMALLINT,
    start_y       SMALLINT,
    end_x         SMALLINT,                     -- NULL when the event has a single position
    end_y         SMALLINT
);
CREATE INDEX IF NOT EXISTS events_match_time_idx ON events (match_id, period, event_sec);
CREATE INDEX IF NOT EXISTS events_player_idx     ON events (player_id);
CREATE INDEX IF NOT EXISTS events_type_idx       ON events (type_id, subtype_id);

CREATE TABLE IF NOT EXISTS event_tags (
    event_id  BIGINT  NOT NULL REFERENCES events ON DELETE CASCADE,
    tag_id    INTEGER NOT NULL,                 -- not enforced: tags2name.csv is optional
    PRIMARY KEY (event_id, tag_id)
);
CREATE INDEX IF NOT EXISTS event_tags_tag_idx ON event_tags (tag_id);

-- ---------------------------------------------------------------- derived: xG (Phase 1)

-- One row per shot (open play, direct free kick, penalty; shoot-outs excluded).
-- Rebuilt in full by `python -m features.xg score`.
CREATE TABLE IF NOT EXISTS shots (
    event_id            BIGINT   PRIMARY KEY REFERENCES events ON DELETE CASCADE,
    match_id            INTEGER  NOT NULL REFERENCES matches ON DELETE CASCADE,
    team_id             INTEGER  NOT NULL,
    player_id           INTEGER,
    period              TEXT     NOT NULL,
    minute              REAL     NOT NULL,
    shot_type           TEXT     NOT NULL,     -- 'open_play' | 'free_kick' | 'penalty'
    x                   REAL,                  -- 0–100, from the shooter's perspective
    y                   REAL,
    distance_m          REAL,                  -- to the centre of the goal
    angle_rad           REAL,                  -- angle subtended by the goal mouth
    is_header           BOOLEAN,
    is_counter          BOOLEAN,
    assist_type         TEXT,                  -- pass, cross, through_ball, dribble, rebound, ...
    set_piece           TEXT,                  -- none, corner, free_kick, throw_in
    same_player_prev    BOOLEAN,
    seconds_since_prev  REAL,
    score_diff          SMALLINT,              -- shooter's goal difference before the shot
    coords_mirrored     BOOLEAN  NOT NULL DEFAULT false,
    is_goal             BOOLEAN  NOT NULL,
    xg                  REAL,
    xg_model            TEXT                   -- '<model>@<trained_at>'
);
CREATE INDEX IF NOT EXISTS shots_match_team_idx ON shots (match_id, team_id);
CREATE INDEX IF NOT EXISTS shots_player_idx     ON shots (player_id);

-- Team xG per match: the target for the passing-network GNN.
CREATE OR REPLACE VIEW team_match_xg AS
SELECT mt.match_id,
       mt.team_id,
       mt.side,
       mt.score                                                          AS goals,
       count(s.event_id)                                                 AS shots,
       coalesce(sum(s.xg), 0)                                            AS xg,
       coalesce(sum(s.xg) FILTER (WHERE s.shot_type <> 'penalty'), 0)    AS npxg
FROM match_teams mt
LEFT JOIN shots s ON s.match_id = mt.match_id AND s.team_id = mt.team_id
GROUP BY mt.match_id, mt.team_id, mt.side, mt.score;

-- ---------------------------------------------------------------- ETL bookkeeping

-- One row per source file: lets the loader skip files whose contents have not changed.
CREATE TABLE IF NOT EXISTS etl_loads (
    source       TEXT        NOT NULL,          -- e.g. 'wyscout'
    file_name    TEXT        NOT NULL,          -- path relative to the source's raw directory
    sha256       TEXT        NOT NULL,
    rows_loaded  BIGINT      NOT NULL,
    loaded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, file_name)
);

-- ---------------------------------------------------------------- Transfermarkt (Phase 3)
-- dcaribou/transfermarkt-datasets (CC0). Each table is replaced as a whole when its file changes.
-- Only columns that are true at the time they describe are kept for players: the snapshot's
-- current club, contract and caps would leak the future into historical samples.

CREATE TABLE IF NOT EXISTS tm_competitions (
    competition_id   TEXT PRIMARY KEY,           -- e.g. 'GB1'
    name             TEXT NOT NULL,
    type             TEXT,                       -- domestic_league, domestic_cup, international_cup, ...
    sub_type         TEXT,
    country_name     TEXT
);

CREATE TABLE IF NOT EXISTS tm_clubs (
    club_id                  INTEGER PRIMARY KEY,
    name                     TEXT NOT NULL,
    domestic_competition_id  TEXT
);

CREATE TABLE IF NOT EXISTS tm_players (
    player_id               INTEGER PRIMARY KEY,
    name                    TEXT NOT NULL,
    date_of_birth           DATE,
    position                TEXT,                -- Attack, Midfield, Defender, Goalkeeper, Missing
    sub_position            TEXT,
    foot                    TEXT,
    height_in_cm            SMALLINT,
    country_of_citizenship  TEXT
);

CREATE TABLE IF NOT EXISTS tm_player_valuations (
    player_id            INTEGER NOT NULL,
    date                 DATE    NOT NULL,
    market_value_in_eur  BIGINT  NOT NULL,
    current_club_id      INTEGER,
    PRIMARY KEY (player_id, date)
);

CREATE TABLE IF NOT EXISTS tm_appearances (
    appearance_id   TEXT     PRIMARY KEY,        -- '<game_id>_<player_id>'
    game_id         BIGINT   NOT NULL,
    player_id       INTEGER  NOT NULL,
    player_club_id  INTEGER,
    date            DATE     NOT NULL,
    competition_id  TEXT     NOT NULL,
    yellow_cards    SMALLINT NOT NULL,
    red_cards       SMALLINT NOT NULL,
    goals           SMALLINT NOT NULL,
    assists         SMALLINT NOT NULL,
    minutes_played  SMALLINT NOT NULL
);
CREATE INDEX IF NOT EXISTS tm_appearances_player_date_idx ON tm_appearances (player_id, date);

CREATE TABLE IF NOT EXISTS tm_transfers (
    player_id            INTEGER NOT NULL,
    transfer_date        DATE    NOT NULL,       -- may lie in the future (pre-agreed moves, loan returns)
    from_club_id         INTEGER NOT NULL,
    to_club_id           INTEGER NOT NULL,
    transfer_fee         NUMERIC,                -- EUR; NULL unknown, 0 free
    market_value_in_eur  NUMERIC,
    PRIMARY KEY (player_id, transfer_date, from_club_id, to_club_id)
);
