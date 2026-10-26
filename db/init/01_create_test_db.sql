-- Runs once, the first time the container's data volume is created.
-- The application schema itself is applied by `python -m etl init-db`.
CREATE DATABASE proscout_test;
