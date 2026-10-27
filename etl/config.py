"""Configuration, read from environment variables (and a .env file at the repo root)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"
DEFAULT_WYSCOUT_DIR = REPO_ROOT / "data" / "raw" / "wyscout"
DEFAULT_TRANSFERMARKT_DIR = REPO_ROOT / "data" / "raw" / "transfermarkt"
DEFAULT_NEWS_DIR = REPO_ROOT / "data" / "raw" / "news"
DEFAULT_DATABASE_URL = "postgresql://proscout:proscout@localhost:5432/proscout"

load_dotenv(REPO_ROOT / ".env")


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
