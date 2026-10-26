from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "wyscout_sample"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def conn():
    """A clean database. Requires TEST_DATABASE_URL; every table in it is dropped."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set; skipping database tests")
    if "test" not in url.rsplit("/", 1)[-1]:
        pytest.fail("TEST_DATABASE_URL must point to a database whose name contains 'test'")

    from etl.db import connect, init_db, reset_db

    with connect(url) as connection:
        reset_db(connection)
        init_db(connection)
        yield connection
