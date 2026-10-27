"""Download the Transfermarkt open dataset.

Source: https://github.com/dcaribou/transfermarkt-datasets (CC0 1.0). The maintainers paused
updates in July 2026, so the files at the stable URLs are a fixed snapshot (valuations up to
2026-06-12). A manifest.json records each file's size and SHA-256, so a later run can tell
whether the snapshot changed.

Only the tables Phase 3 needs are downloaded.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ("competitions", "clubs", "players", "player_valuations", "appearances", "transfers")
USER_AGENT = "pro-scout-dl/0.1 (+https://github.com/HakanGnes)"

Log = Callable[[str], None]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, log: Log) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        done, next_report = 0, 20 << 20
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done >= next_report:
                log(f"    {dest.name}: {done / 2**20:,.0f} MB")
                next_report += 20 << 20
    tmp.replace(dest)


def download_transfermarkt(raw_dir: Path, force: bool = False, log: Log = print) -> Path:
    """Download the tables into raw_dir (skips files already there unless force); returns the manifest."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for table in TABLES:
        name = f"{table}.csv.gz"
        dest = raw_dir / name
        if dest.exists() and not force:
            log(f"  ok     {name} (already downloaded; --force to fetch again)")
        else:
            log(f"  get    {name}")
            _download(f"{BASE_URL}/{name}", dest, log)
        files.append({"table": table, "name": name, "url": f"{BASE_URL}/{name}",
                      "bytes": dest.stat().st_size, "sha256": sha256(dest)})
        log(f"         {files[-1]['bytes'] / 2**20:,.1f} MB  sha256 {files[-1]['sha256'][:12]}...")

    manifest = raw_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "source": "https://github.com/dcaribou/transfermarkt-datasets",
        "license": "CC0-1.0",
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }, indent=2), encoding="utf-8")
    log(f"Wrote {manifest}")
    return manifest
