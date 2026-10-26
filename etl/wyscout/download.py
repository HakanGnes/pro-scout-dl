"""Download the Wyscout public dataset from figshare.

Collection: "Soccer match event dataset", figshare collection 4415000
(doi:10.6084/m9.figshare.c.4415000). The file list is read from the figshare
API at run time, so nothing here depends on hard-coded file ids.

Each file is checked against figshare's MD5, zip archives are extracted next
to it, and a manifest.json records titles, licenses and checksums.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

API = "https://api.figshare.com/v2"
COLLECTION_ID = 4415000
USER_AGENT = "pro-scout-dl/0.1 (+https://github.com/HakanGnes)"

Log = Callable[[str], None]


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(log: Log = print) -> list[dict[str, Any]]:
    """Every file in the collection, with its article's title and license."""
    articles = _get_json(f"{API}/collections/{COLLECTION_ID}/articles?page_size=100")
    files: list[dict[str, Any]] = []
    for article in articles:
        detail = _get_json(f"{API}/articles/{article['id']}")
        license_name = (detail.get("license") or {}).get("name")
        for f in detail.get("files", []):
            files.append({
                "article_id": article["id"],
                "article_title": detail.get("title"),
                "license": license_name,
                "name": f["name"],
                "size": f.get("size"),
                "md5": f.get("computed_md5") or f.get("supplied_md5"),
                "url": f["download_url"],
            })
    log(f"Found {len(files)} files in {len(articles)} figshare articles")
    return files


def _download(url: str, dest: Path, size: int | None, log: Log) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        done, next_report = 0, 50 << 20
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done >= next_report:
                total = f" / {size / 2**20:,.0f}" if size else ""
                log(f"    {dest.name}: {done / 2**20:,.0f}{total} MB")
                next_report += 50 << 20
    tmp.replace(dest)


def _safe_extract(archive: Path, target: Path) -> int:
    """Extract a zip, refusing members that would land outside `target`."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for m in members:
            if not (target / m.filename).resolve().is_relative_to(target):
                raise ValueError(f"Unsafe path in {archive.name}: {m.filename}")
        target.mkdir(parents=True, exist_ok=True)
        zf.extractall(target)
    return len(members)


def download_wyscout(raw_dir: Path, only: list[str] | None = None, log: Log = print) -> Path:
    """Download (or verify) the dataset into raw_dir; returns the manifest path."""
    raw_dir = Path(raw_dir)
    archive_dir = raw_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    files = list_files(log)
    if only:
        wanted = [w.lower() for w in only]
        files = [f for f in files if any(w in f["name"].lower() for w in wanted)]

    for f in files:
        name = Path(f["name"]).name  # never trust a path from the server
        is_zip = name.lower().endswith(".zip")
        dest = (archive_dir if is_zip else raw_dir) / name

        if dest.exists() and f["md5"] and _md5(dest) == f["md5"]:
            log(f"  ok     {name} (already downloaded)")
        else:
            log(f"  get    {name} ({(f['size'] or 0) / 2**20:,.1f} MB, license: {f['license'] or 'unknown'})")
            _download(f["url"], dest, f["size"], log)
            if f["md5"] and _md5(dest) != f["md5"]:
                dest.unlink()
                raise OSError(f"MD5 mismatch for {name}; the partial file was removed, run again")

        if is_zip:
            target = raw_dir / Path(name).stem
            if target.exists():
                shutil.rmtree(target)
            n = _safe_extract(dest, target)
            log(f"         extracted {n} file(s) to {target.relative_to(raw_dir).as_posix()}/")

    manifest = raw_dir / "manifest.json"
    manifest.write_text(json.dumps({"collection_id": COLLECTION_ID, "files": files}, indent=2), encoding="utf-8")
    log(f"Wrote {manifest}")
    return manifest
