"""Command line: python -m models.rumours run | fit | score-archive

run: the pre-registered Phase 4 study (docs/preregistration/2026-10-29-rumour-credibility.md), in one run:
rebuilds the claims from the archive index and Transfermarkt, trains the paper rate, metadata GBM,
TF-IDF and BERTurk models, fuses metadata with each text model, and writes the registered outcomes on
the 2025 test period to reports/rumours/.
fit: the same metadata + BERTurk fusion, trained the same way, saved to models/artifacts/rumours/ for use.
score-archive: score the archived rumours from 2026 up to the end of the Transfermarkt data (June 2026), which
no model has seen -> data/processed/rumours/scored_2026.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from etl.config import DEFAULT_NEWS_DIR, DEFAULT_TRANSFERMARKT_DIR, REPO_ROOT

REPORT_DIR = REPO_ROOT / "reports" / "rumours"
PREDICTIONS = REPO_ROOT / "data" / "processed" / "rumours" / "predictions.csv.gz"
ARTIFACTS = REPO_ROOT / "models" / "artifacts" / "rumours"
SCORED = REPO_ROOT / "data" / "processed" / "rumours" / "scored_2026.csv"
PREREG = "docs/preregistration/2026-10-29-rumour-credibility.md"
SEEDS = [0, 1, 2]            # BERTurk
META_SEEDS = [0, 1, 2, 3, 4]  # LightGBM
N_BOOT, BOOT_SEED = 2000, 0
SITES = ("aspor", "fanatik", "fotomac", "hurriyet_spor", "milliyet_skorer", "ntvspor", "sabah_spor", "sporx")
MIN_TITLE_COVERAGE = 0.95   # share of the study's rumour articles whose archived page was fetched
TM_FILES = ("players.csv.gz", "clubs.csv.gz", "transfers.csv.gz", "player_valuations.csv.gz")
# The inputs the study was registered on (sha256); other inputs would be a different study.
REGISTERED_FILES = {
    "history/aspor/index.jsonl.gz": "381297bbf064e308d55a7ebe0927cb1c90933226bd846f6dd446f0cd80598770",
    "history/fanatik/index.jsonl.gz": "e382c02d9158ed8d9fae6584f1ba8c63a397de5e6c99fa5c4694d37fefa12893",
    "history/fotomac/index.jsonl.gz": "20f08cdb129e7140dd71e4448a380ebee524e65ccbf82559e18a25d14567abfa",
    "history/hurriyet_spor/index.jsonl.gz": "184f57a48ad27a699bbc21eb9225f855aebb2864475aee7fabf4062a9b2448db",
    "history/milliyet_skorer/index.jsonl.gz": "78f9633a3a7abdc438605aeebb544b78924e56207b7c6444a66dd598e1f305bb",
    "history/ntvspor/index.jsonl.gz": "f01f0d1f7039a784442c1823dff65b7162e2f346655ea5a8f415fac0f107792c",
    "history/sabah_spor/index.jsonl.gz": "1d6b376293770ca6a237bb42623b19291f1aeda816a3f34eb20c2e90742e71a7",
    "history/sporx/index.jsonl.gz": "2003c3a8976f23c77b3a85819581116666fb0180e5c1ca3928c96bd21b31e101",
    "transfermarkt/players.csv.gz": "d22e407981d5b51a79bf8ff59835729f3526f7dc3495a6d8ed2f852ed2e86403",
    "transfermarkt/clubs.csv.gz": "aa57ab56e87faffaea1e8697439a32caf10116d8f2f2dfb3a57193c0bb1d641b",
    "transfermarkt/transfers.csv.gz": "90326983daf7e6ac7aabdfe62b90936d9bf1dd2171e53dec75c3751eb1620a83",
    "transfermarkt/player_valuations.csv.gz": "608e32e5ad2231074f3505c469ad727be66d7e6efda6af918afd85379439631d",
}
PRIMARY = ("fusion", "metadata")
SECONDARY = [("bert", "tfidf"), ("bert", "metadata"), ("tfidf", "metadata"), ("fusion_tfidf", "metadata"),
             ("fusion", "fusion_tfidf"), ("metadata", "site"), ("tfidf", "site")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_files(history_dir: Path, tm_dir: Path) -> dict[str, Path]:
    files = {f"history/{s}/index.jsonl.gz": history_dir / s / "index.jsonl.gz" for s in SITES}
    files.update({f"transfermarkt/{n}": tm_dir / n for n in TM_FILES})
    files["history/titles.jsonl.gz"] = history_dir / "titles.jsonl.gz"   # recorded, not registered (fetched later)
    return files


def check_inputs(files: dict[str, Path], strict: bool) -> dict[str, str]:
    hashes = {k: sha256(p) for k, p in files.items() if p.exists()}
    wrong = sorted(k for k, h in REGISTERED_FILES.items() if hashes.get(k) != h)
    if wrong and strict:
        raise SystemExit(f"Inputs differ from the registered ones: {', '.join(wrong)}")
    return hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m models.rumours", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="train and evaluate every model as registered")
    r.add_argument("--history-dir", type=Path, default=DEFAULT_NEWS_DIR / "history")
    r.add_argument("--transfermarkt-dir", type=Path, default=DEFAULT_TRANSFERMARKT_DIR)
    r.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    r.add_argument("--predictions", type=Path, default=PREDICTIONS)
    r.add_argument("--device", default="auto")
    r.add_argument("--bert-model", help="tests only; the registered run uses dbmdz/bert-base-turkish-uncased")
    r.add_argument("--epochs", type=int, help="tests only")
    r.add_argument("--seeds", type=int, nargs="+", help="tests only; the registered run uses 0-2")
    r.add_argument("--n-boot", type=int, default=N_BOOT, help="tests only")
    r.add_argument("--allow-other-inputs", action="store_true", help="tests only")
    f = sub.add_parser("fit", help="train the metadata + BERTurk fusion as registered and save it")
    f.add_argument("--history-dir", type=Path, default=DEFAULT_NEWS_DIR / "history")
    f.add_argument("--transfermarkt-dir", type=Path, default=DEFAULT_TRANSFERMARKT_DIR)
    f.add_argument("--out", type=Path, default=ARTIFACTS)
    f.add_argument("--device", default="auto")
    f.add_argument("--bert-model", help="tests only")
    f.add_argument("--epochs", type=int, help="tests only")
    f.add_argument("--seeds", type=int, nargs="+", help="tests only; the saved model uses 0-2")
    sa = sub.add_parser("score-archive", help="score the 2026 archived rumours (up to the data's end) with the saved model")
    sa.add_argument("--history-dir", type=Path, default=DEFAULT_NEWS_DIR / "history")
    sa.add_argument("--model", type=Path, default=ARTIFACTS)
    sa.add_argument("--start", default="2026-01-01", help="first day (the 2025 test period ends on 2025-12-31)")
    sa.add_argument("--out", type=Path, default=SCORED)
    sa.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "fit":
        return _fit(args)
    if args.command == "score-archive":
        return _score_archive(args)

    import numpy as np

    from etl.transfermarkt.tables import read_raw
    from etl.news.history import iter_jsonl, url_key
    from features.rumours.build import articles, gazetteer, study_urls
    from features.rumours.labels import claims as make_claims

    from . import bert, dataset, report
    from .models import Calibrator, metadata_gbm, site_rates, tfidf_lr

    started = time.perf_counter()
    hashes = check_inputs(input_files(args.history_dir, args.transfermarkt_dir), strict=not args.allow_other_inputs)
    seeds = args.seeds or SEEDS

    arts = articles(args.history_dir)
    g = gazetteer(args.transfermarkt_dir, arts["slug"].tolist())
    claims = make_claims(arts, g)
    fetched = {url_key(t["url"]) for t in iter_jsonl([args.history_dir / "titles.jsonl.gz"])}
    wanted = study_urls(claims, arts)
    coverage = sum(url_key(u) in fetched for u in wanted) / max(1, len(wanted))
    print(f"Archived pages fetched for {coverage:.1%} of {len(wanted):,} rumour articles; "
          f"dates from {', '.join(f'{k} {v:,}' for k, v in arts['date_source'].value_counts().items())}")
    if coverage < MIN_TITLE_COVERAGE and not args.allow_other_inputs:
        raise SystemExit("Finish `python -m etl history-titles --urls data/processed/rumours/title_urls.txt` first.")
    players, vals = read_raw(args.transfermarkt_dir, "players"), read_raw(args.transfermarkt_dir, "player_valuations")
    data = dataset.build(claims, g, players, vals)
    tr, va, te = (data[data["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"Study table: train {len(tr):,}, val {len(va):,}, test {len(te):,} rumour claims "
          f"({time.perf_counter() - started:.0f}s)")
    frames = {"val": va, "test": te}
    raw: dict[str, dict[str, np.ndarray]] = {}
    info: dict[str, dict] = {}

    raw["base"] = {k: np.full(len(f), tr["label"].mean()) for k, f in frames.items()}
    raw["site"] = {k: site_rates(tr, f) for k, f in frames.items()}
    raw["metadata"], info["metadata"] = metadata_gbm(tr, va, frames, seeds=args.seeds or META_SEEDS)
    raw["tfidf"], info["tfidf"] = tfidf_lr(tr, va, frames)
    print("  metadata, tf-idf done")

    runs, preds = [], {k: np.zeros(len(f)) for k, f in frames.items()}
    for s in seeds:
        p, meta = bert.fine_tune(tr["text"].tolist(), tr["label"].to_numpy(float), va["text"].tolist(),
                                 va["label"].to_numpy(), {k: f["text"].tolist() for k, f in frames.items()}, seed=s,
                                 model_name=args.bert_model or bert.MODEL_NAME, device=args.device,
                                 hparams={"epochs": args.epochs} if args.epochs else None)
        runs.append(meta)
        for k in preds:
            preds[k] += p[k] / len(seeds)
    raw["bert"], info["bert"] = preds, {"runs": runs, "hparams": bert.HPARAMS, "model": args.bert_model or bert.MODEL_NAME}

    yv, yt = va["label"].to_numpy(), te["label"].to_numpy()
    final, val_auc = {}, {}
    for m in ("base", "site", "metadata", "tfidf", "bert"):
        if m == "base":
            final[m] = np.full(len(te), yv.mean())
        else:
            final[m] = Calibrator().fit([raw[m]["val"]], yv)([raw[m]["test"]])
            val_auc[m] = report.metrics(yv, raw[m]["val"])["auc"]
    for name, text in (("fusion", "bert"), ("fusion_tfidf", "tfidf")):
        cal = Calibrator().fit([raw["metadata"]["val"], raw[text]["val"]], yv)
        final[name] = cal([raw["metadata"]["test"], raw[text]["test"]])
        info[name] = {"coef_metadata_text_intercept": cal.coef()}

    test = {m: report.metrics(yt, p) for m, p in final.items()}
    diffs = report.auc_differences(yt, final, [PRIMARY, *SECONDARY], te["pair"].to_numpy(), n_boot=args.n_boot,
                                   seed=BOOT_SEED)
    te = te.assign(new_pair=np.where(te["pair_365d"] == 0, "new pair", "pair in the news before"),
                   target_league=np.where(te["target_tr"] == 1, "Turkish club", "foreign club"))
    groups = {c: report.by_group(te, final, c) for c in ("site", "new_pair", "target_league")}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=REPO_ROOT).stdout.strip()
    except OSError:
        commit = ""
    results = {"prereg": PREREG, "commit": commit, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "inputs_sha256": hashes, "title_coverage": coverage,
               "date_sources": arts["date_source"].value_counts().to_dict(), "counts": {"train": len(tr), "val": len(va), "test": len(te)},
               "test_pairs": int(te["pair"].nunique()), "n_boot": args.n_boot, "val_auc_raw": val_auc, "test": test,
               "primary": diffs[" - ".join(PRIMARY)], "secondary": {k: v for k, v in diffs.items()
                                                                    if k != " - ".join(PRIMARY)},
               "models": info}
    path = report.write(args.report_dir, results, groups, report.reliability(yt, final["fusion"]))
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    out = te[["url", "date", "site", "player_id", "club_id", "label", "text"]].copy()
    for m, p in final.items():
        out[f"p_{m}"] = p   # "p_" prefix: the "site" model must not overwrite the site column
    out.to_csv(args.predictions, index=False, compression="gzip")

    d = results["primary"]
    print(f"\nPrimary: AUC(fusion) - AUC(metadata) = {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] -> "
          f"{'SUPPORTED' if d[1] > 0 else 'not supported'}")
    for m, r in test.items():
        print(f"  {m:<13} AUC {r['auc']:.4f}  Brier {r['brier']:.4f}")
    print(f"Wrote {path} ({time.perf_counter() - started:.0f}s)")
    return 0


def _study(history_dir: Path, tm_dir: Path):
    from etl.transfermarkt.tables import read_raw
    from features.rumours.build import articles, gazetteer
    from features.rumours.labels import claims as make_claims

    from . import dataset
    arts = articles(history_dir)
    g = gazetteer(tm_dir, arts["slug"].tolist())
    claims = make_claims(arts, g)
    players, vals = read_raw(tm_dir, "players"), read_raw(tm_dir, "player_valuations")
    return dataset.build(claims, g, players, vals), claims, g, players, vals, club_names(tm_dir)


def club_names(tm_dir: Path) -> dict[int, str]:
    """Club id -> name: the clubs table, else the latest name a transfer gives it."""
    import pandas as pd

    from etl.transfermarkt.tables import read_raw
    t = pd.read_csv(Path(tm_dir) / "transfers.csv.gz", low_memory=False).sort_values("transfer_date")
    names = {}
    for col in ("from", "to"):
        part = t[[f"{col}_club_id", f"{col}_club_name"]].dropna()
        names.update(dict(zip(part[f"{col}_club_id"].astype(int), part[f"{col}_club_name"])))
    clubs = read_raw(tm_dir, "clubs").dropna(subset=["name"])
    names.update(dict(zip(clubs["club_id"].astype(int), clubs["name"])))
    return names


def _fit(args) -> int:
    from . import live
    started = time.perf_counter()
    data, claims, g, players, vals, names = _study(args.history_dir, args.transfermarkt_dir)
    print(f"Study table: {len(data):,} rumour claims ({time.perf_counter() - started:.0f}s); training as registered")
    hp = {"epochs": args.epochs} if args.epochs else None
    summary = live.fit(data, claims, g, players, vals, args.out, seeds=args.seeds or SEEDS,
                       meta_seeds=args.seeds or META_SEEDS, club_names=names, device=args.device,
                       model_name=args.bert_model or live.bert.MODEL_NAME, hparams=hp)
    v = summary["val_auc"]
    print(f"Validation AUC: metadata {v['metadata']:.4f}, BERTurk {v['bert']:.4f}, fusion {v['fusion']:.4f}")
    print(f"Saved to {args.out} ({time.perf_counter() - started:.0f}s)")
    return 0


def _score_archive(args) -> int:
    import json

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from etl.news.history import iter_jsonl, url_key
    from features.rumours.build import articles
    from features.rumours.labels import claims as make_claims

    from . import live
    scorer = live.Scorer.load(args.model, args.device)
    snap = scorer.snapshot
    arts = articles(args.history_dir)
    archive_end = arts["date"].max()
    window = arts[(arts["date"] >= pd.Timestamp(args.start)) & (arts["date"] <= snap)].reset_index(drop=True)
    titles = {url_key(t["url"]): t.get("title") for t in iter_jsonl([args.history_dir / "titles.jsonl.gz"])}
    print(f"{len(window):,} archived transfer articles from {args.start} to {snap:%Y-%m-%d} (end of the Transfermarkt data)")
    scored = scorer.score(window[["url", "site", "date", "slug"]])
    if scored.empty:
        print("No headline names a known player and club.")
        return 0
    # outcomes after the data's end: moves the later archive reports as done (checking only, not an input)
    reported = live.reported_moves(make_claims(arts[arts["date"] > snap], scorer.g), snap)
    res = live.outcome(scored, snap, reported, archive_end)
    scored = scored.assign(title=scored["url"].map(lambda u: titles.get(url_key(u))).fillna(scored["slug"]),
                           player=scored["player_id"].map(scorer.player_name),
                           club=scored["club_id"].map(scorer.club_name),
                           current_club=scored["current_club_id"].map(scorer.club_name),
                           happened=res["happened"], outcome=res["outcome"])
    cols = ["date", "site", "title", "player", "club", "current_club", "stance", "p", "p_metadata", "p_bert",
            "happened", "outcome", "url", "player_id", "club_id"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored[cols].sort_values(["date", "p"], ascending=[False, False]).to_csv(args.out, index=False)
    pair_table = live.pairs(scored)
    pair_table.to_csv(args.out.with_name(args.out.stem + "_pairs.csv"), index=False)

    r = scored[scored["stance"] == "rumour"]
    k = r[r["happened"].notna()]
    check = {"start": args.start, "data_end": f"{snap:%Y-%m-%d}", "archive_end": f"{archive_end:%Y-%m-%d}",
             "claims": int(len(r)), "known": int(len(k)), "happened": int(k["happened"].sum()),
             "reported_moves": int(len(reported))}
    if k["happened"].nunique() == 2:
        y, codes = k["happened"].to_numpy(), pd.factorize(k["player_id"].astype(str) + "-" + k["club_id"].astype(str))[0]
        members = [np.flatnonzero(codes == i) for i in range(codes.max() + 1)]
        rng, boot = np.random.default_rng(0), []
        for _ in range(1000):
            idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            if len(set(y[idx])) == 2:
                boot.append(roc_auc_score(y[idx], k["p"].to_numpy()[idx]))
        check.update({m: round(float(roc_auc_score(y, k[c])), 4) for m, c in
                      (("auc_fusion", "p"), ("auc_metadata", "p_metadata"), ("auc_bert", "p_bert"))})
        check["auc_fusion_95"] = [round(float(v), 4) for v in np.percentile(boot, [2.5, 97.5])]
        check["reliability"] = live.reliability(k["happened"], k["p"]).round(4).reset_index().to_dict("records")
    (args.out.with_name(args.out.stem + "_check.json")).write_text(json.dumps(check, indent=1), encoding="utf-8")

    print(f"{len(r):,} rumour claims ({len(pair_table):,} player-club pairs) scored -> {args.out}")
    print(f"Outcome known for {len(k):,}: {int(k['happened'].sum()):,} happened "
          f"(Transfermarkt to {snap:%d %b %Y}, then moves reported done in the archive to {archive_end:%d %b %Y})")
    if "auc_fusion" in check:
        lo, hi = check["auc_fusion_95"]
        print(f"AUC {check['auc_fusion']:.3f} [{lo:.3f}, {hi:.3f}] (metadata {check['auc_metadata']:.3f}, "
              f"BERTurk {check['auc_bert']:.3f})")
        for row in check["reliability"]:
            print(f"  predicted {row['band']:<14} {row['claims']:5,} claims, mean {row['predicted']:.0%}, "
                  f"happened {row['happened']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
