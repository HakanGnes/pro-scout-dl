"""The Phase 4 study code on synthetic data: masking, metadata, and the whole run with a tiny BERT."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.rumours.dataset import Names, mask

D = pd.Timestamp


def test_mask_keeps_endings_and_prefers_long_names():
    names = Names(player={1: [("mauro", "icardi"), ("icardi",)]},
                  club={141: [("galatasaray",), ("sari", "kirmizili")], 36: [("fenerbahce",)]})
    spans = [(names.player[1], "oyuncu"), (names.club[141], "hedefkulup"), (names.club[36], "mevcutkulup")]
    assert mask("galatasaraydan mauro icardiye teklif", spans) == "hedefkulupdan oyuncuye teklif"
    out = mask("sari kirmizililar icardi icin fenerbahceyi geride birakti", spans)
    assert out.startswith("hedefkulup") and "oyuncu icin mevcutkulupyi" in out


def _fixture(tmp_path, n_articles: int = 900, seed: int = 0):
    """Synthetic archive + Transfermarkt where 'kesinlesti' headlines come true and 'iddia' ones do not."""
    rng = np.random.default_rng(seed)
    tm = tmp_path / "tm"
    tm.mkdir()
    first = ["ali", "veli", "murat", "kerem", "emre", "burak", "hakan", "cenk", "arda", "yusuf"]
    last = ["kaplan", "demirkol", "ozturan", "yilmazer", "sahinler", "korkmazer", "aslaner", "kurtoglu",
            "celikten", "aydinoglu", "ozdemirci", "arslaner"]
    players = pd.DataFrame([{"player_id": i + 1, "name": f"{first[i % 10].title()} {last[i % 12].title()}",
                             "date_of_birth": "1995-01-01", "position": "Attack", "sub_position": "x", "foot": "right",
                             "height_in_cm": 180, "country_of_citizenship": "Turkey"} for i in range(60)])
    # curated club names, so the frequent-word filter (meant for real headlines) keeps them
    clubs = pd.DataFrame([{"club_id": cid, "name": n, "domestic_competition_id": "TR1"}
                          for cid, n in [(100, "Kartalspor"), (141, "Galatasaray"), (36, "Fenerbahce"),
                                         (114, "Besiktas")]])
    sites = ["aspor", "fanatik", "fotomac", "hurriyet_spor", "milliyet_skorer", "ntvspor", "sabah_spor", "sporx"]
    transfers, vals, rows = [], [], []
    start, end = D("2018-01-01"), D("2025-12-01")
    days = (end - start).days
    for p in players.itertuples():
        vals += [{"player_id": p.player_id, "date": f"{y}-06-01", "market_value_in_eur": 1_000_000,
                  "current_club_id": 100} for y in range(2017, 2026)]   # the data "ends" on 2025-06-01
        transfers.append({"player_id": p.player_id, "transfer_date": "2017-07-01", "from_club_id": 999,
                          "to_club_id": 100, "transfer_fee": 0, "market_value_in_eur": 1_000_000,
                          "from_club_name": "Other FC", "to_club_name": "Kartalspor"})
    for i in range(n_articles):
        p = players.iloc[rng.integers(0, len(players))]
        club = clubs.iloc[1 + rng.integers(0, 3)]
        date = start + pd.Timedelta(days=int(rng.integers(0, days)))
        true = rng.random() < 0.3
        word = "teklif kesinlesti" if true and rng.random() < 0.8 else "iddia teklif"
        name = p["name"].lower()
        slug = f"{club['name'].lower()} {name} icin {word} {i}"
        site = sites[i % len(sites)]
        url = f"https://www.{site}.com/{date:%Y/%m/%d}/{slug.replace(' ', '-')}"
        rows.append((site, {"site": site, "url": url, "ts": f"{date:%Y%m%d}120000", "slug": slug}))
        if true:
            transfers.append({"player_id": p["player_id"], "transfer_date": (date + pd.Timedelta(days=40)).date(),
                              "from_club_id": 100, "to_club_id": club["club_id"], "transfer_fee": 0,
                              "market_value_in_eur": 1e6, "from_club_name": "Kartalspor", "to_club_name": club["name"]})
    for name, df in (("players", players), ("clubs", clubs), ("transfers", pd.DataFrame(transfers)),
                     ("player_valuations", pd.DataFrame(vals))):
        df.to_csv(tm / f"{name}.csv.gz", index=False)
    hist = tmp_path / "history"
    for site in sites:
        (hist / site).mkdir(parents=True)
        with gzip.open(hist / site / "index.jsonl.gz", "wt", encoding="utf-8") as f:
            for s, r in rows:
                if s == site:
                    f.write(json.dumps(r) + "\n")
    return hist, tm


def _tiny_bert(path, texts):
    transformers = pytest.importorskip("transformers")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted({w for t in texts for w in t.split()})
    path.mkdir()
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = transformers.BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(path)
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=64, max_position_embeddings=64, num_labels=1)
    transformers.BertForSequenceClassification(cfg).save_pretrained(path)


def test_study_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from models.rumours import __main__ as cli

    hist, tm = _fixture(tmp_path)
    words = "hedefkulup oyuncu mevcutkulup baskakulup baskaoyuncu icin teklif kesinlesti iddia".split()
    _tiny_bert(tmp_path / "bert", words + [str(i) for i in range(1000)])
    rep = tmp_path / "report"
    assert cli.main(["run", "--history-dir", str(hist), "--transfermarkt-dir", str(tm), "--report-dir", str(rep),
                     "--predictions", str(tmp_path / "pred.csv.gz"), "--bert-model", str(tmp_path / "bert"),
                     "--epochs", "2", "--seeds", "0", "--n-boot", "50", "--device", "cpu",
                     "--allow-other-inputs"]) == 0
    res = json.loads((rep / "results.json").read_text())
    assert set(res["test"]) == {"base", "site", "metadata", "tfidf", "bert", "fusion", "fusion_tfidf"}
    assert res["test"]["tfidf"]["auc"] > 0.8          # the wording carries the signal in this fixture
    lo, hi = res["primary"][1], res["primary"][2]
    assert lo <= res["primary"][0] <= hi
    pred = pd.read_csv(tmp_path / "pred.csv.gz")
    assert len(pred) == res["counts"]["test"] and "kesinlesti" in " ".join(pred["text"])
    assert pred["site"].isin(["aspor", "fanatik", "fotomac", "hurriyet_spor", "milliyet_skorer", "ntvspor", "sabah_spor", "sporx"]).all()
    assert not pred["text"].str.contains("kartalspor|galatasaray|fenerbahce|besiktas").any()   # names masked
    assert (rep / "README.md").exists()
    by_site = pd.read_csv(rep / "auc_by_site.csv")
    assert set(by_site["site"]) <= {"aspor", "fanatik", "fotomac", "hurriyet_spor", "milliyet_skorer", "ntvspor",
                                    "sabah_spor", "sporx"} and "auc_site" in by_site


def test_inputs_must_match_registration(tmp_path, monkeypatch):
    from models.rumours import __main__ as cli
    f = tmp_path / "a.txt"
    f.write_text("x")
    monkeypatch.setattr(cli, "REGISTERED_FILES", {"a": "0" * 64})
    with pytest.raises(SystemExit):
        cli.check_inputs({"a": f}, strict=True)
    assert cli.check_inputs({"a": f}, strict=False)["a"] == cli.sha256(f)


def test_headline_slug():
    from models.rumours.live import headline_slug
    assert headline_slug("Galatasaray'dan Icardi'ye teklif!") == "galatasaraydan icardiye teklif"
    assert headline_slug("Fenerbahçe, Şenol Güneş'le görüştü – SON DAKİKA") == "fenerbahce senol gunesle gorustu son dakika"


def test_fit_save_and_score_archive(tmp_path):
    pytest.importorskip("torch")
    from models.rumours import __main__ as cli
    from models.rumours.live import Scorer

    hist, tm = _fixture(tmp_path)
    words = "hedefkulup oyuncu mevcutkulup baskakulup baskaoyuncu icin teklif kesinlesti iddia".split()
    _tiny_bert(tmp_path / "bert", words + [str(i) for i in range(1000)])
    art = tmp_path / "artifacts"
    assert cli.main(["fit", "--history-dir", str(hist), "--transfermarkt-dir", str(tm), "--out", str(art),
                     "--bert-model", str(tmp_path / "bert"), "--epochs", "1", "--seeds", "0", "--device", "cpu"]) == 0
    fit = json.loads((art / "fit.json").read_text())
    assert fit["val"] > 0 and (art / "bert_seed0" / "config.json").exists() and (art / "metadata_0.txt").exists()

    out = tmp_path / "scored.csv"
    assert cli.main(["score-archive", "--history-dir", str(hist), "--model", str(art), "--start", "2025-01-01",
                     "--out", str(out), "--device", "cpu"]) == 0
    scored = pd.read_csv(out, parse_dates=["date"])
    assert len(scored) and scored["date"].between(D("2025-01-01"), D("2025-06-01")).all()   # nothing after the data
    r = scored[scored["stance"] == "rumour"]
    assert r["p"].between(0, 1).all() and r["outcome"].str.match(r"happened|open|did not").all()

    s = Scorer.load(art, "cpu")       # the app's path: score one typed headline
    one = s.score(pd.DataFrame({"url": ["typed"], "site": ["sporx"], "date": [D("2025-05-01")],
                                "title": ["Galatasaray Ali Kaplan icin iddia teklif"]}))
    assert (one["stance"] == "rumour").any() and one["p"].notna().any()
    later = s.score(pd.DataFrame({"url": ["t"], "site": ["sporx"], "date": [D("2025-09-01")],
                                  "title": ["Galatasaray Ali Kaplan icin iddia teklif"]}))
    assert later.empty                 # after the end of the data: the player's club is unknown, so nothing is scored
    none = s.score(pd.DataFrame({"url": ["t2"], "site": ["sporx"], "date": [D("2025-05-01")], "title": ["Derbide gol yagmuru"]}))
    assert none.empty and "p" in none


def test_outcome_labels():
    from models.rumours.live import outcome, pairs, reported_moves
    c = pd.DataFrame({"date": [D("2025-01-01"), D("2025-05-01"), D("2025-05-01"), D("2025-05-01")],
                      "label": [0, 0, 1, 0], "move_date": [pd.NaT, pd.NaT, D("2025-07-01"), pd.NaT],
                      "player_id": [1, 2, 3, 4], "club_id": [10, 10, 10, 36]})
    o = outcome(c, D("2025-08-01"))
    assert o["outcome"].tolist() == ["did not happen", "open: nothing reported by 01 Aug 2025",
                                     "happened (Transfermarkt, 01 Jul 2025)", "open: nothing reported by 01 Aug 2025"]
    assert o["happened"].tolist()[:1] == [0.0] and np.isnan(o["happened"].iloc[1]) and o["happened"].iloc[2] == 1.0
    # after the data ends: a move reported done by 3 articles from 2 papers in completed-move words
    late = pd.DataFrame({"url": ["a", "b", "c", "d"], "site": ["x", "y", "x", "x"],
                         "date": [D("2025-09-01"), D("2025-09-02"), D("2025-09-03"), D("2025-09-04")],
                         "slug": ["fenerbahce resmen acikladi", "fenerbahcenin yeni transferi", "imza toreni",
                                  "anlasma saglandi"], "player_id": [4] * 4, "club_id": [36] * 4,
                         "stance": ["announcement"] * 4})
    rep = reported_moves(late, D("2025-08-01"))
    assert rep[["player_id", "club_id"]].values.tolist() == [[4, 36]] and rep["date"].iloc[0] == D("2025-09-03")
    o = outcome(c, D("2025-08-01"), rep, archive_end=D("2026-01-01"))
    assert o["outcome"].iloc[3] == "happened (reported done in news, 03 Sep 2025)" and o["happened"].iloc[1] == 0.0
    assert reported_moves(late.assign(site="x"), D("2025-08-01")).empty          # one paper is not enough
    sc = c.assign(stance="rumour", url=["u1", "u2", "u3", "u4"], site="s", player=["A", "B", "C", "D"],
                  club="K", current_club="Z", p=[0.1, 0.2, 0.3, 0.4], happened=o["happened"], outcome=o["outcome"])
    pt = pairs(pd.concat([sc, sc.iloc[[3]].assign(url="u5", date=D("2025-05-02"), p=0.6)]))
    d = pt.set_index("player").loc["D"]
    assert d["articles"] == 2 and d["latest_p"] == 0.6 and d["highest_p"] == 0.6 and d["happened"] == 1.0


def test_rumour_page(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("torch")
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from models.rumours import __main__ as cli

    hist, tm = _fixture(tmp_path)
    words = "hedefkulup oyuncu mevcutkulup baskakulup baskaoyuncu icin teklif kesinlesti iddia".split()
    _tiny_bert(tmp_path / "bert", words + [str(i) for i in range(1000)])
    art, rep = tmp_path / "artifacts", tmp_path / "report"
    common = ["--history-dir", str(hist), "--transfermarkt-dir", str(tm), "--bert-model", str(tmp_path / "bert"),
              "--epochs", "1", "--seeds", "0", "--device", "cpu"]
    assert cli.main(["fit", "--out", str(art), *common]) == 0
    assert cli.main(["run", "--report-dir", str(rep), "--predictions", str(tmp_path / "pred.csv.gz"), "--n-boot", "20",
                     "--allow-other-inputs", *common]) == 0
    monkeypatch.setenv("PROSCOUT_RUMOUR_MODEL", str(art))
    monkeypatch.setenv("PROSCOUT_RUMOUR_SCORED", str(tmp_path / "missing.csv"))
    monkeypatch.setenv("PROSCOUT_RUMOUR_DATA_END", "2025-06-01")
    monkeypatch.setenv("PROSCOUT_RUMOUR_PREDICTIONS", str(tmp_path / "pred.csv.gz"))
    monkeypatch.setenv("PROSCOUT_DEVICE", "cpu")
    st.cache_resource.clear()
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "rumours.py"), default_timeout=120).run()
    assert not at.exception
    assert any("Not scored yet" in i.value for i in at.info)                  # 2026 tab without a scored file
    at.text_input(key="headline").set_value("Galatasaray'dan Ali Kaplan için iddia: teklif")
    at.button[0].click().run()
    assert not at.exception
    assert any(m.label == "Probability" for m in at.metric)

    assert cli.main(["score-archive", "--history-dir", str(hist), "--model", str(art), "--start", "2025-01-01",
                     "--out", str(tmp_path / "scored.csv"), "--device", "cpu"]) == 0
    monkeypatch.setenv("PROSCOUT_RUMOUR_SCORED", str(tmp_path / "scored.csv"))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "rumours.py"), default_timeout=120).run()
    assert not at.exception
    table = next(d.value for d in at.dataframe if "latest" in d.value.columns)   # the pair table
    assert len(table) and table["latest"].str.endswith("%").all() and "outcome" in table and "articles" in table
    at.toggle[0].set_value(False).run()                                         # one row per headline
    assert not at.exception
    assert any("probability" in d.value for d in at.dataframe)
    st.cache_resource.clear()
