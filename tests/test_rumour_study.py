"""The Phase 4 study code on synthetic data: masking, metadata, and the whole run with a tiny BERT."""
from __future__ import annotations

import gzip
import json

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
        vals.append({"player_id": p.player_id, "date": "2017-06-01", "market_value_in_eur": 1_000_000,
                     "current_club_id": 100})
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
    assert not pred["text"].str.contains("kartalspor|galatasaray|fenerbahce|besiktas").any()   # names masked
    assert (rep / "README.md").exists()


def test_inputs_must_match_registration(tmp_path, monkeypatch):
    from models.rumours import __main__ as cli
    f = tmp_path / "a.txt"
    f.write_text("x")
    monkeypatch.setattr(cli, "REGISTERED_FILES", {"a": "0" * 64})
    with pytest.raises(SystemExit):
        cli.check_inputs({"a": f}, strict=True)
    assert cli.check_inputs({"a": f}, strict=False)["a"] == cli.sha256(f)
