from __future__ import annotations

import pandas as pd
import pytest

from features.rumours.entities import Gazetteer, stems
from features.rumours.labels import claims, stance

D = pd.Timestamp


@pytest.fixture
def g() -> Gazetteer:
    players = pd.DataFrame({"player_id": [1, 2, 3, 4],
                            "name": ["Mauro Icardi", "Altay Bayındır", "Soner Gönül", "Luka Jović"]})
    clubs = pd.DataFrame({"club_id": [141, 36, 2375, 9], "name": ["Galatasaray", "Fenerbahçe", "Altay SK", "Sona"],
                          "domestic_competition_id": ["TR1", "TR1", "TR1", "X"]})
    transfers = pd.DataFrame({
        "player_id": [1, 1, 2, 4, 4], "transfer_date": [D("2019-07-01"), D("2022-09-01"), D("2019-07-01"),
                                                         D("2019-07-01"), D("2024-01-15")],
        "from_club_id": [9, 5, 2375, 418, 418], "to_club_id": [5, 141, 36, 418, 36],
        "from_club_name": ["Sona", "AC Milan", "Altay SK", "Real Madrid", "Real Madrid"],
        "to_club_name": ["AC Milan", "Galatasaray", "Fenerbahçe", "Real Madrid", "Fenerbahçe"]})
    slugs = ["sona erdi mi"] * 50 + ["gonul rahatligiyla oynadi"] * 50 + ["galatasaray"] * 100
    return Gazetteer.build(players, clubs, transfers, slugs=slugs)


def test_turkish_endings_only():
    assert "galatasaray" in stems("galatasaraya") and "icardi" in stems("icardinin")
    assert stems("transfer") == ["transfer"] and stems("karar") == ["karar"]   # 'trans' + 'fer' is not an ending


def test_link_full_names_before_clubs(g):
    players, clubs = g.link("altay bayindir fenerbahce ile yeni sozlesme", D("2023-01-01"))
    assert [p.id for p in players] == [2] and [c.id for c in clubs] == [36]    # 'altay' is not Altay SK here
    players, clubs = g.link("galatasaraya icardiden kotu haber", D("2023-01-01"))
    assert [(p.id, p.how) for p in players] == [(1, "last")] and [c.id for c in clubs] == [141]


def test_everyday_words_are_not_names(g):
    assert g.link("sona erdi transfer donemi", D("2023-01-01")) == ([], [])          # 'sona' the club is dropped
    players, _ = g.link("halil gonul rahatligiyla fenerbahce", D("2023-01-01"))
    assert players == []                                                           # Soner Gönül is not tied to FB


def test_stance():
    assert stance("galatasaray icardi ile resmen anlasti") == "announcement"
    assert stance("fenerbahce jovic icin teklif yapti") == "rumour"
    assert stance("derbide gol yagmuru") == "other"


def test_claims_labels(g):
    arts = pd.DataFrame({"url": ["u1", "u2", "u3"], "site": "x",
                         "date": [D("2023-12-01"), D("2022-08-20"), D("2023-01-10")],
                         "slug": ["fenerbahce luka jovic icin teklif yapti",            # moves to FB 2024-01-15
                                  "galatasaray mauro icardi ile resmen anlasti",       # moved 2022-09-01
                                  "galatasaray mauro icardi icin yeni teklif"]})       # already at GS: no claim
    c = claims(arts, g).set_index("url")
    assert list(c.index) == ["u1", "u2"]
    assert c.loc["u1", "label"] == 1 and c.loc["u1", "stance"] == "rumour" and c.loc["u1", "current_club_id"] == 418
    assert c.loc["u2", "label"] == 1 and c.loc["u2", "stance"] == "announcement"


def test_stance_press_patterns():
    assert stance("trabzonspor transfer haberleri firtinadan artem dovbyk atagi") == "rumour"
    assert stance("son dakika fenerbahceden gokhan gonul operasyonu") == "rumour"
    assert stance("galatasaraydan elias jelert icin resmi teklif") == "rumour"          # an official bid is not a deal
    assert stance("italyan basini duyurdu galatasaray toure icin gorusmelere basladi") == "rumour"
    assert stance("denizlispor kerem can akyuz ile anlasmaya vardi") == "announcement"


def test_article_dates_and_slugs():
    from features.rumours.build import article_date, clean_slug
    assert article_date("https://www.fotomac.com.tr/gs/2023/02/01/x", "20230203120000") == D("2023-02-01")
    assert article_date("https://www.fanatik.com.tr/x-2169107", "20200907122546") == D("2020-09-07")
    assert clean_slug("abdulkadir icin teklif sxhbq981424sxq") == "abdulkadir icin teklif"
    assert clean_slug("feghouli den aciklama 5cf403f6237d8a159cbc784e 2043608") == "feghouli den aciklama"
    assert clean_slug("u21 takimina 3 transfer") == "u21 takimina 3 transfer"
