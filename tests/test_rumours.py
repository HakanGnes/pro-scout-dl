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
    assert article_date("https://www.fotomac.com.tr/gs/2023/02/01/x", "20230203120000") == (D("2023-02-01"), "url")
    assert article_date("https://www.fanatik.com.tr/x-2169107", "20200907122546") == (D("2020-09-07"), "capture")
    assert article_date("https://www.sporx.com/x", "20180101120000", "2017-07-13 20:21:24") == (D("2017-07-13"),
                                                                                                "published")
    assert article_date("https://www.sporx.com/x", "20180101120000", "2019-01-01")[1] == "capture"   # after capture
    assert article_date("https://www.sporx.com/x", "20180101120000", "garbage")[1] == "capture"
    assert clean_slug("abdulkadir icin teklif sxhbq981424sxq") == "abdulkadir icin teklif"
    assert clean_slug("feghouli den aciklama 5cf403f6237d8a159cbc784e 2043608") == "feghouli den aciklama"
    assert clean_slug("u21 takimina 3 transfer") == "u21 takimina 3 transfer"


def test_first_name_must_be_another_word():
    players = pd.DataFrame({"player_id": [1, 2], "name": ["Said Ahmed Said", "Said El Mala"]})
    clubs = pd.DataFrame({"club_id": [141], "name": ["Galatasaray"], "domestic_competition_id": ["TR1"]})
    transfers = pd.DataFrame({"player_id": [1], "transfer_date": [D("2025-07-01")], "from_club_id": [5],
                              "to_club_id": [7], "from_club_name": ["A"], "to_club_name": ["B"]})
    g = Gazetteer.build(players, clubs, transfers)
    found, _ = g.link("galatasarayin radarinda alman cevheri said el mala", D("2026-10-29"))
    assert [p.id for p in found] == [2]          # "said" is El Mala's first name, not Said Ahmed Said's surname


def test_last_name_links_use_only_the_past():
    players = pd.DataFrame({"player_id": [1, 2, 3, 4], "name": ["Ali Kaplan", "Dusan Tadic", "Dusan Vlahovic",
                                                                "Vanja Vlahovic"]})
    clubs = pd.DataFrame({"club_id": [36, 141, 506], "name": ["Fenerbahce", "Galatasaray", "Juventus"],
                          "domestic_competition_id": ["TR1", "TR1", "IT1"]})
    transfers = pd.DataFrame({
        "player_id": [1, 1, 2, 2, 3], "transfer_date": [D("2020-07-01"), D("2025-08-01"), D("2021-07-01"),
                                                         D("2025-06-01"), D("2021-07-01")],
        "from_club_id": [9, 9, 610, 36, 430], "to_club_id": [9, 36, 36, 1000, 506],
        "from_club_name": ["X", "X", "Ajax", "Fenerbahce", "Fiorentina"],
        "to_club_name": ["X", "Fenerbahce", "Fenerbahce", "Al-Wahda", "Juventus"]})
    vals = pd.DataFrame({"player_id": [1, 2, 3, 3, 4], "date": [D("2025-01-01"), D("2025-01-01"), D("2024-06-01"),
                                                                 D("2025-12-01"), D("2025-12-01")],
                         "market_value_in_eur": [500_000, 3_000_000, 40_000_000, 30_000_000, 6_000_000]})
    g = Gazetteer.build(players, clubs, transfers, vals)
    # Kaplan moves to Fenerbahce in Aug 2025: a headline in March must not be linked through that later move
    found, _ = g.link("fenerbahce kaplan icin teklif", D("2025-03-01"))
    assert found == []
    # Vlahovic: no tie to Fenerbahce, but Dusan (EUR 40m) is worth over 5x Vanja (EUR 6m), the other active one;
    # Tadic left Fenerbahce in the year before and is background when another player is named
    found, clubs_found = g.link("fenerbahceden vlahovic hamlesi transferde tadic detayi", D("2026-01-10"))
    assert [(m.id, m.how) for m in found] == [(3, "last_unique")] and [c.id for c in clubs_found] == [36]
    found, _ = g.link("fenerbahce tadic ile yeniden gorusuyor", D("2026-01-10"))
    assert [(m.id, m.how) for m in found] == [(2, "last_left")]            # alone, the ex-player stays


def test_unique_surname_skips_one_name_players_and_club_words():
    players = pd.DataFrame({"player_id": [1, 2], "name": ["Marcelo", "Valere Germain"]})
    clubs = pd.DataFrame({"club_id": [583, 114], "name": ["Paris Saint-Germain", "Besiktas"],
                          "domestic_competition_id": ["FR1", "TR1"]})
    transfers = pd.DataFrame({"player_id": [9], "transfer_date": [D("2020-01-01")], "from_club_id": [1],
                              "to_club_id": [2], "from_club_name": ["A"], "to_club_name": ["B"]})
    vals = pd.DataFrame({"player_id": [1, 2], "date": [D("2021-01-01")] * 2, "market_value_in_eur": [5e6, 5e6]})
    g = Gazetteer.build(players, clubs, transfers, vals)
    assert g.link("marcelo bielsadan besiktas aciklamasi", D("2021-06-01"))[0] == []
    players_found, clubs_found = g.link("besiktas paris saint germainin yildizini istiyor", D("2021-06-01"))
    assert players_found == [] and {c.id for c in clubs_found} == {114, 583}
