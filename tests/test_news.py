from __future__ import annotations

import gzip
import json

import pytest

from etl.news import collect as c
from etl.news.rss import canonical_url, discover, parse_feed

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel><title>Spor</title>
<item><title><![CDATA[Fenerbahçe'den <b>forvet</b> hamlesi]]></title>
  <link>https://site.example/haber/fb-forvet?utm_source=rss</link>
  <description><![CDATA[<p>Sarı-lacivertliler &amp; golcü için görüşmelere başladı.</p>]]></description>
  <pubDate>Tue, 27 Oct 2026 10:15:00 +0300</pubDate><guid>fb-1</guid><category>Transfer</category></item>
<item><title>Başlıksız link yok</title></item>
<item><title>Galatasaray resmi açıklama</title><link>/haber/gs-resmi</link><pubDate>2026-10-27T09:00:00</pubDate></item>
</channel></rss>"""

ATOM = """<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom"><title>x</title>
<entry><title>Beşiktaş stoper arıyor</title><link rel="alternate" href="https://a.example/bjk"/>
<id>tag:a,1</id><updated>2026-10-26T20:00:00Z</updated><summary>Kartal kış için hazır.</summary>
<category term="futbol"/></entry></feed>"""


def test_parse_rss_cleans_and_resolves():
    items = parse_feed(RSS.encode("utf-8"), "test", "https://site.example/rss")
    assert [i.title for i in items] == ["Fenerbahçe'den forvet hamlesi", "Galatasaray resmi açıklama"]
    first, second = items
    assert first.summary == "Sarı-lacivertliler & golcü için görüşmelere başladı."
    assert first.published_at == "2026-10-27T07:15:00+00:00"
    assert first.categories == ["Transfer"] and first.guid == "fb-1"
    assert second.url == "https://site.example/haber/gs-resmi"            # relative link resolved
    assert second.published_at == "2026-10-27T06:00:00+00:00"            # no zone: Turkish time assumed
    assert canonical_url(first.url) == "https://site.example/haber/fb-forvet"


def test_windows_1254_feed():
    body = RSS.replace('encoding="UTF-8"', 'encoding="windows-1254"').encode("cp1254")
    assert parse_feed(body, "test")[0].title == "Fenerbahçe'den forvet hamlesi"


def test_parse_atom_and_rejects_html():
    item = parse_feed(ATOM.encode(), "a")[0]
    assert (item.title, item.url, item.categories) == ("Beşiktaş stoper arıyor", "https://a.example/bjk", ["futbol"])
    with pytest.raises(ValueError):
        parse_feed(b"<html><body>hi</body></html>", "x")
    with pytest.raises(ValueError):
        parse_feed(b"not xml at all <<<", "x")


def test_discover_feed_links():
    page = b"""<html><head><link rel="alternate" type="application/rss+xml" href="/rss/spor.xml">
               <link rel="stylesheet" href="/a.css"></head></html>"""
    assert discover(page, "https://site.example/") == ["https://site.example/rss/spor.xml"]


def fake_fetcher(pages: dict[str, bytes]):
    def fetch(url, timeout=30):
        if url not in pages:
            raise OSError("404")
        return pages[url], url
    return fetch


def test_probe_keeps_working_feeds(tmp_path):
    pages = {"https://a.example": b'<link rel="alternate" type="application/rss+xml" href="/found.xml">',
             "https://a.example/found.xml": RSS.encode(), "https://b.example/rss": b"<html></html>"}
    out = tmp_path / "feeds.json"
    working = c.probe({"a": {"home": "https://a.example", "urls": ["https://a.example/missing"]},
                       "b": {"urls": ["https://b.example/rss"]}}, out=out, log=lambda _m: None,
                      fetcher=fake_fetcher(pages))
    assert working == {"a": ["https://a.example/found.xml"]}
    assert json.loads(out.read_text(encoding="utf-8"))["feeds"] == working


def test_probe_drops_excluded_feeds(tmp_path, monkeypatch):
    monkeypatch.setitem(c.EXCLUDE, "https://a.example/all.xml", "not sport")
    pages = {"https://a.example/all.xml": RSS.encode(), "https://a.example/spor.xml": RSS.encode()}
    working = c.probe({"a": {"urls": ["https://a.example/all.xml", "https://a.example/spor.xml"]}},
                      out=tmp_path / "f.json", log=lambda _m: None, fetcher=fake_fetcher(pages))
    assert working == {"a": ["https://a.example/spor.xml"]}


def test_collect_archives_only_new_items(tmp_path):
    feeds = {"s": ["https://s.example/rss"], "broken": ["https://down.example/rss"]}
    fetch = fake_fetcher({"https://s.example/rss": RSS.encode()})
    r1 = c.collect(tmp_path, feeds, log=lambda _m: None, fetcher=fetch)
    r2 = c.collect(tmp_path, feeds, log=lambda _m: None, fetcher=fetch)
    assert (r1["new"], r2["new"]) == (2, 0)
    assert "https://down.example/rss" in r1["errors"]
    rows = list(c.iter_archive(tmp_path))
    assert len(rows) == 2 and rows[0]["feed"] == "s" and rows[0]["fetched_at"]
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 2
    # the archive is a valid gzip even after several appends
    path = next(tmp_path.rglob("*.jsonl.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert sum(1 for _ in f) == 2


def test_collect_inserts_and_load_is_idempotent(conn, tmp_path):
    fetch = fake_fetcher({"https://s.example/rss": RSS.encode()})
    r = c.collect(tmp_path, {"s": ["https://s.example/rss"]}, conn=conn, log=lambda _m: None, fetcher=fetch)
    assert r["inserted"] == 2
    conn.execute("DELETE FROM news_items WHERE title LIKE 'Galatasaray%'")
    assert c.load_archive(conn, tmp_path, log=lambda _m: None) == 1      # puts back only the missing row
    assert c.load_archive(conn, tmp_path, log=lambda _m: None) == 0
    title, cats, published = conn.execute(
        "SELECT title, categories, published_at FROM news_items WHERE guid = 'fb-1'").fetchone()
    assert title.startswith("Fenerbahçe") and cats == ["Transfer"] and published.year == 2026


def test_fetch_retries_connection_errors(monkeypatch):
    import urllib.error
    import urllib.request

    from etl.news import rss

    calls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

        def geturl(self):
            return "https://x.example/rss"

    def flaky(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.URLError(ConnectionResetError(10054, "reset"))
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    assert rss.fetch("https://x.example/rss", wait=0) == (b"ok", "https://x.example/rss")
    assert len(calls) == 2

    def not_found(req, timeout):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    calls.clear()
    with pytest.raises(urllib.error.HTTPError):
        rss.fetch("https://x.example/missing", wait=0)
    assert len(calls) == 1  # a 404 is not retried


def test_archive_slugs():
    from etl.news.archive import fold, is_transfer, slug_text

    assert fold("Fenerbahçe İcardi ŞAMPİYON ığ") == "fenerbahce icardi sampiyon ig"
    assert slug_text("https://www.fotomac.com.tr/galatasaray/2023/07/12/icardi-ile-resmi-anlasma-saglandi-123456") \
        == "icardi ile resmi anlasma saglandi"
    assert slug_text("fanatik.com.tr/transfer-haberleri-12345.html") is None      # too few words
    assert slug_text("https://www.aspor.com.tr/futbol/") is None                   # a section, not an article
    assert is_transfer("icardi ile resmi anlasma saglandi") and not is_transfer("derbide gol yagmuru oldu sahada")


def test_archive_probe_counts():
    from etl.news import archive

    rows = [["timestamp", "original"],
            ["20190701000000", "https://fotomac.com.tr/fb/fenerbahce-den-golcuye-resmi-teklif-111111"],
            ["20190705000000", "https://fotomac.com.tr/fb/fenerbahce-den-golcuye-resmi-teklif-111111"],
            ["20210702000000", "https://fotomac.com.tr/fb/derbide-gol-yagmuru-oldu-222222"],
            ["20190703000000", "https://fotomac.com.tr/futbol/"]]
    page = "<html><head><title>Fenerbahçe'den golcüye resmi teklif</title></head></html>".encode()

    def fake(url, timeout=30, **_):
        if "showNumPages" in url:
            return b"10\n", url
        return (json.dumps(rows).encode(), url) if "/cdx/" in url else (page, url)

    r = archive.probe({"fotomac": "fotomac.com.tr/"}, pages_per_site=2, pause=0, log=lambda _m: None,
                      fetcher=fake)["fotomac"]
    # two sampled blocks of ten, each with the same 2 articles (1 transfer): estimate = 1 / 2 * 10
    assert (r["pages"], r["sampled"], r["articles"], r["transfer"], r["estimate"]) == (10, 2, 2, 1, 5)
    assert r["by_year"] == {"2019": 1}
    assert r["titles"] == ["Fenerbahçe'den golcüye resmi teklif"]


def test_fetch_retries_server_errors_only(monkeypatch):
    import urllib.error
    import urllib.request

    from etl.news import rss

    calls = []

    def gateway_timeout(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 504, "Gateway Time-out", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", gateway_timeout)
    with pytest.raises(urllib.error.HTTPError):
        rss.fetch("https://x.example/cdx", retries=2, wait=0)
    assert len(calls) == 3
