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
