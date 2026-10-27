"""Fetch and parse RSS 2.0 / Atom feeds with the standard library only.

Turkish sites sometimes declare windows-1254 or ISO-8859-9, which the XML parser does not support;
such feeds are re-decoded with Python's codecs before parsing.
"""
from __future__ import annotations

import hashlib
import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

USER_AGENT = "pro-scout-dl/0.1 research crawler (+https://github.com/HakanGnes/pro-scout-dl)"
TIMEOUT = 30
_DECL = re.compile(rb"^\s*<\?xml[^>]*?encoding=[\"']([A-Za-z0-9_\-]+)[\"'][^>]*\?>", re.S)
_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
TURKEY = timezone(timedelta(hours=3))  # assumed when a feed gives a time without a zone


@dataclass
class Item:
    feed: str
    url: str
    title: str
    summary: str = ""
    published_at: str | None = None   # ISO 8601, UTC
    guid: str | None = None
    categories: list[str] = field(default_factory=list)

    @property
    def item_id(self) -> str:
        return hashlib.sha1(canonical_url(self.url).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {"item_id": self.item_id, **asdict(self)}


def canonical_url(url: str) -> str:
    """Drop query strings used for tracking and the fragment; keep scheme, host and path."""
    parts = urlsplit(url.strip())
    query = "&".join(q for q in parts.query.split("&") if q and not q.lower().startswith(("utm_", "fbclid", "gclid")))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, query, ""))


def fetch(url: str, timeout: int = TIMEOUT) -> tuple[bytes, str]:
    """Body and final URL (after redirects)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, "
                                                "application/atom+xml, application/xml, text/xml, text/html;q=0.5"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.geturl()


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    return _SPACE.sub(" ", html.unescape(_TAGS.sub(" ", s))).strip()


def _parse_date(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    try:
        d = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=TURKEY)
    return d.astimezone(timezone.utc).isoformat(timespec="seconds")


def _root(body: bytes) -> ET.Element:
    m = _DECL.match(body)
    enc = m.group(1).decode().lower() if m else "utf-8"
    if enc not in ("utf-8", "utf8", "us-ascii", "iso-8859-1", "utf-16"):
        body = body[m.end():].decode(enc, errors="replace").encode("utf-8")
    return ET.fromstring(body)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child(el: ET.Element, *names: str) -> ET.Element | None:
    for c in el:
        if _local(c.tag) in names:
            return c
    return None


def _text(el: ET.Element, *names: str) -> str | None:
    c = _child(el, *names)
    return (c.text or "").strip() if c is not None and c.text else None


def parse_feed(body: bytes, feed: str, base_url: str = "") -> list[Item]:
    """Items of an RSS 2.0, RSS 1.0 (RDF) or Atom feed. Raises ValueError if it is not a feed."""
    try:
        root = _root(body)
    except ET.ParseError as e:
        raise ValueError(f"not XML: {e}") from e
    kind = _local(root.tag)
    if kind == "rss":
        channel = _child(root, "channel")
        entries = [] if channel is None else [c for c in channel if _local(c.tag) == "item"]
    elif kind == "rdf":
        entries = [c for c in root if _local(c.tag) == "item"]
    elif kind == "feed":
        entries = [c for c in root if _local(c.tag) == "entry"]
    else:
        raise ValueError(f"not a feed (root <{kind}>)")

    items = []
    for e in entries:
        if kind == "feed":
            link = next((c.get("href") for c in e if _local(c.tag) == "link" and c.get("rel", "alternate") == "alternate"),
                        None)
            summary = _text(e, "summary", "content")
            date = _text(e, "published", "updated")
            guid = _text(e, "id")
            cats = [c.get("term", "") for c in e if _local(c.tag) == "category"]
        else:
            link = _text(e, "link")
            summary = _text(e, "description", "encoded")
            date = _text(e, "pubdate", "date", "published")
            guid = _text(e, "guid")
            cats = [(c.text or "").strip() for c in e if _local(c.tag) == "category"]
        title = clean_text(_text(e, "title"))
        if not link and guid and guid.startswith("http"):
            link = guid
        if not link or not title:
            continue
        items.append(Item(feed=feed, url=urljoin(base_url, link.strip()), title=title, summary=clean_text(summary)[:2000],
                          published_at=_parse_date(date), guid=guid, categories=[c for c in cats if c]))
    return items


class _LinkFinder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "link" and (a.get("rel") or "").lower() == "alternate" and \
                (a.get("type") or "").lower() in ("application/rss+xml", "application/atom+xml") and a.get("href"):
            self.links.append(a["href"])


def discover(home_html: bytes, base_url: str) -> list[str]:
    """RSS/Atom links a homepage advertises with <link rel="alternate">."""
    finder = _LinkFinder()
    finder.feed(home_html.decode("utf-8", errors="replace"))
    return [urljoin(base_url, h) for h in finder.links]
