"""Turkish sports-news sources.

`CANDIDATES` are guesses to try: a homepage (searched for <link rel="alternate"> RSS links) and URL
patterns these sites have used. `python -m etl probe-news` tries them all, reports which work and
writes the working ones to `feeds.json` next to this file (tracked in git), which is what
`collect-news` reads.
"""
from __future__ import annotations

import json
from pathlib import Path

FEEDS_FILE = Path(__file__).with_name("feeds.json")

CANDIDATES: dict[str, dict] = {
    "ntvspor": {"home": "https://www.ntvspor.net",
                "urls": ["https://www.ntvspor.net/rss", "https://www.ntvspor.net/rss/futbol",
                         "https://www.ntv.com.tr/sporskor.rss", "https://www.ntv.com.tr/spor.rss"]},
    "fotomac": {"home": "https://www.fotomac.com.tr",
                "urls": ["https://www.fotomac.com.tr/rss/anasayfa.xml", "https://www.fotomac.com.tr/rss/futbol.xml",
                         "https://www.fotomac.com.tr/rss/transfer.xml"]},
    "fanatik": {"home": "https://www.fanatik.com.tr",
                "urls": ["https://www.fanatik.com.tr/rss/anasayfa.xml", "https://www.fanatik.com.tr/rss",
                         "https://www.fanatik.com.tr/rss/futbol.xml"]},
    "aspor": {"home": "https://www.aspor.com.tr",
              "urls": ["https://www.aspor.com.tr/rss/anasayfa.xml", "https://www.aspor.com.tr/rss/futbol.xml"]},
    "sporx": {"home": "https://www.sporx.com",
              "urls": ["https://www.sporx.com/_xml/rss.php", "https://www.sporx.com/rss"]},
    "trtspor": {"home": "https://www.trtspor.com.tr",
                "urls": ["https://www.trtspor.com.tr/rss/futbol.rss", "https://www.trthaber.com/spor_articles.rss"]},
    "aa_spor": {"home": "https://www.aa.com.tr/tr/spor",
                "urls": ["https://www.aa.com.tr/tr/rss/default?cat=spor"]},
    "hurriyet_spor": {"home": "https://www.hurriyet.com.tr/sporarena/",
                      "urls": ["https://www.hurriyet.com.tr/rss/spor", "https://www.hurriyet.com.tr/rss/sporarena"]},
    "milliyet_skorer": {"home": "https://www.milliyet.com.tr/skorer/",
                        "urls": ["https://www.milliyet.com.tr/rss/rssnew/sporrss.xml",
                                 "https://www.milliyet.com.tr/rss/rssnew/skorerrss.xml"]},
    "sabah_spor": {"home": "https://www.sabah.com.tr/spor",
                   "urls": ["https://www.sabah.com.tr/rss/spor.xml"]},
    "haberturk_spor": {"home": "https://www.haberturk.com/spor",
                       "urls": ["https://www.haberturk.com/rss/spor.xml"]},
    "sozcu_spor": {"home": "https://www.sozcu.com.tr/spor",
                   "urls": ["https://www.sozcu.com.tr/feeds-rss-category-spor", "https://www.sozcu.com.tr/rss/spor.xml"]},
    "cnnturk_spor": {"home": "https://www.cnnturk.com/spor",
                     "urls": ["https://www.cnnturk.com/feed/rss/spor/news"]},
    "mynet_spor": {"home": "https://www.mynet.com/spor",
                   "urls": ["https://www.mynet.com/spor/rss", "https://spor.mynet.com/rss"]},
    "beinsports": {"home": "https://beinsports.com.tr",
                   "urls": ["https://beinsports.com.tr/rss", "https://beinsports.com.tr/rss/haberler"]},
}

# Feeds that parse but are not wanted, so a new probe does not bring them back.
EXCLUDE: dict[str, str] = {
    "https://www.aa.com.tr/tr/rss/default?cat=guncel": "general news, not sport",
    "https://www.sozcu.com.tr/feeds/rsssubpub": "all sections (650+ items), mostly not sport",
    "https://www.mynet.com/spor/rss": "same items as publisher-spor.rss",
    "https://spor.mynet.com/rss": "same items as publisher-spor.rss",
}


def load_feeds(path: Path = FEEDS_FILE) -> dict[str, list[str]]:
    """name -> working feed URLs, as written by probe-news."""
    if not path.exists():
        raise SystemExit(f"{path.name} not found. Run first: python -m etl probe-news")
    return json.loads(path.read_text(encoding="utf-8"))["feeds"]
