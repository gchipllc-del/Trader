"""
Curated RSS/Atom feed list for market-moving news ingestion.

Feed list adapted from koala73/worldmonitor
(https://github.com/koala73/worldmonitor), © 2024-2026 Elie Habib,
licensed under GNU AGPL-3.0.

Feed URLs are factual data (not copyrightable), but attribution retained
for provenance. See /NOTICE at repo root for full license text reference.

⚠️  AGPL NOTE: traderbot runs in paper mode. Before any commercial / hosted
    deployment, either (a) replace this feed list with independently curated
    sources, or (b) license worldmonitor commercially. Importing the
    worldmonitor code itself would trigger AGPL §13 (network copyleft) on the
    Flask dashboard — we only port the URL list, not any parsing code.

Categories:
  - markets:     Equities, bonds, commodities, crypto, forex-specific
  - macro:       Central banks, rate decisions, CPI/GDP/PMI, Treasury
  - geopolitics: Foreign policy, defense, sanctions, trade wars
  - tech:        Market-moving tech + AI research
  - general:     Broad news that occasionally moves markets
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Category = Literal["markets", "macro", "geopolitics", "tech", "general"]


@dataclass(frozen=True)
class Feed:
    name: str
    url: str
    category: Category


FEEDS: list[Feed] = [
    # --- Wire services & general global news ---
    Feed("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml", "general"),
    Feed("Guardian World", "https://www.theguardian.com/world/rss", "general"),
    Feed("NPR News", "https://feeds.npr.org/1001/rss.xml", "general"),
    Feed("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", "geopolitics"),
    Feed("AP News (via Google News)", "https://news.google.com/rss/search?q=site:apnews.com&hl=en-US&gl=US&ceid=US:en", "general"),
    Feed("Reuters World (via Google News)", "https://news.google.com/rss/search?q=site:reuters.com+world&hl=en-US&gl=US&ceid=US:en", "general"),
    Feed("Reuters Markets (via Google News)", "https://news.google.com/rss/search?q=site:reuters.com+markets+stocks+when:1d&hl=en-US&gl=US&ceid=US:en", "markets"),
    Feed("Reuters Business (via Google News)", "https://news.google.com/rss/search?q=site:reuters.com+business+markets&hl=en-US&gl=US&ceid=US:en", "markets"),
    Feed("Bloomberg Markets (via Google News)", "https://news.google.com/rss/search?q=site:bloomberg.com+markets+when:1d&hl=en-US&gl=US&ceid=US:en", "markets"),

    # --- US financial press ---
    Feed("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html", "markets"),
    Feed("CNBC Tech", "https://www.cnbc.com/id/19854910/device/rss/rss.html", "tech"),
    Feed("Yahoo Finance", "https://finance.yahoo.com/rss/topstories", "markets"),
    Feed("Yahoo Finance Index", "https://finance.yahoo.com/news/rssindex", "markets"),
    Feed("Seeking Alpha Market Currents", "https://seekingalpha.com/market_currents.xml", "markets"),
    Feed("MarketWatch (via Google News)", "https://news.google.com/rss/search?q=site:marketwatch.com+markets+when:1d&hl=en-US&gl=US&ceid=US:en", "markets"),
    Feed("Financial Times Home", "https://www.ft.com/rss/home", "markets"),
    Feed("Wall Street Journal US News", "https://feeds.content.dowjones.io/public/rss/RSSUSnews", "markets"),
    Feed("Investing.com (via Google News)", "https://news.google.com/rss/search?q=site:investing.com+markets+when:1d&hl=en-US&gl=US&ceid=US:en", "markets"),

    # --- Central banks & macro data ---
    Feed("Federal Reserve Press", "https://www.federalreserve.gov/feeds/press_all.xml", "macro"),
    Feed("SEC Press Releases", "https://www.sec.gov/news/pressreleases.rss", "macro"),
    Feed("ECB Watch (via Google News)", "https://news.google.com/rss/search?q=(%22European+Central+Bank%22+OR+ECB+OR+Lagarde)+monetary+policy+when:3d&hl=en-US&gl=US&ceid=US:en", "macro"),
    Feed("BoJ Watch (via Google News)", "https://news.google.com/rss/search?q=(%22Bank+of+Japan%22+OR+BoJ)+monetary+policy+when:3d&hl=en-US&gl=US&ceid=US:en", "macro"),
    Feed("BoE Watch (via Google News)", "https://news.google.com/rss/search?q=(%22Bank+of+England%22+OR+BoE)+monetary+policy+when:3d&hl=en-US&gl=US&ceid=US:en", "macro"),
    Feed("Global Central Banks", "https://news.google.com/rss/search?q=(%22rate+hike%22+OR+%22rate+cut%22+OR+%22interest+rate+decision%22)+central+bank+when:3d&hl=en-US&gl=US&ceid=US:en", "macro"),
    Feed("Economic Data (CPI/GDP/PMI)", "https://news.google.com/rss/search?q=(CPI+OR+inflation+OR+GDP+OR+%22jobs+report%22+OR+%22nonfarm+payrolls%22+OR+PMI)+when:2d&hl=en-US&gl=US&ceid=US:en", "macro"),
    Feed("Treasury Watch", "https://news.google.com/rss/search?q=(%22US+Treasury%22+OR+%22Treasury+auction%22+OR+%2210-year+yield%22+OR+%222-year+yield%22)+when:2d&hl=en-US&gl=US&ceid=US:en", "macro"),
    Feed("Bond Market", "https://news.google.com/rss/search?q=(%22bond+market%22+OR+%22treasury+yields%22+OR+%22bond+yields%22+OR+%22fixed+income%22)+when:2d&hl=en-US&gl=US&ceid=US:en", "macro"),
    Feed("Forex / Dollar Index", "https://news.google.com/rss/search?q=(%22dollar+index%22+OR+DXY+OR+%22US+dollar%22+OR+%22euro+dollar%22)+when:2d&hl=en-US&gl=US&ceid=US:en", "macro"),
    Feed("Trade & Tariffs", "https://news.google.com/rss/search?q=(tariff+OR+%22trade+war%22+OR+%22trade+deficit%22+OR+sanctions)+when:2d&hl=en-US&gl=US&ceid=US:en", "macro"),

    # --- Commodities & energy ---
    Feed("OilPrice.com", "https://oilprice.com/rss/main", "markets"),
    Feed("Rigzone", "https://www.rigzone.com/news/rss/rigzone_latest.aspx", "markets"),
    Feed("EIA Press Room", "https://www.eia.gov/rss/press_room.xml", "macro"),
    Feed("Kitco News", "https://www.kitco.com/rss/KitcoNews.xml", "markets"),
    Feed("Kitco Gold", "https://www.kitco.com/rss/KitcoGold.xml", "markets"),
    Feed("Mining.com", "https://www.mining.com/feed/", "markets"),

    # --- Crypto ---
    Feed("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", "markets"),
    Feed("Cointelegraph", "https://cointelegraph.com/rss", "markets"),
    Feed("Decrypt", "https://decrypt.co/feed", "markets"),
    Feed("Blockworks", "https://blockworks.co/feed", "markets"),
    Feed("The Defiant", "https://thedefiant.io/feed", "markets"),
    Feed("Bitcoin Magazine", "https://bitcoinmagazine.com/feed", "markets"),
    Feed("CryptoSlate", "https://cryptoslate.com/feed/", "markets"),

    # --- Geopolitics, foreign policy, defense ---
    Feed("Foreign Policy", "https://foreignpolicy.com/feed/", "geopolitics"),
    Feed("Foreign Affairs", "https://www.foreignaffairs.com/rss.xml", "geopolitics"),
    Feed("The Diplomat", "https://thediplomat.com/feed/", "geopolitics"),
    Feed("Defense One", "https://www.defenseone.com/rss/all/", "geopolitics"),
    Feed("War on the Rocks", "https://warontherocks.com/feed", "geopolitics"),
    Feed("Crisis Group (CrisisWatch)", "https://www.crisisgroup.org/rss", "geopolitics"),
    Feed("RAND Articles", "https://www.rand.org/pubs/articles.xml", "geopolitics"),
    Feed("Atlantic Council", "https://www.atlanticcouncil.org/feed/", "geopolitics"),
    Feed("Stimson Center", "https://www.stimson.org/feed/", "geopolitics"),
    Feed("Politico Politics", "https://rss.politico.com/politics-news.xml", "geopolitics"),
    Feed("UK MOD", "https://www.gov.uk/government/organisations/ministry-of-defence.atom", "geopolitics"),
    Feed("IAEA Top News", "https://www.iaea.org/feeds/topnews", "geopolitics"),
    Feed("Bellingcat (via Google News)", "https://news.google.com/rss/search?q=site:bellingcat.com+when:30d&hl=en-US&gl=US&ceid=US:en", "geopolitics"),

    # --- Tech (market-moving tech news) ---
    Feed("Hacker News Front Page", "https://hnrss.org/frontpage", "tech"),
    Feed("Ars Technica", "https://feeds.arstechnica.com/arstechnica/technology-lab", "tech"),
    Feed("The Verge", "https://www.theverge.com/rss/index.xml", "tech"),
    Feed("TechCrunch", "https://techcrunch.com/feed/", "tech"),
    Feed("MIT Technology Review", "https://www.technologyreview.com/feed/", "tech"),
    Feed("VentureBeat AI", "https://venturebeat.com/category/ai/feed/", "tech"),
    Feed("ArXiv cs.AI", "https://export.arxiv.org/rss/cs.AI", "tech"),
    Feed("Krebs on Security", "https://krebsonsecurity.com/feed/", "tech"),
]


def feeds_by_category(category: Category) -> list[Feed]:
    return [f for f in FEEDS if f.category == category]


def feed_urls(category: Category | None = None) -> list[str]:
    src = feeds_by_category(category) if category else FEEDS
    return [f.url for f in src]


# Category counts (for quick sanity checks in tests / dashboard):
#   markets:     21
#   macro:       10
#   geopolitics: 14
#   tech:         8
#   general:      6   (Al Jazeera tagged geopolitics, not general)
#   Total:       59
