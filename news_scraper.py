"""
news_scraper.py - Direct web scraper for Indian financial news.

Scrapes RSS feeds from 20+ financial news websites + Google News RSS,
extracts article text with trafilatura, scores with FinBERT, and saves
to data/scraped_news/.

NO API key needed, NO rate limits, NO cost. Full GNews replacement.

Run:
    python news_scraper.py              # one-shot fetch + score
    python news_scraper.py --loop       # continuous polling every 5 min
    python news_scraper.py --integrate  # one-shot + update news_sentiment.json
"""

import os
import csv
import json
import re
import time
import hashlib
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from html import unescape

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = timezone(timedelta(hours=5, minutes=30))

import requests
import feedparser

try:
    import trafilatura
    TRAF_OK = True
except ImportError:
    TRAF_OK = False
    print("[scraper] trafilatura not installed. Run: pip install trafilatura")

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "data" / "scraped_news"
SEEN_FILE = OUTPUT_DIR / "seen_urls.json"
POLL_INTERVAL = 300
FETCH_DELAY = 0.8
REQUEST_TIMEOUT = (5, 15)
MAX_RETRIES = 2
MAX_ARTICLES_PER_FEED = 25

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "scraper.log"),
    ],
)
log = logging.getLogger("news_scraper")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── RSS Feeds ──────────────────────────────────────────────────────────────────
# "financial": True means the source is finance-dedicated, skip relevance filter
RSS_FEEDS = {
    # ━━━ Tier 1: Major financial news ━━━
    "The Economic Times": {
        "financial": True,
        "feeds": [
            "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
            "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        ],
    },
    "Livemint": {
        "financial": True,
        "feeds": [
            "https://www.livemint.com/rss/markets",
            "https://www.livemint.com/rss/money",
            "https://www.livemint.com/rss/economy",
            "https://www.livemint.com/rss/industry",
        ],
    },
    "Moneycontrol": {
        "financial": True,
        "feeds": [
            "https://www.moneycontrol.com/rss/latestnews.xml",
            "https://www.moneycontrol.com/rss/marketreports.xml",
            "https://www.moneycontrol.com/rss/MCtopnews.xml",
            "https://www.moneycontrol.com/rss/buzzingstocks.xml",
        ],
    },
    "NDTV Profit": {
        "financial": True,
        "feeds": [
            "https://feeds.feedburner.com/ndtvprofit-latest",
        ],
    },

    # ━━━ Tier 2: Business newspapers ━━━
    "The Hindu Business Line": {
        "financial": True,
        "feeds": [
            "https://www.thehindubusinessline.com/feeder/default.rss",
            "https://www.thehindubusinessline.com/markets/feeder/default.rss",
            "https://www.thehindubusinessline.com/economy/feeder/default.rss",
        ],
    },
    "Business Today": {
        "financial": True,
        "feeds": [
            "https://www.businesstoday.in/rss/home",
        ],
    },
    "CNBC TV18": {
        "financial": True,
        "feeds": [
            "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
            "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/economy.xml",
        ],
    },
    "Zee Business": {
        "financial": True,
        "feeds": [
            "https://zeenews.india.com/rss/business.xml",
        ],
    },

    # ━━━ Tier 3: General news with business sections ━━━
    "The Times of India": {
        "financial": False,
        "feeds": [
            "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
            "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        ],
    },
    "The Indian Express": {
        "financial": False,
        "feeds": [
            "https://indianexpress.com/section/business/feed/",
            "https://indianexpress.com/section/business/economy/feed/",
            "https://indianexpress.com/section/business/market/feed/",
        ],
    },
    "News18": {
        "financial": False,
        "feeds": [
            "https://www.news18.com/commonfeeds/v1/eng/rss/business.xml",
        ],
    },
    "ThePrint": {
        "financial": False,
        "feeds": [
            "https://theprint.in/category/economy/feed/",
        ],
    },
}

# Google News RSS — catches Business Standard, NDTV, India Today, Hindustan
# Times, Upstox, and dozens of other sources that block direct RSS
GOOGLE_NEWS_QUERIES = [
    "nifty OR sensex OR BSE OR NSE stock market India",
    "RBI monetary policy interest rate India",
    "Indian economy GDP inflation budget",
    "FII DII mutual fund India market",
    "Nifty 50 prediction analysis today",
]

GOOGLE_NEWS_TOPICS = [
    # CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pKVGlnQVAB = Business (India)
    "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pKVGlnQVAB",
]

# Sources to SKIP from Google News (non-financial noise)
SKIP_SOURCES = {
    "rushlane", "bikewale", "carwale", "carandbike", "zigwheels",
    "91mobiles", "gadgets360", "gizmodo", "techradar",
    "sportskeeda", "cricbuzz", "espncricinfo",
    "bollywood", "filmibeat", "pinkvilla",
    "linkedin", "youtube", "reddit",
}

RELEVANCE_KEYWORDS = {
    # Market terms
    "nifty", "sensex", "bse", "nse", "stock market", "share market",
    "sgx nifty", "bank nifty", "nifty 50", "nifty50",
    # Monetary/macro
    "rbi", "reserve bank", "interest rate", "monetary policy", "repo rate",
    "gdp", "inflation", "cpi", "wpi", "fiscal", "budget", "economy",
    "current account", "trade deficit", "trade surplus",
    # Institutional
    "fii", "dii", "mutual fund", "etf", "sebi",
    # Blue chips
    "reliance", "hdfc", "infosys", "tcs", "icici", "sbi", "adani",
    "wipro", "hul", "bajaj", "kotak", "axis bank", "maruti", "tata",
    "bharti airtel", "larsen", "asian paints", "ultratech",
    # Commodities/forex
    "crude oil", "gold price", "silver price", "rupee", "dollar", "forex",
    "commodity", "brent crude",
    # Market events
    "ipo", "earnings", "quarterly results", "profit", "revenue",
    "dividend", "buyback", "merger", "acquisition",
    # Market direction
    "bull", "bear", "rally", "crash", "selloff", "sell-off", "correction",
    "breakout", "support", "resistance",
    # Derivatives
    "vix", "volatility", "options", "futures", "derivative",
    "open interest", "put call ratio",
    # Trade
    "trade", "export", "import", "tariff", "duty",
    # Global
    "fed", "federal reserve", "rate hike", "rate cut",
    "wall street", "dow jones", "s&p 500", "nasdaq",
    # General financial
    "market", "investor", "equity", "bond", "yield",
    "banking", "finance", "insurance", "pharma", "auto", "it sector",
    "india", "indian",
    # Specific sentiment words
    "upgrade", "downgrade", "outperform", "underperform",
    "buy", "sell", "hold", "target price", "accumulate",
    "overweight", "underweight",
}


# ── Dedup ──────────────────────────────────────────────────────────────────────
def _load_seen() -> set:
    if SEEN_FILE.exists():
        try:
            return set(json.loads(SEEN_FILE.read_text()))
        except Exception:
            log.warning("seen_urls.json corrupt; starting fresh")
    return set()


def _save_seen(seen: set):
    tmp = str(SEEN_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(list(seen)[-8000:], f)
    os.replace(tmp, str(SEEN_FILE))


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _title_key(title: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', title.lower().strip())[:80]


# ── HTTP ───────────────────────────────────────────────────────────────────────
def _fetch_feed(url: str) -> feedparser.FeedParserDict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return None
            return feedparser.parse(r.text)
        except requests.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None


def _fetch_html(url: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                log.debug(f"fetch failed: {url} ({e})")
    return None


def _extract_body(url: str) -> tuple[str, str | None]:
    if not TRAF_OK:
        return "", None
    html = _fetch_html(url)
    if not html:
        return "", None
    body = trafilatura.extract(
        html, include_comments=False, include_tables=False,
        favor_precision=True,
    ) or ""
    pub_date = None
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and meta.date:
            pub_date = meta.date
    except Exception:
        pass
    return body.strip(), pub_date


def _parse_rss_time(entry) -> datetime | None:
    parsed = (getattr(entry, "published_parsed", None)
              or getattr(entry, "updated_parsed", None))
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc).astimezone(IST)
        except Exception:
            pass
    return None


def _is_relevant(title: str, summary: str = "") -> bool:
    text = (title + " " + summary).lower()
    return any(kw in text for kw in RELEVANCE_KEYWORDS)


def _clean_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = unescape(text)
    return re.sub(r'\s+', ' ', text).strip()


# ── Core scraping: Direct RSS ─────────────────────────────────────────────────
def scrape_source(source_name: str, config: dict, seen: set,
                  fetch_body: bool = False, max_age_hours: int = 72) -> list[dict]:
    articles = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    is_financial = config.get("financial", False)

    for feed_url in config["feeds"]:
        feed = _fetch_feed(feed_url)
        if not feed or (feed.bozo and not feed.entries):
            continue

        for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
            url = (entry.get("link") or "").strip()
            if not url or _url_hash(url) in seen:
                continue

            title = _clean_html(entry.get("title") or "")
            summary = _clean_html(entry.get("summary") or entry.get("description") or "")

            if not title:
                continue

            pub_dt = _parse_rss_time(entry)
            if pub_dt and pub_dt.astimezone(timezone.utc) < cutoff:
                continue

            if not is_financial and not _is_relevant(title, summary):
                continue

            body = ""
            page_date = None
            if fetch_body:
                body, page_date = _extract_body(url)
                time.sleep(FETCH_DELAY)

            pub_str = pub_dt.isoformat() if pub_dt else datetime.now(IST).isoformat()
            pub_utc = (pub_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                       if pub_dt else "")

            articles.append({
                "title": title,
                "description": summary[:500] if summary else "",
                "url": url,
                "source": {"name": source_name},
                "publishedAt": pub_utc,
                "publishedIST": pub_str,
                "body": body,
                "_query": f"rss:{source_name}",
                "_scraped": True,
            })
            seen.add(_url_hash(url))

    return articles


# ── Core scraping: Google News RSS ─────────────────────────────────────────────
def scrape_google_news(seen: set, max_age_hours: int = 72) -> list[dict]:
    articles = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    feed_urls = []
    for q in GOOGLE_NEWS_QUERIES:
        encoded = requests.utils.quote(q)
        feed_urls.append(
            f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        )
    for topic in GOOGLE_NEWS_TOPICS:
        feed_urls.append(
            f"https://news.google.com/rss/topics/{topic}?hl=en-IN&gl=IN&ceid=IN:en"
        )

    for feed_url in feed_urls:
        feed = _fetch_feed(feed_url)
        if not feed:
            continue

        for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
            url = (entry.get("link") or "").strip()
            if not url or _url_hash(url) in seen:
                continue

            title = _clean_html(entry.get("title") or "")
            if not title:
                continue

            source_name = (entry.get("source") or {}).get("title", "")
            if not source_name:
                source_name = entry.get("source", {}).get("value", "Google News")

            if source_name.lower() in SKIP_SOURCES:
                continue

            pub_dt = _parse_rss_time(entry)
            if pub_dt and pub_dt.astimezone(timezone.utc) < cutoff:
                continue

            pub_str = pub_dt.isoformat() if pub_dt else datetime.now(IST).isoformat()
            pub_utc = (pub_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                       if pub_dt else "")

            articles.append({
                "title": title,
                "description": "",
                "url": url,
                "source": {"name": source_name},
                "publishedAt": pub_utc,
                "publishedIST": pub_str,
                "body": "",
                "_query": "google_news",
                "_scraped": True,
            })
            seen.add(_url_hash(url))

    return articles


# ── Orchestrator ───────────────────────────────────────────────────────────────
def scrape_all(fetch_body: bool = False, max_age_hours: int = 72,
               sources: list[str] | None = None,
               fresh: bool = False) -> list[dict]:
    seen = set() if fresh else _load_seen()
    all_articles = []
    targets = sources or list(RSS_FEEDS.keys())

    # Phase 1: Direct RSS feeds
    log.info(f"Phase 1: scraping {len(targets)} direct RSS sources...")
    for name in targets:
        config = RSS_FEEDS.get(name)
        if not config:
            continue
        arts = scrape_source(name, config, seen, fetch_body=fetch_body,
                             max_age_hours=max_age_hours)
        log.info(f"  [{name:30s}] {len(arts)} articles")
        all_articles.extend(arts)

    direct_count = len(all_articles)

    # Phase 2: Google News RSS (catches Business Standard, NDTV, India Today etc.)
    log.info("Phase 2: scraping Google News RSS...")
    gn_arts = scrape_google_news(seen, max_age_hours=max_age_hours)
    log.info(f"  [Google News RSS            ] {len(gn_arts)} articles")
    all_articles.extend(gn_arts)

    _save_seen(seen)

    # Dedup by title
    unique = []
    title_set = set()
    for a in all_articles:
        key = _title_key(a["title"])
        if key and key not in title_set:
            title_set.add(key)
            unique.append(a)

    log.info(f"total: {direct_count} direct + {len(gn_arts)} google = "
             f"{len(all_articles)} raw -> {len(unique)} unique")
    return unique


# ── Save ───────────────────────────────────────────────────────────────────────
def save_scraped(articles: list[dict]):
    if not articles:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    json_path = OUTPUT_DIR / f"news_{today}.json"
    csv_path = OUTPUT_DIR / f"news_{today}.csv"

    existing = []
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text())
        except Exception:
            existing = []

    existing_urls = {a.get("url") for a in existing}
    new_arts = [a for a in articles if a.get("url") not in existing_urls]
    all_arts = existing + new_arts

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_arts, f, indent=2, ensure_ascii=False, default=str)

    fields = ["publishedIST", "source", "title", "url", "sentiment", "description"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for a in all_arts:
            row = {**a, "source": a.get("source", {}).get("name", "")}
            w.writerow(row)

    log.info(f"saved {len(new_arts)} new to {json_path.name} (total: {len(all_arts)})")


# ── Integration with news_sentiment.py ─────────────────────────────────────────
def score_and_integrate(articles: list[dict]) -> dict:
    import sys
    sys.path.insert(0, str(HERE))
    from news_sentiment import score_articles, aggregate

    scored = score_articles(articles)
    result = aggregate(scored)
    result["source"] = "web_scraper"
    result["from_cache"] = False
    result["fetched_at"] = datetime.now().isoformat()

    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "news_sentiment.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"sentiment: {result['label']} ({result['score']:+.3f}) "
             f"from {result['n_articles']} articles")

    from news_sentiment import _archive_daily
    _archive_daily(result, data_dir)

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Scrape Indian financial news")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--integrate", action="store_true",
                        help="Score with FinBERT and update news_sentiment.json")
    parser.add_argument("--body", action="store_true",
                        help="Fetch full article body (slower)")
    parser.add_argument("--hours", type=int, default=72,
                        help="Max article age in hours (default: 72)")
    parser.add_argument("--sources", nargs="*",
                        help="Specific sources to scrape (default: all)")
    args = parser.parse_args()

    n_src = len(args.sources or RSS_FEEDS)
    print(f"\n{'='*65}")
    print(f"  News Scraper - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Direct RSS: {n_src} | Google News: {len(GOOGLE_NEWS_QUERIES)} queries "
          f"| Body: {args.body} | Max age: {args.hours}h")
    print(f"{'='*65}\n")

    if args.loop:
        while True:
            try:
                articles = scrape_all(fetch_body=args.body,
                                      max_age_hours=args.hours,
                                      sources=args.sources)
                save_scraped(articles)
                if args.integrate and articles:
                    score_and_integrate(articles)
                log.info(f"sleeping {POLL_INTERVAL}s...")
                time.sleep(POLL_INTERVAL)
            except KeyboardInterrupt:
                log.info("stopped")
                break
            except Exception as e:
                log.error(f"cycle error: {e}")
                time.sleep(60)
    else:
        articles = scrape_all(fetch_body=args.body,
                              max_age_hours=args.hours,
                              sources=args.sources)
        save_scraped(articles)

        if args.integrate and articles:
            result = score_and_integrate(articles)
            print(f"\n{'='*65}")
            print(f"  SENTIMENT: {result['label'].upper()} ({result['score']:+.3f})")
            print(f"  Articles: {result['n_articles']} | "
                  f"+ve: {result['n_positive']} | -ve: {result['n_negative']} | "
                  f"neutral: {result['n_neutral']}")
            print(f"{'='*65}\n")
        elif articles:
            from collections import Counter
            counts = Counter(a["source"]["name"] for a in articles)
            print(f"\n  {'Source':<35} {'Count':>5}")
            print(f"  {'-'*40}")
            for src, cnt in counts.most_common():
                print(f"  {src:<35} {cnt:>5}")
            print(f"  {'-'*40}")
            print(f"  {'TOTAL':<35} {len(articles):>5}")
        else:
            print("  No new articles found.")


if __name__ == "__main__":
    main()
