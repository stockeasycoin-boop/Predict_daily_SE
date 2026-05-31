"""
news_scraper.py — Reliable news scraper for The Hindu and Times of India.

Design:
  - RSS feeds for DISCOVERY (headline, url, publish timestamp) — stable, no bot detection.
  - trafilatura for BODY extraction — selector-free, survives site redesigns.
  - Fallback chain so a record is never silently empty.
  - Dedup across restarts via a seen-urls file.
  - Polite fetching: timeouts, retries with backoff, delays.
  - Parallel body extraction (ThreadPoolExecutor) to keep web request snappy.

Module entry points (used by news_sentiment.py):
  fetch_recent(days)            -> list[dict]  : RSS poll + last-N-days cache read
  load_articles_from_cache(days)               : cache-only read (no network)

CLI entry point (for backfill / daemon):
  python news_scraper.py        : run continuous polling loop
"""

from __future__ import annotations
import os
import csv
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytz
import requests
import feedparser
import trafilatura

# ── Config ────────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")
OUTPUT_DIR = "data/raw/news"
SEEN_FILE = os.path.join(OUTPUT_DIR, "seen_urls.json")
POLL_INTERVAL_SECONDS = 300        # daemon mode only
FETCH_DELAY_SECONDS = 0.2          # polite gap between article fetches (sequential mode)
REQUEST_TIMEOUT = (5, 15)          # (connect, read)
MAX_RETRIES = 3
MAX_PARALLEL_FETCHES = 4           # body extraction concurrency
MAX_ARTICLES_PER_FEED = 25         # cap items pulled per RSS feed per cycle

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("news_scraper")

# RSS feeds = discovery layer. Add/remove sections freely; structure is identical.
RSS_FEEDS = {
    "thehindu": [
        "https://www.thehindu.com/feeder/default.rss",                 # top news
        "https://www.thehindu.com/business/feeder/default.rss",        # business
        "https://www.thehindu.com/news/national/feeder/default.rss",   # national
        "https://www.thehindu.com/news/international/feeder/default.rss",
        "https://www.thehindu.com/sci-tech/feeder/default.rss",
    ],
    "timesofindia": [
        "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",  # top stories
        "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",    # business
        "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",  # india
        "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",# world
    ],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ── Dedup state ──────────────────────────────────────────────────────────────
def load_seen() -> set:
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE) as f:
                return set(json.load(f))
        except Exception:
            log.warning("seen_urls.json unreadable; starting fresh")
    return set()


def save_seen(seen: set) -> None:
    tmp = SEEN_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(list(seen), f)
    os.replace(tmp, SEEN_FILE)  # atomic


def url_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


# ── Timestamp parsing ────────────────────────────────────────────────────────
def parse_rss_datetime(entry) -> datetime | None:
    """RSS publish time -> tz-aware IST datetime. None if absent."""
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed:
        return datetime(*parsed[:6], tzinfo=timezone.utc).astimezone(IST)
    return None


# ── HTTP fetch with retries ──────────────────────────────────────────────────
def fetch_html(url: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning(f"fetch attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    return None


# ── Body extraction (selector-free) ──────────────────────────────────────────
def extract_body_and_date(url: str):
    """Returns (body_text, page_published_iso_or_None). Selector-free via trafilatura."""
    html = fetch_html(url)
    if not html:
        return "", None
    body = trafilatura.extract(
        html, include_comments=False, include_tables=False, favor_precision=True,
    ) or ""
    page_date = None
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and meta.date:
            page_date = meta.date
    except Exception:
        pass
    return body.strip(), page_date


# ── Persistence ──────────────────────────────────────────────────────────────
CSV_FIELDS = ["published_ist", "source", "headline", "url", "body", "page_date", "scraped_at"]


def save_article(article: dict) -> None:
    date_str = article["published_ist"][:10]
    json_path = os.path.join(OUTPUT_DIR, f"news_{date_str}.json")
    records = []
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                records = json.load(f)
        except Exception:
            records = []
    records.append(article)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(OUTPUT_DIR, f"news_{date_str}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: article.get(k, "") for k in CSV_FIELDS})


# ── Discovery: build the list of new URLs to extract ─────────────────────────
def _discover_new_entries(seen: set) -> list[tuple[str, str, dict]]:
    """Return list of (source, feed_url, entry) for entries we haven't seen."""
    new = []
    for source, feeds in RSS_FEEDS.items():
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
            except Exception as e:
                log.warning(f"[{source}] parse failed for {feed_url}: {e}")
                continue
            if feed.bozo:
                log.warning(f"[{source}] malformed/unreachable feed: {feed_url}")
                continue
            for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
                url = (entry.get("link") or "").strip()
                if url and url_key(url) not in seen:
                    new.append((source, feed_url, entry))
    return new


def _process_entry(source: str, entry) -> dict | None:
    """Fetch body for one entry, return article dict ready to save (or None)."""
    url = (entry.get("link") or "").strip()
    if not url:
        return None
    headline = (entry.get("title") or "").strip()
    rss_summary = (entry.get("summary") or "").strip()
    pub_dt = parse_rss_datetime(entry) or datetime.now(IST)

    body, page_date = extract_body_and_date(url)
    if not body:
        body = rss_summary
    if not body:
        return None
    return {
        "published_ist": pub_dt.isoformat(),
        "source":        source,
        "headline":      headline,
        "url":           url,
        "body":          body,
        "page_date":     page_date,
        "scraped_at":    datetime.now(IST).isoformat(),
    }


# ── Single-cycle discover-and-extract (used by Streamlit) ────────────────────
def discover_and_extract(parallel: int = MAX_PARALLEL_FETCHES) -> int:
    """Poll all RSS feeds once, extract bodies in parallel, persist new articles.
    Returns number of NEW articles added. Safe to call from a request context."""
    seen = load_seen()
    new_entries = _discover_new_entries(seen)
    if not new_entries:
        return 0

    log.info(f"discovered {len(new_entries)} new entries; extracting bodies "
             f"(parallel={parallel})…")
    count = 0
    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {
                ex.submit(_process_entry, src, ent): (src, ent)
                for src, _feed, ent in new_entries
            }
            for fut in as_completed(futures):
                try:
                    art = fut.result()
                except Exception as e:
                    log.warning(f"entry processing failed: {e}")
                    continue
                if not art:
                    continue
                save_article(art)
                seen.add(url_key(art["url"]))
                count += 1
    else:
        for src, _feed, ent in new_entries:
            art = _process_entry(src, ent)
            if art:
                save_article(art)
                seen.add(url_key(art["url"]))
                count += 1
                time.sleep(FETCH_DELAY_SECONDS)

    save_seen(seen)
    log.info(f"saved {count} new articles to {OUTPUT_DIR}")
    return count


# ── Cache read ───────────────────────────────────────────────────────────────
def load_articles_from_cache(days: int) -> list[dict]:
    """Read daily JSON files for the last `days` (IST) and return concat list."""
    days = max(1, int(days))
    today_ist = datetime.now(IST).date()
    out = []
    for i in range(days):
        d = (today_ist - timedelta(days=i)).isoformat()
        p = os.path.join(OUTPUT_DIR, f"news_{d}.json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    out.extend(json.load(f))
            except Exception as e:
                log.warning(f"cache read failed for {p}: {e}")
    return out


def fetch_recent(days: int) -> list[dict]:
    """Refresh from RSS + return articles published in the last N days."""
    try:
        discover_and_extract()
    except Exception as e:
        log.warning(f"discover_and_extract failed: {e}")
    return load_articles_from_cache(days)


# ── Daemon loop (CLI / backfill use) ─────────────────────────────────────────
def run():
    """Continuous polling loop. Call via `python news_scraper.py`."""
    log.info(f"starting live scraper | output={OUTPUT_DIR} interval={POLL_INTERVAL_SECONDS}s")
    try:
        while True:
            try:
                added = discover_and_extract()
                log.info(f"cycle done | {added} new | sleeping {POLL_INTERVAL_SECONDS}s")
            except Exception as e:
                log.error(f"cycle error: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        log.info("stopped cleanly")


if __name__ == "__main__":
    run()
