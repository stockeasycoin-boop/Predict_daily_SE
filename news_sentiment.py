"""
news_sentiment.py — News sentiment enrichment for Nifty predictions.

Pipeline:
  1. GNews API     → fetch recent India/Nifty/market news headlines
  2. FinBERT       → financial-domain sentiment scoring (primary)
  3. VADER         → lexicon-based fallback when transformers/torch missing

Returns a sentiment score in [-1, +1] aggregated across articles, plus per-article
breakdown. Designed to ENRICH (not replace) model predictions — it adjusts
confidence based on news/model alignment.

Cached to data/news_sentiment.json (4-hour TTL) to respect GNews free-tier
limits (100 requests/day).
"""

from __future__ import annotations
import os
import json
import time
import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

warnings.filterwarnings("ignore")

# ── Silence noisy transformers / HF deprecation chatter ──────────────────────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _noisy in ("transformers", "transformers.modeling_utils",
               "transformers.configuration_utils", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ── Lazy backend detection ───────────────────────────────────────────────────
_FINBERT = None      # cached pipeline (loaded once)
_VADER   = None      # cached analyzer (loaded once)
_BACKEND = None      # "finbert" | "vader" | "none"


def _load_finbert():
    """Lazy-load FinBERT. Returns the pipeline or None if unavailable."""
    global _FINBERT
    if _FINBERT is not None:
        return _FINBERT
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from transformers import pipeline
        _FINBERT = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        print("[news] FinBERT loaded (financial sentiment, GPU/CPU auto)")
        return _FINBERT
    except Exception as e:
        print(f"[news] FinBERT unavailable ({type(e).__name__}: {e}). Falling back to VADER.")
        return None


def _load_vader():
    """Lazy-load VADER. Returns the analyzer or None if unavailable."""
    global _VADER
    if _VADER is not None:
        return _VADER
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
        print("[news] VADER loaded (lexicon-based sentiment)")
        return _VADER
    except Exception as e:
        print(f"[news] VADER unavailable ({type(e).__name__}: {e}). No sentiment backend.")
        return None


def _detect_backend() -> str:
    """Pick the best available backend, lazy-loading it."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    if _load_finbert() is not None:
        _BACKEND = "finbert"
    elif _load_vader() is not None:
        _BACKEND = "vader"
    else:
        _BACKEND = "none"
    return _BACKEND


# ─────────────────────────────────────────────────────────────────────────────
# NEWS FETCH — scraped from The Hindu + TOI via news_scraper.py
# (RSS discovery + trafilatura body extraction; no API key required)
# ─────────────────────────────────────────────────────────────────────────────


def _gnews_cfg():
    """Read news tuning from settings (with safe fallbacks).
    Returns (max_per_query, lookback_days, query_pause, cache_minutes).
    Name kept for backward compatibility with callers."""
    try:
        from settings import (GNEWS_MAX_PER_QUERY, GNEWS_LOOKBACK_DAYS,
                              GNEWS_QUERY_PAUSE, GNEWS_CACHE_MINUTES)
        return (int(GNEWS_MAX_PER_QUERY), int(GNEWS_LOOKBACK_DAYS),
                float(GNEWS_QUERY_PAUSE), int(GNEWS_CACHE_MINUTES))
    except Exception:
        return (50, 3, 0.0, 5)


def _ist_iso_to_utc_z(ist_iso: str) -> str:
    """Convert IST ISO string from scraper -> 'YYYY-MM-DDTHH:MM:SSZ' (UTC)."""
    if not ist_iso:
        return ""
    try:
        dt = datetime.fromisoformat(ist_iso)
        if dt.tzinfo is None:
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ist_iso


GNEWS_URL = "https://gnews.io/api/v4/search"
NIFTY_QUERIES = ["Nifty 50", "Indian stock market", "Sensex", "RBI India", "FII India"]


def _fetch_gnews_articles(api_key: str, days: int, max_per_query: int) -> list[dict]:
    """Fetch fresh articles from GNews. Returns [] on any failure."""
    if not api_key or api_key in ("YOUR_GNEWS_API_KEY", ""):
        return []
    from_dt = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    seen, out = set(), []
    for q in NIFTY_QUERIES:
        try:
            resp = requests.get(GNEWS_URL, timeout=10, params={
                "q": q, "lang": "en", "country": "in",
                "max": max(1, min(max_per_query, 100)),
                "from": from_dt, "sortby": "publishedAt", "apikey": api_key,
            })
            if resp.status_code != 200:
                print(f"[gnews] '{q}' → HTTP {resp.status_code}")
                continue
            for art in resp.json().get("articles", []):
                u = art.get("url")
                if u and u not in seen:
                    seen.add(u)
                    out.append(art)
        except Exception as e:
            print(f"[gnews] '{q}' failed: {e}")
    return out


def fetch_all_news(api_key: str = None, days: int = None) -> list[dict]:
    """
    HYBRID news fetch: scraper (Hindu + TOI + Moneycontrol + LiveMint) UNIONED
    with GNews. Articles deduped by URL. Whichever source is unavailable is
    silently skipped — we use whatever's reachable. Then filter to market-
    relevant only before returning.

    Returns GNews-shape dicts: {title, description, publishedAt, url, source}.
    """
    cfg_max, cfg_days, _, _ = _gnews_cfg()
    if days is None:
        days = cfg_days

    # ── Source 1: RSS scraper (Hindu, TOI, Moneycontrol, LiveMint) ─────────
    scraped: list[dict] = []
    try:
        from news_scraper import fetch_recent, is_market_relevant
        raw = fetch_recent(days)
        for a in raw:
            tag = a.get("market_relevant")
            if tag is None:
                tag = is_market_relevant(a.get("headline", ""), a.get("body", ""))
            if not tag:
                continue
            scraped.append({
                "title":       a.get("headline", ""),
                "description": (a.get("body") or "")[:400],
                "publishedAt": _ist_iso_to_utc_z(a.get("published_ist", "")),
                "url":         a.get("url", ""),
                "source":      {"name": a.get("source", "rss")},
            })
        print(f"[news] scraper: {len(scraped)} market-relevant articles")
    except Exception as e:
        print(f"[news] scraper unavailable: {e}")

    # ── Source 2: GNews API ───────────────────────────────────────────────
    try:
        from news_scraper import is_market_relevant as _rel
    except Exception:
        _rel = lambda *_: True
    gnews_raw = _fetch_gnews_articles(api_key or "", days, cfg_max)
    gnews_filtered = [
        a for a in gnews_raw
        if _rel(a.get("title", ""), a.get("description", ""))
    ]
    print(f"[news] gnews: {len(gnews_filtered)} (of {len(gnews_raw)}) market-relevant")

    # ── Merge + dedupe by URL ─────────────────────────────────────────────
    by_url = {a["url"]: a for a in scraped if a.get("url")}
    for a in gnews_filtered:
        u = a.get("url")
        if u and u not in by_url:
            by_url[u] = a
    combined = list(by_url.values())
    print(f"[news] hybrid total: {len(combined)} unique articles "
          f"(scraper={len(scraped)} + gnews={len(gnews_filtered)} − overlap)")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _score_finbert(texts: list[str]) -> list[float]:
    """Score texts with FinBERT. Returns list of compound scores in [-1, +1]."""
    pipe = _load_finbert()
    if pipe is None:
        return [0.0] * len(texts)
    scores = []
    try:
        # FinBERT returns [{label: 'positive'|'negative'|'neutral', score: 0..1}, ...]
        results = pipe(texts)
        for r in results:
            label = r["label"].lower()
            conf  = float(r["score"])
            if   label == "positive": scores.append(+conf)
            elif label == "negative": scores.append(-conf)
            else:                     scores.append(0.0)        # neutral
    except Exception as e:
        print(f"[news] FinBERT scoring failed: {e}. Returning zeros.")
        scores = [0.0] * len(texts)
    return scores


def _score_vader(texts: list[str]) -> list[float]:
    """Score texts with VADER. Returns list of compound scores in [-1, +1]."""
    analyzer = _load_vader()
    if analyzer is None:
        return [0.0] * len(texts)
    return [float(analyzer.polarity_scores(t)["compound"]) for t in texts]


def score_articles(articles: list[dict]) -> list[dict]:
    """
    Score a list of GNews articles, returning enriched dicts with 'sentiment' field.
    Uses FinBERT if available, else VADER. Combines title + description for context.
    """
    if not articles:
        return []

    backend = _detect_backend()
    if backend == "none":
        # No backend at all — return neutral
        for a in articles:
            a["sentiment"] = 0.0
            a["backend"]   = "none"
        return articles

    texts = [
        (a.get("title", "") + ". " + (a.get("description") or "")).strip()
        for a in articles
    ]
    scores = _score_finbert(texts) if backend == "finbert" else _score_vader(texts)

    for a, s in zip(articles, scores):
        a["sentiment"] = round(float(s), 4)
        a["backend"]   = backend
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE + CACHE
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(scored: list[dict]) -> dict:
    """
    Aggregate per-article scores into a single market sentiment snapshot.
    Recent articles weighted slightly higher (linear decay over `days` window).
    """
    if not scored:
        return {
            "score":          0.0,
            "label":          "neutral",
            "n_articles":     0,
            "n_positive":     0,
            "n_negative":     0,
            "n_neutral":      0,
            "pct_positive":   0.0,
            "pct_negative":   0.0,
            "backend":        _detect_backend(),
            "top_headlines":  [],
        }

    # Recency-weighted mean
    now = datetime.utcnow()
    weighted_sum, weight_total = 0.0, 0.0
    n_pos = n_neg = n_neu = 0
    for a in scored:
        s = a.get("sentiment", 0.0)
        # Parse publishedAt for recency weight
        try:
            published = datetime.strptime(a["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
            hours_old = max(0, (now - published).total_seconds() / 3600)
            w = max(0.3, 1.0 - hours_old / 72)   # 3-day half-decay, floor 0.3
        except Exception:
            w = 0.7
        weighted_sum += s * w
        weight_total += w
        if   s >  0.15: n_pos += 1
        elif s < -0.15: n_neg += 1
        else:           n_neu += 1

    score = weighted_sum / weight_total if weight_total > 0 else 0.0
    n     = len(scored)

    if   score >  0.20: label = "bullish"
    elif score >  0.05: label = "slightly bullish"
    elif score < -0.20: label = "bearish"
    elif score < -0.05: label = "slightly bearish"
    else:               label = "neutral"

    def _to_ist(utc_str: str) -> str:
        """Convert GNews UTC 'YYYY-MM-DDTHH:MM:SSZ' to IST 'YYYY-MM-DD HH:MM IST'."""
        if not utc_str:
            return ""
        try:
            dt = datetime.strptime(utc_str[:19], "%Y-%m-%dT%H:%M:%S")
            ist = dt + timedelta(hours=5, minutes=30)
            return ist.strftime("%Y-%m-%d %H:%M IST")
        except Exception:
            return utc_str

    def _fmt(a):
        return {
            "title":         a.get("title", "")[:140],
            "sentiment":     a.get("sentiment", 0.0),
            "source":        (a.get("source") or {}).get("name", "unknown"),
            "url":           a.get("url", ""),
            "publishedAt":   a.get("publishedAt", ""),          # raw UTC (for sorting)
            "publishedIST":  _to_ist(a.get("publishedAt", "")), # display string
        }

    # Top 5 strongest-signal headlines (by |sentiment|)
    top = sorted(scored, key=lambda a: abs(a.get("sentiment", 0.0)), reverse=True)[:5]
    top_headlines = [_fmt(t) for t in top]

    # Last 5 fetched headlines (most recent by publish time)
    latest = sorted(scored, key=lambda a: a.get("publishedAt", ""), reverse=True)[:5]
    latest_headlines = [_fmt(t) for t in latest]

    return {
        "score":          round(score, 4),
        "label":          label,
        "n_articles":     n,
        "n_positive":     n_pos,
        "n_negative":     n_neg,
        "n_neutral":      n_neu,
        "pct_positive":   round(n_pos / n * 100, 1) if n else 0.0,
        "pct_negative":   round(n_neg / n * 100, 1) if n else 0.0,
        "backend":        scored[0].get("backend", _detect_backend()),
        "top_headlines":  top_headlines,
        "latest_headlines": latest_headlines,
    }


def get_market_sentiment(
    api_key: str,
    days: int = None,
    force_refresh: bool = False,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Main entry point. Fetches news, scores it, aggregates, caches.

    Cache lasts 4 hours — news doesn't change second-to-second, and this
    keeps you under the 100/day GNews free-tier limit.

    Returns the aggregate dict (see `aggregate()`).
    """
    if cache_dir is None:
        try:
            from settings import DATA_DIR
            cache_dir = DATA_DIR
        except Exception:
            cache_dir = Path("data")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "news_sentiment.json"

    # Cache TTL in minutes (small on premium plans = near-realtime).
    _, _, _, cache_minutes = _gnews_cfg()

    # Serve from cache only if still within the (short) TTL
    if cache_file.exists() and not force_refresh and cache_minutes > 0:
        age_min = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 60
        if age_min < cache_minutes:
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                cached["from_cache"] = True
                cached["cache_age_hours"]   = round(age_min / 60, 3)
                cached["cache_age_minutes"] = round(age_min, 1)
                return cached
            except Exception:
                pass   # fall through to refresh

    # Fresh fetch
    if not api_key or api_key in ("YOUR_GNEWS_API_KEY", ""):
        return {
            "score":  0.0,
            "label":  "neutral (no API key)",
            "n_articles": 0,
            "backend": _detect_backend(),
            "error":  "GNews API key not configured. Add it in Settings tab.",
            "top_headlines": [],
        }

    articles = fetch_all_news(api_key, days=days)
    scored   = score_articles(articles)
    result   = aggregate(scored)
    result["fetched_at"] = datetime.utcnow().isoformat()
    result["from_cache"] = False

    try:
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"[news] Cache write failed: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

def adjust_confidence(direction: int, confidence: float, sentiment: dict,
                      max_boost: float = 0.08, max_penalty: float = 0.15) -> tuple[float, str]:
    """
    Adjust the model's confidence based on news/model alignment.

    Rules
    -----
    - News strongly AGREES with model direction → small boost (+max_boost)
    - News strongly DISAGREES with model direction → larger penalty (-max_penalty)
    - News neutral or noisy → no change
    - Asymmetric penalty (disagreement matters more than agreement) — markets
      get hit harder by adverse news than they rally on positive news.

    Parameters
    ----------
    direction   : Model's predicted direction (1 = bullish, 0 = bearish)
    confidence  : Model's confidence (0..1)
    sentiment   : Aggregate dict from get_market_sentiment()
    max_boost   : Max upward adjustment when news strongly agrees
    max_penalty : Max downward adjustment when news strongly disagrees

    Returns
    -------
    (adjusted_confidence, reason_string)
    """
    score = sentiment.get("score", 0.0)
    n     = sentiment.get("n_articles", 0)

    if n < 3 or abs(score) < 0.05:
        return confidence, "News neutral / too few articles — no adjustment"

    # Model says bullish (1), news positive → agree; news negative → disagree
    # Model says bearish (0), news negative → agree; news positive → disagree
    model_bullish = direction == 1
    news_bullish  = score > 0

    aligned = (model_bullish == news_bullish)
    magnitude = min(abs(score), 1.0)              # strength of news signal

    if aligned:
        delta = +max_boost * magnitude
        reason = f"News {sentiment['label']} agrees with model ({score:+.2f}) → +{delta:.3f} boost"
    else:
        delta = -max_penalty * magnitude
        reason = f"News {sentiment['label']} contradicts model ({score:+.2f}) → {delta:+.3f} penalty"

    adjusted = max(0.0, min(1.0, confidence + delta))
    return round(adjusted, 4), reason
