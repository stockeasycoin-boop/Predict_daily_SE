"""
news_sentiment.py — News sentiment enrichment for Nifty predictions.

Pipeline:
  1. GNews API     → fetch India/Nifty/market/macro/global news (1000 req/day plan)
  2. FinBERT       → financial-domain sentiment scoring (primary)
  3. VADER         → lexicon-based fallback when transformers/torch missing

Returns a sentiment score in [-1, +1] aggregated across articles, plus per-article
breakdown. Designed to ENRICH (not replace) model predictions — it adjusts
confidence based on news/model alignment.

Cached to data/news_sentiment.json (short TTL for premium plan).
"""

from __future__ import annotations
import os
import json
import time
import logging
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import requests

warnings.filterwarnings("ignore")

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _noisy in ("transformers", "transformers.modeling_utils",
               "transformers.configuration_utils", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_FINBERT = None
_VADER   = None
_BACKEND = None


def _load_finbert():
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
# GNEWS FETCH — PREMIUM PLAN (1000 req/day)
# ─────────────────────────────────────────────────────────────────────────────

GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"
GNEWS_TOP_URL    = "https://gnews.io/api/v4/top-headlines"

# Comprehensive query set covering all market-moving categories
# Each query costs 1 API call, returns up to 100 articles
NIFTY_QUERIES = [
    # Direct index
    "Nifty 50",
    "Sensex BSE",
    "Indian stock market today",
    # Institutional flows
    "FII India investment",
    "DII mutual fund India",
    # Central bank / monetary policy
    "RBI monetary policy",
    "RBI interest rate India",
    # Government / fiscal
    "India GDP growth",
    "India budget fiscal policy",
    # Global macro that impacts Indian markets
    "US Federal Reserve rate",
    "crude oil price India",
    "dollar rupee exchange",
    # Sector heavyweights (top Nifty weight)
    "Reliance Industries",
    "HDFC Bank results",
    "Infosys TCS IT sector",
    # Volatility / risk
    "India VIX volatility",
    "global recession risk",
    # Geopolitical
    "India trade export",
    "Asia markets today",
]

# Top-headlines categories to also fetch (no query needed, just category)
TOP_HEADLINE_CATEGORIES = ["business", "world"]


def _gnews_cfg():
    try:
        from settings import (GNEWS_MAX_PER_QUERY, GNEWS_LOOKBACK_DAYS,
                              GNEWS_QUERY_PAUSE, GNEWS_CACHE_MINUTES)
        return (int(GNEWS_MAX_PER_QUERY), int(GNEWS_LOOKBACK_DAYS),
                float(GNEWS_QUERY_PAUSE), int(GNEWS_CACHE_MINUTES))
    except Exception:
        return (100, 3, 0.0, 5)


def fetch_gnews(api_key: str, query: str, days: int = None,
                max_results: int = None) -> list[dict]:
    if not api_key or api_key in ("YOUR_GNEWS_API_KEY", ""):
        return []

    cfg_max, cfg_days, _, _ = _gnews_cfg()
    if days is None:
        days = cfg_days
    if max_results is None:
        max_results = cfg_max

    from_dt = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "q":        query,
        "lang":     "en",
        "country":  "in",
        "max":      max(1, min(max_results, 100)),
        "from":     from_dt,
        "sortby":   "publishedAt",
        "apikey":   api_key,
    }
    try:
        resp = requests.get(GNEWS_SEARCH_URL, params=params, timeout=10)
        if resp.status_code in (401, 403, 429):
            print(f"[news] GNews {resp.status_code} for '{query}'")
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])
    except Exception as e:
        print(f"[news] GNews fetch failed for '{query}': {e}")
        return []


def fetch_top_headlines(api_key: str, category: str = "business",
                        max_results: int = 100) -> list[dict]:
    if not api_key:
        return []
    params = {
        "category": category,
        "lang":     "en",
        "country":  "in",
        "max":      max(1, min(max_results, 100)),
        "apikey":   api_key,
    }
    try:
        resp = requests.get(GNEWS_TOP_URL, params=params, timeout=10)
        if resp.status_code in (401, 403, 429):
            return []
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except Exception as e:
        print(f"[news] Top headlines fetch failed ({category}): {e}")
        return []


def fetch_all_news(api_key: str, days: int = None) -> list[dict]:
    """
    Fetch news across all queries + top headlines. Dedupes by URL.
    With 20 queries + 2 top-headline categories = ~22 API calls per refresh.
    At 1000 req/day, can refresh ~45 times/day (every ~20 min during market hours).
    """
    cfg_max, cfg_days, pause, _ = _gnews_cfg()
    if days is None:
        days = cfg_days

    seen = set()
    articles = []

    # Search queries
    for q in NIFTY_QUERIES:
        for art in fetch_gnews(api_key, q, days=days, max_results=cfg_max):
            url = art.get("url")
            if url and url not in seen:
                seen.add(url)
                art["_query"] = q
                articles.append(art)
        if pause > 0:
            time.sleep(pause)

    # Top headlines (business + world for India)
    for cat in TOP_HEADLINE_CATEGORIES:
        for art in fetch_top_headlines(api_key, cat, max_results=cfg_max):
            url = art.get("url")
            if url and url not in seen:
                seen.add(url)
                art["_query"] = f"top:{cat}"
                articles.append(art)
        if pause > 0:
            time.sleep(pause)

    print(f"[news] Fetched {len(articles)} unique articles from {len(NIFTY_QUERIES) + len(TOP_HEADLINE_CATEGORIES)} sources")
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT SCORING
# ─────────────────────────────────────────────────────────────────────────────

# Market-impact keywords that boost/dampen sentiment weight
BULLISH_AMPLIFIERS = {
    "record high", "all-time high", "rally", "surge", "breakout", "upgrade",
    "rate cut", "stimulus", "reform", "inflows", "buying spree", "outperform",
    "strong earnings", "beat estimates", "GDP growth", "bull run",
}
BEARISH_AMPLIFIERS = {
    "crash", "plunge", "selloff", "sell-off", "panic", "recession", "downgrade",
    "rate hike", "outflows", "sanctions", "war", "crisis", "default", "bear market",
    "weak earnings", "miss estimates", "slowdown", "correction", "collapse",
}
MACRO_KEYWORDS = {
    "RBI", "Federal Reserve", "Fed", "GDP", "inflation", "CPI", "interest rate",
    "monetary policy", "fiscal deficit", "crude oil", "rupee", "dollar",
    "FII", "DII", "institutional", "budget", "trade war", "tariff",
}


def _score_finbert(texts: list[str]) -> list[float]:
    pipe = _load_finbert()
    if pipe is None:
        return [0.0] * len(texts)
    scores = []
    try:
        # Batch in chunks of 32 for memory efficiency
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            results = pipe(batch)
            for r in results:
                label = r["label"].lower()
                conf  = float(r["score"])
                if   label == "positive": scores.append(+conf)
                elif label == "negative": scores.append(-conf)
                else:                     scores.append(0.0)
    except Exception as e:
        print(f"[news] FinBERT scoring failed: {e}. Returning zeros.")
        scores = [0.0] * len(texts)
    return scores


def _score_vader(texts: list[str]) -> list[float]:
    analyzer = _load_vader()
    if analyzer is None:
        return [0.0] * len(texts)
    return [float(analyzer.polarity_scores(t)["compound"]) for t in texts]


def _classify_impact(text: str) -> dict:
    """Classify article's market impact type and amplification."""
    text_lower = text.lower()
    is_macro = any(kw.lower() in text_lower for kw in MACRO_KEYWORDS)
    bull_hits = sum(1 for kw in BULLISH_AMPLIFIERS if kw in text_lower)
    bear_hits = sum(1 for kw in BEARISH_AMPLIFIERS if kw in text_lower)

    if is_macro:
        impact_type = "macro"
        weight_mult = 1.5
    elif bull_hits > 0 or bear_hits > 0:
        impact_type = "market_event"
        weight_mult = 1.3
    else:
        impact_type = "general"
        weight_mult = 1.0

    return {
        "impact_type": impact_type,
        "weight_multiplier": weight_mult,
        "bull_keywords": bull_hits,
        "bear_keywords": bear_hits,
        "is_macro": is_macro,
    }


def score_articles(articles: list[dict]) -> list[dict]:
    """
    Score articles with FinBERT/VADER + classify market impact.
    """
    if not articles:
        return []

    backend = _detect_backend()
    if backend == "none":
        print("[news] WARNING: No sentiment backend available. "
              "Run: pip install vaderSentiment")
        for a in articles:
            a["sentiment"] = 0.0
            a["backend"]   = "none"
            a["impact"] = _classify_impact(a.get("title", ""))
        return articles

    texts = [
        (a.get("title", "") + ". " + (a.get("description") or "")).strip()
        for a in articles
    ]
    scores = _score_finbert(texts) if backend == "finbert" else _score_vader(texts)

    for a, s in zip(articles, scores):
        a["sentiment"] = round(float(s), 4)
        a["backend"]   = backend
        a["impact"] = _classify_impact(
            a.get("title", "") + " " + (a.get("description") or "")
        )
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE + CACHE
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(scored: list[dict]) -> dict:
    """
    Aggregate per-article scores into a single market sentiment snapshot.
    Impact-weighted: macro news and market events count more than general articles.
    Recent articles weighted higher (linear decay over lookback window).
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
            "latest_headlines": [],
            "macro_headlines": [],
            "category_breakdown": {},
        }

    now = datetime.utcnow()
    weighted_sum, weight_total = 0.0, 0.0
    n_pos = n_neg = n_neu = 0
    n_macro = n_market = n_general = 0
    macro_sum = market_sum = general_sum = 0.0
    query_scores = {}

    for a in scored:
        s = a.get("sentiment", 0.0)
        impact = a.get("impact", {})
        w_mult = impact.get("weight_multiplier", 1.0)

        # Recency weight
        try:
            published = datetime.strptime(a["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
            hours_old = max(0, (now - published).total_seconds() / 3600)
            w = max(0.3, 1.0 - hours_old / 72)
        except Exception:
            w = 0.7

        # Combined weight = recency × impact
        combined_w = w * w_mult
        weighted_sum += s * combined_w
        weight_total += combined_w

        if   s >  0.15: n_pos += 1
        elif s < -0.15: n_neg += 1
        else:           n_neu += 1

        impact_type = impact.get("impact_type", "general")
        if impact_type == "macro":
            n_macro += 1
            macro_sum += s
        elif impact_type == "market_event":
            n_market += 1
            market_sum += s
        else:
            n_general += 1
            general_sum += s

        q = a.get("_query", "unknown")
        if q not in query_scores:
            query_scores[q] = {"count": 0, "sum": 0.0}
        query_scores[q]["count"] += 1
        query_scores[q]["sum"] += s

    score = weighted_sum / weight_total if weight_total > 0 else 0.0
    n     = len(scored)

    if   score >  0.20: label = "bullish"
    elif score >  0.05: label = "slightly bullish"
    elif score < -0.20: label = "bearish"
    elif score < -0.05: label = "slightly bearish"
    else:               label = "neutral"

    def _to_ist(utc_str: str) -> str:
        if not utc_str:
            return ""
        try:
            dt = datetime.strptime(utc_str[:19], "%Y-%m-%dT%H:%M:%S")
            ist = dt + timedelta(hours=5, minutes=30)
            return ist.strftime("%Y-%m-%d %H:%M IST")
        except Exception:
            return utc_str

    def _fmt(a):
        impact = a.get("impact", {})
        return {
            "title":         a.get("title", "")[:140],
            "sentiment":     a.get("sentiment", 0.0),
            "source":        (a.get("source") or {}).get("name", "unknown"),
            "url":           a.get("url", ""),
            "publishedAt":   a.get("publishedAt", ""),
            "publishedIST":  _to_ist(a.get("publishedAt", "")),
            "impact_type":   impact.get("impact_type", "general"),
            "weight_mult":   impact.get("weight_multiplier", 1.0),
            "query":         a.get("_query", ""),
        }

    # Top 10 strongest-signal headlines (by |sentiment|)
    top = sorted(scored, key=lambda a: abs(a.get("sentiment", 0.0)), reverse=True)[:10]
    top_headlines = [_fmt(t) for t in top]

    # Latest 10 fetched headlines (most recent by publish time)
    latest = sorted(scored, key=lambda a: a.get("publishedAt", ""), reverse=True)[:10]
    latest_headlines = [_fmt(t) for t in latest]

    # Top macro/market-event headlines specifically
    macro_arts = [a for a in scored if a.get("impact", {}).get("impact_type") in ("macro", "market_event")]
    macro_arts.sort(key=lambda a: abs(a.get("sentiment", 0.0)), reverse=True)
    macro_headlines = [_fmt(a) for a in macro_arts[:10]]

    # Category breakdown for LLM context
    category_breakdown = {}
    for q, data in sorted(query_scores.items(), key=lambda x: x[1]["count"], reverse=True):
        avg = data["sum"] / data["count"] if data["count"] > 0 else 0
        category_breakdown[q] = {
            "count": data["count"],
            "avg_sentiment": round(avg, 3),
            "label": "bullish" if avg > 0.1 else "bearish" if avg < -0.1 else "neutral",
        }

    backend_used = scored[0].get("backend", _detect_backend()) if scored else _detect_backend()
    result = {
        "score":              round(score, 4),
        "label":              label,
        "n_articles":         n,
        "n_positive":         n_pos,
        "n_negative":         n_neg,
        "n_neutral":          n_neu,
        "pct_positive":       round(n_pos / n * 100, 1) if n else 0.0,
        "pct_negative":       round(n_neg / n * 100, 1) if n else 0.0,
        "n_macro":            n_macro,
        "n_market_events":    n_market,
        "n_general":          n_general,
        "macro_sentiment":    round(macro_sum / n_macro, 3) if n_macro > 0 else 0.0,
        "market_sentiment":   round(market_sum / n_market, 3) if n_market > 0 else 0.0,
        "general_sentiment":  round(general_sum / n_general, 3) if n_general > 0 else 0.0,
        "backend":            backend_used,
        "top_headlines":      top_headlines,
        "latest_headlines":   latest_headlines,
        "macro_headlines":    macro_headlines,
        "category_breakdown": category_breakdown,
    }
    if backend_used == "none":
        result["error"] = ("Sentiment scoring unavailable — no model installed. "
                           "Run: pip install vaderSentiment")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def get_market_sentiment(
    api_key: str,
    days: int = None,
    force_refresh: bool = False,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Main entry point. Fetches news, scores it, aggregates, caches.
    Premium plan (1000 req/day): short TTL (5 min), 100 articles per query.
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

    _, _, _, cache_minutes = _gnews_cfg()

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
                pass

    if not api_key or api_key in ("YOUR_GNEWS_API_KEY", ""):
        return {
            "score":  0.0,
            "label":  "neutral (no API key)",
            "n_articles": 0,
            "backend": _detect_backend(),
            "error":  "GNews API key not configured. Add it in Settings tab.",
            "top_headlines": [],
            "latest_headlines": [],
            "macro_headlines": [],
            "category_breakdown": {},
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

    _archive_daily(result, cache_dir)
    return result


def _archive_daily(result: dict, cache_dir: Path):
    """
    Append today's sentiment to a persistent daily archive.
    File: data/news_history.json — NEVER deleted, grows over time.
    """
    archive_path = Path(cache_dir) / "news_history.json"
    today_str = date.today().isoformat()

    archive = {}
    if archive_path.exists():
        try:
            with open(archive_path) as f:
                archive = json.load(f)
        except Exception:
            pass

    archive[today_str] = {
        "score":           result.get("score", 0),
        "label":           result.get("label", ""),
        "n_articles":      result.get("n_articles", 0),
        "n_positive":      result.get("n_positive", 0),
        "n_negative":      result.get("n_negative", 0),
        "n_neutral":       result.get("n_neutral", 0),
        "n_macro":         result.get("n_macro", 0),
        "macro_sentiment": result.get("macro_sentiment", 0),
        "backend":         result.get("backend", ""),
        "top_headlines":   result.get("top_headlines", [])[:5],
        "macro_headlines":  result.get("macro_headlines", [])[:5],
        "category_breakdown": result.get("category_breakdown", {}),
        "fetched_at":      result.get("fetched_at", ""),
    }

    try:
        with open(archive_path, "w") as f:
            json.dump(archive, f, indent=2)
    except Exception as e:
        print(f"[news] Archive write failed: {e}")


def load_news_history(cache_dir=None, last_n_days: int = 7) -> dict:
    if cache_dir is None:
        try:
            from settings import DATA_DIR
            cache_dir = DATA_DIR
        except Exception:
            cache_dir = Path("data")
    archive_path = Path(cache_dir) / "news_history.json"
    if not archive_path.exists():
        return {}
    try:
        with open(archive_path) as f:
            archive = json.load(f)
        cutoff = (date.today() - timedelta(days=last_n_days)).isoformat()
        return {k: v for k, v in archive.items() if k >= cutoff}
    except Exception:
        return {}
