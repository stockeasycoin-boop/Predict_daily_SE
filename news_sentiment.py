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
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

warnings.filterwarnings("ignore")

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
# GNEWS FETCH
# ─────────────────────────────────────────────────────────────────────────────

GNEWS_URL = "https://gnews.io/api/v4/search"

# India/Nifty-focused search queries (combined with OR via comma-quote trick)
NIFTY_QUERIES = [
    "Nifty 50",
    "Indian stock market",
    "Sensex",
    "RBI India",
    "FII India",
]


def fetch_gnews(api_key: str, query: str, days: int = 2, max_results: int = 10) -> list[dict]:
    """
    Fetch news articles from GNews API for a given query.

    Parameters
    ----------
    api_key      : Your GNews API key (free at gnews.io)
    query        : Search query (e.g. "Nifty 50")
    days         : Look back this many days
    max_results  : Up to 10 articles per query on free tier

    Returns
    -------
    List of article dicts with keys: title, description, publishedAt, source, url
    """
    if not api_key or api_key in ("YOUR_GNEWS_API_KEY", ""):
        return []

    from_dt = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "q":        query,
        "lang":     "en",
        "country":  "in",          # India-focused
        "max":      min(max_results, 10),
        "from":     from_dt,
        "sortby":   "publishedAt",
        "apikey":   api_key,
    }
    try:
        resp = requests.get(GNEWS_URL, params=params, timeout=10)
        if resp.status_code == 403:
            print("[news] GNews 403 — API key invalid or quota exhausted (100/day on free tier).")
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])
    except Exception as e:
        print(f"[news] GNews fetch failed for '{query}': {e}")
        return []


def fetch_all_news(api_key: str, days: int = 2) -> list[dict]:
    """
    Fetch news for all Nifty-related queries and dedupe by URL.
    Costs ~5 GNews calls (one per query in NIFTY_QUERIES).
    """
    seen = set()
    articles = []
    for q in NIFTY_QUERIES:
        for art in fetch_gnews(api_key, q, days=days, max_results=10):
            url = art.get("url")
            if url and url not in seen:
                seen.add(url)
                articles.append(art)
        time.sleep(0.3)   # be polite to the API
    return articles


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

    # Top 5 strongest-signal headlines (by |sentiment|)
    top = sorted(scored, key=lambda a: abs(a.get("sentiment", 0.0)), reverse=True)[:5]
    top_headlines = [
        {
            "title":     t.get("title", "")[:140],
            "sentiment": t.get("sentiment", 0.0),
            "source":    (t.get("source") or {}).get("name", "unknown"),
            "url":       t.get("url", ""),
        }
        for t in top
    ]

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
    }


def get_market_sentiment(
    api_key: str,
    days: int = 2,
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

    # Serve from cache if fresh
    if cache_file.exists() and not force_refresh:
        age_hrs = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
        if age_hrs < 4:
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                cached["from_cache"] = True
                cached["cache_age_hours"] = round(age_hrs, 2)
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
