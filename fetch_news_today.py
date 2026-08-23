"""
fetch_news_today.py — One-shot script: call GNews with all NIFTY_QUERIES,
score with VADER/FinBERT, print a trading-ready table.
Run: python fetch_news_today.py
"""
import json, sys, os, time
from datetime import datetime, timedelta
from pathlib import Path

# ── Load API key from settings.json ──────────────────────────────────────────
HERE = Path(__file__).parent
try:
    cfg = json.loads((HERE / "settings.json").read_text())
    API_KEY = cfg.get("gnews_api_key", "")
except Exception as e:
    sys.exit(f"Could not read settings.json: {e}")

if not API_KEY:
    sys.exit("gnews_api_key not set in settings.json")

# ── Import news_sentiment module ──────────────────────────────────────────────
sys.path.insert(0, str(HERE))
from news_sentiment import (
    NIFTY_QUERIES, TOP_HEADLINE_CATEGORIES,
    fetch_gnews, fetch_top_headlines, score_articles,
)

# ── Fetch ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  GNews Fetch — {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {len(NIFTY_QUERIES)} queries + {len(TOP_HEADLINE_CATEGORIES)} top-headline categories")
print(f"{'='*70}\n")

seen = set()
all_articles = []

for q in NIFTY_QUERIES:
    arts = fetch_gnews(API_KEY, q, days=1, max_results=10)
    for a in arts:
        url = a.get("url", "")
        if url and url not in seen:
            seen.add(url)
            a["_query"] = q
            all_articles.append(a)
    print(f"  [{q:30s}] -> {len(arts)} articles (unique so far: {len(all_articles)})")
    time.sleep(0.15)   # gentle rate limit

for cat in TOP_HEADLINE_CATEGORIES:
    arts = fetch_top_headlines(API_KEY, cat, max_results=10)
    for a in arts:
        url = a.get("url", "")
        if url and url not in seen:
            seen.add(url)
            a["_query"] = f"top:{cat}"
            all_articles.append(a)
    print(f"  [top:{cat:26s}] -> {len(arts)} articles (unique so far: {len(all_articles)})")
    time.sleep(0.15)

print(f"\n  Total unique articles: {len(all_articles)}\n")

# ── Score ─────────────────────────────────────────────────────────────────────
scored = score_articles(all_articles)

# ── Sort by publishedAt desc ───────────────────────────────────────────────────
scored.sort(key=lambda a: a.get("publishedAt", ""), reverse=True)

# ── Print table ───────────────────────────────────────────────────────────────
def sentiment_label(s):
    if s > 0.25:  return "[+] BULLISH"
    if s > 0.05:  return "[+] Sl.Bull"
    if s < -0.25: return "[-] BEARISH"
    if s < -0.05: return "[-] Sl.Bear"
    return "[ ] Neutral"

def to_ist(utc_str):
    try:
        dt = datetime.strptime(utc_str[:19], "%Y-%m-%dT%H:%M:%S")
        return (dt + timedelta(hours=5, minutes=30)).strftime("%H:%M IST")
    except Exception:
        return utc_str[:16]

print(f"{'='*70}")
print(f"  TRADING NEWS TABLE — {datetime.now().strftime('%d %b %Y')}")
print(f"{'='*70}\n")

# Group by query for organised output
from collections import defaultdict
by_query = defaultdict(list)
for a in scored:
    by_query[a.get("_query", "?")].append(a)

query_order = list(NIFTY_QUERIES) + [f"top:{c}" for c in TOP_HEADLINE_CATEGORIES]
row_num = 0
summary_rows = []

for q in query_order:
    arts = by_query.get(q, [])
    if not arts:
        continue
    print(f"\n── {q} ──")
    for a in arts[:5]:   # max 5 per query in console
        row_num += 1
        title   = a.get("title", "")[:100]
        source  = (a.get("source") or {}).get("name", "?")[:20]
        s       = a.get("sentiment", 0.0)
        sl      = sentiment_label(s)
        t       = to_ist(a.get("publishedAt", ""))
        url     = a.get("url", "")
        impact  = a.get("impact", {}).get("impact_type", "general")
        wt      = a.get("impact", {}).get("weight_multiplier", 1.0)

        print(f"  {row_num:3d}. [{sl}] ({s:+.2f}, {impact}, {wt}x)  {t}")
        print(f"       {title}")
        print(f"       Source: {source}  |  {url}")
        summary_rows.append({
            "query": q, "title": title, "source": source,
            "sentiment": s, "label": sl, "impact": impact,
            "weight": wt, "time": t, "url": url,
        })

# ── Aggregate stats ────────────────────────────────────────────────────────────
if scored:
    from news_sentiment import aggregate
    agg = aggregate(scored)
    print(f"\n{'='*70}")
    print(f"  AGGREGATE SENTIMENT: {agg['label'].upper()}  (score={agg['score']:+.3f})")
    print(f"  Articles: {agg['n_articles']}  |  +ve: {agg['n_positive']}  -ve: {agg['n_negative']}  neutral: {agg['n_neutral']}")
    print(f"  Macro: {agg['n_macro']}  Market events: {agg['n_market_events']}  General: {agg['n_general']}")
    print(f"  Backend: {agg['backend']}")
    print(f"{'='*70}\n")

    # Save full result to data/
    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / "news_today.json"
    agg["articles"] = summary_rows
    agg["fetched_at"] = datetime.utcnow().isoformat()
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  Full result saved → {out_path}")
