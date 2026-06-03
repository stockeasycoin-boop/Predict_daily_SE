"""
fetch_5min_history.py — Pull 3 years of 5-min Nifty OHLCV via Fyers or Breeze.

Why a separate CLI:
  - The fetch involves ~70+ API calls (Fyers caps 5-min history to ~10 days/call).
  - Takes 5-30 min depending on the provider; not appropriate to do inside a
    Streamlit request.
  - One-time job; result is cached to data/intraday_5min_3yr.csv.

Usage:
    .venv\\Scripts\\python.exe fetch_5min_history.py             # auto-pick provider
    .venv\\Scripts\\python.exe fetch_5min_history.py --provider fyers
    .venv\\Scripts\\python.exe fetch_5min_history.py --provider breeze --days 1095

Symbol fallback for Fyers:
  - Index symbols (NSE:NIFTY50-INDEX) return very few 5-min candles.
  - We use the current Nifty FUTURES symbol for dense data.
  - For historical data spanning >3 months we'd need rolling futures — out of
    scope here; users typically only need ~2-3 years for LSTM training.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

OUTPUT = Path("data") / "intraday_5min_3yr.csv"


def _load_settings() -> dict:
    p = Path("settings.json")
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return {}
    return {}


def fetch_via_fyers(days: int) -> pd.DataFrame | None:
    from fyers_data import (
        connect_from_settings, fetch_history_fyers, fetch_nifty_futures_symbol,
    )
    f = connect_from_settings()
    if f is None:
        return None
    # Try futures first (dense), fall back to index
    fut = fetch_nifty_futures_symbol()
    print(f"[fetch] trying Fyers futures: {fut}")
    df = fetch_history_fyers(f, fut, "5", days=days)
    if df is None or len(df) < days * 20:
        print(f"[fetch] futures sparse ({0 if df is None else len(df)} rows); trying index")
        df_idx = fetch_history_fyers(f, "NIFTY", "5", days=days)
        if df_idx is not None and (df is None or len(df_idx) > len(df)):
            df = df_idx
    return df


def fetch_via_breeze(days: int) -> pd.DataFrame | None:
    """Breeze caps intraday history at 60 days per call; we loop."""
    from data_fetcher import init_breeze, fetch_intraday_breeze
    s = _load_settings()
    api_k = s.get("api_key")
    api_s = s.get("api_secret")
    tok   = s.get("session_token")
    if not (api_k and api_s and tok):
        print("[fetch] Breeze creds missing in settings.json")
        return None
    try:
        bz = init_breeze(api_k, api_s, tok)
    except Exception as e:
        print(f"[fetch] Breeze init failed: {e}")
        return None
    frames = []
    cursor = datetime.now()
    target = cursor - timedelta(days=days)
    while cursor > target:
        chunk = fetch_intraday_breeze(bz, "NIFTY", days_back=60)
        if chunk is None or chunk.empty:
            break
        frames.append(chunk)
        # Move cursor back; Breeze always returns "last N days", no offset support
        # → can't chunk further with this SDK call. Break after one fetch.
        break
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date"]) if frames else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["fyers", "breeze", "auto"], default="auto")
    p.add_argument("--days", type=int, default=3 * 365,
                   help="Calendar days of history (default 1095 = ~3 years)")
    args = p.parse_args()

    t0 = time.time()
    df = None
    if args.provider in ("fyers", "auto"):
        try:
            df = fetch_via_fyers(args.days)
        except Exception as e:
            print(f"[fetch] Fyers path failed: {e}")
    if (df is None or len(df) < 5000) and args.provider in ("breeze", "auto"):
        print("[fetch] falling back to Breeze")
        df = fetch_via_breeze(args.days)

    if df is None or df.empty:
        print("[fetch] no data acquired from any provider.")
        return 1

    df = df.sort_values("date").drop_duplicates(["date"]).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    secs = time.time() - t0
    print(f"\n[fetch] saved {len(df):,} 5-min bars to {OUTPUT}")
    print(f"        range: {df['date'].min()} → {df['date'].max()}")
    print(f"        ~{len(df) / max(1, df['date'].astype(str).str[:10].nunique()):.0f} bars/day")
    print(f"        elapsed: {secs:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
