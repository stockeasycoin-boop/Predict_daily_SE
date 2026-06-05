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


def _breeze_chunk(bz, from_dt: datetime, to_dt: datetime) -> pd.DataFrame | None:
    """One direct Breeze /historical_data_v2 call for a custom date range."""
    try:
        resp = bz.get_historical_data_v2(
            interval="5minute",
            from_date=from_dt.strftime("%Y-%m-%dT07:00:00.000Z"),
            to_date=to_dt.strftime("%Y-%m-%dT07:00:00.000Z"),
            stock_code="NIFTY",
            exchange_code="NSE",
            product_type="cash",
        )
        if resp.get("Status") != 200 or not resp.get("Success"):
            return None
        df = pd.DataFrame(resp["Success"])
        df["date"] = pd.to_datetime(df["datetime"])
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
        return df[["date", "open", "high", "low", "close", "volume"]] \
                 .dropna(subset=["open", "close"])
    except Exception as e:
        print(f"[breeze] chunk {from_dt.date()}→{to_dt.date()} failed: {e}")
        return None


def fetch_via_breeze(days: int, chunk_days: int = 55) -> pd.DataFrame | None:
    """
    Walk backwards in `chunk_days`-day windows so we span the full `days`
    history. Breeze's per-call cap is ~60 days for 5-min — we leave headroom.
    """
    from data_fetcher import init_breeze
    s = _load_settings()
    api_k = s.get("api_key")
    api_s = s.get("api_secret")
    tok   = s.get("session_token")
    if not (api_k and api_s and tok):
        print("[fetch] Breeze creds missing in settings.json")
        return None
    try:
        bz = init_breeze(api_k, api_s, tok)
        print("[breeze] connected.")
    except Exception as e:
        print(f"[fetch] Breeze init failed: {e}")
        return None

    cursor = datetime.now()
    target = cursor - timedelta(days=days)
    n_chunks = (days + chunk_days - 1) // chunk_days
    print(f"[breeze] will issue ~{n_chunks} chunked 5-min calls "
          f"(window={chunk_days}d, range={target.date()}→{cursor.date()})")

    frames = []
    chunk_idx = 0
    while cursor > target:
        chunk_idx += 1
        chunk_start = max(cursor - timedelta(days=chunk_days), target)
        df = _breeze_chunk(bz, chunk_start, cursor)
        n = 0 if df is None else len(df)
        print(f"  [{chunk_idx:>3d}/{n_chunks}] {chunk_start.date()}→{cursor.date()}: {n} bars")
        if df is not None and n:
            frames.append(df)
        elif df is None:
            # transient failure — short backoff, single retry
            time.sleep(1.5)
            df = _breeze_chunk(bz, chunk_start, cursor)
            if df is not None and len(df):
                frames.append(df)
                print(f"        retry ok: {len(df)} bars")
        cursor = chunk_start - timedelta(seconds=1)
        time.sleep(0.4)   # polite throttle

    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True) \
            .drop_duplicates(["date"]).sort_values("date").reset_index(drop=True)
    return out


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
