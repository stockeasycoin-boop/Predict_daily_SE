"""
train_intraday_full.py -- Train ALL models using 2-year 5-min intraday data.

Unified pipeline: ALL features (intraday + daily) are derived from 5-min candles.
No daily feature_engineering.py dependency.

Features:
  - Incremental fetch: loads existing cache, detects gap, fetches only missing days
  - Full chunked fetch only on first run (bypasses Breeze 60-day cap)
  - Daily models trained from end-of-day snapshots of 5-min features
  - 7 intraday horizon models trained on raw 5-min candles

Usage:
  python train_intraday_full.py                    # full run
  python train_intraday_full.py --days 730         # custom lookback
  python train_intraday_full.py --no-optuna        # skip hyperparameter search
  python train_intraday_full.py --cache-only       # train from cached data only
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

CACHE_FILE = Path("data/nifty_5min_2yr.csv")


def _connect_breeze():
    """Connect to Breeze using settings.json credentials."""
    settings_path = Path(__file__).parent / "settings.json"
    if not settings_path.exists():
        raise FileNotFoundError("settings.json not found — run auth_breeze.py first")
    creds = json.load(open(settings_path))
    api_key = creds.get("api_key", "")
    api_secret = creds.get("api_secret", "")
    session_token = creds.get("session_token", "")
    if not api_key or not session_token:
        raise ValueError("api_key / session_token missing in settings.json")
    from data_fetcher import init_breeze
    return init_breeze(api_key, api_secret, session_token)


def _load_cache() -> pd.DataFrame | None:
    """Load existing 5-min cache if it exists."""
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
        print(f"[Cache] Loaded {len(df)} existing candles "
              f"({df['date'].dt.date.min()} to {df['date'].dt.date.max()})")
        return df
    return None


def _incremental_fetch(breeze, total_days: int = 730) -> pd.DataFrame:
    """
    Incremental fetch: if cache exists, only fetch the gap and append.
    NEVER purges existing data.
    """
    from data_fetcher import fetch_intraday_chunked, _filter_market_hours
    import pytz

    existing = _load_cache()
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)

    if existing is not None and len(existing) > 0:
        last_date = existing["date"].max()
        if hasattr(last_date, 'to_pydatetime'):
            last_date = last_date.to_pydatetime()
        if last_date.tzinfo is None:
            last_date = ist.localize(last_date)
        gap_days = (now_ist - last_date).days
        if gap_days <= 1:
            print(f"[Cache] Already up to date (last candle: {last_date.date()})")
            return existing
        print(f"[Incremental] Gap: {gap_days} days — fetching only missing data")
        new_data = fetch_intraday_chunked(breeze, "NIFTY", total_days=gap_days + 2, chunk_days=12)
        if new_data is not None and len(new_data) > 0:
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(CACHE_FILE, index=False)
            print(f"[Incremental] Appended {len(new_data)} candles, total {len(combined)}")
            return combined
        return existing
    else:
        print(f"[Full Fetch] No cache found — fetching {total_days} days of 5-min data")
        df = fetch_intraday_chunked(breeze, "NIFTY", total_days=total_days, chunk_days=12)
        if df is not None and len(df) > 0:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CACHE_FILE, index=False)
            print(f"[Full Fetch] Saved {len(df)} candles to {CACHE_FILE}")
        return df



def _fetch_daily_context(breeze) -> pd.DataFrame:
    """Fetch VIX/FII supplementary data and return as a daily context DataFrame."""
    rows = []
    try:
        from data_fetcher import load_vix_data, load_fii_dii_data
        vix = load_vix_data(breeze)
        fii = load_fii_dii_data()

        if vix is not None and "india_vix" in vix.columns:
            vix["date"] = pd.to_datetime(vix["date"])
            rows.append(vix[["date", "india_vix"]])
            print(f"[VIX] Loaded {len(vix)} rows")

        if fii is not None and "fii_net" in fii.columns:
            fii["date"] = pd.to_datetime(fii["date"])
            if rows:
                rows[0] = rows[0].merge(fii[["date", "fii_net"]], on="date", how="outer")
            else:
                rows.append(fii[["date", "fii_net"]])
            print(f"[FII] Loaded {len(fii)} rows")
    except Exception as e:
        print(f"[Supplementary] Some data unavailable: {e}")

    if rows:
        ctx = rows[0]
        for col in ctx.select_dtypes(include=[np.floating]).columns:
            ctx[col] = ctx[col].ffill().bfill().fillna(0)
        return ctx
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Train models with 2-year 5-min intraday data (unified pipeline)")
    parser.add_argument("--days", type=int, default=730, help="Total lookback days")
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna hyperparameter search")
    parser.add_argument("--cache-only", action="store_true", help="Train from cached data only, no API fetch")
    args = parser.parse_args()

    print("=" * 70)
    print("NIFTY 50 UNIFIED 5-MIN TRAINING PIPELINE")
    print("ALL features derived from 5-min candles (no daily features)")
    print(f"Lookback: {args.days} days | Cache-only: {args.cache_only}")
    print("=" * 70)

    # Step 1: Get 5-min data (incremental)
    breeze = None
    if args.cache_only:
        intraday = _load_cache()
        if intraday is None or len(intraday) == 0:
            print("ERROR: No cached data found. Run without --cache-only first.")
            return
    else:
        breeze = _connect_breeze()
        intraday = _incremental_fetch(breeze, total_days=args.days)

    if intraday is None or len(intraday) == 0:
        print("ERROR: No intraday data available.")
        return

    n_days = intraday["date"].dt.date.nunique()
    print(f"\n[Data] {len(intraday)} five-minute candles across {n_days} trading days")

    # Step 2: Fetch daily context (VIX, FII) if not cache-only
    daily_context = pd.DataFrame()
    if not args.cache_only and breeze is not None:
        print("\n[Context] Fetching supplementary data (VIX, FII)...")
        daily_context = _fetch_daily_context(breeze)

    # Step 3: Train daily models from 5-min EOD snapshots
    print("\n" + "=" * 70)
    print("TRAINING DAILY MODELS (from 5-min features)")
    print("=" * 70)
    from intraday_predictor import train_daily_models_from_5min
    daily_results = train_daily_models_from_5min(
        intraday, model_dir="models", daily_context=daily_context, verbose=True
    )
    if daily_results:
        print(f"\n[Training] Daily models saved to models/")

    # Step 4: Train intraday horizon models
    print("\n" + "=" * 70)
    print("TRAINING INTRADAY HORIZON MODELS")
    print("=" * 70)
    try:
        from intraday_predictor import train_intraday_models
        intra_results = train_intraday_models(
            intraday, model_dir="models", verbose=True
        )
        if intra_results:
            print(f"\n[Training] Intraday models saved to models/")
    except Exception as e:
        print(f"[Intraday] Training error: {e}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE (unified 5-min pipeline)")
    print("=" * 70)


if __name__ == "__main__":
    main()
