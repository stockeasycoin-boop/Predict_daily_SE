r"""
train_intraday_full.py — Train models using 2 years of 5-min intraday data
                         with ALL 87+ features (daily indicators + VIX + sectors
                         + FII + GIFT + PCR + intraday patterns).

How it works:
  1. Fetch 2 years of 5-min candles in 55-day chunks from Breeze API
  2. Aggregate 5-min candles into daily OHLCV bars
  3. Fetch daily VIX, sector indices, FII/DII, GIFT Nifty, PCR
  4. Build all 87 daily features (same as feature_engineering.build_features)
  5. ALSO build per-day intraday features from the raw 5-min candles
  6. Train open/close/high/low models with Optuna tuning

Run:
  python train_intraday_full.py
  python train_intraday_full.py --days 365    # 1 year only
  python train_intraday_full.py --no-optuna   # skip Optuna (faster)
"""

import sys
import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import settings as cfg

# ─────────────────────────────────────────────────────────────────────────────
# PARSE ARGS
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train models on 2yr 5-min intraday data")
parser.add_argument("--days", type=int, default=730, help="Calendar days of intraday data (default 730 = 2yr)")
parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna hyperparameter tuning")
parser.add_argument("--cache-only", action="store_true", help="Use cached data only, don't fetch from API")
args = parser.parse_args()

TOTAL_DAYS = args.days
USE_OPTUNA = not args.no_optuna
CACHE_FILE = cfg.DATA_DIR / "nifty_5min_2yr.csv"
DAILY_CACHE = cfg.DATA_DIR / "nifty_daily_from_5min.csv"

print(f"{'='*60}")
print(f"  INTRADAY FULL-FEATURE TRAINING")
print(f"  Days: {TOTAL_DAYS} | Optuna: {USE_OPTUNA}")
print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: FETCH / LOAD 5-MIN DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Step 1/6 — Loading 5-min intraday data...")

import data_fetcher as df_mod

df_5min = None
breeze = None

def _connect_breeze():
    """Connect to Breeze API. Returns client or exits."""
    import json as _json
    from breeze_connect import BreezeConnect
    with open(cfg.SETTINGS_FILE) as _f:
        s = _json.load(_f)
    api_key = s.get("api_key", "")
    api_secret = s.get("api_secret", "")
    session_token = s.get("session_token", "")
    if not (api_key and api_secret and session_token):
        print("  ERROR: Breeze API credentials not found in settings.json")
        sys.exit(1)
    bz = BreezeConnect(api_key=api_key)
    bz.generate_session(api_secret=api_secret, session_token=session_token)
    print("  Breeze connected")
    return bz


if args.cache_only and CACHE_FILE.exists():
    df_5min = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    print(f"  Loaded from cache: {len(df_5min)} candles")
elif CACHE_FILE.exists():
    # ── INCREMENTAL MODE: load cached data, fetch only the gap ──────────
    cached = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    cached_last = cached["date"].max()
    today = pd.Timestamp.now().normalize()
    gap_days = (today - cached_last.normalize()).days

    if gap_days <= 0:
        print(f"  Cache is up-to-date (last candle: {cached_last})")
        df_5min = cached
    else:
        print(f"  Cache has {len(cached)} candles up to {cached_last.date()}")
        print(f"  Gap: {gap_days} days — fetching only the missing data...")
        try:
            breeze = _connect_breeze()
            # Fetch the gap (max 60 days, which covers any practical gap)
            fresh = df_mod.fetch_intraday_breeze(breeze, "NIFTY",
                                                 days_back=min(gap_days + 2, 60))
            if fresh is not None and len(fresh) > 0:
                # Keep only candles newer than what we have
                fresh = fresh[fresh["date"] > cached_last]
                if len(fresh) > 0:
                    df_5min = pd.concat([cached, fresh], ignore_index=True)
                    df_5min = df_5min.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
                    df_5min.to_csv(CACHE_FILE, index=False)
                    print(f"  Appended {len(fresh)} new candles → total {len(df_5min)}")
                else:
                    print(f"  No new candles found (market closed?)")
                    df_5min = cached
            else:
                print(f"  Fresh fetch returned nothing — using cache as-is")
                df_5min = cached
        except Exception as e:
            print(f"  WARNING: Could not fetch fresh data ({e}), using cache")
            df_5min = cached
else:
    # ── FULL FETCH: no cache exists, download everything ────────────────
    try:
        breeze = _connect_breeze()
    except Exception as e:
        print(f"  ERROR connecting to Breeze: {e}")
        sys.exit(1)

    print(f"  No cache found — fetching {TOTAL_DAYS} days of 5-min data in chunks...")
    df_5min = df_mod.fetch_intraday_chunked(breeze, "NIFTY", total_days=TOTAL_DAYS)

    if df_5min is None or len(df_5min) < 100:
        print("  ERROR: Not enough intraday data fetched")
        sys.exit(1)

    df_5min.to_csv(CACHE_FILE, index=False)
    print(f"  Cached {len(df_5min)} candles to {CACHE_FILE}")

n_trading_days = df_5min["date"].dt.date.nunique()
print(f"  Total: {len(df_5min)} candles across {n_trading_days} trading days")
print(f"  Range: {df_5min['date'].dt.date.min()} → {df_5min['date'].dt.date.max()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: AGGREGATE 5-MIN → DAILY OHLCV
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2/6 — Aggregating 5-min candles into daily OHLCV bars...")

df_5min["trading_date"] = df_5min["date"].dt.normalize()

daily_agg = df_5min.groupby("trading_date").agg(
    open  =("open",   "first"),
    high  =("high",   "max"),
    low   =("low",    "min"),
    close =("close",  "last"),
    volume=("volume", "sum"),
).reset_index().rename(columns={"trading_date": "date"})

daily_agg = daily_agg.sort_values("date").reset_index(drop=True)
daily_agg.to_csv(DAILY_CACHE, index=False)
print(f"  Daily bars: {len(daily_agg)} rows "
      f"({daily_agg['date'].iloc[0].date()} → {daily_agg['date'].iloc[-1].date()})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: FETCH SUPPLEMENTARY DATA (VIX, sectors, FII, GIFT, PCR)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3/6 — Fetching VIX, sectors, FII/DII, GIFT, PCR...")

_breeze_ref = breeze if not args.cache_only else None
vix_df = df_mod.load_vix_data(_breeze_ref, force_refresh=not args.cache_only)
if vix_df is not None:
    print(f"  VIX: {len(vix_df)} rows")
else:
    print("  VIX: not available (will use default=16)")

# Correlated instruments (BankNifty, sectors)
corr_dict = df_mod.load_correlated_data(_breeze_ref, force_refresh=not args.cache_only)
print(f"  Correlated indices: {list(corr_dict.keys()) if corr_dict else 'none'}")

fii_df = df_mod.load_fii_dii_data(force_refresh=not args.cache_only)
print(f"  FII/DII: {0 if fii_df is None else len(fii_df)} rows")

gift_df = df_mod.load_gift_data(_breeze_ref, force_refresh=not args.cache_only)
print(f"  GIFT Nifty: {0 if gift_df is None else len(gift_df)} rows")

pcr_df = df_mod.load_pcr_data(force_refresh=not args.cache_only)
print(f"  PCR: {0 if pcr_df is None else len(pcr_df)} rows")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: BUILD FULL FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 4/6 — Building features (daily from 5-min + all enrichments)...")

import feature_engineering as fe

feat_df = fe.build_features(
    nifty_df    = daily_agg,
    vix_df      = vix_df,
    global_df   = None,
    fii_df      = fii_df,
    gift_df     = gift_df,
    pcr_df      = pcr_df,
    intraday_df = df_5min,
    corr_dict   = corr_dict,
)

print(f"  Feature matrix: {feat_df.shape[0]} rows × {feat_df.shape[1]} cols")
print(f"  Date range: {feat_df['date'].iloc[0].date()} → {feat_df['date'].iloc[-1].date()}")

# Show feature coverage
from feature_engineering import FEATURE_COLS
avail = [f for f in FEATURE_COLS if f in feat_df.columns]
missing = [f for f in FEATURE_COLS if f not in feat_df.columns]
print(f"  Features available: {len(avail)}/{len(FEATURE_COLS)}")
if missing:
    print(f"  Missing features: {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nStep 5/6 — Training models (Optuna={'ON' if USE_OPTUNA else 'OFF'})...")

import model_trainer as mt

model_dir = str(cfg.MODEL_DIR)
*_, meta = mt.train_model(feat_df, model_dir, verbose=True, use_optuna=USE_OPTUNA)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: ALSO RETRAIN INTRADAY HORIZON MODELS (with 2yr of 5-min data)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nStep 6/6 — Training intraday horizon models (7 horizons)...")

import intraday_predictor as ip

intra_meta = ip.train_intraday_models(df_5min, model_dir, verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Data source:        {TOTAL_DAYS} days of 5-min candles")
print(f"  Total candles:      {len(df_5min)}")
print(f"  Trading days:       {n_trading_days}")
print(f"  Daily features:     {len(avail)}")
print(f"  Training samples:   {meta.get('n_samples', '?')}")
print(f"")
print(f"  DAILY MODEL:")
print(f"    Open  CV accuracy: {meta.get('cv_open', 0):.1%}")
print(f"    Close CV accuracy: {meta.get('cv_close', 0):.1%}")
print(f"    Open  MAE:         {meta.get('mae_open_pct', 0):.3f}%")
print(f"    Close MAE:         {meta.get('mae_close_pct', 0):.3f}%")
print(f"    High  MAE:         {meta.get('mae_high_pct', 0):.3f}%")
print(f"    Low   MAE:         {meta.get('mae_low_pct', 0):.3f}%")
print(f"")
print(f"  INTRADAY MODELS:")
for h, info in intra_meta.items():
    if isinstance(info, dict) and "cv_accuracy" in info:
        print(f"    {h:>6s} CV accuracy: {info['cv_accuracy']:.1%} ({info.get('n_samples', '?')} samples)")
print(f"{'='*60}")
