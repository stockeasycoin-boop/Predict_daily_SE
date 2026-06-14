"""
train_intraday_full.py — Train models using 2-year 5-min intraday data.

Features:
  - Incremental fetch: loads existing cache, detects gap, fetches only missing days
  - Full chunked fetch only on first run (bypasses Breeze 60-day cap)
  - Aggregates 5-min → daily OHLCV, builds all 87 features
  - Trains daily open/close models + 7 intraday horizon models

Usage:
  python train_intraday_full.py                    # full run
  python train_intraday_full.py --days 730         # custom lookback
  python train_intraday_full.py --no-optuna        # skip hyperparameter search
  python train_intraday_full.py --cache-only       # train from cached data only
"""

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
              f"({df['date'].dt.date.min()} → {df['date'].dt.date.max()})")
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
        new_data = fetch_intraday_chunked(breeze, "NIFTY", total_days=gap_days + 2, chunk_days=55)
        if new_data is not None and len(new_data) > 0:
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(CACHE_FILE, index=False)
            print(f"[Incremental] Appended {len(new_data)} candles → total {len(combined)}")
            return combined
        return existing
    else:
        print(f"[Full Fetch] No cache found — fetching {total_days} days of 5-min data")
        df = fetch_intraday_chunked(breeze, "NIFTY", total_days=total_days, chunk_days=55)
        if df is not None and len(df) > 0:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CACHE_FILE, index=False)
            print(f"[Full Fetch] Saved {len(df)} candles to {CACHE_FILE}")
        return df


def _aggregate_to_daily(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-min candles to daily OHLCV bars."""
    df = intraday_df.copy()
    df["trade_date"] = df["date"].dt.date
    daily = df.groupby("trade_date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    daily.rename(columns={"trade_date": "date"}, inplace=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    print(f"[Aggregate] {len(daily)} trading days from intraday data")
    return daily


def _build_intraday_features(intraday_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build intraday pattern features from 5-min data and merge with daily.
    Adds features like: first-hour momentum, lunch-hour range, power-hour volume, etc.
    """
    df5 = intraday_df.copy()
    df5["trade_date"] = df5["date"].dt.date
    df5["hour"] = df5["date"].dt.hour
    df5["minute"] = df5["date"].dt.minute

    features_list = []
    for td, group in df5.groupby("trade_date"):
        if len(group) < 10:
            continue
        row = {"date": pd.Timestamp(td)}
        o = group["open"].iloc[0]
        c = group["close"].iloc[-1]
        h = group["high"].max()
        l = group["low"].min()
        rng = h - l if h != l else 1

        first_hour = group[(group["hour"] == 9) | (group["hour"] == 10)]
        lunch = group[(group["hour"] >= 12) & (group["hour"] <= 13)]
        power = group[group["hour"] >= 14]

        row["intra_first_hour_ret"] = (first_hour["close"].iloc[-1] / o - 1) * 100 if len(first_hour) > 0 else 0
        row["intra_lunch_range"] = (lunch["high"].max() - lunch["low"].min()) / rng * 100 if len(lunch) > 0 else 0
        row["intra_power_hour_ret"] = (power["close"].iloc[-1] / power["open"].iloc[0] - 1) * 100 if len(power) > 0 else 0
        row["intra_power_vol_ratio"] = power["volume"].sum() / max(group["volume"].sum(), 1) if len(power) > 0 else 0
        row["intra_body_ratio"] = abs(c - o) / rng * 100
        row["intra_upper_wick"] = (h - max(o, c)) / rng * 100
        row["intra_lower_wick"] = (min(o, c) - l) / rng * 100
        row["intra_candle_count"] = len(group)
        row["intra_vol_concentration"] = group["volume"].nlargest(5).sum() / max(group["volume"].sum(), 1)
        rets = group["close"].pct_change().dropna()
        row["intra_volatility"] = rets.std() * 100 if len(rets) > 0 else 0
        row["intra_max_drawdown"] = ((group["close"].cummax() - group["close"]) / group["close"].cummax()).max() * 100
        row["intra_trend_strength"] = abs(c - o) / max(group["high"].diff().abs().sum() + group["low"].diff().abs().sum(), 1)
        features_list.append(row)

    intra_feat = pd.DataFrame(features_list)
    intra_feat["date"] = pd.to_datetime(intra_feat["date"])
    daily_df["date"] = pd.to_datetime(daily_df["date"].dt.date if hasattr(daily_df["date"].dt, "date") else daily_df["date"])
    merged = daily_df.merge(intra_feat, on="date", how="left")
    for col in intra_feat.columns:
        if col != "date":
            merged[col] = merged[col].fillna(0)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Train models with 2-year 5-min intraday data")
    parser.add_argument("--days", type=int, default=730, help="Total lookback days")
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna hyperparameter search")
    parser.add_argument("--cache-only", action="store_true", help="Train from cached data only, no API fetch")
    args = parser.parse_args()

    print("=" * 70)
    print("NIFTY 50 FULL INTRADAY TRAINING PIPELINE")
    print(f"Lookback: {args.days} days | Optuna: {not args.no_optuna} | Cache-only: {args.cache_only}")
    print("=" * 70)

    # Step 1: Get 5-min data (incremental)
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

    print(f"\n[Data] {len(intraday)} five-minute candles loaded")

    # Step 2: Aggregate to daily
    daily = _aggregate_to_daily(intraday)
    if len(daily) < 30:
        print(f"ERROR: Only {len(daily)} trading days — need at least 30.")
        return

    # Step 3: Build features (daily + intraday patterns)
    print("\n[Features] Building feature set...")
    from feature_engineering import build_features
    daily_features = build_features(daily)

    daily_with_intra = _build_intraday_features(intraday, daily)
    intra_cols = [c for c in daily_with_intra.columns if c.startswith("intra_")]
    for col in intra_cols:
        if col not in daily_features.columns:
            daily_features[col] = daily_with_intra.set_index("date").reindex(daily_features["date"]).reset_index(drop=True)[col]

    feat_count = len([c for c in daily_features.columns if c not in ["date", "open", "high", "low", "close", "volume"]])
    print(f"[Features] {feat_count} features built on {len(daily_features)} trading days")

    # Step 4: Also fetch supplementary data if not cache-only
    if not args.cache_only:
        try:
            from data_fetcher import load_vix_data, load_correlated_data, load_fii_dii_data
            vix = load_vix_data(breeze)
            if vix is not None and "india_vix" in vix.columns:
                daily_features = daily_features.merge(
                    vix[["date", "india_vix"]], on="date", how="left"
                )
                daily_features["india_vix"] = daily_features["india_vix"].ffill().bfill()
                print(f"[VIX] Merged {len(vix)} VIX rows")

            correlated = load_correlated_data(breeze)
            for sym, cdf in correlated.items():
                col_name = f"{sym.lower()}_close"
                if col_name not in daily_features.columns:
                    daily_features = daily_features.merge(
                        cdf[["date", "close"]].rename(columns={"close": col_name}),
                        on="date", how="left"
                    )
                    daily_features[col_name] = daily_features[col_name].ffill().bfill()

            fii = load_fii_dii_data()
            if fii is not None:
                daily_features = daily_features.merge(fii[["date", "fii_net"]], on="date", how="left")
                daily_features["fii_net"] = daily_features["fii_net"].fillna(0)
                print(f"[FII] Merged {len(fii)} FII rows")
        except Exception as e:
            print(f"[Supplementary] Some data unavailable: {e}")

    # Step 5: Train daily models
    print("\n" + "=" * 70)
    print("TRAINING DAILY MODELS")
    print("=" * 70)
    from model_trainer import train_model
    use_optuna = not args.no_optuna
    results = train_model(daily_features, model_dir="models", verbose=True, use_optuna=use_optuna)
    if results:
        print(f"\n[Training] Daily models saved to models/")

    # Step 6: Train intraday horizon models
    print("\n" + "=" * 70)
    print("TRAINING INTRADAY HORIZON MODELS")
    print("=" * 70)
    try:
        from intraday_predictor import train_intraday_models
        intra_results = train_intraday_models(
            intraday, model_dir="models", use_optuna=use_optuna
        )
        if intra_results:
            print(f"\n[Training] Intraday models saved to models/")
    except ImportError:
        print("[Skip] intraday_predictor not available — skipping horizon models")
    except Exception as e:
        print(f"[Intraday] Training error: {e}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
