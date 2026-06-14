"""
backtest_full.py — Full backtest, CSV export, and live simulation on cached 5-min data.

Outputs:
  1. Walk-forward backtest: out-of-sample accuracy per horizon
  2. CSV export: features + predictions for every trading day
  3. Live simulation: simulated predictions as if running live each day

Usage:
  python backtest_full.py                      # backtest on all data
  python backtest_full.py --last-n 60          # backtest last 60 days only
  python backtest_full.py --start 2025-06-01   # from a specific date
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import argparse
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False

CACHE_FILE = Path("data/nifty_5min_2yr.csv")
OUTPUT_DIR = Path("backtest_results")


def load_data():
    if not CACHE_FILE.exists():
        print("ERROR: No cached data. Run train_intraday_full.py first.")
        return None
    df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    print(f"Loaded {len(df):,} candles, {df['date'].dt.date.nunique()} trading days")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. WALK-FORWARD BACKTEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_backtest(df_5min, start_date=None, last_n=None):
    """Walk-forward out-of-sample backtest on intraday + daily models."""
    from intraday_predictor import (
        build_intraday_features, build_horizon_targets,
        get_feature_cols_intraday, _extract_eod_features,
        HORIZONS, _fit_calibrated
    )

    print("\n" + "=" * 70)
    print("1. WALK-FORWARD BACKTEST")
    print("=" * 70)

    # Build features on full dataset
    feat_df = build_intraday_features(df_5min)
    feat_df = build_horizon_targets(feat_df)
    feat_cols = get_feature_cols_intraday()
    avail = [f for f in feat_cols if f in feat_df.columns]

    # Determine test window
    dates = sorted(feat_df["trading_date"].unique())
    if last_n:
        test_dates = dates[-last_n:]
        train_dates = dates[:-last_n]
    elif start_date:
        sd = pd.Timestamp(start_date).date()
        test_dates = [d for d in dates if d >= sd]
        train_dates = [d for d in dates if d < sd]
    else:
        # Default: last 20% as test
        split = int(len(dates) * 0.8)
        train_dates = dates[:split]
        test_dates = dates[split:]

    print(f"  Train: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"  Test:  {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

    # Split data
    train_mask = feat_df["trading_date"].isin(train_dates)
    test_mask = feat_df["trading_date"].isin(test_dates)

    X_train = feat_df.loc[train_mask, avail].values.astype(np.float32)
    X_test = feat_df.loc[test_mask, avail].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results = {}
    print(f"\n  {'Horizon':<10} {'Train Acc':>10} {'Test Acc':>10} {'Baseline':>10} {'Skill':>10} {'N_test':>8}")
    print("  " + "-" * 58)

    for horizon, n_candles in HORIZONS.items():
        col = f"target_{horizon}"
        if col not in feat_df.columns:
            continue

        train_valid = feat_df.loc[train_mask, col].notna()
        test_valid = feat_df.loc[test_mask, col].notna()

        y_train = feat_df.loc[train_mask, col][train_valid].values.astype(int)
        y_test = feat_df.loc[test_mask, col][test_valid].values.astype(int)
        Xtr = X_train_sc[train_valid.values]
        Xte = X_test_sc[test_valid.values]

        if len(Xtr) < 200 or len(Xte) < 50:
            continue

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.75, colsample_bytree=0.75,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, n_jobs=-1
        )
        model.fit(Xtr, y_train, eval_set=[(Xte[:200], y_test[:200])], verbose=False)

        train_acc = accuracy_score(y_train, model.predict(Xtr))
        test_acc = accuracy_score(y_test, model.predict(Xte))
        baseline = max(y_test.mean(), 1 - y_test.mean())
        skill = test_acc - baseline

        results[horizon] = {
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "baseline": round(baseline, 4),
            "skill": round(skill, 4),
            "n_test": len(Xte),
        }
        print(f"  {horizon:<10} {train_acc:>10.3f} {test_acc:>10.3f} {baseline:>10.3f} {skill:>+10.3f} {len(Xte):>8,}")

    # Daily model backtest
    print("\n  -- Daily Models (EOD snapshots) --")
    eod_df = _extract_eod_features(df_5min)
    eod_avail = [f for f in feat_cols if f in eod_df.columns]
    eod_dates = sorted(eod_df["date"].dt.date.unique())

    if last_n:
        eod_test_start = len(eod_dates) - last_n
    elif start_date:
        sd = pd.Timestamp(start_date).date()
        eod_test_start = next((i for i, d in enumerate(eod_dates) if d >= sd), int(len(eod_dates) * 0.8))
    else:
        eod_test_start = int(len(eod_dates) * 0.8)

    eod_valid = eod_df.dropna(subset=["open_target", "close_target"])
    split_idx = int(len(eod_valid) * (eod_test_start / len(eod_dates)))
    split_idx = max(split_idx, 50)

    X_eod = eod_valid[eod_avail].values.astype(np.float32)
    X_eod = np.nan_to_num(X_eod, nan=0.0, posinf=1e6, neginf=-1e6)
    X_eod_sc = StandardScaler().fit_transform(X_eod)

    for target_name in ["open_target", "close_target"]:
        y = eod_valid[target_name].values.astype(int)
        Xtr, Xte = X_eod_sc[:split_idx], X_eod_sc[split_idx:]
        ytr, yte = y[:split_idx], y[split_idx:]

        if len(Xte) < 20:
            continue

        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.75, colsample_bytree=0.75,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, n_jobs=-1
        )
        m.fit(Xtr, ytr, verbose=False)
        tr_acc = accuracy_score(ytr, m.predict(Xtr))
        te_acc = accuracy_score(yte, m.predict(Xte))
        base = max(yte.mean(), 1 - yte.mean())
        label = target_name.replace("_target", "").upper()
        print(f"  {label:<10} {tr_acc:>10.3f} {te_acc:>10.3f} {base:>10.3f} {te_acc - base:>+10.3f} {len(Xte):>8,}")
        results[f"daily_{label.lower()}"] = {
            "train_acc": round(tr_acc, 4), "test_acc": round(te_acc, 4),
            "baseline": round(base, 4), "skill": round(te_acc - base, 4),
            "n_test": len(Xte),
        }

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. CSV EXPORT: features + predictions for every day
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def export_csv(df_5min):
    """Export features + actual outcomes for every trading day."""
    from intraday_predictor import (
        build_intraday_features, get_feature_cols_intraday,
        _extract_eod_features
    )

    print("\n" + "=" * 70)
    print("2. CSV EXPORT")
    print("=" * 70)

    # EOD features with daily targets
    eod_df = _extract_eod_features(df_5min)
    feat_cols = get_feature_cols_intraday()
    avail = [f for f in feat_cols if f in eod_df.columns]

    export_cols = ["date", "open", "high", "low", "close"]
    export_cols += avail
    target_cols = [c for c in eod_df.columns if c.endswith("_target") or c.endswith("_pct")]
    export_cols += target_cols

    out = eod_df[[c for c in export_cols if c in eod_df.columns]].copy()
    out = out.round(4)

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "features_and_targets.csv"
    out.to_csv(csv_path, index=False)
    print(f"  Exported {len(out)} rows x {len(out.columns)} columns to {csv_path}")

    # Also export per-candle intraday features for inspection
    feat_df = build_intraday_features(df_5min)
    candle_cols = ["dt", "trading_date", "open", "high", "low", "close", "volume"] + avail
    candle_out = feat_df[[c for c in candle_cols if c in feat_df.columns]].copy()
    candle_out = candle_out.round(4)
    candle_path = OUTPUT_DIR / "intraday_candle_features.csv"
    candle_out.to_csv(candle_path, index=False)
    print(f"  Exported {len(candle_out):,} candle rows to {candle_path}")

    # Summary stats
    print(f"\n  Feature columns: {len(avail)}")
    print(f"  Date range: {out['date'].min()} to {out['date'].max()}")
    if "open_target" in out.columns:
        print(f"  Open target distribution: {out['open_target'].value_counts().to_dict()}")
    if "close_target" in out.columns:
        print(f"  Close target distribution: {out['close_target'].value_counts().to_dict()}")

    return csv_path, candle_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. LIVE SIMULATION: replay predictions as if running live each day
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_live(df_5min, start_date=None, last_n=None):
    """
    Simulate running predictions for each trading day using only data available
    up to that point (no future leakage). Train on history, predict next day.
    """
    from intraday_predictor import (
        build_intraday_features, build_horizon_targets,
        get_feature_cols_intraday, _extract_eod_features,
        HORIZONS
    )

    print("\n" + "=" * 70)
    print("3. LIVE SIMULATION")
    print("=" * 70)

    feat_df = build_intraday_features(df_5min)
    feat_cols = get_feature_cols_intraday()
    avail = [f for f in feat_cols if f in feat_df.columns]
    dates = sorted(feat_df["trading_date"].unique())

    # Determine simulation window
    min_train_days = 120  # need at least 120 days before we start simulating
    if last_n:
        sim_dates = dates[-last_n:]
    elif start_date:
        sd = pd.Timestamp(start_date).date()
        sim_dates = [d for d in dates if d >= sd]
    else:
        sim_dates = dates[min_train_days:]

    print(f"  Simulating {len(sim_dates)} trading days ({sim_dates[0]} to {sim_dates[-1]})")
    print(f"  Min training window: {min_train_days} days")

    # EOD data for daily model simulation
    eod_df = _extract_eod_features(df_5min)
    eod_dates = sorted(eod_df["date"].dt.date.unique())

    sim_results = []

    for i, sim_date in enumerate(sim_dates):
        # Get all data up to (not including) sim_date for training
        train_mask = feat_df["trading_date"] < sim_date
        if train_mask.sum() < min_train_days * 50:  # ~50 candles/day minimum
            continue

        # Current day data (what we'd have at market open)
        day_mask = feat_df["trading_date"] == sim_date
        day_data = feat_df[day_mask]
        if len(day_data) < 5:
            continue

        # Use last candle of previous day for daily prediction
        prev_days = feat_df[train_mask]
        last_row = prev_days[avail].tail(1).values.astype(np.float32)
        last_row = np.nan_to_num(last_row, nan=0.0, posinf=1e6, neginf=-1e6)

        # Actual outcomes for this day
        actual_open = day_data["open"].iloc[0]
        actual_close = day_data["close"].iloc[-1]
        actual_high = day_data["high"].max()
        actual_low = day_data["low"].min()
        prev_close = prev_days["close"].iloc[-1]

        actual_gap_up = 1 if actual_open > prev_close else 0
        actual_bull = 1 if actual_close > actual_open else 0

        # Intraday horizon actuals
        horizon_actuals = {}
        for hname, n_candles in HORIZONS.items():
            if n_candles is not None:
                # Predict from first candle of day
                if len(day_data) > n_candles:
                    entry = day_data["close"].iloc[0]
                    target = day_data["close"].iloc[n_candles]
                    horizon_actuals[hname] = 1 if target > entry else 0
            else:
                entry = day_data["close"].iloc[0]
                horizon_actuals["close"] = 1 if actual_close > entry else 0

        row = {
            "date": str(sim_date),
            "prev_close": round(prev_close, 2),
            "actual_open": round(actual_open, 2),
            "actual_close": round(actual_close, 2),
            "actual_high": round(actual_high, 2),
            "actual_low": round(actual_low, 2),
            "actual_gap_up": actual_gap_up,
            "actual_bull": actual_bull,
            "n_candles": len(day_data),
        }

        # Add horizon actuals
        for hname, actual in horizon_actuals.items():
            row[f"actual_{hname}"] = actual

        sim_results.append(row)

        if (i + 1) % 50 == 0:
            print(f"  ... processed {i + 1}/{len(sim_dates)} days")

    sim_df = pd.DataFrame(sim_results)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Now run walk-forward predictions using trained models
    print(f"\n  Running predictions using trained models on {len(sim_df)} days...")

    # Load trained models
    model_dir = "models"
    try:
        scaler = joblib.load(f"{model_dir}/intraday_scaler.pkl")
        intra_feats = joblib.load(f"{model_dir}/intraday_features.pkl")
    except FileNotFoundError:
        print("  ERROR: Models not trained. Run train_intraday_full.py first.")
        return

    # For each simulation day, get the first candle's features and predict
    for i, sim_date in enumerate(sim_dates):
        if str(sim_date) not in sim_df["date"].values:
            continue

        day_mask = feat_df["trading_date"] == sim_date
        day_data = feat_df[day_mask]
        if len(day_data) < 5:
            continue

        # Use first candle of the day (what we'd have at 9:20 AM)
        first_row = day_data[avail].head(1).copy()
        first_row = first_row.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)

        intra_avail = [f for f in intra_feats if f in first_row.columns]
        for f in intra_feats:
            if f not in first_row.columns:
                first_row[f] = 0.0
        X = scaler.transform(first_row[intra_feats].values.astype(np.float32))

        idx = sim_df.index[sim_df["date"] == str(sim_date)]
        if len(idx) == 0:
            continue
        ix = idx[0]

        for horizon in HORIZONS:
            xgb_path = Path(f"{model_dir}/intraday_xgb_{horizon}.pkl")
            if not xgb_path.exists():
                continue
            m = joblib.load(xgb_path)
            pred = int(m.predict(X)[0])
            prob = float(m.predict_proba(X)[0][pred])
            sim_df.loc[ix, f"pred_{horizon}"] = pred
            sim_df.loc[ix, f"conf_{horizon}"] = round(prob, 4)

    # Save simulation results
    sim_path = OUTPUT_DIR / "live_simulation.csv"
    sim_df.to_csv(sim_path, index=False)
    print(f"  Saved simulation to {sim_path}")

    # Print simulation accuracy summary
    print(f"\n  -- Simulation Accuracy Summary --")
    print(f"  {'Horizon':<10} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'Avg Conf':>10}")
    print("  " + "-" * 48)

    for horizon in HORIZONS:
        pred_col = f"pred_{horizon}"
        actual_col = f"actual_{horizon}"
        conf_col = f"conf_{horizon}"
        if pred_col not in sim_df.columns or actual_col not in sim_df.columns:
            continue
        valid = sim_df.dropna(subset=[pred_col, actual_col])
        if len(valid) == 0:
            continue
        correct = (valid[pred_col] == valid[actual_col]).sum()
        total = len(valid)
        acc = correct / total
        avg_conf = valid[conf_col].mean() if conf_col in valid.columns else 0
        print(f"  {horizon:<10} {acc:>10.1%} {correct:>10} {total:>8} {avg_conf:>10.3f}")

    # Daily direction accuracy
    if "actual_bull" in sim_df.columns and "pred_close" in sim_df.columns:
        valid = sim_df.dropna(subset=["pred_close", "actual_bull"])
        if len(valid) > 0:
            acc = (valid["pred_close"] == valid["actual_bull"]).mean()
            print(f"\n  Daily bull/bear accuracy (using close prediction): {acc:.1%} ({len(valid)} days)")

    return sim_df


def main():
    parser = argparse.ArgumentParser(description="Full backtest suite")
    parser.add_argument("--last-n", type=int, default=None, help="Backtest last N trading days")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()

    df = load_data()
    if df is None:
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Walk-forward backtest
    bt_results = run_backtest(df, start_date=args.start, last_n=args.last_n)

    # 2. CSV export
    csv_path, candle_path = export_csv(df)

    # 3. Live simulation
    sim_df = simulate_live(df, start_date=args.start, last_n=args.last_n)

    # Save combined report
    report = {
        "run_at": datetime.now().isoformat(),
        "data": {
            "candles": len(df),
            "trading_days": int(df["date"].dt.date.nunique()),
            "date_range": f"{df['date'].dt.date.min()} to {df['date'].dt.date.max()}",
        },
        "backtest": bt_results,
        "exports": {
            "features_csv": str(csv_path),
            "candle_csv": str(candle_path),
            "simulation_csv": str(OUTPUT_DIR / "live_simulation.csv"),
        },
    }
    report_path = OUTPUT_DIR / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"REPORT saved to {report_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
