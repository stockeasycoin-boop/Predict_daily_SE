"""Smoke test for Pred_new_model — no live API needed."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("=== Module imports ===")
import settings as cfg;          print("settings OK — MODEL_DIR:", cfg.MODEL_DIR)
import feature_engineering as fe; print("feature_engineering OK — FEATURE_COLS:", len(fe.FEATURE_COLS))
import data_fetcher;              print("data_fetcher OK")
import model_trainer;             print("model_trainer OK")
import news_sentiment;            print("news_sentiment OK")
import options_engine;            print("options_engine OK")
import tracker;                   print("tracker OK")
import intraday_predictor as ip;  print("intraday_predictor OK — horizons:", list(ip.HORIZONS.keys()))
import live_engine as le;         print("live_engine OK")
import groww_connector as gc;     print("groww_connector OK — groww_available:", gc.groww_available())

print("\n=== Synthetic 5-min data ===")
np.random.seed(42)
n = 4500
prices = 24000.0 + np.cumsum(np.random.randn(n) * 20)
times = []
t = datetime(2026, 1, 2, 9, 15)
for i in range(n):
    times.append(t)
    t += timedelta(minutes=5)
    if t.hour == 15 and t.minute >= 30:
        t = t.replace(hour=9, minute=15) + timedelta(days=1)
        while t.weekday() >= 5:
            t += timedelta(days=1)

df5 = pd.DataFrame({
    "date":   times,
    "open":   prices * (1 + np.random.randn(n) * 0.0002),
    "high":   prices * (1 + np.abs(np.random.randn(n)) * 0.0005),
    "low":    prices * (1 - np.abs(np.random.randn(n)) * 0.0005),
    "close":  prices,
    "volume": np.random.randint(50000, 500000, n),
})
print("Shape:", df5.shape)

print("\n=== Feature engineering ===")
feat = ip.build_intraday_features(df5)
present = [c for c in ip.get_feature_cols_intraday() if c in feat.columns]
print(f"Features: {len(present)}/{len(ip.get_feature_cols_intraday())} present")

tgt = ip.build_horizon_targets(feat)
target_cols = [c for c in tgt.columns if c.startswith("target_")]
print("Target cols:", target_cols)

print("\n=== Train intraday models (synthetic data) ===")
os.makedirs("models", exist_ok=True)
results = ip.train_intraday_models(df5, model_dir="models", verbose=True)
print("Training complete.")
for hz, r in results.items():
    print(f"  {hz:8s}: CV acc={r['cv_accuracy']:.3f}  n={r['n_samples']:,}")

print("\n=== Predict on latest candle ===")
preds = ip.predict_all_horizons(df5, model_dir="models")
print("_live_price:", preds.get("_live_price"))
print("_minutes_elapsed:", preds.get("_minutes_elapsed"))
horizon_preds = {k: v for k, v in preds.items() if not k.startswith("_")}
print("Active horizons:", list(horizon_preds.keys()))
for hz, p in horizon_preds.items():
    print(f"  {hz:8s}: {p['label']}  conf={p['confidence']:.2f}  agree={p['ensemble_agree']}")

print("\n=== OFI logic (mock order book) ===")
mock_book = {"bids": [{"quantity": 150}, {"quantity": 200}],
             "asks": [{"quantity": 80}]}
ofi = gc.compute_ofi(mock_book)
print(f"OFI from mock book: {ofi:.4f}")
adj_agree     = gc.ofi_confidence_adjustment(0.4, 1)
adj_contradict = gc.ofi_confidence_adjustment(-0.5, 1)
print(f"Confidence adj (OFI agrees):      {adj_agree:.2f}x")
print(f"Confidence adj (OFI contradicts): {adj_contradict:.2f}x")

print("\n=== Live engine state ===")
print("is_trading_day():", le.is_trading_day())
print("is_market_open():", le.is_market_open())
print("next_refresh_seconds():", le.next_refresh_seconds())
print("get_live_accuracy_stats():", le.get_live_accuracy_stats())
print("get_pending_predictions():", le.get_pending_predictions())

print("\n=== SMOKE TEST PASSED ===")
