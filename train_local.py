r"""
train_local.py — Retrain models locally (mirrors the app's Model Health → Train flow).
Run:  .venv\Scripts\python.exe train_local.py
"""
import data_fetcher as df_mod
import feature_engineering as fe
import model_trainer as mt
import settings as cfg

import sys
sys.stdout.reconfigure(encoding="utf-8")

print("Step 1/4 - downloading Nifty OHLCV (free sources)...")
nifty = df_mod.load_nifty_data(None, force_refresh=True)
print(f"  Nifty rows: {0 if nifty is None else len(nifty)}")

print("Step 2/4 - downloading VIX / global / FII / GIFT / PCR...")
vix    = df_mod.load_vix_data(None, force_refresh=True)
glob   = df_mod.load_global_data(force_refresh=True)
fii    = df_mod.load_fii_dii_data(force_refresh=True)
gift   = df_mod.load_gift_data(force_refresh=True)
pcr    = df_mod.load_pcr_data(force_refresh=True)

print("Step 3/4 - building features...")
# Note: intraday and corr_dict require a live Breeze session — skipped for local train
feat = fe.build_features(nifty, vix, glob, fii, gift, pcr,
                         intraday_df=None, corr_dict=None)
print(f"  Feature matrix: {feat.shape[0]} rows x {feat.shape[1]} cols")

print("Step 4/4 - training models (chained open->close->high->low)...")
*_, meta = mt.train_model(feat, str(cfg.MODEL_DIR), verbose=True, use_optuna=True)

print("\nDone. Metadata:")
for k in ("n_samples","cv_open","cv_close","mae_open_pct","mae_close_pct",
          "mae_high_pct","mae_low_pct","train_start","train_end"):
    print(f"  {k}: {meta.get(k)}")
