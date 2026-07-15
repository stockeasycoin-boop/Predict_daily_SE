"""
bootstrap.py — one-shot generator for all model artifacts.

The repo gitignores models/ and data/ (weights + caches are regenerated, never
committed). Run this once on a fresh checkout to produce everything the app needs:

  1. 5-min candle data          -> data/nifty_5min_2yr.csv   (cache or Breeze fetch)
  2. daily open/close models    -> models/*.pkl
  3. intraday horizon models    -> models/intraday_*.pkl
  4. confidence-gate config      -> models/gate_config.json
  5. meta-labeling models+config -> models/intraday_meta_*.pkl, models/meta_config.json

Usage:
  python bootstrap.py                # cache if present, else fetch via Breeze
  python bootstrap.py --cache-only   # never hit the API (fails if no cache)
  python bootstrap.py --days 730     # lookback for the initial fetch
  python bootstrap.py --force        # rebuild even if models already exist
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"

# Confidence-gate operating points from the leakage-free walk-forward backtest
# (scratchpad/gated_backtest.py). Static — a coarse fallback used only when the
# meta gate is unavailable for a horizon.
GATE_CONFIG = {
    "_comment": "Confidence-gate operating points from walk-forward backtest on "
                "3yr of 5-min data. conf_gate = min confidence (prob on predicted "
                "side) for a high-conviction signal; exp_acc/coverage measured OOS. "
                "Fallback for when the meta gate is unavailable.",
    "horizons": {
        "5min":   {"conf_gate": 0.585, "exp_acc": 64, "coverage": 0.05},
        "15min":  {"conf_gate": 0.605, "exp_acc": 68, "coverage": 0.20},
        "30min":  {"conf_gate": 0.641, "exp_acc": 74, "coverage": 0.20},
        "60min":  {"conf_gate": 0.685, "exp_acc": 83, "coverage": 0.20},
        "120min": {"conf_gate": 0.680, "exp_acc": 83, "coverage": 0.30},
        "180min": {"conf_gate": 0.672, "exp_acc": 78, "coverage": 0.30},
        "close":  {"conf_gate": 0.693, "exp_acc": 77, "coverage": 0.20},
    },
}


def _write_gate_config():
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "gate_config.json", "w") as f:
        json.dump(GATE_CONFIG, f, indent=2)
    print("[gate] wrote models/gate_config.json")


def _models_present() -> bool:
    need = ["intraday_xgb_close.pkl", "intraday_meta_close.pkl",
            "gate_config.json", "meta_config.json"]
    return all((MODEL_DIR / n).exists() for n in need)


def main():
    ap = argparse.ArgumentParser(description="Generate all model artifacts.")
    ap.add_argument("--days", type=int, default=730, help="Lookback days for initial fetch")
    ap.add_argument("--cache-only", action="store_true", help="Never hit the Breeze API")
    ap.add_argument("--force", action="store_true", help="Rebuild even if models exist")
    args = ap.parse_args()

    if _models_present() and not args.force:
        print("[bootstrap] All artifacts already present — nothing to do "
              "(use --force to rebuild).")
        return

    print("=" * 68)
    print("BOOTSTRAP — generating model artifacts")
    print("=" * 68)

    # 1. Data (reuse the existing training pipeline's fetch/cache helpers)
    import train_intraday_full as T
    breeze = None
    if args.cache_only:
        intraday = T._load_cache()
        if intraday is None or len(intraday) == 0:
            print("ERROR: --cache-only set but data/nifty_5min_2yr.csv is missing.\n"
                  "       Run without --cache-only to fetch it via Breeze "
                  "(needs credentials in settings.json).")
            sys.exit(1)
    else:
        try:
            breeze = T._connect_breeze()
        except Exception as e:
            print(f"ERROR: could not connect to Breeze ({e}).\n"
                  "       Fill api_key/api_secret/session_token in settings.json, "
                  "or use --cache-only if you already have data/nifty_5min_2yr.csv.")
            sys.exit(1)
        intraday = T._incremental_fetch(breeze, total_days=args.days)

    if intraday is None or len(intraday) == 0:
        print("ERROR: no 5-min data available — cannot train.")
        sys.exit(1)

    n_days = intraday["date"].dt.date.nunique()
    print(f"[data] {len(intraday):,} candles across {n_days} trading days")

    daily_context = pd.DataFrame()
    if breeze is not None:
        daily_context = T._fetch_daily_context(breeze)

    # 2. Daily models
    print("\n[train] daily open/close models …")
    from intraday_predictor import train_daily_models_from_5min, train_intraday_models
    train_daily_models_from_5min(intraday, model_dir="models",
                                 daily_context=daily_context, verbose=True)

    # 3. Intraday horizon models
    print("\n[train] intraday horizon models …")
    train_intraday_models(intraday, model_dir="models", verbose=True)

    # 4. Confidence-gate config
    _write_gate_config()

    # 5. Meta-labeling models + config (must run AFTER intraday models exist so the
    #    scaler matches).
    print("\n[train] meta-labeling models …")
    from meta_labeling import train_meta_models
    train_meta_models(intraday, model_dir="models", verbose=True)

    print("\n" + "=" * 68)
    print("BOOTSTRAP COMPLETE — models/ is ready. Launch with: streamlit run app.py")
    print("=" * 68)


if __name__ == "__main__":
    main()
