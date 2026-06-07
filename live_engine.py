"""
live_engine.py — Live market loop, auto-verification feedback, EOD retraining.

THE FEEDBACK LOOP (the key new capability):
  1. At each refresh, every horizon prediction is logged with:
       - timestamp, horizon, direction, confidence, entry_price, target_time
  2. On the NEXT refresh, any prediction whose target time has passed is
     automatically verified against the current live price:
       - correct = did price move in the predicted direction?
  3. Verified results update the rolling accuracy stats shown on the dashboard.

This creates a continuous self-checking system: the model grades its own
predictions in real time and the dashboard shows how it's doing today.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import json
from pathlib import Path
import pytz

IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN  = time(9, 15)
MARKET_CLOSE = time(15, 30)
EOD_RETRAIN  = time(15, 40)

# How many minutes ahead each horizon predicts
HORIZON_MINUTES = {
    "5min":   5,   "15min":  15,  "30min":  30,
    "60min":  60,  "120min": 120, "180min": 180,
    "close":  None,   # special: target is 3:15 PM
}


# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS
# ─────────────────────────────────────────────────────────────────────────────

def now_ist() -> datetime:
    return datetime.now(IST)

def is_trading_day() -> bool:
    """NSE is closed on weekends (Sat=5, Sun=6). Does not account for holidays."""
    return now_ist().weekday() < 5


def is_market_open() -> bool:
    """Market is open only on weekdays during 9:15 AM - 3:30 PM IST."""
    if not is_trading_day():
        return False
    return MARKET_OPEN <= now_ist().time() <= MARKET_CLOSE


def market_status_text() -> str:
    """Human-readable market status for the dashboard banner."""
    now = now_ist()
    if not is_trading_day():
        day_name = now.strftime("%A")
        return f"🔴 Market closed — {day_name} (NSE is closed on weekends)"
    t = now.time()
    if t < MARKET_OPEN:
        return "🟡 Pre-market — NSE opens at 9:15 AM IST"
    elif t > MARKET_CLOSE:
        return "🔴 Market closed — NSE closed at 3:30 PM IST"
    else:
        return "🟢 Market open — live trading in progress"

def is_eod_retrain_window() -> bool:
    return EOD_RETRAIN <= now_ist().time() <= time(16, 30)

def minutes_to_next_candle() -> int:
    m = now_ist().minute % 5
    return (5 - m) if m > 0 else 5

def next_refresh_seconds() -> int:
    if not is_market_open():
        return 300
    return minutes_to_next_candle() * 60 + 15


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOG  (JSON-lines file, one record per prediction)
# ─────────────────────────────────────────────────────────────────────────────

LIVE_LOG_FILE = Path("trades") / "live_predictions.jsonl"


def log_predictions_batch(predictions: dict, entry_price: float) -> int:
    """
    Log all horizon predictions from one refresh.
    Deduplicates: won't log the same horizon twice within the same 5-min candle.
    Returns number of new predictions logged.
    """
    LIVE_LOG_FILE.parent.mkdir(exist_ok=True)
    now = now_ist()
    candle_id = now.strftime("%Y-%m-%d %H:") + f"{(now.minute // 5) * 5:02d}"

    # Read existing to dedupe by (candle_id, horizon)
    existing = set()
    if LIVE_LOG_FILE.exists():
        with open(LIVE_LOG_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing.add((r.get("candle_id"), r.get("horizon")))
                except Exception:
                    pass

    n_logged = 0
    with open(LIVE_LOG_FILE, "a") as f:
        for horizon, p in predictions.items():
            if horizon.startswith("_"):
                continue
            if (candle_id, horizon) in existing:
                continue   # already logged this candle

            # Prefer target_iso from prediction (already IST-correct); else compute
            target_iso = p.get("target_iso")
            if not target_iso:
                mins = HORIZON_MINUTES.get(horizon)
                if mins is not None:
                    target_iso = (now + timedelta(minutes=mins)).isoformat()
                else:
                    target_iso = now.replace(hour=15, minute=15, second=0,
                                             microsecond=0).isoformat()

            record = {
                "ts":           now.isoformat(),
                "candle_id":    candle_id,
                "horizon":      horizon,
                "direction":    p["direction"],
                "confidence":   p["confidence"],
                "entry_price":  round(entry_price, 2),
                "target_price": p.get("target_price", 0),
                "target_ts":    target_iso,
                "actual_price": None,
                "correct":      None,
                "verified_ts":  None,
            }
            f.write(json.dumps(record) + "\n")
            n_logged += 1

    return n_logged


def verify_due_predictions(current_price: float) -> list:
    """
    THE AUTO-FEEDBACK STEP — call this on every refresh.

    Checks all unverified predictions whose target time has now passed,
    and grades them against the current live price.

    Returns a list of newly-verified records (for display).
    """
    if not LIVE_LOG_FILE.exists() or current_price is None or current_price <= 0:
        return []

    now = now_ist()
    rows, newly_verified = [], []

    with open(LIVE_LOG_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue

            # Verify if: not yet verified AND target time has passed
            if r.get("correct") is None and r.get("target_ts"):
                try:
                    target_dt = datetime.fromisoformat(r["target_ts"])
                    if now >= target_dt:
                        entry = r["entry_price"]
                        r["actual_price"] = round(current_price, 2)
                        r["correct"] = int(
                            (r["direction"] == 1 and current_price > entry) or
                            (r["direction"] == 0 and current_price <= entry)
                        )
                        r["verified_ts"] = now.isoformat()
                        r["actual_move_pct"] = round((current_price - entry) / entry * 100, 3)
                        newly_verified.append(r)
                except Exception:
                    pass
            rows.append(r)

    # Rewrite log with verifications
    if newly_verified:
        with open(LIVE_LOG_FILE, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    return newly_verified


def get_live_accuracy_stats() -> dict:
    """Rolling accuracy per horizon from verified predictions."""
    if not LIVE_LOG_FILE.exists():
        return {}
    rows = []
    with open(LIVE_LOG_FILE) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    df = df[df["correct"].notna()]
    if len(df) == 0:
        return {}

    today_str = date.today().isoformat()
    stats = {}
    for horizon in df["horizon"].unique():
        sub = df[df["horizon"] == horizon]
        sub_today = sub[sub["ts"].str[:10] == today_str]
        recent = sub.tail(50)
        stats[horizon] = {
            "n":            len(recent),
            "accuracy":     round(float(recent["correct"].mean()) * 100, 1),
            "n_today":      len(sub_today),
            "accuracy_today": round(float(sub_today["correct"].mean()) * 100, 1) if len(sub_today) else None,
        }
    return stats


def get_recent_verifications(limit: int = 8) -> list:
    """Most recent verified predictions for the live feedback feed."""
    if not LIVE_LOG_FILE.exists():
        return []
    rows = []
    with open(LIVE_LOG_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("correct") is not None:
                    rows.append(r)
            except Exception:
                pass
    rows.sort(key=lambda r: r.get("verified_ts", ""), reverse=True)
    return rows[:limit]


def get_pending_predictions() -> list:
    """Predictions logged but not yet verified (awaiting their target time)."""
    if not LIVE_LOG_FILE.exists():
        return []
    rows = []
    with open(LIVE_LOG_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("correct") is None:
                    rows.append(r)
            except Exception:
                pass
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# DAILY CALIBRATION  (predicted vs actual — full history per day)
# ─────────────────────────────────────────────────────────────────────────────

def get_daily_history(target_date: str = None) -> pd.DataFrame:
    """
    Full prediction history for a given day (default: today).
    Returns DataFrame with: time, horizon, direction, confidence,
    entry_price, target_price, actual_price, correct, move_pct.

    This is the predicted-vs-actual record for calibration analysis.
    """
    if target_date is None:
        target_date = date.today().isoformat()
    if not LIVE_LOG_FILE.exists():
        return pd.DataFrame()

    rows = []
    with open(LIVE_LOG_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("ts", "")[:10] == target_date:
                    rows.append(r)
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time"]      = df["ts"].str[11:16]
    df["pred_dir"]  = df["direction"].map({1: "↑ Up", 0: "↓ Down"})
    df["status"]    = df["correct"].map({1: "✅ Correct", 0: "❌ Wrong"})
    df["status"]    = df["status"].fillna("⏳ Pending")
    return df.sort_values("ts", ascending=False).reset_index(drop=True)


def get_calibration_summary(target_date: str = None) -> dict:
    """
    Calibration metrics for a day: for each confidence bucket, what was the
    actual hit rate? A well-calibrated model's 70%-confidence predictions
    should be correct ~70% of the time.

    Returns dict with per-horizon accuracy + confidence-bucket calibration.
    """
    df = get_daily_history(target_date)
    if len(df) == 0:
        return {}

    verified = df[df["correct"].notna()].copy()
    if len(verified) == 0:
        return {"n_total": len(df), "n_verified": 0}

    verified["correct"] = pd.to_numeric(verified["correct"], errors="coerce")
    verified["confidence"] = pd.to_numeric(verified["confidence"], errors="coerce")

    # Per-horizon accuracy
    per_horizon = {}
    for hz in verified["horizon"].unique():
        sub = verified[verified["horizon"] == hz]
        per_horizon[hz] = {
            "n":        len(sub),
            "accuracy": round(float(sub["correct"].mean()) * 100, 1),
        }

    # Confidence-bucket calibration
    buckets = {"50-60%": (0.50, 0.60), "60-70%": (0.60, 0.70),
               "70-80%": (0.70, 0.80), "80-100%": (0.80, 1.01)}
    calibration = {}
    for name, (lo, hi) in buckets.items():
        sub = verified[(verified["confidence"] >= lo) & (verified["confidence"] < hi)]
        if len(sub) > 0:
            calibration[name] = {
                "n":            len(sub),
                "actual_acc":   round(float(sub["correct"].mean()) * 100, 1),
                "expected_acc": round((lo + hi) / 2 * 100, 0),
            }

    return {
        "n_total":     len(df),
        "n_verified":  len(verified),
        "overall_acc": round(float(verified["correct"].mean()) * 100, 1),
        "per_horizon": per_horizon,
        "calibration": calibration,
    }


def list_history_dates() -> list:
    """All dates that have logged predictions (for the history dropdown)."""
    if not LIVE_LOG_FILE.exists():
        return []
    dates = set()
    with open(LIVE_LOG_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                dates.add(r.get("ts", "")[:10])
            except Exception:
                pass
    return sorted([d for d in dates if d], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# END-OF-DAY RETRAINING
# ─────────────────────────────────────────────────────────────────────────────

def should_retrain_today(model_dir: str = "models") -> bool:
    meta_file = Path(model_dir) / "intraday_metadata.json"
    if not meta_file.exists():
        return True
    try:
        with open(meta_file) as f:
            meta = json.load(f)
        return meta.get("trained_at", "")[:10] != date.today().isoformat()
    except Exception:
        return True


def run_eod_retrain(breeze, model_dir: str = "models", verbose: bool = True) -> dict:
    import data_fetcher as df_mod
    import intraday_predictor as ip
    if verbose:
        print("[EOD] Fetching latest 5-min candles…")
    df_5min = df_mod.load_intraday_data(breeze, force_refresh=True)
    if df_5min is None or len(df_5min) < 100:
        return {"error": "Not enough intraday data"}
    if verbose:
        print(f"[EOD] Retraining on {len(df_5min)} candles…")
    return ip.train_intraday_models(df_5min, model_dir=model_dir, verbose=verbose)
