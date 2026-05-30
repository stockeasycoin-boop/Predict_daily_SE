"""
tracker.py — Trade log, accuracy tracking, and P&L computation.
All trades are stored in trades/trades.csv.
You update the outcome at end of day (via the dashboard or manually).
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
from settings import TRADES_FILE, NIFTY_LOT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

TRADE_COLS = [
    "date",
    "signal",           # BUY_CE / BUY_PE / NO_TRADE
    "option_type",      # CE / PE
    "strike",
    "expiry",
    "confidence",       # Model confidence at signal time
    "spot_price",       # Nifty spot at signal time
    "premium_entry",    # Recommended entry premium
    "target_premium",   # Target exit premium
    "sl_premium",       # Stop-loss premium
    "lots",
    "capital_used",
    "predicted_direction",   # 1=bullish 0=bearish
    "actual_direction",      # Filled at day-end: 1=bullish 0=bearish
    "exit_premium",          # Actual exit premium (filled at day-end)
    "exit_reason",           # TARGET / STOP_LOSS / TIME_EXIT / MANUAL
    "pnl",                   # Realised P&L in ₹ (filled at day-end)
    "direction_correct",     # 1 if direction matched, 0 if not
    "notes",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD / SAVE
# ─────────────────────────────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    """Load trade log. Returns empty DataFrame if file does not exist."""
    TRADES_FILE.parent.mkdir(exist_ok=True)
    if TRADES_FILE.exists():
        df = pd.read_csv(TRADES_FILE)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    return pd.DataFrame(columns=TRADE_COLS)


def _save(df: pd.DataFrame) -> None:
    df.to_csv(TRADES_FILE, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# WRITE SUGGESTION
# ─────────────────────────────────────────────────────────────────────────────

def log_suggestion(suggestion: dict, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Append today's suggestion to the trade log.
    Only writes if signal is BUY_CE or BUY_PE (NO_TRADE is not logged).
    Skips if today already has an entry.
    """
    if suggestion.get("signal") == "NO_TRADE":
        return df if df is not None else load_trades()

    if df is None:
        df = load_trades()

    today = date.today()
    if len(df) > 0 and today in df["date"].values:
        return df   # Already logged today

    row = {
        "date":               today,
        "signal":             suggestion.get("signal", ""),
        "option_type":        suggestion.get("option_type", ""),
        "strike":             suggestion.get("strike", ""),
        "expiry":             suggestion.get("expiry", ""),
        "confidence":         round(float(suggestion.get("confidence", 0)), 4),
        "spot_price":         suggestion.get("spot_price", 0),
        "premium_entry":      suggestion.get("premium_entry", 0),
        "target_premium":     suggestion.get("target_premium", 0),
        "sl_premium":         suggestion.get("sl_premium", 0),
        "lots":               suggestion.get("lots", 1),
        "capital_used":       suggestion.get("capital_used", 0),
        "predicted_direction": suggestion.get("direction", ""),
        "actual_direction":   "",
        "exit_premium":       "",
        "exit_reason":        "",
        "pnl":                "",
        "direction_correct":  "",
        "notes":              "",
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE OUTCOME (filled at end of day)
# ─────────────────────────────────────────────────────────────────────────────

def update_outcome(
    trade_date,
    actual_nifty_close: float,
    exit_premium: float,
    exit_reason: str,
    notes: str = "",
) -> pd.DataFrame:
    """
    Record the actual result for a given day's trade.

    Parameters
    ----------
    trade_date          : date or 'YYYY-MM-DD' string
    actual_nifty_close  : What Nifty actually closed at (to determine actual direction)
    exit_premium        : The premium at which you exited the option
    exit_reason         : 'TARGET' | 'STOP_LOSS' | 'TIME_EXIT' | 'MANUAL'
    notes               : Optional remarks
    """
    df = load_trades()

    if isinstance(trade_date, str):
        trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()

    mask = df["date"] == trade_date
    if not mask.any():
        print(f"No trade found for {trade_date}")
        return df

    idx = df[mask].index[0]

    # Determine actual direction from Nifty close vs entry spot
    spot  = float(df.loc[idx, "spot_price"])
    actual_dir = 1 if actual_nifty_close > spot else 0

    # P&L
    predicted = int(df.loc[idx, "predicted_direction"])
    entry     = float(df.loc[idx, "premium_entry"])
    lots      = int(df.loc[idx, "lots"])
    pnl       = round((float(exit_premium) - entry) * lots * NIFTY_LOT_SIZE)

    df.loc[idx, "actual_direction"]  = actual_dir
    df.loc[idx, "exit_premium"]      = float(exit_premium)
    df.loc[idx, "exit_reason"]       = exit_reason
    df.loc[idx, "pnl"]               = pnl
    df.loc[idx, "direction_correct"] = int(predicted == actual_dir)
    df.loc[idx, "notes"]             = notes

    _save(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> dict:
    """
    Compute accuracy and P&L statistics from completed trades.
    A trade is 'completed' when direction_correct is filled in.
    """
    empty = {
        "total_trades": 0, "completed": 0,
        "accuracy_all": 0.0, "accuracy_7d": 0.0, "accuracy_30d": 0.0,
        "win_rate": 0.0, "total_pnl": 0,
        "avg_win": 0, "avg_loss": 0, "risk_reward": 0.0,
        "max_drawdown": 0, "streak": 0,
        "cum_pnl": [], "completed_df": pd.DataFrame(),
    }
    if df is None or len(df) == 0:
        return empty

    done = df[df["direction_correct"].notna() & (df["direction_correct"] != "")].copy()
    done["direction_correct"] = pd.to_numeric(done["direction_correct"], errors="coerce")
    done["pnl"]               = pd.to_numeric(done["pnl"],               errors="coerce")
    done = done.dropna(subset=["direction_correct", "pnl"])

    if len(done) == 0:
        return {**empty, "total_trades": len(df)}

    n        = len(done)
    acc_all  = round(float(done["direction_correct"].mean()) * 100, 1)
    acc_7d   = round(float(done.tail(7)["direction_correct"].mean()) * 100, 1)
    acc_30d  = round(float(done.tail(30)["direction_correct"].mean()) * 100, 1)

    winners  = done[done["pnl"] > 0]
    losers   = done[done["pnl"] <= 0]
    win_rate = round(len(winners) / n * 100, 1)
    avg_win  = round(float(winners["pnl"].mean())) if len(winners) > 0 else 0
    avg_loss = round(float(losers["pnl"].mean()))  if len(losers)  > 0 else 0
    rr       = round(abs(avg_win / avg_loss), 2)   if avg_loss != 0 else 0.0

    cum = done["pnl"].cumsum().tolist()
    peak = max(cum) if cum else 0
    dd   = max(peak - v for v in cum) if cum else 0

    # Current streak (consecutive wins or losses)
    corr = done["direction_correct"].tolist()
    streak = 0
    if corr:
        last = corr[-1]
        for v in reversed(corr):
            if v == last:
                streak += 1
            else:
                break
        streak = streak if last == 1 else -streak

    return {
        "total_trades":   len(df),
        "completed":      n,
        "accuracy_all":   acc_all,
        "accuracy_7d":    acc_7d,
        "accuracy_30d":   acc_30d,
        "win_rate":       win_rate,
        "total_pnl":      int(done["pnl"].sum()),
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "risk_reward":    rr,
        "max_drawdown":   int(dd),
        "streak":         streak,
        "cum_pnl":        cum,
        "completed_df":   done,
    }
