"""
lstm_intraday.py — LSTM-based 10-minute intraday forecast for Nifty.

What it does:
  1. Reads the 5-min OHLCV history fetched by fetch_5min_history.py.
  2. Aggregates 5-min → 10-min bars (open, high, low, close, volume).
  3. Builds (X_seq, X_static, y) training tuples per trading day.
  4. Trains a sequence-to-sequence LSTM that maps:
       past_N_10min_bars + daily_features  →  next 38 ten-min OHLC bars
  5. Saves model + scalers under models/lstm/.
  6. predict_next_day() returns a DataFrame of predicted 10-min OHLC for the
     upcoming trading session (9:15–15:25, 38 bars at 10-min cadence).

The LSTM is a SECOND prediction layer alongside the existing XGBoost daily
model — they complement each other:
  - XGBoost: gap / close / high / low direction + magnitude for the whole day.
  - LSTM:    minute-by-minute path through the session.

Usage (CLI):
    python lstm_intraday.py train      # fit + save (30 min – 2 hrs)
    python lstm_intraday.py predict    # quick test of the saved model

Import (from app.py):
    from lstm_intraday import predict_next_day_intraday
    df_pred = predict_next_day_intraday(daily_features_row, news_score)
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[lstm] PyTorch not installed. Run: pip install torch")

DATA_FILE     = Path("data") / "intraday_5min_3yr.csv"
MODEL_DIR     = Path("models") / "lstm"
MODEL_FILE    = MODEL_DIR / "lstm_intraday.pt"
SCALER_FILE   = MODEL_DIR / "scalers.pt"
META_FILE     = MODEL_DIR / "meta.json"

BARS_PER_DAY    = 38           # 9:15..15:25 at 10-min cadence
SEQ_BARS        = 76           # lookback = 2 trading days
PRED_BARS       = BARS_PER_DAY
N_BAR_FEATURES  = 5            # open_ret, high_ret, low_ret, close_ret, vol_log
N_STATIC_FEATS  = 8            # see _build_static_features
HIDDEN          = 64
LSTM_LAYERS     = 2
BATCH           = 64
EPOCHS          = 40
LR              = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREP
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_to_10min(df_5: pd.DataFrame) -> pd.DataFrame:
    """5-min OHLCV → 10-min OHLCV grouped into floor('10min') buckets."""
    df = df_5.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["bucket"] = df["date"].dt.floor("10min")
    g = df.groupby("bucket", as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low",  "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).rename(columns={"bucket": "date"})
    g = g.sort_values("date").reset_index(drop=True)
    # Keep only market-hour bars: 9:15..15:25
    h, m = g["date"].dt.hour, g["date"].dt.minute
    g = g[(h > 9) | ((h == 9) & (m >= 15))]
    g = g[(h < 15) | ((h == 15) & (m <= 25))]
    return g.reset_index(drop=True)


def _bar_features(df_10: pd.DataFrame) -> np.ndarray:
    """Per-10min-bar features used by the LSTM encoder. Shape: (N, n_features)."""
    o, h, l, c, v = (df_10[k].values for k in ("open", "high", "low", "close", "volume"))
    bar_open = o
    eps = 1e-9
    open_ret  = (o - bar_open) / (bar_open + eps)
    high_ret  = (h - bar_open) / (bar_open + eps)
    low_ret   = (l - bar_open) / (bar_open + eps)
    close_ret = (c - bar_open) / (bar_open + eps)
    vol_log   = np.log1p(v.astype(float))
    return np.stack([open_ret, high_ret, low_ret, close_ret, vol_log], axis=1)


def _build_static_features(day_row: pd.Series, news_score: float = 0.0) -> np.ndarray:
    """Static per-day features prepended to the LSTM (gap, regime, news, etc.).
    Length must equal N_STATIC_FEATS — update in lockstep."""
    feats = [
        float(day_row.get("ret_1d",        0.0)),
        float(day_row.get("ret_5d",        0.0)),
        float(day_row.get("india_vix",    16.0)) / 50.0,    # normalised
        float(day_row.get("fii_net",       0.0)) / 5000.0,  # normalised cr
        float(day_row.get("above_ema21",   1.0)),
        float(day_row.get("c_vs_ema21",    0.0)),
        float(day_row.get("day_of_week",   2)) / 4.0,
        float(news_score),
    ]
    return np.array(feats, dtype=np.float32)


def build_training_tensors(df_10: pd.DataFrame,
                           daily_feat_df: pd.DataFrame | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_seq, X_static, y) where
      X_seq    : (N_days, SEQ_BARS, N_BAR_FEATURES)
      X_static : (N_days, N_STATIC_FEATS)
      y        : (N_days, PRED_BARS, 4)   — open_ret, high_ret, low_ret, close_ret
                 measured vs each day's first bar's open
    """
    df = df_10.copy()
    df["trading_date"] = df["date"].dt.date
    days = sorted(df["trading_date"].unique())

    bar_feats = _bar_features(df)
    df_idx = df.reset_index()

    X_seq_list, X_stat_list, y_list = [], [], []
    for i, day in enumerate(days):
        if i < 2:
            continue   # need 2 prior days of context
        prev_days = days[max(0, i - 5):i]            # gather up to 5 prior days
        seq_mask = df["trading_date"].isin(prev_days)
        seq_feats = bar_feats[seq_mask.values][-SEQ_BARS:]
        if len(seq_feats) < SEQ_BARS:
            continue

        today_mask = df["trading_date"] == day
        today_bars = df[today_mask]
        if len(today_bars) < PRED_BARS:
            continue
        today_bars = today_bars.iloc[:PRED_BARS]

        day_open = float(today_bars["open"].iloc[0])
        eps = 1e-9
        y_bars = np.stack([
            (today_bars["open"].values  - day_open) / (day_open + eps),
            (today_bars["high"].values  - day_open) / (day_open + eps),
            (today_bars["low"].values   - day_open) / (day_open + eps),
            (today_bars["close"].values - day_open) / (day_open + eps),
        ], axis=1)

        static = np.zeros(N_STATIC_FEATS, dtype=np.float32)
        if daily_feat_df is not None:
            row = daily_feat_df[daily_feat_df["date"].dt.date == day]
            if len(row):
                static = _build_static_features(row.iloc[0])

        X_seq_list.append(seq_feats.astype(np.float32))
        X_stat_list.append(static)
        y_list.append(y_bars.astype(np.float32))

    if not y_list:
        raise RuntimeError("No training samples built — check 5-min data depth.")
    return np.stack(X_seq_list), np.stack(X_stat_list), np.stack(y_list)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_OK:
    class IntraDayLSTM(nn.Module):
        def __init__(self, n_bar_feats=N_BAR_FEATURES, n_static=N_STATIC_FEATS,
                     hidden=HIDDEN, layers=LSTM_LAYERS, pred_bars=PRED_BARS):
            super().__init__()
            self.lstm = nn.LSTM(n_bar_feats, hidden, layers,
                                batch_first=True, dropout=0.15)
            self.static_proj = nn.Sequential(
                nn.Linear(n_static, hidden), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden * 2, 256), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(256, pred_bars * 4),
            )
            self.pred_bars = pred_bars

        def forward(self, x_seq, x_stat):
            _, (h, _) = self.lstm(x_seq)
            h_last = h[-1]                                # (B, hidden)
            s = self.static_proj(x_stat)                  # (B, hidden)
            z = torch.cat([h_last, s], dim=-1)
            out = self.decoder(z)                         # (B, pred_bars*4)
            return out.view(-1, self.pred_bars, 4)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_cli():
    if not TORCH_OK:
        print("PyTorch missing. pip install torch")
        return 1
    if not DATA_FILE.exists():
        print(f"Missing {DATA_FILE}. Run fetch_5min_history.py first.")
        return 1

    print(f"[train] reading {DATA_FILE}…")
    df5 = pd.read_csv(DATA_FILE, parse_dates=["date"])
    print(f"[train] 5-min bars: {len(df5):,}")

    df10 = aggregate_to_10min(df5)
    print(f"[train] 10-min bars: {len(df10):,}  ({df10['date'].dt.date.nunique()} trading days)")

    # Build static features from a cached daily feature file if available
    daily_feat_df = None
    daily_csv = Path("data") / "nifty_ohlcv.csv"
    if daily_csv.exists():
        try:
            daily_feat_df = pd.read_csv(daily_csv, parse_dates=["date"])
        except Exception:
            pass

    X_seq, X_stat, y = build_training_tensors(df10, daily_feat_df)
    print(f"[train] samples: X_seq={X_seq.shape}  X_stat={X_stat.shape}  y={y.shape}")

    # Normalise the LSTM bar inputs (per-feature mean/std on training portion)
    n_train = int(len(X_seq) * 0.85)
    bar_mean = X_seq[:n_train].reshape(-1, N_BAR_FEATURES).mean(axis=0)
    bar_std  = X_seq[:n_train].reshape(-1, N_BAR_FEATURES).std(axis=0) + 1e-9
    X_seq = (X_seq - bar_mean) / bar_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    model = IntraDayLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    Xs_t = torch.tensor(X_seq[:n_train],  dtype=torch.float32, device=device)
    Xt_t = torch.tensor(X_stat[:n_train], dtype=torch.float32, device=device)
    y_t  = torch.tensor(y[:n_train],      dtype=torch.float32, device=device)
    Xs_v = torch.tensor(X_seq[n_train:],  dtype=torch.float32, device=device)
    Xt_v = torch.tensor(X_stat[n_train:], dtype=torch.float32, device=device)
    y_v  = torch.tensor(y[n_train:],      dtype=torch.float32, device=device)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xs_t))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            pred = model(Xs_t[idx], Xt_t[idx])
            loss = loss_fn(pred, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        train_l = total / len(perm)

        model.eval()
        with torch.no_grad():
            vpred = model(Xs_v, Xt_v)
            val_l = loss_fn(vpred, y_v).item()
            # Direction accuracy on close_ret
            dir_acc = (torch.sign(vpred[..., 3]) == torch.sign(y_v[..., 3])).float().mean().item()
        print(f"epoch {epoch+1:02d}/{EPOCHS}  train={train_l:.5f}  val={val_l:.5f}  "
              f"close-dir-acc={dir_acc:.2%}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE)
    torch.save({"bar_mean": bar_mean, "bar_std": bar_std}, SCALER_FILE)
    META_FILE.write_text(json.dumps({
        "n_train": int(n_train), "n_val": int(len(X_seq) - n_train),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seq_bars": SEQ_BARS, "pred_bars": PRED_BARS,
        "n_bar_features": N_BAR_FEATURES, "n_static": N_STATIC_FEATS,
    }, indent=2))
    print(f"\n[train] saved → {MODEL_FILE}, {SCALER_FILE}, {META_FILE}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_next_day_intraday(daily_feat_row: pd.Series,
                              news_score: float = 0.0,
                              ref_open_price: float | None = None,
                              df_10_recent: pd.DataFrame | None = None
                              ) -> pd.DataFrame | None:
    """
    Predict the next session's 10-minute OHLC path.

    Parameters
    ----------
    daily_feat_row : a row from feature_engineering.build_features()'s output
                     (the latest row — used for static features).
    news_score     : sentiment score in [-1, 1].
    ref_open_price : anchor for converting predicted returns to ₹ prices.
                     Defaults to daily_feat_row['close'].
    df_10_recent   : optional 10-min DataFrame for the encoder sequence.
                     If None we read from the cached 5-min file.

    Returns
    -------
    DataFrame with columns: time, open, high, low, close (₹), or None if the
    LSTM model isn't trained yet.
    """
    if not TORCH_OK or not MODEL_FILE.exists():
        return None

    if df_10_recent is None:
        if not DATA_FILE.exists():
            return None
        df5 = pd.read_csv(DATA_FILE, parse_dates=["date"])
        df_10_recent = aggregate_to_10min(df5)

    bar_feats = _bar_features(df_10_recent)
    if len(bar_feats) < SEQ_BARS:
        return None
    seq = bar_feats[-SEQ_BARS:].astype(np.float32)

    scalers = torch.load(SCALER_FILE, weights_only=False)
    seq = (seq - scalers["bar_mean"]) / scalers["bar_std"]

    static = _build_static_features(daily_feat_row, news_score)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntraDayLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=False))
    model.eval()

    with torch.no_grad():
        pred = model(
            torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(static, dtype=torch.float32, device=device).unsqueeze(0),
        ).cpu().numpy()[0]                    # (PRED_BARS, 4)

    # Reconstruct ₹ prices from returns
    if ref_open_price is None:
        ref_open_price = float(daily_feat_row.get("close", 23000.0))
    open_ret, high_ret, low_ret, close_ret = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    o_px = ref_open_price * (1 + open_ret)
    h_px = ref_open_price * (1 + high_ret)
    l_px = ref_open_price * (1 + low_ret)
    c_px = ref_open_price * (1 + close_ret)

    # Build the bar timestamps for the next trading day
    next_day = (pd.Timestamp(date.today()) + pd.Timedelta(days=1))
    times = pd.date_range(
        start=next_day.replace(hour=9, minute=15),
        periods=PRED_BARS, freq="10min",
    )
    return pd.DataFrame({
        "time":  times,
        "open":  np.round(o_px, 2),
        "high":  np.round(h_px, 2),
        "low":   np.round(l_px, 2),
        "close": np.round(c_px, 2),
    })


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY-SHAPE PREDICT — drop-in for model_trainer.predict_today()
# Derives daily-level metrics (open/close/high/low + direction + confidence)
# from the 38-bar 10-minute forecast.
# ─────────────────────────────────────────────────────────────────────────────

def predict_today_compat(feature_df: pd.DataFrame, news_score: float = 0.0
                         ) -> dict:
    """Mirror model_trainer.predict_today's return shape using ONLY the LSTM."""
    if not TORCH_OK or not MODEL_FILE.exists():
        raise RuntimeError(
            "LSTM model not trained yet. Run:\n"
            "  python fetch_5min_history.py\n"
            "  python lstm_intraday.py train"
        )

    last_row   = feature_df.iloc[-1]
    last_close = float(feature_df["close"].iloc[-1]) if "close" in feature_df else 23000.0
    atr_pct    = float(last_row.get("atr_pct",   0.8))
    india_vix  = float(last_row.get("india_vix", 16.0))

    df_pred = predict_next_day_intraday(
        last_row, news_score=news_score, ref_open_price=last_close,
    )
    if df_pred is None or df_pred.empty:
        raise RuntimeError("LSTM produced no forecast — check 5-min cache.")

    pred_open  = float(df_pred["open"].iloc[0])
    pred_close = float(df_pred["close"].iloc[-1])
    pred_high  = float(df_pred["high"].max())
    pred_low   = float(df_pred["low"].min())

    open_ret_pct  = (pred_open  - last_close) / (last_close + 1e-9) * 100
    close_ret_pct = (pred_close - pred_open)  / (pred_open  + 1e-9) * 100
    high_ret_pct  = (pred_high  - pred_open)  / (pred_open  + 1e-9) * 100
    low_ret_pct   = (pred_low   - pred_open)  / (pred_open  + 1e-9) * 100

    open_dir  = int(pred_open  > last_close * 1.0015)
    close_dir = int(pred_close > pred_open)

    # Confidence proxy: magnitude of predicted return / typical daily ATR.
    open_conf  = min(0.99, 0.50 + abs(open_ret_pct)  / max(atr_pct * 2, 1e-6) * 0.20)
    close_conf = min(0.99, 0.50 + abs(close_ret_pct) / max(atr_pct * 2, 1e-6) * 0.20)

    return {
        "open_direction":   open_dir,
        "open_confidence":  round(open_conf, 4),
        "open_pred_pct":    round(open_ret_pct, 3),
        "open_range":       (round(pred_open * (1 - atr_pct * 0.25 / 100)),
                             round(pred_open * (1 + atr_pct * 0.25 / 100))),
        "open_agree":       True,

        "close_direction":  close_dir,
        "close_confidence": round(close_conf, 4),
        "close_pred_pct":   round(close_ret_pct, 3),
        "close_range":      (round(pred_close * (1 - atr_pct * 0.35 / 100)),
                             round(pred_close * (1 + atr_pct * 0.35 / 100))),
        "close_agree":      True,

        "ensemble_agree":   True,
        "trade_signal":     "BUY_CE" if close_dir == 1 else "BUY_PE",
        "signal_reason":    "",

        "last_close":       round(last_close, 2),
        "predicted_open":   round(pred_open,  2),
        "predicted_close":  round(pred_close, 2),

        "high_pred_pct":    round(high_ret_pct, 3),
        "low_pred_pct":     round(low_ret_pct, 3),
        "predicted_high":   round(pred_high),
        "predicted_low":    round(pred_low),
        "daily_range":      (round(pred_low), round(pred_high)),

        "atr_pct":          round(atr_pct, 3),
        "india_vix":        round(india_vix, 2),

        # Legacy keys for options_engine compat
        "direction":          close_dir,
        "confidence":         round(close_conf, 4),
        "predicted_move_pct": round(abs(close_ret_pct), 3),

        # Carry the full 10-min path for the UI
        "_intraday_path":     df_pred,
        "_model_backend":     "lstm",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "predict"])
    args = p.parse_args()
    if args.mode == "train":
        return train_cli()
    else:
        # quick smoke-test prediction
        if not DATA_FILE.exists():
            print(f"Need {DATA_FILE} first.")
            return 1
        df5 = pd.read_csv(DATA_FILE, parse_dates=["date"])
        df10 = aggregate_to_10min(df5)
        fake_row = pd.Series({"close": float(df10["close"].iloc[-1]),
                              "ret_1d": 0.0, "ret_5d": 0.0,
                              "india_vix": 14.0, "fii_net": 0.0,
                              "above_ema21": 1.0, "c_vs_ema21": 0.5,
                              "day_of_week": 2})
        df_pred = predict_next_day_intraday(fake_row, news_score=0.0,
                                            df_10_recent=df10)
        if df_pred is None:
            print("Model not trained yet.")
            return 1
        print(df_pred.head(10))
        print(f"… ({len(df_pred)} bars total)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
