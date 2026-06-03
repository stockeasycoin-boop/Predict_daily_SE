"""
arima_model.py — ARIMA / SARIMAX next-day OHLC forecaster.

Fast classical alternative to the LSTM. Trains in seconds, runs inside the
Streamlit request (no separate training CLI needed). Uses statsmodels SARIMAX
with exogenous variables so the model can react to VIX, FII flows, and the
day's news sentiment.

Returns the SAME dict shape as model_trainer.predict_today() and
lstm_intraday.predict_today_compat(), so app.py can swap backends with no
other code changes.
"""

from __future__ import annotations
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SM_OK = True
except ImportError:
    SM_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# CORE — fit SARIMAX on log-returns of daily close, optionally with exog vars.
# Returns mean and std of the next-step log-return forecast.
# ─────────────────────────────────────────────────────────────────────────────

def _fit_and_forecast(
    series: np.ndarray,
    exog: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    order=(1, 0, 1),
    seasonal_order=(0, 0, 0, 0),
) -> tuple[float, float]:
    """Returns (forecast_mean, forecast_std) for one step ahead."""
    if not SM_OK:
        raise ImportError("statsmodels not installed. Run: pip install statsmodels")
    model = SARIMAX(
        series, exog=exog, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=1, exog=exog_future)
    mean = float(np.atleast_1d(fc.predicted_mean)[0])
    se   = float(np.sqrt(np.atleast_1d(fc.var_pred_mean)[0]))
    return mean, se


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in predict — same dict shape as predict_today_compat()
# ─────────────────────────────────────────────────────────────────────────────

def predict_today_compat(feature_df: pd.DataFrame, news_score: float = 0.0,
                         use_seasonal: bool = False) -> dict:
    """
    ARIMA(1,0,1) or SARIMA(1,0,1)(1,0,1,5) forecast for next-day OHLC.

    We forecast in LOG-RETURN space (stable, stationary-ish) and reconstruct
    prices. Exogenous regressors: india_vix, fii_net, news_score.
    """
    if not SM_OK:
        raise RuntimeError(
            "statsmodels missing. Add `statsmodels>=0.14` to requirements.txt "
            "and `pip install statsmodels`."
        )
    if feature_df is None or len(feature_df) < 100:
        raise RuntimeError("Need at least 100 daily rows for ARIMA fit.")

    df = feature_df.dropna(subset=["close"]).copy()
    last_close = float(df["close"].iloc[-1])

    # Log-returns of close
    log_close = np.log(df["close"].values)
    log_ret   = np.diff(log_close)

    # Exogenous: VIX (level), FII (cr), news (current value)
    exog_full = np.column_stack([
        df["india_vix"].fillna(16.0).values[1:],
        df.get("fii_net", pd.Series([0.0] * len(df))).fillna(0).values[1:] / 5000.0,
        np.full(len(df) - 1, news_score),
    ])

    order = (1, 0, 1)
    seasonal = (1, 0, 1, 5) if use_seasonal else (0, 0, 0, 0)

    # Forecast next-day close log-return
    mean_lr, se_lr = _fit_and_forecast(
        log_ret,
        exog=exog_full,
        exog_future=np.array([[
            float(df["india_vix"].iloc[-1] or 16.0),
            float(df.get("fii_net", pd.Series([0.0])).iloc[-1] or 0) / 5000.0,
            float(news_score),
        ]]),
        order=order, seasonal_order=seasonal,
    )

    pred_close = last_close * float(np.exp(mean_lr))
    # Approximate open as last_close × exp(½ predicted return) (drift midpoint)
    pred_open  = last_close * float(np.exp(mean_lr * 0.4))
    # Approximate intraday range via forecast std translated to %
    atr_pct    = float(df.get("atr_pct", pd.Series([0.8])).iloc[-1])
    band       = max(atr_pct, se_lr * 100 * 0.8)
    pred_high  = max(pred_open, pred_close) * (1 + band / 100 * 0.5)
    pred_low   = min(pred_open, pred_close) * (1 - band / 100 * 0.5)

    open_ret_pct  = (pred_open  - last_close) / last_close * 100
    close_ret_pct = (pred_close - pred_open)  / pred_open  * 100
    high_ret_pct  = (pred_high  - pred_open)  / pred_open  * 100
    low_ret_pct   = (pred_low   - pred_open)  / pred_open  * 100

    open_dir  = int(pred_open  > last_close * 1.0015)
    close_dir = int(pred_close > pred_open)

    # Confidence from SE — narrower forecast band ⇒ higher conviction
    z = abs(mean_lr) / max(se_lr, 1e-6)
    conf = min(0.99, 0.50 + 0.18 * float(np.tanh(z)))

    india_vix = float(df.get("india_vix", pd.Series([16.0])).iloc[-1])
    return {
        "open_direction":   open_dir,
        "open_confidence":  round(conf, 4),
        "open_pred_pct":    round(open_ret_pct, 3),
        "open_range":       (round(pred_open * (1 - atr_pct * 0.25 / 100)),
                             round(pred_open * (1 + atr_pct * 0.25 / 100))),
        "open_agree":       True,

        "close_direction":  close_dir,
        "close_confidence": round(conf, 4),
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
        "low_pred_pct":     round(low_ret_pct,  3),
        "predicted_high":   round(pred_high),
        "predicted_low":    round(pred_low),
        "daily_range":      (round(pred_low), round(pred_high)),

        "atr_pct":          round(atr_pct, 3),
        "india_vix":        round(india_vix, 2),

        "direction":          close_dir,
        "confidence":         round(conf, 4),
        "predicted_move_pct": round(abs(close_ret_pct), 3),

        "_model_backend":     "sarima" if use_seasonal else "arima",
        "_forecast_se":       round(se_lr, 6),
    }
