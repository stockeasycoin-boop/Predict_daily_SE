"""
feature_engineering.py — Full feature matrix for open + close prediction.

Feature groups:
  A. Daily OHLCV — gap, candle shape, returns, moving averages, oscillators
  B. Intraday pattern — yesterday's 5-min candle features (morning/afternoon trend,
     volume profile, reversal patterns)
  C. Bank Nifty + sector — spread, divergence, lead/lag signals
  D. VIX — level, change, spike/calm flags
  E. FII/DII — net flows, trend, derivatives positioning
  F. GIFT Nifty — pre-market gap signal
  G. PCR — contrarian sentiment
  H. Calendar — expiry proximity, day-of-week effects

Two targets:
  open_target  : 1 = next-day open > today's close by ≥0.15% (gap-up)
  close_target : 1 = next-day close > next-day open (intraday bullish)
  close_ret    : float — predicted open-to-close return % (regression)
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _macd(s: pd.Series, f=12, sl=26, sig=9):
    line = _ema(s, f) - _ema(s, sl)
    signal = _ema(line, sig)
    return line, signal, line - signal

def _atr(h, l, c, n=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _days_to_tuesday(d) -> int:
    """Days until next Tuesday (Nifty weekly expiry)."""
    if hasattr(d, "date"): d = d.date()
    elif isinstance(d, str): d = pd.to_datetime(d).date()
    n = (1 - d.weekday()) % 7
    return n if n > 0 else 7

# Keep alias for backward compatibility
days_to_thursday = _days_to_tuesday


# ─────────────────────────────────────────────────────────────────────────────
# GROUP B — INTRADAY PATTERN FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_intraday_features(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-day intraday features from 5-min candles.

    Features extracted:
      morning_ret     : 9:15–11:00 return (first 2hrs direction)
      afternoon_ret   : 11:00–15:30 return (afternoon direction)
      morning_vol_ratio: morning volume vs full-day volume
      reversal        : 1 if morning and afternoon had opposite directions
      breakout        : 1 if close > intraday high of first hour
      close_strength  : close position within day's range (0=low, 1=high)
      intraday_range  : (high-low)/open %
      vol_spike_time  : hour of highest volume (buyer/seller urgency timing)
    """
    if intraday_df is None or len(intraday_df) < 10:
        return pd.DataFrame()

    df = intraday_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["trading_date"] = df["date"].dt.normalize()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute

    records = []
    for day, grp in df.groupby("trading_date"):
        if len(grp) < 10:
            continue
        grp = grp.sort_values("date")

        day_open  = grp["open"].iloc[0]
        day_close = grp["close"].iloc[-1]
        day_high  = grp["high"].max()
        day_low   = grp["low"].min()
        day_vol   = grp["volume"].sum()

        # Morning session: 9:15 to 11:00
        morning = grp[(grp["hour"] == 9) | (grp["hour"] == 10)]
        morning_ret = 0.0
        morning_vol = 0.0
        first_hr_high = day_high
        if len(morning) > 0:
            morning_ret = (morning["close"].iloc[-1] - morning["open"].iloc[0]) / (morning["open"].iloc[0] + 1e-9) * 100
            morning_vol = morning["volume"].sum()
            first_hr    = grp[grp["hour"] == 9]
            if len(first_hr) > 0:
                first_hr_high = first_hr["high"].max()

        # Afternoon session: 11:00 to 15:30
        afternoon = grp[grp["hour"] >= 11]
        afternoon_ret = 0.0
        if len(afternoon) > 0:
            afternoon_ret = (afternoon["close"].iloc[-1] - afternoon["open"].iloc[0]) / (afternoon["open"].iloc[0] + 1e-9) * 100

        # Derived features
        reversal     = int(np.sign(morning_ret) != np.sign(afternoon_ret) and abs(morning_ret) > 0.1)
        breakout     = int(day_close > first_hr_high * 1.001)
        close_str    = (day_close - day_low) / (day_high - day_low + 1e-9)
        intra_range  = (day_high - day_low) / (day_open + 1e-9) * 100
        morn_vol_r   = morning_vol / (day_vol + 1e-9)

        # Hour of peak volume (urgency signal)
        vol_by_hour  = grp.groupby("hour")["volume"].sum()
        peak_hr      = int(vol_by_hour.idxmax()) if len(vol_by_hour) > 0 else 11

        records.append({
            "date":             pd.Timestamp(day),
            "morning_ret":      round(morning_ret, 4),
            "afternoon_ret":    round(afternoon_ret, 4),
            "morning_vol_ratio":round(morn_vol_r, 4),
            "reversal":         reversal,
            "breakout":         breakout,
            "close_strength":   round(close_str, 4),
            "intraday_range":   round(intra_range, 4),
            "vol_spike_hour":   peak_hr,
            "day_oc_ret":       round((day_close - day_open) / (day_open + 1e-9) * 100, 4),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP C — BANK NIFTY + SECTOR FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_correlated_features(nifty_df: pd.DataFrame,
                                  corr_dict: dict) -> pd.DataFrame:
    """
    Compute Bank Nifty and sector index features relative to Nifty 50.

    Features per instrument:
      {sym}_ret_1d    : 1-day return
      {sym}_vs_nifty  : spread vs Nifty return (divergence signal)
      {sym}_bull      : 1 if above 21-day EMA
      {sym}_leads     : 1 if {sym} moved before Nifty (today lead-lag)
    """
    if not corr_dict:
        return pd.DataFrame()

    base = nifty_df[["date","close"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base["nifty_ret"] = base["close"].pct_change() * 100

    result = base[["date"]].copy()

    for sym, df in corr_dict.items():
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date","open","high","low","close"]].sort_values("date")
        sym_l = sym.lower()[:8]   # shorten for column names

        df[f"{sym_l}_ret"]      = df["close"].pct_change() * 100
        df[f"{sym_l}_oc_ret"]   = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100
        ema21                    = _ema(df["close"], 21)
        df[f"{sym_l}_bull"]     = (df["close"] > ema21).astype(int)
        df[f"{sym_l}_rsi"]      = _rsi(df["close"], 14)

        merged = pd.merge(result, df[["date", f"{sym_l}_ret", f"{sym_l}_oc_ret",
                                       f"{sym_l}_bull", f"{sym_l}_rsi"]],
                          on="date", how="left")
        result = merged

    # Bank Nifty vs Nifty spread (if available)
    bnf_col = next((c for c in result.columns if "banknif" in c and "_ret" in c), None)
    if bnf_col:
        merged2 = pd.merge(result, base[["date","nifty_ret"]], on="date", how="left")
        merged2["bnf_vs_nifty"] = merged2[bnf_col] - merged2["nifty_ret"]
        merged2["bnf_leads_bull"] = (
            (merged2[bnf_col] > 0.3) & (merged2["nifty_ret"] < 0.1)
        ).astype(int)
        merged2["bnf_leads_bear"] = (
            (merged2[bnf_col] < -0.3) & (merged2["nifty_ret"] > -0.1)
        ).astype(int)
        result = merged2.drop(columns=["nifty_ret"])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    nifty_df:     pd.DataFrame,
    vix_df:       pd.DataFrame        = None,
    global_df:    pd.DataFrame        = None,
    fii_df:       pd.DataFrame        = None,
    gift_df:      pd.DataFrame        = None,
    pcr_df:       pd.DataFrame        = None,
    intraday_df:  pd.DataFrame        = None,   # 5-min candles
    corr_dict:    dict                = None,   # {sym: daily_df}
) -> pd.DataFrame:
    """
    Build the full feature matrix.

    Target variables (all based on NEXT trading day):
      open_target   : 1 = next open > today close × 1.0015  (gap-up ≥0.15%)
      close_target  : 1 = next close > next open            (intraday bullish)
      close_ret_pct : float = (next close - next open) / next open × 100
      open_ret_pct  : float = (next open  - today close) / today close × 100
    """
    df = nifty_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    c, h, l, o, vol = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # ── A. DAILY OHLCV FEATURES ──────────────────────────────────────────────

    # Gap vs previous close
    df["prev_close"]     = c.shift(1)
    df["prev_open"]      = o.shift(1)
    df["gap_pct"]        = (o - c.shift(1)) / (c.shift(1) + 1e-9) * 100
    df["gap_up"]         = (df["gap_pct"] >  0.30).astype(int)
    df["gap_down"]       = (df["gap_pct"] < -0.30).astype(int)
    df["gap_size"]       = df["gap_pct"].abs()

    # Candle characteristics
    df["prev_oc_ret"]    = ((c.shift(1) - o.shift(1)) / (o.shift(1) + 1e-9) * 100)
    df["prev_oc_bull"]   = (df["prev_oc_ret"] > 0).astype(int)
    df["prev_range_pct"] = ((h.shift(1) - l.shift(1)) / (o.shift(1) + 1e-9) * 100)
    prev_top             = pd.concat([c.shift(1), o.shift(1)], axis=1).max(axis=1)
    prev_bot             = pd.concat([c.shift(1), o.shift(1)], axis=1).min(axis=1)
    df["prev_wick_up"]   = (h.shift(1) - prev_top) / (o.shift(1) + 1e-9) * 100
    df["prev_wick_dn"]   = (prev_bot - l.shift(1))  / (o.shift(1) + 1e-9) * 100
    df["close_strength"] = (c - l) / (h - l + 1e-9)   # where close is in day range

    # Multi-day returns
    for n in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{n}d"] = c.pct_change(n) * 100

    # Consecutive days up/down
    df["consec_up"]   = (
        df["prev_oc_bull"].rolling(3).sum().fillna(0).astype(int)
    )
    df["consec_down"] = (
        (1 - df["prev_oc_bull"]).rolling(3).sum().fillna(0).astype(int)
    )

    # Trend — EMAs
    for n in [9, 21, 50, 200]:
        df[f"ema_{n}"] = _ema(c, n)
    df["c_vs_ema9"]    = (c - df["ema_9"])   / c * 100
    df["c_vs_ema21"]   = (c - df["ema_21"])  / c * 100
    df["c_vs_ema50"]   = (c - df["ema_50"])  / c * 100
    df["ema9_21"]      = (df["ema_9"] - df["ema_21"]) / df["ema_21"] * 100
    df["above_ema21"]  = (c > df["ema_21"]).astype(int)   # regime filter
    df["above_ema50"]  = (c > df["ema_50"]).astype(int)
    df["above_ema200"] = (c > df["ema_200"]).astype(int)

    # RSI
    df["rsi_14"]       = _rsi(c, 14)
    df["rsi_7"]        = _rsi(c, 7)
    df["rsi_ob"]       = (df["rsi_14"] > 70).astype(int)
    df["rsi_os"]       = (df["rsi_14"] < 30).astype(int)
    df["rsi_cross50"]  = (
        (df["rsi_14"] > 50) & (df["rsi_14"].shift(1) <= 50)
    ).astype(int) - (
        (df["rsi_14"] < 50) & (df["rsi_14"].shift(1) >= 50)
    ).astype(int)

    # MACD
    ml, ms, mh         = _macd(c)
    df["macd_hist"]    = mh
    df["macd_bull"]    = (ml > ms).astype(int)
    df["macd_hist_chg"]= mh.diff()
    df["macd_xbull"]   = ((ml > ms) & (ml.shift(1) <= ms.shift(1))).astype(int)
    df["macd_xbear"]   = ((ml < ms) & (ml.shift(1) >= ms.shift(1))).astype(int)

    # Bollinger
    bb_ma = c.rolling(20).mean()
    bb_std= c.rolling(20).std()
    bb_up = bb_ma + 2 * bb_std
    bb_dn = bb_ma - 2 * bb_std
    df["bb_pct_b"]     = (c - bb_dn) / (bb_up - bb_dn + 1e-9)
    df["bb_width"]     = (bb_up - bb_dn) / (bb_ma + 1e-9)
    df["bb_squeeze"]   = (df["bb_width"] < df["bb_width"].rolling(20).mean()).astype(int)
    df["bb_upper_touch"]= (c > bb_up * 0.998).astype(int)
    df["bb_lower_touch"]= (c < bb_dn * 1.002).astype(int)

    # ATR / volatility
    df["atr_14"]       = _atr(h, l, c, 14)
    df["atr_pct"]      = df["atr_14"] / c * 100
    df["vol_5d"]       = c.pct_change().rolling(5).std() * 100
    df["vol_20d"]      = c.pct_change().rolling(20).std() * 100
    df["vol_regime"]   = df["vol_5d"] / (df["vol_20d"] + 1e-9)

    # Volume
    df["vol_ma5"]      = vol.rolling(5).mean()
    df["vol_ratio"]    = vol / (df["vol_ma5"] + 1e-9)
    df["vol_surge"]    = (df["vol_ratio"] > 1.5).astype(int)

    # Support / resistance proximity
    df["near_52w_high"]= (c >= c.rolling(252).max() * 0.99).astype(int)
    df["near_52w_low"] = (c <= c.rolling(252).min() * 1.01).astype(int)

    # ── B. INTRADAY FEATURES ─────────────────────────────────────────────────
    if intraday_df is not None and len(intraday_df) > 0:
        intra_feat = extract_intraday_features(intraday_df)
        if len(intra_feat) > 0:
            intra_feat["date"] = pd.to_datetime(intra_feat["date"])
            # Shift by 1: yesterday's intraday → today's prediction input
            intra_shifted = intra_feat.copy()
            intra_shifted["date"] = intra_shifted["date"] + pd.Timedelta(days=1)
            # Align to next trading day
            df = pd.merge(df, intra_shifted.add_prefix("prev_intra_").rename(
                columns={"prev_intra_date":"date"}), on="date", how="left")
            # Forward-fill gaps (weekends/holidays)
            intra_cols = [c_ for c_ in df.columns if c_.startswith("prev_intra_")]
            df[intra_cols] = df[intra_cols].ffill()
    else:
        for col in ["prev_intra_morning_ret","prev_intra_afternoon_ret",
                    "prev_intra_reversal","prev_intra_breakout",
                    "prev_intra_close_strength","prev_intra_intraday_range",
                    "prev_intra_day_oc_ret","prev_intra_morning_vol_ratio",
                    "prev_intra_vol_spike_hour"]:
            df[col] = 0.0

    # ── C. BANK NIFTY + SECTOR ───────────────────────────────────────────────
    if corr_dict:
        corr_feat = extract_correlated_features(nifty_df, corr_dict)
        if len(corr_feat) > 0:
            corr_feat["date"] = pd.to_datetime(corr_feat["date"])
            df = pd.merge(df, corr_feat, on="date", how="left")
            corr_cols = [c_ for c_ in df.columns
                         if any(s.lower()[:8] in c_ for s in corr_dict.keys())]
            df[corr_cols] = df[corr_cols].ffill().fillna(0)

    # ── D. INDIA VIX ─────────────────────────────────────────────────────────
    if vix_df is not None:
        vix = vix_df.copy()
        vix["date"] = pd.to_datetime(vix["date"])
        df = pd.merge(df, vix[["date","india_vix"]], on="date", how="left")
        df["india_vix"] = df["india_vix"].ffill().fillna(16.0)
    else:
        df["india_vix"] = 16.0

    df["vix_change"]    = df["india_vix"].diff()
    df["vix_pct_chg"]   = df["india_vix"].pct_change() * 100
    df["vix_spike"]     = (df["vix_change"] >  1.5).astype(int)
    df["vix_calm"]      = (df["vix_change"] < -1.0).astype(int)
    df["vix_high"]      = (df["india_vix"] > 20).astype(int)
    df["vix_extreme"]   = (df["india_vix"] > 25).astype(int)
    df["vix_3d_avg"]    = df["india_vix"].rolling(3).mean()
    df["vix_above_avg"] = (df["india_vix"] > df["vix_3d_avg"]).astype(int)

    # ── E. FII / DII ─────────────────────────────────────────────────────────
    if fii_df is not None:
        fii = fii_df.copy()
        fii["date"] = pd.to_datetime(fii["date"])
        df = pd.merge(df, fii[["date","fii_net","dii_net"]], on="date", how="left")
        df["fii_net"] = df["fii_net"].ffill().fillna(0)
        df["dii_net"] = df["dii_net"].ffill().fillna(0)
        df["fii_bull"]   = (df["fii_net"] >  500).astype(int)
        df["fii_bear"]   = (df["fii_net"] < -500).astype(int)
        df["fii_5d_ma"]  = df["fii_net"].rolling(5).mean().fillna(0)
        df["fii_trend"]  = (df["fii_5d_ma"] > 0).astype(int)
        df["fii_3d_sum"] = df["fii_net"].rolling(3).sum().fillna(0)
    else:
        for col in ["fii_net","dii_net","fii_bull","fii_bear",
                    "fii_5d_ma","fii_trend","fii_3d_sum"]:
            df[col] = 0.0

    # ── F. GIFT NIFTY ─────────────────────────────────────────────────────────
    if gift_df is not None:
        gft = gift_df.copy()
        gft["date"] = pd.to_datetime(gft["date"])
        df = pd.merge(df, gft[["date","gift_close"]], on="date", how="left")
        df["gift_close"] = df["gift_close"].ffill()
        df["gift_vs_prev"] = (df["gift_close"] - c.shift(1)) / (c.shift(1) + 1e-9) * 100
        df["gift_bull"]    = (df["gift_vs_prev"] >  0.2).astype(int)
        df["gift_bear"]    = (df["gift_vs_prev"] < -0.2).astype(int)
    else:
        df["gift_vs_prev"] = df["gap_pct"].fillna(0)
        df["gift_bull"]    = df["gap_up"]
        df["gift_bear"]    = df["gap_down"]

    # ── G. PCR ───────────────────────────────────────────────────────────────
    if pcr_df is not None:
        pcr = pcr_df.copy()
        pcr["date"] = pd.to_datetime(pcr["date"])
        df = pd.merge(df, pcr[["date","pcr"]], on="date", how="left")
        df["pcr"] = df["pcr"].ffill().fillna(1.0)
    else:
        df["pcr"] = 1.0
    df["pcr_high"] = (df["pcr"] > 1.2).astype(int)
    df["pcr_low"]  = (df["pcr"] < 0.8).astype(int)

    # ── H. CALENDAR ──────────────────────────────────────────────────────────
    df["day_of_week"]    = df["date"].dt.dayofweek
    df["month"]          = df["date"].dt.month
    df["is_monday"]      = (df["day_of_week"] == 0).astype(int)
    df["is_tuesday"]     = (df["day_of_week"] == 1).astype(int)
    df["is_friday"]      = (df["day_of_week"] == 4).astype(int)
    df["days_to_expiry"] = df["date"].apply(_days_to_tuesday)
    df["expiry_week"]    = (df["days_to_expiry"] <= 2).astype(int)
    df["month_end"]      = (df["date"].dt.is_month_end).astype(int)
    df["month_start"]    = (df["date"].dt.is_month_start).astype(int)

    # ── TARGET VARIABLES ─────────────────────────────────────────────────────
    next_open  = o.shift(-1)
    next_close = c.shift(-1)

    # Open target: is next-day open a gap-up (≥0.15% above today's close)?
    df["open_target"]    = (next_open > c * 1.0015).astype(int)
    df["open_ret_pct"]   = (next_open - c) / (c + 1e-9) * 100

    # Close target: does next day close above its open (intraday bullish)?
    df["close_target"]   = (next_close > next_open).astype(int)
    df["close_ret_pct"]  = (next_close - next_open) / (next_open + 1e-9) * 100

    # High / Low targets: % move of next-day high/low vs next-day open.
    # high_ret_pct is typically ≥ 0, low_ret_pct typically ≤ 0.
    next_high = h.shift(-1)
    next_low  = l.shift(-1)
    df["high_ret_pct"]   = (next_high - next_open) / (next_open + 1e-9) * 100
    df["low_ret_pct"]    = (next_low  - next_open) / (next_open + 1e-9) * 100

    df = df.dropna(subset=["open_target","close_target","rsi_14","ema_21"])
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS LIST
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Gap & prev candle
    "gap_pct","gap_up","gap_down","gap_size",
    "prev_oc_ret","prev_oc_bull","prev_range_pct",
    "prev_wick_up","prev_wick_dn","close_strength",
    # Returns
    "ret_1d","ret_2d","ret_3d","ret_5d","ret_10d","ret_20d",
    "consec_up","consec_down",
    # Trend
    "c_vs_ema9","c_vs_ema21","c_vs_ema50","ema9_21",
    "above_ema21","above_ema50","above_ema200",
    # RSI
    "rsi_14","rsi_7","rsi_ob","rsi_os","rsi_cross50",
    # MACD
    "macd_hist","macd_bull","macd_hist_chg","macd_xbull","macd_xbear",
    # Bollinger
    "bb_pct_b","bb_width","bb_squeeze","bb_upper_touch","bb_lower_touch",
    # Volatility
    "atr_pct","vol_5d","vol_regime","vol_ratio","vol_surge",
    # S/R
    "near_52w_high","near_52w_low",
    # Intraday (yesterday's pattern)
    "prev_intra_morning_ret","prev_intra_afternoon_ret",
    "prev_intra_reversal","prev_intra_breakout",
    "prev_intra_close_strength","prev_intra_intraday_range",
    "prev_intra_day_oc_ret","prev_intra_morning_vol_ratio",
    # VIX
    "india_vix","vix_change","vix_pct_chg",
    "vix_spike","vix_calm","vix_high","vix_extreme","vix_above_avg",
    # FII
    "fii_net","fii_bull","fii_bear","fii_5d_ma","fii_trend","fii_3d_sum",
    # GIFT Nifty
    "gift_vs_prev","gift_bull","gift_bear",
    # PCR
    "pcr","pcr_high","pcr_low",
    # Calendar
    "day_of_week","month","is_monday","is_tuesday","is_friday",
    "days_to_expiry","expiry_week","month_end","month_start",
]

# Feature labels for the reasoning panel
FEATURE_LABELS = {
    "gap_pct":                    "Opening gap vs previous close",
    "gap_up":                     "Gap-up at open",
    "gap_down":                   "Gap-down at open",
    "prev_oc_ret":                "Yesterday open-to-close return",
    "prev_oc_bull":               "Yesterday was bullish intraday",
    "prev_range_pct":             "Yesterday's total candle range",
    "close_strength":             "Close position in day's range",
    "ret_1d":                     "1-day return",
    "ret_3d":                     "3-day momentum",
    "ret_5d":                     "5-day momentum",
    "ret_10d":                    "10-day trend",
    "consec_up":                  "Consecutive bullish days (3-day)",
    "consec_down":                "Consecutive bearish days (3-day)",
    "c_vs_ema9":                  "Price vs 9-day EMA",
    "c_vs_ema21":                 "Price vs 21-day EMA",
    "c_vs_ema50":                 "Price vs 50-day EMA",
    "above_ema21":                "Price above 21-EMA (trend up)",
    "above_ema50":                "Price above 50-EMA (bull market)",
    "above_ema200":               "Price above 200-EMA (long-term bull)",
    "rsi_14":                     "RSI 14-day",
    "rsi_ob":                     "RSI overbought (>70)",
    "rsi_os":                     "RSI oversold (<30)",
    "macd_hist":                  "MACD histogram",
    "macd_bull":                  "MACD bullish",
    "macd_xbull":                 "MACD bullish crossover",
    "macd_xbear":                 "MACD bearish crossover",
    "bb_pct_b":                   "Bollinger %B position",
    "bb_squeeze":                 "Bollinger Band squeeze",
    "bb_upper_touch":             "Price touching upper Bollinger Band",
    "bb_lower_touch":             "Price touching lower Bollinger Band",
    "atr_pct":                    "ATR — expected daily range %",
    "vol_surge":                  "Unusual volume spike",
    "near_52w_high":              "Near 52-week high",
    "near_52w_low":               "Near 52-week low",
    "prev_intra_morning_ret":     "Yesterday morning session return",
    "prev_intra_afternoon_ret":   "Yesterday afternoon session return",
    "prev_intra_reversal":        "Yesterday had morning-afternoon reversal",
    "prev_intra_breakout":        "Yesterday broke out of first-hour range",
    "prev_intra_close_strength":  "Yesterday closed strong (high in range)",
    "prev_intra_day_oc_ret":      "Yesterday full intraday return",
    "india_vix":                  "India VIX level",
    "vix_change":                 "VIX day-over-day change",
    "vix_spike":                  "VIX spike — fear in market",
    "vix_calm":                   "VIX falling — market calming",
    "fii_net":                    "FII net buy/sell (₹ cr)",
    "fii_bull":                   "FIIs net buyers >₹500 cr",
    "fii_bear":                   "FIIs net sellers >₹500 cr",
    "fii_trend":                  "FII 5-day trend positive",
    "gift_vs_prev":               "GIFT Nifty vs previous close",
    "gift_bull":                  "GIFT Nifty signals gap-up",
    "gift_bear":                  "GIFT Nifty signals gap-down",
    "pcr":                        "Put-Call Ratio",
    "pcr_high":                   "High PCR — contrarian bullish",
    "pcr_low":                    "Low PCR — contrarian bearish",
    "is_tuesday":                 "Tuesday (expiry day)",
    "days_to_expiry":             "Days until weekly expiry",
    "expiry_week":                "Near expiry (≤2 days)",
}
