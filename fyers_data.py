"""
fyers_data.py — Fyers API data pipeline (alternative to Breeze).

Provides the same DataFrame shape as data_fetcher.fetch_nifty_breeze:
    columns = ["date", "open", "high", "low", "close", "volume"]

Auth model:
  - Fyers uses OAuth: client_id + secret -> auth_code (browser) -> access_token.
  - Access tokens expire DAILY (similar friction to Breeze session tokens).
  - Run `python auth_fyers.py` once each morning to refresh.

Entry points:
    init_fyers(client_id, access_token)              -> FyersModel client
    fetch_nifty_fyers(fyers, days=730)               -> daily Nifty OHLCV
    fetch_history_fyers(fyers, symbol, resolution,
                        days, from_date, to_date)    -> generic OHLCV fetch
    fyers_symbol(name)                               -> map "NIFTY"->"NSE:NIFTY50-INDEX" etc.
"""

from __future__ import annotations
import time
import warnings
from datetime import datetime, timedelta

import pandas as pd

warnings.filterwarnings("ignore")

try:
    from fyers_apiv3 import fyersModel
    FYERS_OK = True
except ImportError:
    FYERS_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL MAPPING — translate friendly names → Fyers symbol strings
# ─────────────────────────────────────────────────────────────────────────────

FYERS_SYMBOLS = {
    # Indices — Fyers uses the modern NSE names ("NIFTYIT" not "CNXIT").
    # The settings.py CORRELATED_INSTRUMENTS list still uses legacy CNX names
    # for backward-compatibility with the Breeze pipeline; we translate here.
    "NIFTY":       "NSE:NIFTY50-INDEX",
    "BANKNIFTY":   "NSE:NIFTYBANK-INDEX",
    "CNXIT":       "NSE:NIFTYIT-INDEX",       # was CNXIT pre-2013
    "CNXAUTO":     "NSE:NIFTYAUTO-INDEX",     # was CNXAUTO pre-2013
    "CNXFMCG":     "NSE:NIFTYFMCG-INDEX",     # was CNXFMCG pre-2013
    "CNXPHARMA":   "NSE:NIFTYPHARMA-INDEX",
    "NIFTYIT":     "NSE:NIFTYIT-INDEX",
    "NIFTYAUTO":   "NSE:NIFTYAUTO-INDEX",
    "NIFTYFMCG":   "NSE:NIFTYFMCG-INDEX",
    "NIFTYPHARMA": "NSE:NIFTYPHARMA-INDEX",
    "INDIAVIX":    "NSE:INDIAVIX-INDEX",
}


def fyers_symbol(name: str) -> str:
    """Map a friendly name to a Fyers symbol; passthrough if already Fyers-formatted."""
    if ":" in name:
        return name        # already a Fyers symbol like "NSE:NIFTY50-INDEX"
    return FYERS_SYMBOLS.get(name.upper(), f"NSE:{name.upper()}-INDEX")


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT INIT
# ─────────────────────────────────────────────────────────────────────────────

def init_fyers(client_id: str, access_token: str, log_path: str = "logs"):
    """
    Create a FyersModel client for API calls.

    Parameters
    ----------
    client_id    : Your Fyers App ID, format "XXXXXXXX-100"
    access_token : Daily token from auth_fyers.py (or Streamlit secrets)
    log_path     : Where the SDK writes its own logs (created if missing)
    """
    if not FYERS_OK:
        raise ImportError("Run: pip install fyers-apiv3")
    if not client_id or "-100" not in client_id:
        raise ValueError("Fyers client_id missing or malformed (expected 'XXXXXXXX-100').")
    if not access_token:
        raise ValueError(
            "Fyers access_token empty. Run `python auth_fyers.py` to generate a fresh "
            "token (tokens expire daily)."
        )
    import os
    os.makedirs(log_path, exist_ok=True)
    return fyersModel.FyersModel(
        client_id=client_id,
        token=access_token,
        log_path=log_path,
        is_async=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — chunked historical fetch
# Fyers caps a single history call at ~100 days for daily / smaller for intraday,
# so for multi-year history we walk backwards in windows.
# ─────────────────────────────────────────────────────────────────────────────

_RESOLUTION_CHUNK_DAYS = {
    "D":  365,    # daily — Fyers allows up to ~366 days per request
    "60": 60,     # 60-min
    "30": 30,
    "15": 20,
    "5":  10,
    "1":  5,
}


def _fyers_one_chunk(fyers, symbol: str, resolution: str,
                     from_date: str, to_date: str,
                     retries: int = 2) -> pd.DataFrame | None:
    """Single Fyers history call -> DataFrame (or None on failure)."""
    payload = {
        "symbol":      symbol,
        "resolution":  resolution,
        "date_format": "1",
        "range_from":  from_date,    # YYYY-MM-DD
        "range_to":    to_date,
        "cont_flag":   "1",
    }
    for attempt in range(retries + 1):
        try:
            resp = fyers.history(payload)
            if resp.get("s") == "ok" and resp.get("candles"):
                df = pd.DataFrame(
                    resp["candles"],
                    columns=["epoch", "open", "high", "low", "close", "volume"],
                )
                df["date"] = pd.to_datetime(df["epoch"], unit="s")
                if resolution == "D":
                    df["date"] = df["date"].dt.normalize()
                for c in ("open", "high", "low", "close"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
                df = df[["date", "open", "high", "low", "close", "volume"]] \
                       .dropna(subset=["open", "close"])
                return df
            elif resp.get("s") == "no_data":
                return pd.DataFrame(
                    columns=["date", "open", "high", "low", "close", "volume"])
            else:
                if attempt < retries:
                    time.sleep(1)
                else:
                    print(f"[Fyers] {symbol} chunk failed: {resp}")
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"[Fyers] {symbol} chunk exception: {e}")
    return None


def fetch_history_fyers(fyers, symbol: str = "NIFTY", resolution: str = "D",
                        days: int = 730) -> pd.DataFrame | None:
    """
    Fetch OHLCV history. Walks the window in chunks so Fyers' per-call cap
    doesn't limit total range. Symbol may be friendly ("NIFTY") or fully
    qualified ("NSE:NIFTY50-INDEX").
    """
    if not FYERS_OK:
        raise ImportError("Run: pip install fyers-apiv3")
    sym = fyers_symbol(symbol)
    chunk = _RESOLUTION_CHUNK_DAYS.get(resolution, 100)

    end = datetime.now()
    start_target = end - timedelta(days=days)
    cursor = end
    frames: list[pd.DataFrame] = []

    while cursor > start_target:
        chunk_start = max(cursor - timedelta(days=chunk), start_target)
        df = _fyers_one_chunk(
            fyers, sym, resolution,
            chunk_start.strftime("%Y-%m-%d"),
            cursor.strftime("%Y-%m-%d"),
        )
        if df is None:
            print(f"[Fyers] aborting chunked fetch after failure at "
                  f"{chunk_start.date()} → {cursor.date()}")
            break
        if not df.empty:
            frames.append(df)
        cursor = chunk_start - timedelta(days=1)
        time.sleep(0.25)   # polite throttling

    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True) \
            .drop_duplicates(subset=["date"]) \
            .sort_values("date") \
            .reset_index(drop=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE — Nifty daily OHLCV (same shape as fetch_nifty_breeze)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nifty_fyers(fyers, days: int = 730) -> pd.DataFrame | None:
    df = fetch_history_fyers(fyers, "NIFTY", "D", days)
    if df is not None and len(df):
        print(f"[Fyers] Nifty daily: {len(df)} rows "
              f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
    return df


def fetch_vix_fyers(fyers, days: int = 730) -> pd.DataFrame | None:
    """India VIX daily — drop-in alternative to data_fetcher.fetch_vix_breeze."""
    df = fetch_history_fyers(fyers, "INDIAVIX", "D", days)
    if df is not None and len(df):
        df = df.rename(columns={"close": "india_vix"})[["date", "india_vix"]]
        print(f"[Fyers] India VIX daily: {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INTRADAY 5-min — drop-in for data_fetcher.fetch_intraday_breeze
# ─────────────────────────────────────────────────────────────────────────────

def fetch_intraday_fyers(fyers, symbol: str = "NIFTY",
                         days_back: int = 60) -> pd.DataFrame | None:
    """
    5-min intraday candles, market hours only.

    NOTE on sparse data: Fyers' history endpoint returns *very* few 5-min
    candles for index symbols (NSE:NIFTY50-INDEX) — often just a handful
    per day. For dense intraday you need the Nifty futures contract
    (e.g. NSE:NIFTY25JUNFUT). We log both the requested symbol and the
    actual candle count so the caller can decide whether to fall back.
    """
    df = fetch_history_fyers(fyers, symbol, "5", days_back)
    if df is None or df.empty:
        print(f"[Fyers] Intraday {symbol} 5min: 0 candles returned by API.")
        return None
    # Keep only NSE market hours: 9:15 to 15:30 IST
    h, m = df["date"].dt.hour, df["date"].dt.minute
    df = df[(h > 9) | ((h == 9) & (m >= 15))]
    df = df[(h < 15) | ((h == 15) & (m <= 30))]
    n_days = df["date"].dt.date.nunique() if len(df) else 0
    per_day = (len(df) / n_days) if n_days else 0
    print(f"[Fyers] Intraday {symbol} 5min: {len(df)} candles across "
          f"{n_days} days (~{per_day:.0f}/day; healthy = ~75/day). "
          f"Range: {df['date'].dt.date.min() if len(df) else None} → "
          f"{df['date'].dt.date.max() if len(df) else None}")
    return df.reset_index(drop=True)


def fetch_nifty_futures_symbol() -> str:
    """
    Build the Fyers symbol for the current near-month Nifty futures contract.
    Format: NSE:NIFTY{YY}{MMM}FUT  e.g. NSE:NIFTY25JUNFUT
    """
    today = datetime.now()
    # Last Thursday of the month is expiry; if past, use next month.
    yr = today.year
    mo = today.month
    # Simple heuristic: if today is in the last week of the month, roll to next.
    if today.day >= 25:
        mo += 1
        if mo > 12:
            mo = 1
            yr += 1
    months = ["JAN","FEB","MAR","APR","MAY","JUN",
              "JUL","AUG","SEP","OCT","NOV","DEC"]
    return f"NSE:NIFTY{yr % 100:02d}{months[mo-1]}FUT"


# ─────────────────────────────────────────────────────────────────────────────
# BANK NIFTY + SECTOR INDICES — drop-in for fetch_correlated_daily
# ─────────────────────────────────────────────────────────────────────────────

def fetch_correlated_fyers(fyers, days: int = 730) -> dict[str, pd.DataFrame]:
    """Daily OHLCV for Bank Nifty and sector indices configured in settings."""
    from settings import CORRELATED_INSTRUMENTS
    out = {}
    for sym in CORRELATED_INSTRUMENTS:
        df = fetch_history_fyers(fyers, sym, "D", days)
        if df is not None and len(df) > 30:
            out[sym] = df
            print(f"[Fyers] {sym} daily: {len(df)} rows")
        else:
            print(f"[Fyers] {sym} — no data (symbol not found via Fyers)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LIVE QUOTE — drop-in for fetch_live_quote_breeze
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_quote_fyers(fyers, symbol: str = "NIFTY") -> dict | None:
    sym = fyers_symbol(symbol)
    try:
        resp = fyers.quotes({"symbols": sym})
        if resp.get("s") == "ok" and resp.get("d"):
            v = resp["d"][0].get("v", {})
            return {
                "ltp":        float(v.get("lp", 0) or 0),
                "open":       float(v.get("open_price",       0) or 0),
                "high":       float(v.get("high_price",       0) or 0),
                "low":        float(v.get("low_price",        0) or 0),
                "prev_close": float(v.get("prev_close_price", 0) or 0),
            }
        print(f"[Fyers] live quote response: {resp}")
    except Exception as e:
        print(f"[Fyers] live quote failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS CHAIN — drop-in for fetch_options_chain_breeze
# Returns (DataFrame[strike,type,ltp,bid,ask,oi,volume,iv], pcr)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_options_chain_fyers(fyers, expiry_str: str = "",
                              spot: float = 0.0,
                              strikecount: int = 10
                              ) -> tuple["pd.DataFrame | None", float]:
    """
    Fetch Nifty options chain via Fyers.

    Fyers' option_chain endpoint returns symmetric strikes around spot
    (controlled by strikecount = strikes on either side, total = 2*strikecount+1).
    `expiry_str` is currently advisory — Fyers picks the nearest expiry by default.
    """
    try:
        payload = {
            "symbol":      fyers_symbol("NIFTY"),
            "strikecount": int(strikecount),
            "timestamp":   "",            # "" = current/nearest expiry
        }
        resp = fyers.optionchain(data=payload)
    except Exception as e:
        print(f"[Fyers] options chain failed: {e}")
        return None, 1.0

    if not isinstance(resp, dict) or resp.get("s") != "ok":
        print(f"[Fyers] options chain bad response: {resp}")
        return None, 1.0

    data = resp.get("data", {}) or {}
    rows = data.get("optionsChain") or data.get("optionchain") or []
    ce_oi = pe_oi = 0.0
    out_rows = []
    for r in rows:
        opt_type = (r.get("option_type") or r.get("optionType") or "").upper()
        if opt_type not in ("CE", "PE"):
            # Index spot row — skip
            continue
        strike = float(r.get("strike_price") or r.get("strikePrice") or 0)
        ltp    = float(r.get("ltp") or r.get("lastPrice") or 0)
        oi     = float(r.get("oi") or r.get("openInterest") or 0)
        volume = float(r.get("volume") or 0)
        iv     = float(r.get("iv") or r.get("impliedVolatility") or 0)
        bid    = float(r.get("bid") or r.get("bestBid") or 0)
        ask    = float(r.get("ask") or r.get("bestAsk") or 0)
        if opt_type == "CE":
            ce_oi += oi
        else:
            pe_oi += oi
        out_rows.append({
            "strike": strike, "type": opt_type,
            "ltp": ltp, "bid": bid, "ask": ask,
            "oi": oi, "volume": volume, "iv": iv,
        })

    pcr = round(pe_oi / ce_oi, 3) if ce_oi > 0 else 1.0
    return (pd.DataFrame(out_rows) if out_rows else None), pcr


# ─────────────────────────────────────────────────────────────────────────────
# CACHE-AWARE WRAPPERS for intraday & correlated — same shape as data_fetcher
# ─────────────────────────────────────────────────────────────────────────────

def load_intraday_data_fyers(fyers=None, force_refresh: bool = False,
                             days_back: int = None) -> pd.DataFrame | None:
    """
    5-min Nifty intraday with disk cache (mirrors load_intraday_data).

    Auto-promotes from index symbol to FUTURES if the index returns sparse
    data (Fyers' /history endpoint serves very few 5-min candles for indices).
    """
    from settings import DATA_DIR, INTRADAY_DAYS_BACK
    if days_back is None:
        days_back = INTRADAY_DAYS_BACK
    cache = DATA_DIR / "intraday_nifty.csv"

    if cache.exists() and not force_refresh:
        age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        if age < 4:
            df = pd.read_csv(cache, parse_dates=["date"])
            print(f"[Cache] Intraday Nifty (fyers source): {len(df)} candles")
            return df.sort_values("date").reset_index(drop=True)

    if fyers is None:
        if cache.exists():
            df = pd.read_csv(cache, parse_dates=["date"])
            print(f"[Cache] Using stale intraday cache: {len(df)} candles")
            return df.sort_values("date").reset_index(drop=True)
        return None

    # Try the index first
    df = fetch_intraday_fyers(fyers, "NIFTY", days_back)
    sparse_threshold = max(20 * days_back, 200)   # expect ~75/day; require >=20/day
    if df is None or len(df) < sparse_threshold:
        n = 0 if df is None else len(df)
        fut_sym = fetch_nifty_futures_symbol()
        print(f"[Fyers] Index intraday too sparse ({n} candles); "
              f"trying futures {fut_sym}…")
        df_fut = fetch_intraday_fyers(fyers, fut_sym, days_back)
        if df_fut is not None and len(df_fut) > (len(df) if df is not None else 0):
            df = df_fut

    if df is not None and len(df):
        df.to_csv(cache, index=False)
    return df


def load_correlated_data_fyers(fyers=None,
                               force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Bank Nifty + sector indices via Fyers, per-symbol cache."""
    from settings import DATA_DIR, CORRELATED_INSTRUMENTS, TRAINING_DAYS
    out: dict[str, pd.DataFrame] = {}

    for sym in CORRELATED_INSTRUMENTS:
        cache = DATA_DIR / f"corr_{sym.lower()}.csv"
        if cache.exists() and not force_refresh:
            age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
            if age < 8:
                df = pd.read_csv(cache, parse_dates=["date"])
                print(f"[Cache] {sym} (fyers source): {len(df)} rows")
                out[sym] = df.sort_values("date").reset_index(drop=True)
                continue
        if fyers is None:
            if cache.exists():
                out[sym] = pd.read_csv(cache, parse_dates=["date"]) \
                            .sort_values("date").reset_index(drop=True)
            continue
        df = fetch_history_fyers(fyers, sym, "D", TRAINING_DAYS)
        if df is not None and len(df) > 30:
            df.to_csv(cache, index=False)
            out[sym] = df
            print(f"[Fyers] {sym} daily: {len(df)} rows")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CACHE-AWARE WRAPPER — drop-in for data_fetcher.load_nifty_data
# ─────────────────────────────────────────────────────────────────────────────

def load_nifty_data_fyers(fyers=None, force_refresh: bool = False,
                          days: int = None) -> pd.DataFrame | None:
    """
    Cache-aware Nifty daily loader using Fyers as source.
    Same caching semantics as data_fetcher.load_nifty_data:
      - serves disk cache if < 8 hours old (unless force_refresh)
      - fresh fetch via Fyers when stale or missing
      - falls back to stale cache if Fyers unavailable
    """
    from settings import DATA_DIR, TRAINING_DAYS
    from datetime import datetime as _dt
    if days is None:
        days = TRAINING_DAYS
    cache = DATA_DIR / "nifty_ohlcv.csv"

    if cache.exists() and not force_refresh:
        age = (_dt.now().timestamp() - cache.stat().st_mtime) / 3600
        if age < 8:
            df = pd.read_csv(cache, parse_dates=["date"])
            print(f"[Cache] Nifty (fyers source): {len(df)} rows")
            return df.sort_values("date").reset_index(drop=True)

    if fyers is None:
        print("[Fyers] No client — cannot fetch fresh data.")
        if cache.exists():
            df = pd.read_csv(cache, parse_dates=["date"])
            print(f"[Cache] Using stale cache: {len(df)} rows")
            return df.sort_values("date").reset_index(drop=True)
        return None

    df = fetch_nifty_fyers(fyers, days)
    if df is not None and len(df):
        df.to_csv(cache, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CONNECT HELPER — returns a usable Fyers client or None
# Mirrors how Breeze is initialised; respects settings priority chain.
# ─────────────────────────────────────────────────────────────────────────────

def connect_from_settings():
    """
    One-call helper to spin up a Fyers client from stored credentials.
    Returns the FyersModel client or None (with a printed reason).
    """
    cid, _, tok = get_fyers_credentials()
    if not cid:
        print("[Fyers] client_id not configured.")
        return None
    if not tok:
        print("[Fyers] access_token empty — run `python auth_fyers.py` to refresh.")
        return None
    try:
        return init_fyers(cid, tok)
    except Exception as e:
        print(f"[Fyers] connect failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG HELPERS — read credentials with the same priority as the rest of
# the app: env vars → Streamlit secrets → settings.json → settings.py defaults
# ─────────────────────────────────────────────────────────────────────────────

def get_fyers_credentials() -> tuple[str, str, str]:
    """
    Return (client_id, secret_id, access_token).
    Pulls from env first, then Streamlit secrets, then settings.json.
    Missing values come back as empty strings.
    """
    import os, json
    from pathlib import Path

    def _from_secrets(key):
        try:
            import streamlit as st
            return st.secrets.get(key, "")
        except Exception:
            return ""

    def _from_json(key):
        try:
            data = json.load(open(Path(__file__).parent / "settings.json"))
            return data.get(key, "")
        except Exception:
            return ""

    def _resolve(*names):
        for n in names:
            v = os.getenv(n) or _from_secrets(n) or _from_json(n.lower())
            if v:
                return v
        return ""

    client_id    = _resolve("FYERS_CLIENT_ID")
    secret_id    = _resolve("FYERS_SECRET_ID")
    access_token = _resolve("FYERS_ACCESS_TOKEN")
    return client_id, secret_id, access_token
