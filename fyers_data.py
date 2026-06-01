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
    # Indices
    "NIFTY":      "NSE:NIFTY50-INDEX",
    "BANKNIFTY":  "NSE:NIFTYBANK-INDEX",
    "CNXIT":      "NSE:CNXIT-INDEX",
    "CNXAUTO":    "NSE:CNXAUTO-INDEX",
    "CNXFMCG":    "NSE:CNXFMCG-INDEX",
    "INDIAVIX":   "NSE:INDIAVIX-INDEX",
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
