"""
data_fetcher.py — All Breeze API data fetching.

Fetches:
  1. Nifty 50 daily OHLCV (primary training data)
  2. Intraday 5-min candles (yesterday's intraday pattern features)
  3. Bank Nifty + correlated sector indices (leading indicator features)
  4. India VIX (volatility regime)
  5. FII derivatives participant OI (institutional positioning)
  6. Live quotes + options chain + PCR (signal generation)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import warnings, requests, io, time
warnings.filterwarnings("ignore")

try:
    from breeze_connect import BreezeConnect
    BREEZE_OK = True
except ImportError:
    BREEZE_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# BREEZE INIT
# ─────────────────────────────────────────────────────────────────────────────

def init_breeze(api_key: str, api_secret: str, session_token: str):
    """
    Connect to ICICI Breeze API.
    Session token must be freshly generated each morning:
      1. Visit https://api.icicidirect.com/apiuser/login?api_key=YOUR_KEY
      2. Login → copy the apisession= value from the redirected URL
      3. Paste into Settings tab
    """
    if not BREEZE_OK:
        raise ImportError("Run: pip install breeze-connect")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("Breeze API key not configured in Settings.")
    if not session_token:
        raise ValueError("Session token empty — generate one from the login URL.")
    breeze = BreezeConnect(api_key=api_key)
    breeze.generate_session(api_secret=api_secret, session_token=session_token)
    return breeze


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — raw historical fetch with retry
# ─────────────────────────────────────────────────────────────────────────────

def _breeze_hist(breeze, stock_code: str, interval: str = "1day",
                 days: int = 730, retries: int = 2) -> pd.DataFrame | None:
    """
    Generic Breeze historical data fetch.
    Breeze caps at ~730 days for daily data, ~60 days for intraday.
    """
    days  = min(days, 730) if interval == "1day" else min(days, 60)
    end   = datetime.now()
    start = end - timedelta(days=days)
    for attempt in range(retries + 1):
        try:
            resp = breeze.get_historical_data_v2(
                interval=interval,
                from_date=start.strftime("%Y-%m-%dT07:00:00.000Z"),
                to_date=end.strftime("%Y-%m-%dT07:00:00.000Z"),
                stock_code=stock_code,
                exchange_code="NSE",
                product_type="cash",
            )
            if resp.get("Status") == 200 and resp.get("Success"):
                df = pd.DataFrame(resp["Success"])
                df["datetime_raw"] = pd.to_datetime(df["datetime"])
                if interval == "1day":
                    df["date"] = df["datetime_raw"].dt.normalize()
                else:
                    df["date"] = df["datetime_raw"]   # keep full timestamp for intraday
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["volume"] = pd.to_numeric(
                    df.get("volume", 0), errors="coerce").fillna(0)
                cols = ["date","open","high","low","close","volume"]
                df = df[cols].dropna(subset=["open","close"])
                return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"[Breeze] {stock_code} ({interval}) failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. NIFTY 50 DAILY OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nifty_breeze(breeze, days: int = 730) -> pd.DataFrame | None:
    df = _breeze_hist(breeze, "NIFTY", "1day", min(days, 730))
    if df is not None:
        print(f"[Breeze] Nifty daily: {len(df)} rows "
              f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
    return df


def fetch_vix_breeze(breeze, days: int = 730) -> pd.DataFrame | None:
    df = _breeze_hist(breeze, "INDIAVIX", "1day", min(days, 730))
    if df is not None:
        out = df[["date","close"]].rename(columns={"close":"india_vix"})
        print(f"[Breeze] VIX: {len(out)} rows")
        return out
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. INTRADAY 5-MIN CANDLES  (last N trading days)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_intraday_breeze(breeze, stock_code: str = "NIFTY",
                          days_back: int = 60) -> pd.DataFrame | None:
    """
    Fetch 5-minute intraday candles for the past `days_back` calendar days.
    Breeze caps at 60 days for intraday data.
    Returns DataFrame with columns: date (timestamp), open, high, low, close, volume
    """
    df = _breeze_hist(breeze, stock_code, "5minute", min(days_back, 60))
    if df is not None:
        # Keep only market hours: 9:15 to 15:30 IST
        df = df[
            (df["date"].dt.hour > 9) |
            ((df["date"].dt.hour == 9) & (df["date"].dt.minute >= 15))
        ]
        df = df[
            (df["date"].dt.hour < 15) |
            ((df["date"].dt.hour == 15) & (df["date"].dt.minute <= 30))
        ]
        print(f"[Breeze] Intraday {stock_code} 5min: {len(df)} candles "
              f"({df['date'].dt.date.min()} → {df['date'].dt.date.max()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. BANK NIFTY + SECTOR INDICES
# ─────────────────────────────────────────────────────────────────────────────

def fetch_correlated_daily(breeze, days: int = 730) -> dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV for Bank Nifty and key sector indices.
    Returns dict: {symbol: DataFrame}
    Available on Breeze: BANKNIFTY, CNXIT, CNXAUTO, CNXFMCG, CNXPHARMA
    """
    from settings import CORRELATED_INSTRUMENTS
    result = {}
    for sym in CORRELATED_INSTRUMENTS:
        df = _breeze_hist(breeze, sym, "1day", min(days, 730))
        if df is not None and len(df) > 30:
            result[sym] = df
            print(f"[Breeze] {sym} daily: {len(df)} rows")
        else:
            print(f"[Breeze] {sym} — no data (may not be available on your account)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. FII DERIVATIVES PARTICIPANT DATA
# Source: Breeze provides F&O participant OI data
# FII long/short ratio in index futures = strongest institutional signal
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fii_derivatives_breeze(breeze) -> pd.DataFrame | None:
    """
    Fetch FII/DII/Client/Pro F&O participant-wise OI from Breeze.
    This is far more powerful than equity FII flows because it shows
    institutional directional bets in the derivatives market directly.

    FII index futures long% > 60%  → strong bullish institutional bias
    FII index futures long% < 40%  → strong bearish institutional bias
    """
    try:
        resp = breeze.get_names(exchange_code="NFO", stock_code="NIFTY")
        if resp.get("Status") != 200:
            return None

        # Try participant OI endpoint
        resp2 = breeze.get_option_chain_quotes(
            stock_code="NIFTY",
            exchange_code="NFO",
            product_type="futures",
            expiry_date="",
            right="",
            strike_price="",
        )
        if resp2.get("Status") == 200 and resp2.get("Success"):
            raw = resp2["Success"]
            if isinstance(raw, list) and len(raw) > 0:
                df = pd.DataFrame(raw)
                df["date"] = date.today()
                return df
    except Exception as e:
        print(f"[Breeze] FII derivatives failed: {e}")
    return None


def fetch_fii_nsdl(days: int = 730) -> pd.DataFrame | None:
    """Fallback: NSDL FII equity flows (less precise but reliable)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,*/*;q=0.8",
    }
    end, start = datetime.now(), datetime.now() - timedelta(days=days)
    url = (
        "https://www.nsdl.co.in/nsdlcms/fii/fiiDailyActivity.php"
        f"?startDate={start.strftime('%d-%m-%Y')}&endDate={end.strftime('%d-%m-%Y')}"
        "&type=equity&submit=Get+Data&format=csv"
    )
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200 and len(resp.content) > 200:
            df = pd.read_csv(io.StringIO(resp.text), skip_blank_lines=True)
            df.columns = [c.strip().lower() for c in df.columns]
            for col in df.columns:
                if "date" in col: df = df.rename(columns={col: "date"})
                if "net"  in col: df = df.rename(columns={col: "fii_net"})
            df["date"]    = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df["fii_net"] = pd.to_numeric(df.get("fii_net", 0), errors="coerce").fillna(0)
            df["dii_net"] = 0.0
            df = df[["date","fii_net","dii_net"]].dropna(subset=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) > 5:
                print(f"[NSDL] FII equity: {len(df)} rows")
                return df
    except Exception as e:
        print(f"[NSDL] failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. LIVE QUOTES & OPTIONS CHAIN
# ─────────────────────────────────────────────────────────────────────────────

def _breeze_get_quotes_with_retry(breeze, max_retries: int = 3, delay: float = 0.5, **kwargs) -> dict:
    """
    Wrapper around breeze.get_quotes() with retry on 503 / empty-body errors.

    On every failure prints a debug line with:
      - exchange, product, stock, strike, right  (what was requested)
      - the Breeze REST endpoint being hit
      - the HTTP status code and raw error

    Breeze REST base: https://api.icicidirect.com/breezeapi/api/v1/quotes
    """
    BREEZE_QUOTES_URL = (
        "https://api.icicidirect.com/breezeapi/api/v1/quotes"
        "?stock_code={stock_code}&exchange_code={exchange_code}"
        "&product_type={product_type}&expiry_date={expiry_date}"
        "&right={right}&strike_price={strike_price}"
    ).format(
        stock_code    = kwargs.get("stock_code",    ""),
        exchange_code = kwargs.get("exchange_code", ""),
        product_type  = kwargs.get("product_type",  ""),
        expiry_date   = kwargs.get("expiry_date",   ""),
        right         = kwargs.get("right",         ""),
        strike_price  = kwargs.get("strike_price",  ""),
    )

    label = (
        f"exchange={kwargs.get('exchange_code','?')} "
        f"product={kwargs.get('product_type','?')} "
        f"stock={kwargs.get('stock_code','?')} "
        f"strike={kwargs.get('strike_price','') or 'N/A'} "
        f"right={kwargs.get('right','') or 'N/A'}"
    )

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = breeze.get_quotes(**kwargs)
            status = resp.get("Status") if isinstance(resp, dict) else None

            if status == 200:
                return resp                          # success

            # API-level error (e.g. 500 "Check stock code")
            err_msg = resp.get("Error", "unknown API error") if isinstance(resp, dict) else str(resp)
            print(
                f"[Breeze] FAILED (attempt {attempt}/{max_retries}) | "
                f"{label} | HTTP {status} | error: {err_msg} | "
                f"URL: {BREEZE_QUOTES_URL}"
            )
            last_exc = Exception(f"Status {status}: {err_msg}")
            break   # API errors (4xx/5xx with body) won't improve on retry

        except Exception as e:
            err_str = str(e)
            # "Expecting value: line 1 column 1" = empty body = 503 transient
            is_transient = "Expecting value" in err_str or "503" in err_str
            print(
                f"[Breeze] FAILED (attempt {attempt}/{max_retries}) | "
                f"{label} | {'503 empty-body (transient)' if is_transient else 'exception'}: {err_str} | "
                f"URL: {BREEZE_QUOTES_URL}"
            )
            last_exc = e
            if is_transient and attempt < max_retries:
                time.sleep(delay)
                continue
            break   # non-transient exception — no point retrying

    return {}   # all attempts exhausted


def fetch_live_quote_breeze(breeze) -> dict | None:
    resp = _breeze_get_quotes_with_retry(
        breeze,
        stock_code="NIFTY", exchange_code="NSE",
        product_type="cash", expiry_date="", right="", strike_price="",
    )
    if resp.get("Status") == 200 and resp.get("Success"):
        d = resp["Success"][0]
        return {
            "ltp":        float(d.get("ltp",              0) or 0),
            "open":       float(d.get("open",             0) or 0),
            "high":       float(d.get("high",             0) or 0),
            "low":        float(d.get("low",              0) or 0),
            "prev_close": float(d.get("previous_close",   0) or 0),
        }
    return None


def fetch_gift_nifty_breeze(breeze) -> float | None:
    """
    Fetch the live GIFT Nifty (SGX Nifty) futures price from Breeze.
    Used at 8:45 AM to check if GIFT contradicts the model's direction.
    Tries multiple product/exchange combos since GIFT listing varies.
    Returns the LTP as a float, or None if unavailable.
    """
    combos = [
        {"stock_code": "GIFTNIFTY", "exchange_code": "NSE",  "product_type": "futures"},
        {"stock_code": "NIFTY",     "exchange_code": "NFO",  "product_type": "futures"},
    ]
    for params in combos:
        resp = _breeze_get_quotes_with_retry(
            breeze,
            stock_code=params["stock_code"],
            exchange_code=params["exchange_code"],
            product_type=params["product_type"],
            expiry_date="", right="", strike_price="",
        )
        if resp.get("Status") == 200 and resp.get("Success"):
            ltp = float(resp["Success"][0].get("ltp", 0) or 0)
            if ltp > 0:
                return ltp
    return None


def fetch_options_chain_breeze(breeze, expiry_str: str,
                                spot: float) -> tuple[pd.DataFrame | None, float]:
    from settings import NIFTY_STRIKE_GAP
    atm     = int(round(spot / NIFTY_STRIKE_GAP) * NIFTY_STRIKE_GAP)
    strikes = [atm + i * NIFTY_STRIKE_GAP for i in range(-5, 6)]
    rows, ce_oi, pe_oi = [], 0.0, 0.0
    ok_count, fail_count = 0, 0

    for strike in strikes:
        for right in ["call", "put"]:
            resp = _breeze_get_quotes_with_retry(
                breeze,
                stock_code="NIFTY", exchange_code="NFO",
                product_type="options", expiry_date=expiry_str,
                right=right, strike_price=str(int(strike)),
            )
            if resp.get("Status") == 200 and resp.get("Success"):
                d    = resp["Success"][0]
                oi   = float(d.get("open_interest",       0) or 0)
                ltp  = float(d.get("ltp",                 0) or 0)
                opt  = "CE" if right == "call" else "PE"
                if opt == "CE": ce_oi += oi
                else:           pe_oi += oi
                rows.append({
                    "strike": strike, "type":   opt,    "ltp":    ltp,
                    "bid":   float(d.get("best_bid_price",    0) or 0),
                    "ask":   float(d.get("best_offer_price",  0) or 0),
                    "oi":    oi,
                    "volume":float(d.get("volume",            0) or 0),
                    "iv":    float(d.get("implied_volatility", 0) or 0),
                })
                ok_count += 1
            else:
                fail_count += 1

    print(
        f"[Breeze] options chain scan complete: "
        f"{ok_count} OK, {fail_count} failed "
        f"(expiry={expiry_str}, ATM={atm}, strikes={strikes[0]}–{strikes[-1]})"
    )
    pcr = round(pe_oi / ce_oi, 3) if ce_oi > 0 else 1.0
    return (pd.DataFrame(rows) if rows else None), pcr


# ─────────────────────────────────────────────────────────────────────────────
# SMART LOADERS  (cache to disk, source = Breeze only)
# ─────────────────────────────────────────────────────────────────────────────

def load_nifty_data(breeze=None, force_refresh: bool = False,
                    days: int = None) -> pd.DataFrame | None:
    from settings import DATA_DIR, TRAINING_DAYS
    if days is None: days = TRAINING_DAYS
    cache = DATA_DIR / "nifty_ohlcv.csv"

    if cache.exists() and not force_refresh:
        age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        if age < 8:
            df = pd.read_csv(cache, parse_dates=["date"])
            print(f"[Cache] Nifty: {len(df)} rows")
            return df.sort_values("date").reset_index(drop=True)

    if breeze is None:
        print("[Nifty] No Breeze session — cannot fetch data without API connection.")
        # Return from cache even if stale
        if cache.exists():
            df = pd.read_csv(cache, parse_dates=["date"])
            print(f"[Cache] Using stale cache: {len(df)} rows")
            return df.sort_values("date").reset_index(drop=True)
        return None

    df = fetch_nifty_breeze(breeze, days)
    if df is not None:
        df.to_csv(cache, index=False)
    return df


def load_vix_data(breeze=None, force_refresh: bool = False) -> pd.DataFrame | None:
    from settings import DATA_DIR, TRAINING_DAYS
    cache = DATA_DIR / "india_vix.csv"

    if cache.exists() and not force_refresh:
        age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        if age < 8:
            return pd.read_csv(cache, parse_dates=["date"])

    if breeze is None:
        if cache.exists():
            return pd.read_csv(cache, parse_dates=["date"])
        return None

    df = fetch_vix_breeze(breeze, TRAINING_DAYS)
    if df is not None:
        df.to_csv(cache, index=False)
    return df


def load_intraday_data(breeze=None, force_refresh: bool = False,
                       stock_code: str = "NIFTY") -> pd.DataFrame | None:
    """Load intraday 5-min candles. Re-fetched daily (stale after 8hrs)."""
    from settings import DATA_DIR, INTRADAY_DAYS_BACK
    cache = DATA_DIR / f"intraday_{stock_code.lower()}.csv"

    if cache.exists() and not force_refresh:
        age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        if age < 8:
            df = pd.read_csv(cache, parse_dates=["date"])
            return df

    if breeze is None:
        if cache.exists():
            return pd.read_csv(cache, parse_dates=["date"])
        return None

    df = fetch_intraday_breeze(breeze, stock_code, days_back=60)
    if df is not None:
        df.to_csv(cache, index=False)
    return df


def load_correlated_data(breeze=None,
                         force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Load Bank Nifty + sector indices. Returns dict {symbol: df}."""
    from settings import DATA_DIR, TRAINING_DAYS, CORRELATED_INSTRUMENTS
    result = {}

    for sym in CORRELATED_INSTRUMENTS:
        cache = DATA_DIR / f"corr_{sym.lower()}.csv"

        if cache.exists() and not force_refresh:
            age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
            if age < 8:
                result[sym] = pd.read_csv(cache, parse_dates=["date"])
                continue

        if breeze is not None:
            df = _breeze_hist(breeze, sym, "1day", min(TRAINING_DAYS, 730))
            if df is not None and len(df) > 30:
                df.to_csv(cache, index=False)
                result[sym] = df
        elif cache.exists():
            result[sym] = pd.read_csv(cache, parse_dates=["date"])

    return result


def load_global_data(force_refresh: bool = False) -> pd.DataFrame | None:
    """Global cues — try NSDL then return cached if unavailable."""
    from settings import DATA_DIR
    cache = DATA_DIR / "global_cues.csv"
    if cache.exists() and not force_refresh:
        age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        if age < 8:
            return pd.read_csv(cache, parse_dates=["date"])
    # Build minimal global from what we can get
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
    return None


def load_fii_dii_data(force_refresh: bool = False) -> pd.DataFrame | None:
    from settings import DATA_DIR, TRAINING_DAYS
    manual = DATA_DIR / "fii_dii.csv"
    cache  = DATA_DIR / "fii_dii_cache.csv"

    if manual.exists():
        try:
            df = pd.read_csv(manual, parse_dates=["date"])
            df.columns = [c.lower().strip() for c in df.columns]
            for col in df.columns:
                if "fii" in col and "net" in col: df = df.rename(columns={col:"fii_net"})
                if "dii" in col and "net" in col: df = df.rename(columns={col:"dii_net"})
            if "fii_net" in df.columns:
                if "dii_net" not in df.columns: df["dii_net"] = 0.0
                df["fii_net"] = pd.to_numeric(df["fii_net"], errors="coerce").fillna(0)
                df["dii_net"] = pd.to_numeric(df["dii_net"], errors="coerce").fillna(0)
                return df[["date","fii_net","dii_net"]].sort_values("date").reset_index(drop=True)
        except Exception: pass

    if cache.exists() and not force_refresh:
        age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        if age < 8:
            return pd.read_csv(cache, parse_dates=["date"])

    df = fetch_fii_nsdl(TRAINING_DAYS)
    if df is not None:
        df.to_csv(cache, index=False)
        return df

    # Zero stub
    dates = pd.date_range(datetime.now() - timedelta(days=TRAINING_DAYS),
                          datetime.now(), freq="B")
    df = pd.DataFrame({"date": dates, "fii_net": 0.0, "dii_net": 0.0})
    df.to_csv(cache, index=False)
    return df


def load_gift_data(breeze=None, force_refresh: bool = False) -> pd.DataFrame | None:
    from settings import DATA_DIR
    cache = DATA_DIR / "gift_nifty.csv"
    if cache.exists() and not force_refresh:
        age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        if age < 8:
            return pd.read_csv(cache, parse_dates=["date"])
    if breeze is not None:
        try:
            raw = _breeze_hist(breeze, "GIFTNIFTY", "1day", 730)
            if raw is not None:
                df = raw[["date","close"]].rename(columns={"close":"gift_close"})
                df.to_csv(cache, index=False)
                return df
        except Exception: pass
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
    return None


def load_pcr_data(force_refresh: bool = False) -> pd.DataFrame | None:
    from settings import DATA_DIR
    manual = DATA_DIR / "pcr.csv"
    if manual.exists():
        try:
            df = pd.read_csv(manual, parse_dates=["date"])
            df.columns = [c.lower().strip() for c in df.columns]
            if "pcr" in df.columns:
                df["pcr"] = pd.to_numeric(df["pcr"], errors="coerce").fillna(1.0)
                return df[["date","pcr"]].sort_values("date").reset_index(drop=True)
        except Exception: pass
    return None
