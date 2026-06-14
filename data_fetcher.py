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
    # Use IST for date boundaries — NSE operates in IST. Using the server's
    # local time (which may be US/UTC) can make "today" resolve to yesterday,
    # excluding the current day's intraday candles entirely.
    import pytz
    _ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(_ist)
    # +1 day buffer on the end date guarantees today's candles are captured
    end   = now_ist + timedelta(days=1)
    start = now_ist - timedelta(days=days)
    # Intraday: from start of range to tomorrow (captures all of today).
    # Daily: 07:00 UTC anchor is fine since we only need the date.
    if interval == "1day":
        from_str = start.strftime("%Y-%m-%dT07:00:00.000Z")
        to_str   = end.strftime("%Y-%m-%dT07:00:00.000Z")
    else:
        from_str = start.strftime("%Y-%m-%dT00:00:00.000Z")
        to_str   = end.strftime("%Y-%m-%dT07:00:00.000Z")
    for attempt in range(retries + 1):
        try:
            resp = breeze.get_historical_data_v2(
                interval=interval,
                from_date=from_str,
                to_date=to_str,
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
              f"({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()})")
    return df


# Last error per source — surfaced in the Data Sources tab for diagnosis
LAST_FETCH_ERRORS = {}

def fetch_vix_breeze(breeze, days: int = 730) -> pd.DataFrame | None:
    """India VIX history. Breeze security-master codes vary by version, so we
    try the known aliases in order: INDVIX, INDIAVIX, INDIA VIX."""
    for code in ("INDVIX", "INDIAVIX", "INDIA VIX", "INDIAVIX-INDEX"):
        try:
            df = _breeze_hist(breeze, code, "1day", min(days, 730))
            if df is not None and len(df) > 0:
                out = df[["date","close"]].rename(columns={"close":"india_vix"})
                print(f"[Breeze] VIX via '{code}': {len(out)} rows")
                LAST_FETCH_ERRORS.pop("vix", None)
                return out
        except Exception as e:
            LAST_FETCH_ERRORS["vix"] = f"{code}: {e}"
            continue
    if "vix" not in LAST_FETCH_ERRORS:
        LAST_FETCH_ERRORS["vix"] = "All VIX stock codes returned empty (INDVIX/INDIAVIX/INDIA VIX)"
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
              f"({df['date'].dt.date.min()} to {df['date'].dt.date.max()})")
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

def fetch_live_quote_breeze(breeze) -> dict | None:
    try:
        resp = breeze.get_quotes(
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
    except Exception as e:
        print(f"[Breeze] live quote failed: {e}")
    return None


def _next_monthly_futures_expiry() -> str:
    """Last Tuesday of the current month (Nifty monthly F&O expiry).
    If already past, roll to next month. Breeze ISO format."""
    from datetime import date as _d, timedelta as _td
    import calendar as _cal
    today = _d.today()
    def last_tuesday(year, month):
        last_day = _cal.monthrange(year, month)[1]
        d = _d(year, month, last_day)
        while d.weekday() != 1:   # 1 = Tuesday
            d -= _td(days=1)
        return d
    exp = last_tuesday(today.year, today.month)
    if exp < today:
        ny, nm = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        exp = last_tuesday(ny, nm)
    return exp.strftime("%Y-%m-%dT07:00:00.000Z")


def fetch_gift_nifty_breeze(breeze) -> float | None:
    """
    Pre-market / live forward-looking Nifty level.

    True GIFT Nifty trades on NSE-IX (GIFT City), which Breeze generally does
    NOT carry. Attempt order:
      1. GIFTNIFTY listings (in case the account has access)
      2. NIFTY current-month futures on NFO — a near-equivalent lead indicator
         during market hours (futures basis ≈ GIFT basis intraday).
    Returns the LTP as a float, or None. Last error in LAST_FETCH_ERRORS["gift"].
    """
    expiry = _next_monthly_futures_expiry()
    attempts = [
        {"stock_code": "GIFTNIFTY", "exchange_code": "NSE", "product_type": "futures", "expiry_date": ""},
        {"stock_code": "GIFTNIFTY", "exchange_code": "NFO", "product_type": "futures", "expiry_date": expiry},
        {"stock_code": "NIFTY",     "exchange_code": "NFO", "product_type": "futures", "expiry_date": expiry},
    ]
    for params in attempts:
        try:
            resp = breeze.get_quotes(
                stock_code=params["stock_code"],
                exchange_code=params["exchange_code"],
                product_type=params["product_type"],
                expiry_date=params["expiry_date"], right="", strike_price="",
            )
            if resp.get("Status") == 200 and resp.get("Success"):
                d   = resp["Success"][0]
                ltp = float(d.get("ltp", 0) or 0)
                if ltp > 0:
                    LAST_FETCH_ERRORS.pop("gift", None)
                    if params["stock_code"] == "NIFTY":
                        LAST_FETCH_ERRORS["gift_note"] = "Using NIFTY futures as GIFT proxy"
                    return ltp
            else:
                LAST_FETCH_ERRORS["gift"] = str(resp.get("Error") or resp.get("Status"))[:140]
        except Exception as e:
            LAST_FETCH_ERRORS["gift"] = f"{params['stock_code']}@{params['exchange_code']}: {e}"[:140]
            continue
    return None


def fetch_options_chain_breeze(breeze, expiry_str: str,
                                spot: float) -> tuple[pd.DataFrame | None, float]:
    from settings import NIFTY_STRIKE_GAP
    atm     = int(round(spot / NIFTY_STRIKE_GAP) * NIFTY_STRIKE_GAP)
    strikes = [atm + i * NIFTY_STRIKE_GAP for i in range(-5, 6)]
    rows, ce_oi, pe_oi = [], 0.0, 0.0
    for strike in strikes:
        for right in ["call", "put"]:
            try:
                resp = breeze.get_quotes(
                    stock_code="NIFTY", exchange_code="NFO",
                    product_type="options", expiry_date=expiry_str,
                    right=right, strike_price=str(int(strike)),
                )
                if resp.get("Status") == 200 and resp.get("Success"):
                    d    = resp["Success"][0]
                    oi   = float(d.get("open_interest",      0) or 0)
                    ltp  = float(d.get("ltp",                0) or 0)
                    opt  = "CE" if right == "call" else "PE"
                    if opt == "CE": ce_oi += oi
                    else:           pe_oi += oi
                    rows.append({
                        "strike": strike, "type":   opt,    "ltp":    ltp,
                        "bid":   float(d.get("best_bid_price",   0) or 0),
                        "ask":   float(d.get("best_offer_price", 0) or 0),
                        "oi":    oi,
                        "volume":float(d.get("volume",           0) or 0),
                        "iv":    float(d.get("implied_volatility",0) or 0),
                    })
            except Exception:
                pass
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
    """
    Load intraday 5-min candles.

    NOTE on Breeze real-time limitation:
      Breeze's historical 5-min endpoint often does NOT return the current
      day's candles in real time — they appear with a delay or only after
      market close. The LIVE spot price (from get_quotes) is real-time, which
      is why predictions anchor to that instead of the candle close.

    Cache strategy:
      - During market hours: short 3-min cache so we keep polling for fresh
        candles as Breeze makes them available.
      - Outside market hours: 8-hour cache (candles won't change).
    """
    from settings import DATA_DIR
    import pytz
    cache = DATA_DIR / f"intraday_{stock_code.lower()}.csv"

    # Determine cache freshness window based on market hours
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)
    is_mkt = (now_ist.weekday() < 5 and
              now_ist.replace(hour=9, minute=15) <= now_ist <= now_ist.replace(hour=15, minute=30))
    cache_hrs = 0.05 if is_mkt else 8   # 0.05 hr = 3 min during market hours

    if cache.exists() and not force_refresh:
        age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        if age < cache_hrs:
            return pd.read_csv(cache, parse_dates=["date"])

    if breeze is None:
        if cache.exists():
            return pd.read_csv(cache, parse_dates=["date"])
        return None

    df = fetch_intraday_breeze(breeze, stock_code, days_back=60)
    if df is not None and len(df) > 0:
        df.to_csv(cache, index=False)
        return df
    # If fetch failed but we have cache, return it rather than nothing
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
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
        raw = None
        for _code, _exch, _prod in [
            ("GIFTNIFTY", "NSE", "cash"),
            ("GIFTNIFTY", "NFO", "futures"),
            ("NIFTY", "NFO", "futures"),
        ]:
            try:
                expiry = _next_monthly_futures_expiry() if _prod == "futures" else ""
                import pytz
                _ist = pytz.timezone("Asia/Kolkata")
                now_ist = datetime.now(_ist)
                end = now_ist + timedelta(days=1)
                start = now_ist - timedelta(days=730)
                resp = breeze.get_historical_data_v2(
                    interval="1day",
                    from_date=start.strftime("%Y-%m-%dT07:00:00.000Z"),
                    to_date=end.strftime("%Y-%m-%dT07:00:00.000Z"),
                    stock_code=_code, exchange_code=_exch,
                    product_type=_prod, expiry_date=expiry,
                    right="", strike_price="",
                )
                if resp.get("Status") == 200 and resp.get("Success"):
                    df = pd.DataFrame(resp["Success"])
                    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
                    df["close"] = pd.to_numeric(df["close"], errors="coerce")
                    df = df.dropna(subset=["close"])
                    if len(df) > 0:
                        raw = df[["date", "close"]].rename(columns={"close": "gift_close"})
                        raw = raw.sort_values("date").drop_duplicates("date").reset_index(drop=True)
                        LAST_FETCH_ERRORS.pop("gift_hist", None)
                        if _code == "NIFTY":
                            LAST_FETCH_ERRORS["gift_hist_note"] = "Using NIFTY futures daily as GIFT proxy"
                        break
            except Exception as e:
                LAST_FETCH_ERRORS["gift_hist"] = f"{_code}@{_exch}: {e}"[:140]
                continue
        if raw is not None and len(raw) > 0:
            raw.to_csv(cache, index=False)
            return raw
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKED INTRADAY FETCH  (bypass 60-day cap)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:15–15:30 IST candles."""
    df = df[(df["date"].dt.hour > 9) | ((df["date"].dt.hour == 9) & (df["date"].dt.minute >= 15))]
    df = df[(df["date"].dt.hour < 15) | ((df["date"].dt.hour == 15) & (df["date"].dt.minute <= 30))]
    return df


def fetch_intraday_chunked(breeze, stock_code: str = "NIFTY",
                           total_days: int = 730, chunk_days: int = 12) -> pd.DataFrame:
    """
    Fetch 5-min intraday candles in small windows to bypass Breeze's 1000-row cap.
    Default chunk_days=12 keeps each request under 1000 candles (12 days * 75 candles).
    Loops backwards from today, deduplicates, returns combined DataFrame sorted by date.
    """
    import pytz
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)
    all_chunks = []
    cursor = now_ist
    oldest = now_ist - timedelta(days=total_days)

    while cursor > oldest:
        end_dt = cursor + timedelta(days=1)
        start_dt = cursor - timedelta(days=chunk_days)
        if start_dt < oldest:
            start_dt = oldest
        from_str = start_dt.strftime("%Y-%m-%dT00:00:00.000Z")
        to_str = end_dt.strftime("%Y-%m-%dT07:00:00.000Z")
        try:
            resp = breeze.get_historical_data_v2(
                interval="5minute", from_date=from_str, to_date=to_str,
                stock_code=stock_code, exchange_code="NSE", product_type="cash",
            )
            if resp.get("Status") == 200 and resp.get("Success"):
                chunk = pd.DataFrame(resp["Success"])
                chunk["date"] = pd.to_datetime(chunk["datetime"])
                for col in ["open", "high", "low", "close"]:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                chunk["volume"] = pd.to_numeric(chunk.get("volume", 0), errors="coerce").fillna(0)
                chunk = chunk[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "close"])
                all_chunks.append(chunk)
                print(f"[Chunked] {stock_code} {start_dt.date()} to {cursor.date()}: {len(chunk)} candles")
        except Exception as e:
            print(f"[Chunked] {stock_code} chunk error: {e}")
        cursor = start_dt - timedelta(days=1)
        time.sleep(0.5)

    if not all_chunks:
        return pd.DataFrame()
    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    combined = _filter_market_hours(combined)
    print(f"[Chunked] Total {stock_code}: {len(combined)} candles "
          f"({combined['date'].dt.date.min()} to {combined['date'].dt.date.max()})")
    return combined


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
