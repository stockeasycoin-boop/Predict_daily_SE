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

_BREEZE_MAX_CANDLES = 1000

def _parse_breeze_response(resp: dict, interval: str) -> pd.DataFrame | None:
    if resp.get("Status") == 200 and resp.get("Success"):
        df = pd.DataFrame(resp["Success"])
        df["datetime_raw"] = pd.to_datetime(df["datetime"])
        if interval == "1day":
            df["date"] = df["datetime_raw"].dt.normalize()
        else:
            df["date"] = df["datetime_raw"]
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(
            df.get("volume", 0), errors="coerce").fillna(0)
        cols = ["date", "open", "high", "low", "close", "volume"]
        return df[cols].dropna(subset=["open", "close"])
    return None


def _breeze_single_request(breeze, stock_code: str, interval: str,
                           start: datetime, end: datetime,
                           retries: int = 2) -> pd.DataFrame | None:
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
            df = _parse_breeze_response(resp, interval)
            if df is not None:
                return df
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"[Breeze] {stock_code} ({interval}) failed: {e}")
    return None


def _breeze_hist(breeze, stock_code: str, interval: str = "1day",
                 days: int = 730, retries: int = 2) -> pd.DataFrame | None:
    """
    Generic Breeze historical data fetch.
    Daily data uses a single request (API returns up to 730 days).
    Intraday uses chunked requests (API caps at 1000 candles/request).
    """
    end = datetime.now()
    start = end - timedelta(days=min(days, 730))

    if interval == "1day":
        return _breeze_single_request(breeze, stock_code, interval,
                                      start, end, retries)

    return _breeze_hist_chunked(breeze, stock_code, interval,
                                start, end, retries)


def _breeze_hist_chunked(breeze, stock_code: str, interval: str,
                         start: datetime, end: datetime,
                         retries: int = 2) -> pd.DataFrame | None:
    """
    Fetch intraday data in small chunks to stay under the 1000-candle
    per-request limit.  15 calendar days ~ 10 trading days ~ 750 candles.
    """
    chunk_days = 15
    all_chunks = []
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        df = _breeze_single_request(breeze, stock_code, interval,
                                    current, chunk_end, retries)
        if df is not None and len(df) > 0:
            all_chunks.append(df)
        current = chunk_end
        time.sleep(0.5)

    if not all_chunks:
        return None

    merged = pd.concat(all_chunks, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
    return merged.reset_index(drop=True)


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

def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 9:15 AM - 15:30 PM IST candles."""
    mask = (
        ((df["date"].dt.hour > 9) |
         ((df["date"].dt.hour == 9) & (df["date"].dt.minute >= 15))) &
        ((df["date"].dt.hour < 15) |
         ((df["date"].dt.hour == 15) & (df["date"].dt.minute <= 30)))
    )
    return df[mask].reset_index(drop=True)


def fetch_intraday_breeze(breeze, stock_code: str = "NIFTY",
                          days_back: int = 730) -> pd.DataFrame | None:
    """
    Fetch 5-minute intraday candles for the past `days_back` calendar days.
    Uses chunked requests to fetch up to 2 years of data.
    """
    df = _breeze_hist(breeze, stock_code, "5minute", days_back)
    if df is not None and len(df) > 0:
        df = _filter_market_hours(df)
        if len(df) > 0:
            print(f"[Breeze] Intraday {stock_code} 5min: {len(df)} candles "
                  f"({df['date'].dt.date.min()} -> {df['date'].dt.date.max()})")
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

    cached_df = None
    if cache.exists():
        cached_df = pd.read_csv(cache, parse_dates=["date"])
        if not force_refresh:
            last_cached = cached_df["date"].max()
            hours_since = (datetime.now() - last_cached).total_seconds() / 3600
            if hours_since < 8:
                print(f"[Cache] Nifty: {len(cached_df)} rows (fresh)")
                return cached_df.sort_values("date").reset_index(drop=True)

    if breeze is None:
        if cached_df is not None:
            print(f"[Cache] Using stale cache: {len(cached_df)} rows")
            return cached_df.sort_values("date").reset_index(drop=True)
        print("[Nifty] No Breeze session and no cache.")
        return None

    if cached_df is not None and not force_refresh and len(cached_df) > 0:
        last_date = cached_df["date"].max()
        gap_days = (datetime.now() - last_date).days + 1
        if gap_days <= 1:
            return cached_df.sort_values("date").reset_index(drop=True)
        print(f"[Breeze] Nifty incremental: {gap_days} days since {last_date.date()}")
        new_df = fetch_nifty_breeze(breeze, gap_days)
        if new_df is not None and len(new_df) > 0:
            merged = pd.concat([cached_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
            merged = merged.reset_index(drop=True)
            merged.to_csv(cache, index=False)
            print(f"[Cache] Nifty: updated {len(cached_df)} -> {len(merged)} rows")
            return merged
        return cached_df.sort_values("date").reset_index(drop=True)

    df = fetch_nifty_breeze(breeze, days)
    if df is not None:
        df.to_csv(cache, index=False)
    return df


def load_vix_data(breeze=None, force_refresh: bool = False) -> pd.DataFrame | None:
    from settings import DATA_DIR, TRAINING_DAYS
    cache = DATA_DIR / "india_vix.csv"

    cached_df = None
    if cache.exists():
        cached_df = pd.read_csv(cache, parse_dates=["date"])
        if not force_refresh:
            last_cached = cached_df["date"].max()
            hours_since = (datetime.now() - last_cached).total_seconds() / 3600
            if hours_since < 8:
                return cached_df.sort_values("date").reset_index(drop=True)

    if breeze is None:
        if cached_df is not None:
            return cached_df.sort_values("date").reset_index(drop=True)
        return None

    if cached_df is not None and not force_refresh and len(cached_df) > 0:
        last_date = cached_df["date"].max()
        gap_days = (datetime.now() - last_date).days + 1
        if gap_days <= 1:
            return cached_df.sort_values("date").reset_index(drop=True)
        new_df = fetch_vix_breeze(breeze, gap_days)
        if new_df is not None and len(new_df) > 0:
            merged = pd.concat([cached_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
            merged = merged.reset_index(drop=True)
            merged.to_csv(cache, index=False)
            return merged
        return cached_df.sort_values("date").reset_index(drop=True)

    df = fetch_vix_breeze(breeze, TRAINING_DAYS)
    if df is not None:
        df.to_csv(cache, index=False)
    return df


def load_intraday_data(breeze=None, force_refresh: bool = False,
                       stock_code: str = "NIFTY",
                       days_back: int = 730) -> pd.DataFrame | None:
    """
    Load intraday 5-min candles with incremental caching.

    First call fetches the full history (up to `days_back` calendar days).
    Subsequent calls only fetch the gap since the last cached date and
    append to the existing cache file, avoiding redundant API calls.
    """
    from settings import DATA_DIR
    cache = DATA_DIR / f"intraday_{stock_code.lower()}.csv"

    cached_df = None
    if cache.exists():
        cached_df = pd.read_csv(cache, parse_dates=["date"])
        if not force_refresh:
            last_cached = cached_df["date"].max()
            hours_since = (datetime.now() - last_cached).total_seconds() / 3600
            if hours_since < 8:
                print(f"[Cache] Intraday {stock_code}: {len(cached_df)} candles "
                      f"(fresh, last={last_cached.date()})")
                return cached_df.sort_values("date").reset_index(drop=True)

    if breeze is None:
        if cached_df is not None:
            return cached_df.sort_values("date").reset_index(drop=True)
        return None

    if cached_df is not None and not force_refresh and len(cached_df) > 0:
        last_date = cached_df["date"].max()
        gap_days = (datetime.now() - last_date).days + 1
        if gap_days <= 1:
            return cached_df.sort_values("date").reset_index(drop=True)
        print(f"[Breeze] Incremental fetch: {gap_days} days since {last_date.date()}")
        new_df = fetch_intraday_breeze(breeze, stock_code, days_back=gap_days)
        if new_df is not None and len(new_df) > 0:
            merged = pd.concat([cached_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
            merged = merged.reset_index(drop=True)
            merged.to_csv(cache, index=False)
            print(f"[Cache] Intraday {stock_code}: updated {len(cached_df)} -> "
                  f"{len(merged)} candles")
            return merged
        return cached_df.sort_values("date").reset_index(drop=True)

    df = fetch_intraday_breeze(breeze, stock_code, days_back=days_back)
    if df is not None and len(df) > 0:
        df.to_csv(cache, index=False)
        print(f"[Cache] Intraday {stock_code}: saved {len(df)} candles")
    return df


def load_correlated_data(breeze=None,
                         force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Load Bank Nifty + sector indices with incremental caching."""
    from settings import DATA_DIR, TRAINING_DAYS, CORRELATED_INSTRUMENTS
    result = {}

    for sym in CORRELATED_INSTRUMENTS:
        cache = DATA_DIR / f"corr_{sym.lower()}.csv"

        cached_df = None
        if cache.exists():
            cached_df = pd.read_csv(cache, parse_dates=["date"])
            if not force_refresh:
                last_cached = cached_df["date"].max()
                hours_since = (datetime.now() - last_cached).total_seconds() / 3600
                if hours_since < 8:
                    result[sym] = cached_df.sort_values("date").reset_index(drop=True)
                    continue

        if breeze is not None:
            if cached_df is not None and not force_refresh and len(cached_df) > 0:
                last_date = cached_df["date"].max()
                gap_days = (datetime.now() - last_date).days + 1
                if gap_days <= 1:
                    result[sym] = cached_df.sort_values("date").reset_index(drop=True)
                    continue
                new_df = _breeze_hist(breeze, sym, "1day", gap_days)
                if new_df is not None and len(new_df) > 0:
                    merged = pd.concat([cached_df, new_df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
                    merged = merged.reset_index(drop=True)
                    merged.to_csv(cache, index=False)
                    result[sym] = merged
                    continue
                result[sym] = cached_df.sort_values("date").reset_index(drop=True)
                continue

            df = _breeze_hist(breeze, sym, "1day", min(TRAINING_DAYS, 730))
            if df is not None and len(df) > 30:
                df.to_csv(cache, index=False)
                result[sym] = df
        elif cached_df is not None:
            result[sym] = cached_df.sort_values("date").reset_index(drop=True)

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
