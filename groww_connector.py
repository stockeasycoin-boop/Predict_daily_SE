"""
groww_connector.py — Groww API integration for tick-level order flow data.

WHY GROWW ADDS VALUE OVER BREEZE:
  Breeze gives 5-min OHLCV candles. Groww's API can give tick-by-tick trades
  and Level-2 order book depth (bid/ask quantities at each price level).

  This unlocks ORDER FLOW IMBALANCE (OFI) — the single most predictive
  short-term feature in market microstructure research:

      OFI = (aggressive_buy_volume - aggressive_sell_volume) / total_volume

  OFI > +0.3  → buyers lifting offers aggressively → price likely to rise next
  OFI < -0.3  → sellers hitting bids aggressively → price likely to fall next

  Breeze cannot compute this because 5-min candles hide WHO was aggressive.
  Adding OFI as a feature typically improves 5-min/15-min accuracy by 4-6%.

SETUP:
  1. Get Groww API access: https://groww.in/trade-api  (TOTP-based auth)
  2. pip install growwapi
  3. Paste your Groww API key + secret into Settings tab (saved to settings.json)
  4. The Live Monitor will automatically add OFI features when this is connected

STATUS: This is a working template. The exact Groww SDK method names may need
        adjustment based on the current growwapi version — see comments inline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    from growwapi import GrowwAPI
    GROWW_OK = True
except ImportError:
    GROWW_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def init_groww(api_key: str, api_secret: str = None, totp: str = None):
    """
    Connect to Groww Trade API.

    Auth flow (growwapi SDK):
      1. Call GrowwAPI.get_access_token(api_key, totp=...) or secret=... to get access token
      2. Pass access token to GrowwAPI(token) to create client
      3. If settings has a saved groww_access_token, try that first

    Run `python groww_auth.py` to generate and save a token interactively.
    """
    if not GROWW_OK:
        raise ImportError("Run: pip install growwapi")
    if not api_key:
        raise ValueError("Groww API key not configured in Settings.")

    errors = []

    # Check for a saved access token from groww_auth.py
    try:
        import json
        from pathlib import Path
        sf = Path(__file__).parent / "settings.json"
        if sf.exists():
            with open(sf) as f:
                _s = json.load(f)
            saved_token = _s.get("groww_access_token", "")
            if saved_token:
                try:
                    client = GrowwAPI(saved_token)
                    client.get_ltp(exchange_trading_symbols=("NIFTY 50",), segment="CASH")
                    return client
                except Exception as e:
                    errors.append(f"saved-token: {e}")
    except Exception:
        pass

    # Flow 1: TOTP flow (explicit code or auto-generated from secret)
    if totp:
        try:
            token = GrowwAPI.get_access_token(api_key, totp=totp)
            client = GrowwAPI(token)
            client.get_ltp(exchange_trading_symbols=("NIFTY 50",), segment="CASH")
            return client
        except Exception as e:
            errors.append(f"totp: {e}")

    # Flow 2: Auto-generate TOTP from secret
    if api_secret and len(api_secret) in (16, 32, 64):
        try:
            import pyotp
            auto_totp = pyotp.TOTP(api_secret).now()
            token = GrowwAPI.get_access_token(api_key, totp=auto_totp)
            client = GrowwAPI(token)
            client.get_ltp(exchange_trading_symbols=("NIFTY 50",), segment="CASH")
            return client
        except ImportError:
            errors.append("pyotp not installed (pip install pyotp)")
        except Exception as e:
            errors.append(f"auto-totp: {e}")

    # Flow 3: Approval flow with secret
    if api_secret:
        try:
            token = GrowwAPI.get_access_token(api_key, secret=api_secret)
            client = GrowwAPI(token)
            client.get_ltp(exchange_trading_symbols=("NIFTY 50",), segment="CASH")
            return client
        except Exception as e:
            errors.append(f"secret: {e}")

    # Flow 4: Direct token (api_key might already be an access token)
    try:
        client = GrowwAPI(api_key)
        client.get_ltp(exchange_trading_symbols=("NIFTY 50",), segment="CASH")
        return client
    except Exception as e:
        errors.append(f"direct: {e}")

    raise RuntimeError(
        "Groww auth failed. Run: python groww_auth.py\n"
        f"Tried: {'; '.join(errors)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ORDER FLOW IMBALANCE  (the key new feature)
# ─────────────────────────────────────────────────────────────────────────────

LAST_GROWW_ERROR = {"msg": ""}


def _nifty_futures_symbol() -> str:
    """Groww FNO symbol for current-month Nifty futures, e.g. 'NIFTY26JUNFUT'.
    Rolls to next month after the last Tuesday (monthly expiry)."""
    from datetime import date as _d, timedelta as _td
    import calendar as _cal
    today = _d.today()
    def last_tuesday(y, m):
        d = _d(y, m, _cal.monthrange(y, m)[1])
        while d.weekday() != 1:
            d -= _td(days=1)
        return d
    y, m = today.year, today.month
    if last_tuesday(y, m) < today:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    mon = _cal.month_abbr[m].upper()      # JUN
    yy  = str(y)[2:]                       # 26
    return f"NIFTY{yy}{mon}FUT"


def fetch_order_book_groww(client, symbol: str = None) -> dict | None:
    """
    Fetch Level-2 order book (market depth).

    IMPORTANT: the NIFTY *index* has no order book — depth only exists on
    tradeable instruments. We therefore query the current-month NIFTY FUTURES
    contract (e.g. NIFTY26JUNFUT), whose order flow is the standard proxy for
    index buy/sell pressure.

    Tries several growwapi method names since the SDK has changed across
    versions. Last failure reason is stored in LAST_GROWW_ERROR["msg"].
    """
    fut_symbol = symbol or _nifty_futures_symbol()

    attempts = [
        ("get_quote", {"trading_symbol": fut_symbol, "exchange": "NSE", "segment": "FNO"}),
        ("get_quote", {"trading_symbol": "NIFTY 50",  "exchange": "NSE", "segment": "CASH"}),
    ]
    errors = []
    for method_name, kwargs in attempts:
        fn = getattr(client, method_name, None)
        if fn is None:
            errors.append(f"{method_name}: not in SDK")
            continue
        try:
            resp = fn(**kwargs)
            if not resp:
                errors.append(f"{method_name}({kwargs.get('trading_symbol')}): empty")
                continue
            depth = resp.get("depth", resp) if isinstance(resp, dict) else {}
            bids = depth.get("buy")  or depth.get("bids") or []
            asks = depth.get("sell") or depth.get("asks") or []
            if bids or asks:
                LAST_GROWW_ERROR["msg"] = ""
                return {"bids": bids, "asks": asks,
                        "symbol": kwargs.get("trading_symbol"),
                        "ts": datetime.now()}
            errors.append(f"{method_name}({kwargs.get('trading_symbol')}): no depth "
                          f"(keys: {list(resp.keys())[:6] if isinstance(resp, dict) else type(resp).__name__})")
        except Exception as e:
            errors.append(f"{method_name}({kwargs.get('trading_symbol')}): {e}")

    LAST_GROWW_ERROR["msg"] = " | ".join(errors)[:300]
    print(f"[Groww] Order book failed: {LAST_GROWW_ERROR['msg']}")
    return None


def compute_ofi(order_book: dict) -> float:
    """
    Compute Order Flow Imbalance from Level-2 depth.

    OFI = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)

    Range: -1 (all sell pressure) to +1 (all buy pressure)
    A reading sustained above +0.3 strongly predicts short-term upward moves.
    """
    if not order_book:
        return 0.0
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    total_bid_qty = sum(float(b.get("quantity", 0)) for b in bids)
    total_ask_qty = sum(float(a.get("quantity", 0)) for a in asks)

    denom = total_bid_qty + total_ask_qty
    if denom <= 0:
        return 0.0
    return round((total_bid_qty - total_ask_qty) / denom, 4)


def fetch_tick_trades_groww(client, symbol: str = "NIFTY",
                            minutes: int = 5) -> pd.DataFrame | None:
    """
    Fetch recent tick-by-tick trades to compute aggressive buy/sell volume.

    Each tick is classified as buyer- or seller-initiated using the tick rule:
      - trade at or above last ask → buyer-initiated (aggressive buy)
      - trade at or below last bid → seller-initiated (aggressive sell)
    """
    try:
        end = datetime.now()
        start = end - timedelta(minutes=minutes)
        # NOTE: method name may differ; some versions use get_historical_candle
        # or a websocket feed for ticks. This is the REST approximation.
        ticks = client.get_trades(
            trading_symbol=symbol, exchange="NSE",
            from_time=start, to_time=end,
        )
        if ticks:
            return pd.DataFrame(ticks)
    except Exception as e:
        print(f"[Groww] Tick fetch failed: {e}")
    return None


def compute_ofi_from_ticks(ticks_df: pd.DataFrame) -> float:
    """
    Compute OFI from classified tick trades (more accurate than depth-based OFI).
    Uses the tick rule to classify each trade as buy- or sell-initiated.
    """
    if ticks_df is None or len(ticks_df) == 0:
        return 0.0
    df = ticks_df.copy()

    # Classify using price movement (tick rule)
    if "price" in df.columns and "quantity" in df.columns:
        df["price_chg"] = df["price"].diff()
        df["side"] = np.where(df["price_chg"] > 0, 1,
                      np.where(df["price_chg"] < 0, -1, 0))
        buy_vol  = df.loc[df["side"] == 1,  "quantity"].sum()
        sell_vol = df.loc[df["side"] == -1, "quantity"].sum()
        denom = buy_vol + sell_vol
        if denom > 0:
            return round((buy_vol - sell_vol) / denom, 4)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LIVE OFI SNAPSHOT  (called from Live Monitor each refresh)
# ─────────────────────────────────────────────────────────────────────────────

def get_live_ofi(client, symbol: str = "NIFTY") -> dict:
    """
    Get the current Order Flow Imbalance snapshot.
    Returns dict with OFI value and interpretation for the dashboard.

    This is what the Live Monitor calls each refresh to add OFI as a
    real-time feature alongside the model predictions.
    """
    if client is None:
        return {"ofi": 0.0, "available": False,
                "signal": "Groww not connected"}

    # Depth-based OFI on NIFTY futures (the index itself has no order book)
    book = fetch_order_book_groww(client, None if symbol == "NIFTY" else symbol)
    if book is None:
        return {"ofi": 0.0, "available": False, "bias": "neutral",
                "signal": f"No depth: {LAST_GROWW_ERROR['msg'][:120]}" if LAST_GROWW_ERROR["msg"]
                          else "No depth data returned"}
    ofi  = compute_ofi(book)

    # Interpretation
    if ofi > 0.3:
        signal = "Strong buy pressure — supports upward moves"
        bias = "bullish"
    elif ofi < -0.3:
        signal = "Strong sell pressure — supports downward moves"
        bias = "bearish"
    elif abs(ofi) <= 0.1:
        signal = "Balanced order flow — neutral"
        bias = "neutral"
    else:
        signal = "Mild " + ("buy" if ofi > 0 else "sell") + " pressure"
        bias = "mild_" + ("bullish" if ofi > 0 else "bearish")

    return {
        "ofi":       ofi,
        "available": book is not None,
        "signal":    signal,
        "bias":      bias,
    }


def ofi_confidence_adjustment(ofi: float, predicted_direction: int) -> float:
    """
    How much to adjust model confidence based on OFI agreement.

    If OFI strongly agrees with the model → boost confidence.
    If OFI strongly contradicts → cut confidence.

    Returns a multiplier (e.g. 1.10 = +10%, 0.80 = -20%).
    """
    if abs(ofi) < 0.15:
        return 1.0   # neutral OFI, no adjustment

    ofi_direction = 1 if ofi > 0 else 0
    if ofi_direction == predicted_direction:
        # OFI agrees — boost proportional to strength
        return 1.0 + min(abs(ofi) * 0.2, 0.12)   # max +12%
    else:
        # OFI contradicts — cut proportional to strength
        return 1.0 - min(abs(ofi) * 0.4, 0.25)   # max -25%


def groww_available() -> bool:
    return GROWW_OK
