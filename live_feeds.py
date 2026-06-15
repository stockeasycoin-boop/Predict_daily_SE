"""
live_feeds.py — Breeze WebSocket live tick stream for GIFT Nifty + Nifty spot.

Instead of polling get_quotes() on every signal refresh, this module opens
a single WebSocket connection and keeps the latest tick prices in memory.
Any part of the app can read the latest price via get_latest().

Usage in app.py:
    import live_feeds
    live_feeds.start(breeze)          # once, after Breeze session is live
    gift = live_feeds.get_latest("gift")   # instant — no API call
    nifty = live_feeds.get_latest("nifty") # instant
"""

import threading
import time as _time
from datetime import datetime

_lock = threading.Lock()
_latest = {}   # {"gift": {"ltp": ..., "ts": ...}, "nifty": {...}, "nifty_fut": {...}}
_running = False
_breeze_ref = None


def _on_ticks(data: dict):
    """Callback fired by Breeze WebSocket on every tick."""
    if not data or not isinstance(data, dict):
        return
    ltp = data.get("ltp") or data.get("last")
    if not ltp:
        return
    try:
        ltp = float(ltp)
    except (ValueError, TypeError):
        return
    if ltp <= 0:
        return

    stock = (data.get("stock_code") or "").upper()
    exch = (data.get("exchange_code") or "").upper()
    prod = (data.get("product_type") or "").lower()

    ts = datetime.now().isoformat()
    tick = {
        "ltp": ltp,
        "ts": ts,
        "stock": stock,
        "exchange": exch,
        "open": _safe_float(data.get("open")),
        "high": _safe_float(data.get("high")),
        "low": _safe_float(data.get("low")),
        "prev_close": _safe_float(data.get("previous_close")),
        "volume": _safe_float(data.get("total_quantity_traded") or data.get("volume")),
    }

    with _lock:
        if stock == "GIFTNIFTY":
            _latest["gift"] = tick
        elif stock == "NIFTY" and prod == "futures":
            _latest["nifty_fut"] = tick
            if "gift" not in _latest:
                _latest["gift"] = tick
        elif stock == "NIFTY" and (exch == "NSE" or prod == "cash"):
            _latest["nifty"] = tick


def _safe_float(val):
    if val is None:
        return None
    try:
        v = float(val)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def _next_monthly_futures_expiry() -> str:
    from datetime import date as _d, timedelta as _td
    import calendar as _cal
    today = _d.today()
    def last_tuesday(year, month):
        last_day = _cal.monthrange(year, month)[1]
        d = _d(year, month, last_day)
        while d.weekday() != 1:
            d -= _td(days=1)
        return d
    exp = last_tuesday(today.year, today.month)
    if exp < today:
        ny, nm = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        exp = last_tuesday(ny, nm)
    return exp.strftime("%Y-%m-%dT07:00:00.000Z")


def start(breeze) -> bool:
    """
    Start the WebSocket stream and subscribe to GIFT Nifty + Nifty spot.
    Safe to call multiple times — only starts once.
    Returns True if started, False if already running.
    """
    global _running, _breeze_ref
    if _running:
        return False

    _breeze_ref = breeze
    expiry = _next_monthly_futures_expiry()

    try:
        breeze.on_ticks = _on_ticks
        breeze.ws_connect()

        # Subscribe to Nifty spot (cash)
        try:
            breeze.subscribe_feeds(
                stock_code="NIFTY", exchange_code="NSE",
                product_type="cash",
                get_exchange_quotes=True, get_market_depth=False,
            )
        except Exception as e:
            print(f"[LiveFeeds] Nifty spot subscribe failed: {e}")

        # Subscribe to GIFTNIFTY if available
        for code, exch in [("GIFTNIFTY", "NSE"), ("GIFTNIFTY", "NFO")]:
            try:
                breeze.subscribe_feeds(
                    stock_code=code, exchange_code=exch,
                    product_type="futures", expiry_date=expiry,
                    get_exchange_quotes=True, get_market_depth=False,
                )
                break
            except Exception:
                continue

        # Subscribe to Nifty futures as GIFT proxy
        try:
            breeze.subscribe_feeds(
                stock_code="NIFTY", exchange_code="NFO",
                product_type="futures", expiry_date=expiry,
                get_exchange_quotes=True, get_market_depth=False,
            )
        except Exception as e:
            print(f"[LiveFeeds] Nifty futures subscribe failed: {e}")

        _running = True
        print("[LiveFeeds] WebSocket stream started — GIFT + Nifty subscribed")
        return True

    except Exception as e:
        print(f"[LiveFeeds] WebSocket connect failed: {e}")
        return False


def stop():
    """Disconnect the WebSocket stream."""
    global _running
    if _breeze_ref and _running:
        try:
            _breeze_ref.ws_disconnect()
        except Exception:
            pass
    _running = False


def get_latest(key: str = "gift") -> dict | None:
    """
    Get the latest tick for a key: "gift", "nifty", "nifty_fut".
    Returns dict with ltp, ts, open, high, low, prev_close, volume
    or None if no tick received yet.
    """
    with _lock:
        return _latest.get(key)


def get_gift_price() -> float | None:
    """Get latest GIFT Nifty / futures LTP. Returns float or None."""
    tick = get_latest("gift")
    return tick["ltp"] if tick else None


def get_nifty_spot() -> dict | None:
    """Get latest Nifty spot tick with all fields."""
    return get_latest("nifty")


def get_all() -> dict:
    """Get all latest ticks."""
    with _lock:
        return dict(_latest)


def is_streaming() -> bool:
    """Check if WebSocket is connected and running."""
    return _running


def age_seconds(key: str = "gift") -> float | None:
    """Seconds since last tick for a key. None if no tick received."""
    tick = get_latest(key)
    if not tick:
        return None
    try:
        ts = datetime.fromisoformat(tick["ts"])
        return (datetime.now() - ts).total_seconds()
    except Exception:
        return None
