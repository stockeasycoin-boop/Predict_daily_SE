"""
ofi_logger.py — persist live Order-Flow-Imbalance + options-flow snapshots.

The trained models were built on OHLCV only (no order-flow history exists yet).
This appends each live OFI snapshot to data/ofi_log.csv so that, after a few
weeks of collection, OFI / options-flow can be joined to the 5-min candles by
timestamp and included as real features in a future retrain.

Design: append-only, best-effort, never raises into the caller. One row per
refresh; deduped by (minute, source) so rapid reruns don't spam duplicates.
"""
import csv
from pathlib import Path
from datetime import datetime

try:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
except Exception:
    _IST = None

_LOG_PATH = Path(__file__).parent / "data" / "ofi_log.csv"
_FIELDS = ["ts_ist", "minute_key", "source", "available", "spot", "ofi", "bias",
           "pcr", "max_pain", "max_ce_strike", "max_pe_strike", "signal"]
_last_key = {"v": None}


def _now_ist() -> datetime:
    if _IST is not None:
        return datetime.now(_IST)
    return datetime.now()


def log_ofi_snapshot(ofi_data: dict, spot: float = None) -> None:
    """Append one OFI/options-flow snapshot. Silent on any error."""
    try:
        if not ofi_data:
            return
        now = _now_ist()
        minute_key = now.strftime("%Y-%m-%d %H:%M")
        source = "groww" if "signal" in ofi_data and ofi_data.get("bias") in (
            "bullish", "bearish", "neutral", "mild_bullish", "mild_bearish") and \
            "pcr" not in ofi_data else ("options_chain" if "pcr" in ofi_data else "groww")
        key = (minute_key, source)
        if _last_key["v"] == key:      # already logged this minute+source
            return

        row = {
            "ts_ist":        now.strftime("%Y-%m-%d %H:%M:%S"),
            "minute_key":    minute_key,
            "source":        source,
            "available":     int(bool(ofi_data.get("available"))),
            "spot":          round(float(spot), 2) if spot else "",
            "ofi":           ofi_data.get("ofi", ""),
            "bias":          ofi_data.get("bias", ""),
            "pcr":           ofi_data.get("pcr", ""),
            "max_pain":      ofi_data.get("max_pain", ""),
            "max_ce_strike": ofi_data.get("max_ce_strike", ""),
            "max_pe_strike": ofi_data.get("max_pe_strike", ""),
            "signal":        str(ofi_data.get("signal", ""))[:160],
        }

        _LOG_PATH.parent.mkdir(exist_ok=True)
        new_file = not _LOG_PATH.exists()
        with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        _last_key["v"] = key
    except Exception:
        pass   # logging must never break the live monitor
