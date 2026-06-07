"""
options_engine.py — Converts model predictions into actionable options trades.

Given direction + confidence + market data, produces:
  - CE or PE to buy (or NO_TRADE if confidence is too low / VIX too high)
  - Strike price selection
  - Recommended lot size
  - Entry window, target premium, stop-loss premium
  - Full risk/reward breakdown
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

from settings import (
    CAPITAL_MIN, CAPITAL_MAX, MAX_LOSS_PCT,
    NIFTY_LOT_SIZE, NIFTY_STRIKE_GAP,
    TARGET_PCT, STOP_LOSS_PCT,
    TIME_EXIT_HOUR, TIME_EXIT_MIN,
    ENTRY_START_HOUR, ENTRY_START_MIN,
    ENTRY_END_HOUR, ENTRY_END_MIN,
    MIN_CONFIDENCE, MAX_VIX_FOR_TRADE,
)


# ─────────────────────────────────────────────────────────────────────────────
# EXPIRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def next_expiry(from_date: date = None) -> date:
    """Return the date of the next Tuesday (Nifty 50 weekly expiry).
    NSE moved Nifty weekly expiry from Thursday to Tuesday (effective Oct 2023).
    If today is Tuesday and market hasn't closed yet, still return today's expiry.
    """
    if from_date is None:
        from_date = date.today()
    # Tuesday = weekday 1
    days = (1 - from_date.weekday()) % 7
    return from_date + timedelta(days=days if days > 0 else 7)

# Keep old name as alias so nothing else breaks
next_thursday = next_expiry


def breeze_expiry_format(d: date) -> str:
    """Format date as Breeze API expects: '2024-11-28T07:00:00.000Z'"""
    return d.strftime("%Y-%m-%dT07:00:00.000Z")


# ─────────────────────────────────────────────────────────────────────────────
# STRIKE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_strike(spot: float, direction: int, predicted_move_pct: float) -> tuple[int, int, int]:
    """
    Choose the best strike based on expected move size.

    Rules
    -----
    Expected move < 0.5%  → ATM  (0 strikes OTM)
    Expected move 0.5–1%  → 1 OTM
    Expected move > 1%    → 2 OTM

    For CE (bullish) OTM = higher strike.
    For PE (bearish) OTM = lower strike.

    Returns: (selected_strike, atm_strike, otm_levels)
    """
    atm = int(round(spot / NIFTY_STRIKE_GAP) * NIFTY_STRIKE_GAP)

    if predicted_move_pct < 0.5:
        otm = 0
    elif predicted_move_pct < 1.0:
        otm = 1
    else:
        otm = 2

    if direction == 1:   # Bullish → CE → move strike higher
        strike = atm + otm * NIFTY_STRIKE_GAP
    else:                # Bearish → PE → move strike lower
        strike = atm - otm * NIFTY_STRIKE_GAP

    return int(strike), atm, otm


# ─────────────────────────────────────────────────────────────────────────────
# LOT SIZING
# ─────────────────────────────────────────────────────────────────────────────

def calc_lots(premium: float, capital: float) -> int:
    """
    Calculate how many lots to buy, respecting both capital and daily risk limits.

    Rules
    -----
    - Never spend more than capital on premium (cost = premium × lot_size × lots)
    - Max loss per lot = premium × STOP_LOSS_PCT × lot_size
    - Total max loss ≤ capital × MAX_LOSS_PCT
    - Hard cap: 3 lots (for safety during live trading)
    """
    if premium <= 0:
        return 1

    cost_per_lot     = premium * NIFTY_LOT_SIZE
    loss_per_lot     = premium * STOP_LOSS_PCT * NIFTY_LOT_SIZE
    daily_loss_limit = capital * MAX_LOSS_PCT

    max_by_capital = max(1, int(capital // cost_per_lot))
    max_by_risk    = max(1, int(daily_loss_limit // loss_per_lot))

    return min(max_by_capital, max_by_risk, 3)


# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM ESTIMATOR (fallback when live options chain unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_premium(spot: float, strike: int, option_type: str,
                     india_vix: float, days_to_expiry: int) -> float:
    """
    Simplified premium estimate using VIX as implied volatility proxy.
    Used only when Breeze options chain is not available.
    """
    iv    = india_vix / 100
    t     = max(days_to_expiry, 1) / 252
    moneyness = abs(spot - strike) / spot

    # ATM premium ≈ spot × IV × sqrt(T) × 0.4  (rough BSM at-the-money)
    atm_prem = spot * iv * np.sqrt(t) * 0.4

    # OTM adjustment: reduce premium as we go further OTM
    otm_factor = max(0.15, 1.0 - moneyness * 3)

    raw = atm_prem * otm_factor
    return max(round(raw / 0.5) * 0.5, 15.0)   # round to nearest 0.5, floor ₹15


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SUGGESTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_suggestion(
    direction: int,
    confidence: float,
    spot_price: float,
    atr_pct: float,
    india_vix: float,
    capital: float = None,
    options_df: pd.DataFrame = None,
) -> dict:
    """
    Generate a complete options trade suggestion (or NO_TRADE if conditions not met).

    Parameters
    ----------
    direction   : 1 = bullish, 0 = bearish
    confidence  : model probability (0–1)
    spot_price  : current Nifty spot price
    atr_pct     : ATR as % of price (proxy for expected move)
    india_vix   : India VIX value
    capital     : trade capital in ₹ (defaults to CAPITAL_MIN)
    options_df  : live options chain DataFrame (optional)

    Returns
    -------
    dict with all trade parameters (or signal='NO_TRADE' with reason)
    """
    if capital is None:
        capital = CAPITAL_MIN

    # ── NO TRADE conditions ───────────────────────────────────────────────
    if confidence < MIN_CONFIDENCE:
        return _no_trade(
            direction, confidence,
            f"Confidence {confidence:.0%} below threshold ({MIN_CONFIDENCE:.0%}). "
            f"Model is not sure enough today — skip and wait.",
        )

    if india_vix > MAX_VIX_FOR_TRADE:
        return _no_trade(
            direction, confidence,
            f"India VIX = {india_vix:.1f} (above {MAX_VIX_FOR_TRADE}). "
            f"Options premiums are very expensive in high-VIX markets — skip.",
        )

    # ── Signal ────────────────────────────────────────────────────────────
    opt_type  = "CE" if direction == 1 else "PE"
    signal    = f"BUY_{opt_type}"

    # Predicted move = 70% of ATR (conservative; ATR is the full average range)
    pred_move = round(min(atr_pct * 0.70, 2.5), 2)

    # Strike selection
    strike, atm, otm_level = select_strike(spot_price, direction, pred_move)

    # ── Premium ───────────────────────────────────────────────────────────
    premium = None
    expiry_date = next_expiry()

    if options_df is not None and not options_df.empty:
        row = options_df[
            (options_df["strike"] == strike) &
            (options_df["type"]   == opt_type)
        ]
        if not row.empty and float(row.iloc[0]["ltp"]) > 0:
            premium = float(row.iloc[0]["ltp"])

    if premium is None:
        days_left = (expiry_date - date.today()).days
        premium = estimate_premium(spot_price, strike, opt_type, india_vix, days_left)

    premium = max(round(premium * 2) / 2, 15.0)   # round to ₹0.5

    # ── Targets & stop loss ───────────────────────────────────────────────
    target_premium = round(premium * (1 + TARGET_PCT),  1)
    sl_premium     = round(premium * (1 - STOP_LOSS_PCT), 1)

    # ── Lot sizing ────────────────────────────────────────────────────────
    lots        = calc_lots(premium, capital)
    cost        = round(lots * NIFTY_LOT_SIZE * premium)
    max_profit  = round(lots * NIFTY_LOT_SIZE * (target_premium - premium))
    max_loss    = round(lots * NIFTY_LOT_SIZE * (premium - sl_premium))
    rr          = round((target_premium - premium) / max(premium - sl_premium, 0.1), 2)

    # ── Direction label ───────────────────────────────────────────────────
    dir_label   = "Bullish" if direction == 1 else "Bearish"
    gap_label   = "Gap-up expected" if direction == 1 else "Gap-down expected"

    # ── OTM label (extracted to avoid backslash-in-f-string, Python <3.12) ──
    otm_label   = "ATM" if otm_level == 0 else f"{otm_level} OTM"

    return {
        "signal":           signal,
        "option_type":      opt_type,
        "direction":        direction,
        "confidence":       round(confidence, 4),
        "spot_price":       round(spot_price, 2),
        "atm_strike":       atm,
        "strike":           strike,
        "otm_level":        otm_level,
        "expiry":           expiry_date.strftime("%d %b %Y"),
        "expiry_date":      expiry_date,

        # Entry / exit levels
        "entry_window":     f"{ENTRY_START_HOUR}:{ENTRY_START_MIN:02d} AM – "
                            f"{ENTRY_END_HOUR}:{ENTRY_END_MIN:02d} AM",
        "premium_entry":    premium,
        "target_premium":   target_premium,
        "sl_premium":       sl_premium,
        "time_exit":        f"{TIME_EXIT_HOUR}:{TIME_EXIT_MIN:02d} PM",

        # Risk
        "lots":             lots,
        "lot_size":         NIFTY_LOT_SIZE,
        "capital_used":     cost,
        "max_profit_inr":   max_profit,
        "max_loss_inr":     max_loss,
        "risk_reward":      rr,

        # Context
        "predicted_move_pct": pred_move,
        "india_vix":          round(india_vix, 2),
        "reason": (
            f"{dir_label} signal — {confidence:.0%} confidence. "
            f"{gap_label}. Expected move: ~{pred_move:.1f}%. "
            f"Strike: {strike} {opt_type} ({otm_label})."
        ),
    }


def _no_trade(direction: int, confidence: float, reason: str) -> dict:
    return {
        "signal":     "NO_TRADE",
        "direction":  direction,
        "confidence": round(confidence, 4),
        "reason":     reason,
    }
