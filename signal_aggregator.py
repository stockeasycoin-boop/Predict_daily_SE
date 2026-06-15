"""
signal_aggregator.py — Multi-source signal consensus engine.

Direction is decided by WEIGHTED VOTE across all available signals,
not by the model alone. Each source contributes a directional vote
with a strength, and the final direction + confidence comes from
the weighted consensus.

Sources and default weights:
  Model (XGB+LGB):  0.45  — trained on 2yr data, primary but not sole
  GIFT Nifty gap:    0.20  — pre-market gap is a strong lead indicator
  Groww OFI:         0.20  — real-time buy/sell pressure from order book
  News sentiment:    0.15  — FinBERT scored, noisy but important

Agreement bonus: when 3+ sources agree on direction, confidence gets
a +5% boost. When sources split 2v2, confidence is capped at 60%.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# Default weights — must sum to 1.0
DEFAULT_WEIGHTS = {
    "model":  0.45,
    "gift":   0.20,
    "ofi":    0.20,
    "news":   0.15,
}


@dataclass
class SignalVote:
    source: str
    direction: Optional[int]   # 1=bullish, 0=bearish, None=abstain
    strength: float            # 0.0-1.0 how strong this signal is
    reason: str = ""
    available: bool = True


@dataclass
class AggregatedSignal:
    direction: int             # 1=bullish, 0=bearish
    confidence: float          # 0.0-1.0
    votes: list = field(default_factory=list)
    agreement_ratio: float = 0.0
    summary: str = ""


def vote_from_model(preds: dict) -> SignalVote:
    direction = preds.get("close_direction", preds.get("direction", None))
    confidence = preds.get("close_confidence", preds.get("confidence", 0.5))
    agree = preds.get("ensemble_agree", preds.get("close_agree", False))

    if direction is None:
        return SignalVote("model", None, 0.0, "Model prediction unavailable", False)

    strength = confidence if agree else confidence * 0.6
    reason = f"{'Bullish' if direction == 1 else 'Bearish'} {confidence:.0%}"
    if not agree:
        reason += " (XGB/LGB disagree)"

    return SignalVote("model", int(direction), float(strength), reason)


def vote_from_gift(gift_live: float, prev_close: float) -> SignalVote:
    if not gift_live or gift_live <= 0 or not prev_close or prev_close <= 0:
        return SignalVote("gift", None, 0.0, "GIFT Nifty unavailable", False)

    gap_pct = (gift_live - prev_close) / prev_close * 100

    if abs(gap_pct) < 0.10:
        return SignalVote("gift", None, 0.1, f"GIFT gap flat ({gap_pct:+.2f}%)")

    direction = 1 if gap_pct > 0 else 0
    strength = min(abs(gap_pct) / 1.5, 1.0)
    reason = f"GIFT gap {gap_pct:+.2f}% → {'bullish' if direction == 1 else 'bearish'}"

    return SignalVote("gift", direction, strength, reason)


def vote_from_ofi(ofi_data: dict) -> SignalVote:
    if not ofi_data or not ofi_data.get("available", False):
        return SignalVote("ofi", None, 0.0, "OFI unavailable", False)

    ofi = ofi_data.get("ofi", 0.0)
    src = ofi_data.get("source", "groww")

    # Options chain composite uses multiple sub-signals so even small values are meaningful
    threshold = 0.03 if src == "options_chain" else 0.10

    if abs(ofi) < threshold:
        return SignalVote("ofi", None, 0.1,
                          f"OFI neutral ({ofi:+.3f}) — {ofi_data.get('signal', '')}")

    direction = 1 if ofi > 0 else 0
    # Scale strength: options_chain signal is already composite, so scale more aggressively
    if src == "options_chain":
        strength = min(abs(ofi) / 0.4, 1.0)
    else:
        strength = min(abs(ofi) / 0.8, 1.0)
    strength = max(0.15, strength)

    bias = ofi_data.get("bias", "buy" if direction == 1 else "sell")
    signal_detail = ofi_data.get("signal", "")
    if src == "options_chain":
        reason = f"Options: {bias} ({ofi:+.3f}) — {signal_detail}"
    elif src == "pcr":
        reason = f"PCR {signal_detail} → {'buy' if direction == 1 else 'sell'} pressure"
    else:
        reason = f"OFI {ofi:+.2f} → {'buy' if direction == 1 else 'sell'} pressure"

    return SignalVote("ofi", direction, strength, reason)


def vote_from_news(news: dict) -> SignalVote:
    if not news or news.get("n_articles", 0) < 3:
        return SignalVote("news", None, 0.0,
                         f"Insufficient articles ({news.get('n_articles', 0) if news else 0})",
                         False)

    score = news.get("score", 0.0)

    if abs(score) < 0.05:
        return SignalVote("news", None, 0.1, f"News neutral ({score:+.2f})")

    direction = 1 if score > 0 else 0
    strength = min(abs(score) / 0.6, 1.0)
    reason = f"News {news.get('label', '?')} ({score:+.2f}, {news.get('n_articles', 0)} articles)"

    return SignalVote("news", direction, strength, reason)


def aggregate_signals(
    votes: list[SignalVote],
    weights: dict = None,
) -> AggregatedSignal:
    """
    Weighted consensus across all signal sources.

    Each source that has a direction votes with:
      weighted_score = weight × strength × direction_sign (+1 or -1)

    Sources that abstain (direction=None) have their weight
    redistributed proportionally to the active voters.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    active = [v for v in votes if v.direction is not None and v.available]
    if not active:
        return AggregatedSignal(
            direction=1, confidence=0.50,
            votes=votes, agreement_ratio=0.0,
            summary="No signals available — defaulting to neutral"
        )

    total_active_weight = sum(weights.get(v.source, 0) for v in active)
    if total_active_weight <= 0:
        total_active_weight = 1.0

    weighted_sum = 0.0
    weighted_conf = 0.0

    for v in active:
        w = weights.get(v.source, 0) / total_active_weight
        sign = 1.0 if v.direction == 1 else -1.0
        weighted_sum += w * sign * v.strength
        weighted_conf += w * v.strength

    direction = 1 if weighted_sum >= 0 else 0

    n_bull = sum(1 for v in active if v.direction == 1)
    n_bear = sum(1 for v in active if v.direction == 0)
    agreement_ratio = max(n_bull, n_bear) / len(active) if active else 0

    confidence = abs(weighted_sum) * 0.5 + weighted_conf * 0.5
    confidence = max(0.45, min(confidence, 0.98))

    if agreement_ratio >= 0.75 and len(active) >= 3:
        confidence = min(confidence + 0.05, 0.98)
    elif n_bull == n_bear and len(active) >= 2:
        confidence = min(confidence, 0.60)

    parts = []
    for v in votes:
        if not v.available:
            parts.append(f"{v.source}: unavailable")
        elif v.direction is None:
            parts.append(f"{v.source}: neutral")
        else:
            d = "↑" if v.direction == 1 else "↓"
            parts.append(f"{v.source}: {d} {v.strength:.0%}")

    dir_label = "BULLISH" if direction == 1 else "BEARISH"
    agree_pct = agreement_ratio * 100
    summary = (f"Consensus: {dir_label} ({confidence:.0%}) — "
               f"{n_bull} bullish / {n_bear} bearish / "
               f"{len(votes) - len(active)} abstain | "
               + " | ".join(parts))

    return AggregatedSignal(
        direction=direction,
        confidence=round(confidence, 4),
        votes=votes,
        agreement_ratio=round(agreement_ratio, 3),
        summary=summary,
    )
