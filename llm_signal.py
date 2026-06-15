"""
llm_signal.py — LLM-powered signal aggregation via Groq.

Takes all signal sources (model prediction, GIFT gap, OFI/PCR, news sentiment)
and asks an LLM to reason about the market direction and confidence.
Falls back to weighted math (signal_aggregator.py) if the LLM call fails.
"""

import json
import requests
import signal_aggregator as sa

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


def _build_prompt(votes: list[sa.SignalVote], market_context: dict) -> str:
    signal_lines = []
    for v in votes:
        if not v.available:
            signal_lines.append(f"- {v.source}: UNAVAILABLE")
        elif v.direction is None:
            signal_lines.append(f"- {v.source}: NEUTRAL (strength={v.strength:.2f}) — {v.reason}")
        else:
            d = "BULLISH" if v.direction == 1 else "BEARISH"
            signal_lines.append(f"- {v.source}: {d} (strength={v.strength:.2f}) — {v.reason}")

    spot = market_context.get("spot", "N/A")
    vix = market_context.get("vix", "N/A")
    atr_pct = market_context.get("atr_pct", "N/A")
    prev_close = market_context.get("prev_close", "N/A")
    gift_gap = market_context.get("gift_gap_pct", "N/A")
    pcr = market_context.get("live_pcr", "N/A")
    news_score = market_context.get("news_score", "N/A")
    news_count = market_context.get("news_count", 0)

    return f"""You are an expert Indian equity market analyst specializing in Nifty 50 index direction prediction.

Analyze the following real-time signals and market context to predict Nifty 50's direction for today.

## Market Context
- Nifty Spot: {spot}
- Previous Close: {prev_close}
- India VIX: {vix}
- ATR%: {atr_pct}
- GIFT/Futures Gap: {gift_gap}%
- Put-Call Ratio: {pcr}
- News articles analyzed: {news_count}, composite score: {news_score}

## Signal Sources
{chr(10).join(signal_lines)}

## Your Task
1. Weigh each signal's reliability and strength
2. Consider how signals reinforce or contradict each other
3. Factor in market volatility (VIX) — high VIX means less certainty
4. Consider PCR levels — above 1.2 is bearish, below 0.8 is bullish
5. GIFT/futures gap direction is a strong pre-market lead indicator

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "direction": 1 or 0,
  "confidence": 0.45 to 0.95,
  "reasoning": "2-3 sentence explanation of your analysis",
  "key_factors": ["factor1", "factor2", "factor3"]
}}

direction: 1 = BULLISH (Nifty will close higher), 0 = BEARISH (Nifty will close lower)
confidence: your conviction level (0.45 = coin flip, 0.95 = very strong signal alignment)

Be calibrated — if signals conflict, keep confidence below 0.60. Only go above 0.80 if 3+ signals strongly agree."""


def llm_aggregate(
    votes: list[sa.SignalVote],
    market_context: dict,
    groq_api_key: str,
    timeout: float = 15.0,
) -> dict | None:
    """
    Call Groq LLM to analyze signals and return direction + confidence.
    Returns dict with direction, confidence, reasoning, key_factors
    or None if the call fails.
    """
    if not groq_api_key:
        return None

    prompt = _build_prompt(votes, market_context)

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 400,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        result = json.loads(content)

        direction = int(result.get("direction", 1))
        confidence = float(result.get("confidence", 0.50))
        confidence = max(0.45, min(confidence, 0.95))

        return {
            "direction": direction,
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
            "key_factors": result.get("key_factors", []),
            "model_used": MODEL,
            "source": "groq_llm",
        }
    except Exception:
        return None


def aggregate_with_llm(
    votes: list[sa.SignalVote],
    market_context: dict,
    groq_api_key: str,
) -> sa.AggregatedSignal:
    """
    Primary entry point. Tries LLM first, falls back to weighted math.
    Returns an AggregatedSignal either way.
    """
    llm_result = llm_aggregate(votes, market_context, groq_api_key)

    if llm_result:
        direction = llm_result["direction"]
        confidence = llm_result["confidence"]

        active = [v for v in votes if v.direction is not None and v.available]
        n_bull = sum(1 for v in active if v.direction == 1)
        n_bear = sum(1 for v in active if v.direction == 0)
        agreement_ratio = max(n_bull, n_bear) / len(active) if active else 0

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
        reasoning_short = llm_result["reasoning"][:200]
        summary = (f"LLM Consensus: {dir_label} ({confidence:.0%}) — "
                   f"{reasoning_short} | "
                   + " | ".join(parts))

        return sa.AggregatedSignal(
            direction=direction,
            confidence=round(confidence, 4),
            votes=votes,
            agreement_ratio=round(agreement_ratio, 3),
            summary=summary,
        )

    return sa.aggregate_signals(votes)
