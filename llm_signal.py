"""
llm_signal.py — LLM-powered signal aggregation via Groq.

Feeds the LLM raw parameters from every signal source so it can
reason from first principles — not just bullish/bearish labels.
Falls back to weighted math (signal_aggregator.py) if the call fails.
"""

import json
import requests
import signal_aggregator as sa

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a senior quantitative analyst at a top Indian hedge fund, specializing in Nifty 50 intraday and positional trading. You have deep expertise in:
- Technical analysis (RSI, MACD, Bollinger Bands, EMA crossovers, volume analysis)
- Options market microstructure (PCR, max pain, OI buildup, IV skew)
- Pre-market indicators (GIFT Nifty / SGX Nifty gap analysis)
- News sentiment impact on Indian markets
- Volatility regimes (India VIX interpretation)
- Institutional flow analysis (FII/DII)

Your job: analyze ALL the raw data provided and predict Nifty 50's direction for today's session.
Think step by step. Consider how signals reinforce or contradict each other.
Be brutally honest about uncertainty — markets are noisy."""


def _build_prompt(market_context: dict) -> str:
    m = market_context

    # ── Model predictions ─────────────────────────────────────────────
    model_section = "## ML Model Predictions (XGBoost + LightGBM ensemble, trained on 2yr 5-min data)\n"
    mp = m.get("model_params", {})
    if mp:
        model_section += f"""- Close direction: {"BULLISH (1)" if mp.get("close_direction") == 1 else "BEARISH (0)"}
- Close confidence: {mp.get("close_confidence", "N/A")}
- XGB and LGB agree on close: {mp.get("close_agree", "N/A")}
- Close predicted move: {mp.get("close_pred_pct", "N/A")}%
- Predicted close price: {mp.get("predicted_close", "N/A")}
- Open direction: {"UP (1)" if mp.get("open_direction") == 1 else "DOWN (0)"}
- Open confidence: {mp.get("open_confidence", "N/A")}
- XGB and LGB agree on open: {mp.get("open_agree", "N/A")}
- Open predicted move: {mp.get("open_pred_pct", "N/A")}%
- Predicted open price: {mp.get("predicted_open", "N/A")}
- Full ensemble agree (open+close): {mp.get("ensemble_agree", "N/A")}
- High predicted move: {mp.get("high_pred_pct", "N/A")}%
- Low predicted move: {mp.get("low_pred_pct", "N/A")}%
- Predicted high: {mp.get("predicted_high", "N/A")}
- Predicted low: {mp.get("predicted_low", "N/A")}
- Predicted daily range: {mp.get("daily_range", "N/A")}
- Flat prediction (move < 0.2%): {mp.get("flat_prediction", "N/A")}
- Model's own trade signal: {mp.get("trade_signal", "N/A")}
- Signal reason: {mp.get("signal_reason", "")}
"""
    else:
        model_section += "- Model predictions unavailable\n"

    # ── Market context ────────────────────────────────────────────────
    market_section = f"""## Market Context
- Nifty Spot (current/last): {m.get("spot", "N/A")}
- Previous session close: {m.get("prev_close", "N/A")}
- Last close from 5-min data: {mp.get("last_close", m.get("prev_close", "N/A"))}
- India VIX: {m.get("vix", "N/A")} (>20 = high vol regime, <14 = complacency)
- ATR%: {m.get("atr_pct", "N/A")} (average true range as % of price)
"""

    # ── GIFT Nifty / Futures gap ──────────────────────────────────────
    gift_section = "## GIFT Nifty / Pre-market Gap\n"
    gp = m.get("gift_params", {})
    if gp.get("available"):
        gift_section += f"""- GIFT/Futures price: {gp.get("gift_price", "N/A")}
- Previous close: {gp.get("prev_close", "N/A")}
- Gap %: {gp.get("gap_pct", "N/A")}%
- Gap direction: {"UP (bullish)" if gp.get("gap_pct", 0) > 0 else "DOWN (bearish)" if gp.get("gap_pct", 0) < 0 else "FLAT"}
- Source: {gp.get("source", "N/A")} (live=real-time, historical=cached, from_5min_cache=proxy)
- Interpretation: Gap > +0.3% = strong bullish open expected; Gap < -0.3% = strong bearish open expected; within ±0.1% = flat open
"""
    else:
        gift_section += "- GIFT Nifty data unavailable\n"

    # ── OFI / PCR (Order Flow / Put-Call Ratio) ───────────────────────
    ofi_section = "## Order Flow Imbalance / Put-Call Ratio\n"
    op = m.get("ofi_params", {})
    if op.get("available"):
        ofi_section += f"""- OFI value: {op.get("ofi", "N/A")} (positive = buy pressure, negative = sell pressure)
- Source: {op.get("source", "N/A")} (groww = live order book, pcr = derived from options chain)
- Signal: {op.get("signal", "N/A")}
- Bias: {op.get("bias", "N/A")}
"""
        if op.get("source") == "pcr":
            ofi_section += f"""- Raw PCR value: {op.get("raw_pcr", "N/A")}
- PCR interpretation: >1.2 = excessive puts (bearish sentiment but can mean support/contrarian bullish),
  <0.8 = excessive calls (bullish sentiment but can mean resistance/contrarian bearish),
  0.8-1.2 = balanced
"""
    else:
        ofi_section += "- OFI/PCR data unavailable\n"

    # ── Options chain data ────────────────────────────────────────────
    opts = m.get("options_params", {})
    opts_section = "## Options Chain Analysis\n"
    if opts.get("available"):
        opts_section += f"""- Live PCR (OI-based): {opts.get("pcr", "N/A")}
- Total CE OI: {opts.get("total_ce_oi", "N/A")}
- Total PE OI: {opts.get("total_pe_oi", "N/A")}
- ATM strike: {opts.get("atm_strike", "N/A")}
- ATM CE premium: {opts.get("atm_ce_premium", "N/A")}
- ATM PE premium: {opts.get("atm_pe_premium", "N/A")}
- ATM CE IV: {opts.get("atm_ce_iv", "N/A")}%
- ATM PE IV: {opts.get("atm_pe_iv", "N/A")}%
- IV skew (PE IV - CE IV): {opts.get("iv_skew", "N/A")} (positive = fear premium in puts)
- Max pain estimate: {opts.get("max_pain", "N/A")}
- Highest CE OI strike: {opts.get("max_ce_oi_strike", "N/A")} (resistance)
- Highest PE OI strike: {opts.get("max_pe_oi_strike", "N/A")} (support)
"""
    else:
        opts_section += "- Options chain data unavailable\n"

    # ── News sentiment ────────────────────────────────────────────────
    news_section = "## News Sentiment Analysis\n"
    np_ = m.get("news_params", {})
    if np_.get("n_articles", 0) > 0:
        news_section += f"""- Composite sentiment score: {np_.get("score", "N/A")} (range: -1.0 bearish to +1.0 bullish)
- Sentiment label: {np_.get("label", "N/A")}
- Total articles analyzed: {np_.get("n_articles", 0)}
- Positive articles: {np_.get("n_positive", 0)} ({np_.get("pct_positive", 0)}%)
- Negative articles: {np_.get("n_negative", 0)} ({np_.get("pct_negative", 0)}%)
- Neutral articles: {np_.get("n_neutral", 0)}
- Scoring backend: {np_.get("backend", "N/A")} (finbert = high accuracy, vader = basic)

### Breakdown by impact type (macro news matters most for index direction):
- Macro/policy news: {np_.get("n_macro", 0)} articles, avg sentiment: {np_.get("macro_sentiment", "N/A")}
- Market events: {np_.get("n_market_events", 0)} articles, avg sentiment: {np_.get("market_sentiment", "N/A")}
- General news: {np_.get("n_general", 0)} articles, avg sentiment: {np_.get("general_sentiment", "N/A")}
"""
        # Category breakdown
        cat_breakdown = np_.get("category_breakdown", {})
        if cat_breakdown:
            news_section += "\n### Sentiment by category:\n"
            for cat, data in list(cat_breakdown.items())[:12]:
                news_section += f"  - {cat}: {data.get('count', 0)} articles, avg: {data.get('avg_sentiment', 0):+.3f} ({data.get('label', 'neutral')})\n"

        # Macro/market-event headlines (most impactful)
        macro_hl = np_.get("macro_headlines", [])
        if macro_hl:
            news_section += "\n### Key macro/market-event headlines (highest market impact):\n"
            for i, h in enumerate(macro_hl[:7], 1):
                s = h.get("sentiment", 0)
                tag = "BULLISH" if s > 0.15 else "BEARISH" if s < -0.15 else "NEUTRAL"
                news_section += f"  {i}. [{tag} {s:+.2f}] [{h.get('impact_type', '')}] {h.get('title', 'N/A')} — {h.get('source', '')} ({h.get('publishedIST', '')})\n"

        # Top sentiment headlines
        headlines = np_.get("top_headlines", [])
        if headlines:
            news_section += "\n### Top sentiment-moving headlines (by strength):\n"
            for i, h in enumerate(headlines[:7], 1):
                s = h.get("sentiment", 0)
                tag = "BULLISH" if s > 0.15 else "BEARISH" if s < -0.15 else "NEUTRAL"
                news_section += f"  {i}. [{tag} {s:+.2f}] {h.get('title', 'N/A')} — {h.get('source', '')} ({h.get('publishedIST', '')})\n"

        # Latest breaking headlines
        latest_hl = np_.get("latest_headlines", [])
        if latest_hl:
            news_section += "\n### Latest breaking headlines (most recent first):\n"
            for i, h in enumerate(latest_hl[:5], 1):
                s = h.get("sentiment", 0)
                tag = "BULLISH" if s > 0.15 else "BEARISH" if s < -0.15 else "NEUTRAL"
                news_section += f"  {i}. [{tag} {s:+.2f}] {h.get('title', 'N/A')} — {h.get('source', '')} ({h.get('publishedIST', '')})\n"
    else:
        news_section += f"- News unavailable (articles: {np_.get('n_articles', 0)}, error: {np_.get('error', 'N/A')})\n"

    return f"""{model_section}
{market_section}
{gift_section}
{ofi_section}
{opts_section}
{news_section}
## Your Analysis Required

Based on ALL the data above, determine Nifty 50's most likely direction for today.

Think about:
1. Do ML models and market signals agree? If the model says bullish but GIFT gap is negative and PCR is high, that's conflicting — lower your confidence.
2. Is the VIX in a high-vol regime? High VIX (>20) means wider swings and less predictable direction — reduce confidence.
3. What is the options market telling you? High PCR with heavy PE OI at a support level can be contrarian bullish. Low PCR with heavy CE OI at resistance can be contrarian bearish.
4. Are the news headlines market-moving? Macro events (RBI policy, US Fed, global events) matter more than company-specific news for index direction.
5. Is the predicted move flat (<0.2%)? If yes, it's essentially a coin flip — confidence should be low.
6. Do XGB and LGB ensemble models agree? Disagreement = lower conviction.
7. GIFT gap direction has ~65% hit rate for open direction — give it weight but don't over-rely.

Respond with ONLY a JSON object (no markdown, no code fences):
{{
  "direction": 1,
  "confidence": 0.72,
  "reasoning": "Brief 2-3 sentence analysis connecting the key signals to your conclusion.",
  "key_factors": ["factor1", "factor2", "factor3"],
  "risk_factors": ["risk1", "risk2"]
}}

direction: 1 = BULLISH (Nifty will close above previous close), 0 = BEARISH (Nifty will close below previous close)
confidence: 0.45 (pure guess) to 0.95 (extreme conviction with all signals aligned)

Calibration rules:
- All signals align strongly → 0.80-0.95
- Most signals align, minor conflicts → 0.65-0.80
- Mixed signals, some conflict → 0.50-0.65
- Strong conflicts between signals → 0.45-0.55
- Flat prediction from model → cap at 0.55 regardless"""


def _extract_options_params(opts_df) -> dict:
    """Extract key options chain metrics from the raw DataFrame."""
    if opts_df is None or len(opts_df) == 0:
        return {"available": False}

    try:
        ce = opts_df[opts_df["type"] == "CE"]
        pe = opts_df[opts_df["type"] == "PE"]

        total_ce_oi = ce["oi"].sum()
        total_pe_oi = pe["oi"].sum()
        pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 1.0

        max_ce_oi_row = ce.loc[ce["oi"].idxmax()] if len(ce) > 0 else None
        max_pe_oi_row = pe.loc[pe["oi"].idxmax()] if len(pe) > 0 else None

        strikes = sorted(opts_df["strike"].unique())
        atm_strike = strikes[len(strikes) // 2] if strikes else None

        atm_ce = ce[ce["strike"] == atm_strike] if atm_strike else None
        atm_pe = pe[pe["strike"] == atm_strike] if atm_strike else None

        atm_ce_ltp = float(atm_ce["ltp"].iloc[0]) if atm_ce is not None and len(atm_ce) > 0 else None
        atm_pe_ltp = float(atm_pe["ltp"].iloc[0]) if atm_pe is not None and len(atm_pe) > 0 else None
        atm_ce_iv = float(atm_ce["iv"].iloc[0]) if atm_ce is not None and len(atm_ce) > 0 else None
        atm_pe_iv = float(atm_pe["iv"].iloc[0]) if atm_pe is not None and len(atm_pe) > 0 else None

        iv_skew = round(atm_pe_iv - atm_ce_iv, 2) if atm_pe_iv and atm_ce_iv else None

        max_pain = None
        if len(ce) > 0 and len(pe) > 0:
            pain = {}
            for s in strikes:
                ce_pain = ce[ce["strike"] <= s]["oi"].sum() * (s - ce[ce["strike"] <= s]["strike"]).sum() if len(ce[ce["strike"] <= s]) > 0 else 0
                pe_pain = pe[pe["strike"] >= s]["oi"].sum() * (pe[pe["strike"] >= s]["strike"] - s).sum() if len(pe[pe["strike"] >= s]) > 0 else 0
                pain[s] = ce_pain + pe_pain
            if pain:
                max_pain = min(pain, key=pain.get)

        return {
            "available": True,
            "pcr": pcr,
            "total_ce_oi": int(total_ce_oi),
            "total_pe_oi": int(total_pe_oi),
            "atm_strike": atm_strike,
            "atm_ce_premium": atm_ce_ltp,
            "atm_pe_premium": atm_pe_ltp,
            "atm_ce_iv": round(atm_ce_iv, 1) if atm_ce_iv else None,
            "atm_pe_iv": round(atm_pe_iv, 1) if atm_pe_iv else None,
            "iv_skew": iv_skew,
            "max_pain": max_pain,
            "max_ce_oi_strike": int(max_ce_oi_row["strike"]) if max_ce_oi_row is not None else None,
            "max_pe_oi_strike": int(max_pe_oi_row["strike"]) if max_pe_oi_row is not None else None,
        }
    except Exception:
        return {"available": False}


def build_market_context(
    preds: dict,
    spot: float,
    prev_close: float,
    vix: float,
    atr_pct: float,
    gift_live: float,
    gift_gap_pct: float,
    gift_status: str,
    ofi_data: dict,
    live_pcr: float,
    news: dict,
    opts_df=None,
) -> dict:
    """Build the comprehensive market context dict from all raw sources."""
    return {
        "spot": spot,
        "prev_close": prev_close,
        "vix": vix,
        "atr_pct": atr_pct,
        "model_params": {
            "close_direction": preds.get("close_direction", preds.get("direction")),
            "close_confidence": preds.get("close_confidence", preds.get("confidence")),
            "close_pred_pct": preds.get("close_pred_pct"),
            "close_agree": preds.get("close_agree"),
            "predicted_close": preds.get("predicted_close"),
            "open_direction": preds.get("open_direction"),
            "open_confidence": preds.get("open_confidence"),
            "open_pred_pct": preds.get("open_pred_pct"),
            "open_agree": preds.get("open_agree"),
            "predicted_open": preds.get("predicted_open"),
            "ensemble_agree": preds.get("ensemble_agree"),
            "high_pred_pct": preds.get("high_pred_pct"),
            "low_pred_pct": preds.get("low_pred_pct"),
            "predicted_high": preds.get("predicted_high"),
            "predicted_low": preds.get("predicted_low"),
            "daily_range": preds.get("daily_range"),
            "flat_prediction": preds.get("flat_prediction"),
            "trade_signal": preds.get("trade_signal"),
            "signal_reason": preds.get("signal_reason", ""),
            "last_close": preds.get("last_close"),
        },
        "gift_params": {
            "available": gift_live is not None and gift_live > 0,
            "gift_price": gift_live,
            "prev_close": prev_close,
            "gap_pct": round(gift_gap_pct, 3) if gift_gap_pct else 0,
            "source": gift_status,
        },
        "ofi_params": {
            "available": ofi_data.get("available", False),
            "ofi": ofi_data.get("ofi", 0),
            "source": ofi_data.get("source", "groww"),
            "signal": ofi_data.get("signal", ""),
            "bias": ofi_data.get("bias", ""),
            "raw_pcr": float(live_pcr) if live_pcr else None,
        },
        "options_params": _extract_options_params(opts_df),
        "news_params": {
            "score": news.get("score", 0),
            "label": news.get("label", ""),
            "n_articles": news.get("n_articles", 0),
            "n_positive": news.get("n_positive", 0),
            "n_negative": news.get("n_negative", 0),
            "n_neutral": news.get("n_neutral", 0),
            "pct_positive": news.get("pct_positive", 0),
            "pct_negative": news.get("pct_negative", 0),
            "n_macro": news.get("n_macro", 0),
            "n_market_events": news.get("n_market_events", 0),
            "n_general": news.get("n_general", 0),
            "macro_sentiment": news.get("macro_sentiment", 0),
            "market_sentiment": news.get("market_sentiment", 0),
            "general_sentiment": news.get("general_sentiment", 0),
            "backend": news.get("backend", "none"),
            "top_headlines": news.get("top_headlines", []),
            "latest_headlines": news.get("latest_headlines", []),
            "macro_headlines": news.get("macro_headlines", []),
            "category_breakdown": news.get("category_breakdown", {}),
            "error": news.get("error", ""),
        },
    }


def llm_aggregate(
    votes: list[sa.SignalVote],
    market_context: dict,
    groq_api_key: str,
    timeout: float = 20.0,
) -> dict | None:
    if not groq_api_key:
        return None

    prompt = _build_prompt(market_context)

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.15,
                "max_tokens": 600,
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
            "risk_factors": result.get("risk_factors", []),
            "model_used": MODEL,
            "source": "groq_llm",
        }
    except Exception as e:
        print(f"[LLM Signal] Groq call failed: {e}")
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
        reasoning_short = llm_result["reasoning"][:300]
        risk_str = ", ".join(llm_result.get("risk_factors", [])[:2])
        summary = (f"LLM Consensus: {dir_label} ({confidence:.0%}) — "
                   f"{reasoning_short}"
                   + (f" | Risks: {risk_str}" if risk_str else "")
                   + " | " + " | ".join(parts))

        return sa.AggregatedSignal(
            direction=direction,
            confidence=round(confidence, 4),
            votes=votes,
            agreement_ratio=round(agreement_ratio, 3),
            summary=summary,
        )

    return sa.aggregate_signals(votes)
