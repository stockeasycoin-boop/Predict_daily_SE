# Nifty AI Options Trader

Predicts Nifty 50 open and close direction using XGBoost + LightGBM ensemble with Optuna tuning. Suggests intraday CE/PE trades with news sentiment enrichment. Tracks accuracy daily via Streamlit dashboard.

---

## Quick start

```bash
pip install -r requirements.txt
pip install lightgbm optuna
streamlit run app.py
```

Then go to **Settings** tab → paste Breeze API key, secret, and session token.
Go to **Model Health** tab → click **Train model now**.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard — 4 tabs |
| `settings.py` | All configuration |
| `data_fetcher.py` | Breeze API data (OHLCV, VIX, intraday, sectors, GIFT) |
| `feature_engineering.py` | 70+ features including intraday patterns and new mean-reversion signals |
| `model_trainer.py` | XGBoost + LightGBM ensemble, Optuna, open/close dual models |
| `news_sentiment.py` | GNews + FinBERT/VADER sentiment scoring |
| `options_engine.py` | Strike selection, lot sizing, entry/exit rules |
| `tracker.py` | Trade log, accuracy, P&L |
| `train_local.py` | CLI training script (mirrors Model Health tab) |

---

## Daily workflow

| Time | Action |
|------|--------|
| 8:30 AM | Paste today's Breeze session token in Settings |
| 8:45 AM | Today's Signal → Generate. Check GIFT Nifty status and news sentiment |
| 9:15 AM | Enter actual open in recalibration form → updated close target |
| 9:20–9:30 AM | Enter trade if signal + first candle confirm direction |
| 1:30 PM | Exit if target/SL not hit |
| 3:30 PM | Record outcome in dashboard |

---

## Active safety filters

1. Tuesday expiry day block
2. Ensemble agreement gate (XGBoost + LightGBM must agree)
3. Confidence threshold ≥ 70%
4. Live GIFT Nifty override (cuts confidence 30% on contradiction)
5. News sentiment adjustment (FinBERT or VADER)
6. Regime filter (trade with 21-EMA trend only)
7. Mean-reversion override (blocks bearish in oversold+compressed market)
8. Monday reversal filter (blocks bearish PE after 3 consecutive down days)
9. High VIX block (> 25)

---

## Breeze session token (daily step)

```
https://api.icicidirect.com/apiuser/login?api_key=YOUR_KEY
```
Login → copy `apisession=` value from redirected URL → paste in Settings tab.
