"""
config.py — All settings for Nifty AI Trader.
Edit the values in the ICICI BREEZE API section before first run.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# ICICI BREEZE API CREDENTIALS
# Step 1: Login to https://api.icicidirect.com/ and register your app
# Step 2: Copy API Key and Secret below
# Step 3: Each morning, generate a session token:
#          Visit: https://api.icicidirect.com/apiuser/login?api_key=YOUR_KEY
#          Login → the URL will contain ?apisession=XXXXXXX
#          Copy that token and paste it in the dashboard Settings tab
# ─────────────────────────────────────────────────────────────────────
BREEZE_API_KEY     = os.getenv("BREEZE_API_KEY",     "YOUR_API_KEY_HERE")
BREEZE_API_SECRET  = os.getenv("BREEZE_API_SECRET",  "YOUR_API_SECRET_HERE")
BREEZE_SESSION_TOKEN = os.getenv("BREEZE_SESSION_TOKEN", "")  # Refresh daily

# ─────────────────────────────────────────────────────────────────────
# GNEWS API (news sentiment enrichment) — free 100 req/day at gnews.io
# Paste key into Settings tab; saved to settings.json
# ─────────────────────────────────────────────────────────────────────
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")

# ── GNews PREMIUM / realtime tuning ──────────────────────────────────────
# These are sized for a PAID GNews plan (higher rate limits, up to 100
# articles/request, near-realtime indexing). On the free tier, lower
# GNEWS_MAX_PER_QUERY back to 10 and raise GNEWS_CACHE_MINUTES to ~240.
GNEWS_CACHE_MINUTES = int(os.getenv("GNEWS_CACHE_MINUTES", "5"))   # cache TTL; 5 ≈ realtime
GNEWS_MAX_PER_QUERY = int(os.getenv("GNEWS_MAX_PER_QUERY", "50"))  # premium allows up to 100
GNEWS_LOOKBACK_DAYS = int(os.getenv("GNEWS_LOOKBACK_DAYS", "3"))   # days of news to analyse
GNEWS_QUERY_PAUSE   = float(os.getenv("GNEWS_QUERY_PAUSE", "0.0")) # premium: no throttle needed

# News-sentiment adjustment caps (used by news_sentiment.adjust_confidence)
NEWS_MAX_BOOST   = 0.08   # max upward adjustment when news agrees with model
NEWS_MAX_PENALTY = 0.15   # max downward adjustment when news disagrees

# ─────────────────────────────────────────────────────────────────────
# CAPITAL & RISK SETTINGS
# ─────────────────────────────────────────────────────────────────────
CAPITAL_MIN     = 20_000    # ₹ minimum capital per trade
CAPITAL_MAX     = 50_000    # ₹ maximum capital per trade
MAX_LOSS_PCT    = 0.10      # 10% max daily loss

# ─────────────────────────────────────────────────────────────────────
# SIGNAL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────
RETRAIN_THRESHOLD     = 0.50   # Auto-flag retraining if 7-day accuracy < 50%
MAX_VIX_FOR_TRADE     = 25.0   # Skip trading if India VIX > 25 (options too expensive)

# ─────────────────────────────────────────────────────────────────────
# NIFTY OPTIONS PARAMETERS
# ─────────────────────────────────────────────────────────────────────
NIFTY_LOT_SIZE   = 75      # NSE Nifty lot size (as of 2024)
NIFTY_STRIKE_GAP = 50      # ₹50 gap between Nifty strikes
TARGET_PCT       = 0.80    # Target: +80% on premium (e.g. buy at ₹100, target ₹180)
STOP_LOSS_PCT    = 0.30    # Stop loss: -30% on premium (e.g. buy at ₹100, SL at ₹70)
TIME_EXIT_HOUR   = 13      # Time-based exit at 1:30 PM (before EOD volatility)
TIME_EXIT_MIN    = 30
ENTRY_START_HOUR = 9
ENTRY_START_MIN  = 30      # Enter only after 9:30 AM (let opening volatility settle)
ENTRY_END_HOUR   = 9
ENTRY_END_MIN    = 45      # Enter before 9:45 AM

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = BASE_DIR / "models"
TRADES_FILE = BASE_DIR / "trades" / "trades.csv"
SETTINGS_FILE = BASE_DIR / "settings.json"

for _d in [DATA_DIR, MODEL_DIR, BASE_DIR / "trades"]:
    _d.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA DEPTH
# ─────────────────────────────────────────────────────────────────────────────
# How many calendar days of history to fetch for model training.
# 5475 ≈ 15 years ≈ ~3750 trading days (removing weekends + holidays).
# Stooq and tvDatafeed support up to 20+ years for Nifty.
# Breeze API is limited to ~2 years — used only for live top-up.
# Increase this freely; more data = better model on longer market cycles.
TRAINING_DAYS      = 5475   # ~15 years
TRAINING_DAYS_VIX  = 5475   # VIX data available since 2008 (~17 years)
TRAINING_DAYS_GIFT = 3650   # GIFT Nifty (SGX) data reliable from ~2014
TRAINING_DAYS_FII  = 3650   # NSDL FII data availability

# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE & MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
ENSEMBLE_AGREE_ONLY   = True    # Only signal when XGBoost + LightGBM agree
MIN_CONFIDENCE        = 0.70    # Raised from 0.65 → trade only strongest signals
OPTUNA_TRIALS         = 60      # Hyperparameter search trials (more = better, slower)

# ─────────────────────────────────────────────────────────────────────────────
# INTRADAY FEATURE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
INTRADAY_INTERVAL     = "5minute"   # Breeze interval for intraday candles
INTRADAY_DAYS_BACK    = 5           # How many past days of intraday to fetch for features

# ─────────────────────────────────────────────────────────────────────────────
# INSTRUMENTS TO FETCH FROM BREEZE
# ─────────────────────────────────────────────────────────────────────────────
CORRELATED_INSTRUMENTS = [
    "BANKNIFTY",   # Strongest Nifty correlate, leads/lags by minutes
    "CNXIT",       # IT sector — sensitive to USD/global tech
    "CNXAUTO",     # Auto sector — domestic demand proxy
    "CNXFMCG",     # Defensive sector — safe-haven signal
]

# ─────────────────────────────────────────────────────────────────────────────
# REGIME FILTER
# ─────────────────────────────────────────────────────────────────────────────
REGIME_EMA            = 21     # Only trade in direction of this EMA
SKIP_EXPIRY_DAY       = True   # Skip Tuesday (expiry day) — max-pain pinning
