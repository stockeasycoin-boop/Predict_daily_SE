"""
app.py — Nifty AI Options Trader Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
import json
import hashlib
import os
import sys
import logging
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
# Stream to stdout so Streamlit Community Cloud captures it in "Manage app" logs.
# force=True overrides handlers Streamlit/uvicorn may have installed first.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("nifty_app")

# Silence noisy third-party logs (Breeze 503 retries, HuggingFace HTTP, etc.)
for _noisy in ["breeze_connect", "urllib3", "requests", "httpx", "httpcore",
               "APILogger", "huggingface_hub", "transformers", "filelock"]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


def log_step(msg: str, level: str = "info") -> None:
    """Log a single step to stdout (visible in Streamlit Cloud logs)."""
    getattr(log, level, log.info)(msg)


log_step("🚀 App started / script rerun")

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="Nifty AI Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .signal-box {
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 1rem;
    }
    .signal-buy-ce  { background: #e8f5e9; border: 2px solid #2e7d32; }
    .signal-buy-pe  { background: #fce4ec; border: 2px solid #b71c1c; }
    .signal-no-trade{ background: #fff8e1; border: 2px solid #f9a825; }
    .step-box {
        background: #f1f3f4;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.9rem;
    }
    div[data-testid="stTab"] button { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH GATE — Google sign-in (native OIDC) with password fallback
# ──────────────────────────────────────────────────────────────────────────────
# Two modes, auto-detected:
#
#  1. GOOGLE SIGN-IN (preferred) — active when .streamlit/secrets.toml has an
#     [auth] section. Uses Streamlit's native st.login()/st.user (requires
#     Streamlit >= 1.42). Only emails in ALLOWED_EMAILS may enter.
#     FREE: Google charges nothing for OIDC sign-in.
#
#  2. PASSWORD FALLBACK — used when Google isn't configured yet. SHA-256 hashed
#     username/password. Lightweight; not strong security.
#
# ── Google setup (one time, free) ─────────────────────────────────────────────
#  a. console.cloud.google.com → new project
#  b. APIs & Services → Credentials → Create OAuth client ID → Web application
#  c. Authorized redirect URI:  http://localhost:8501/oauth2callback
#  d. Copy Client ID + Client Secret into .streamlit/secrets.toml (see template
#     printed by the app, or the secrets_template.toml file shipped alongside).
# ══════════════════════════════════════════════════════════════════════════════

# Whitelist — only these Google emails may access the app. Add yours here.
# Read from (in priority): st.secrets, then ALLOWED_EMAILS env var.
def _load_allowed_emails() -> list:
    raw = None
    try:
        if "ALLOWED_EMAILS" in st.secrets:
            raw = st.secrets["ALLOWED_EMAILS"]
    except Exception:
        pass
    if raw is None or raw == "":
        raw = os.getenv("ALLOWED_EMAILS", "")
    # raw may be a TOML list (Python list) or a comma-separated string
    if isinstance(raw, (list, tuple, set)):
        items = [str(e) for e in raw]
    else:
        items = str(raw).split(",")
    return [e.strip().strip("'\"[]").lower() for e in items if e.strip().strip("'\"[]")]

ALLOWED_EMAILS = _load_allowed_emails()

# Password fallback credentials (used only if Google auth isn't configured)
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "stockeasy")
# SHA-256 of "letsconquer"
AUTH_PASSWORD_SHA256 = os.getenv(
    "AUTH_PASSWORD_SHA256",
    "076baa2c57a6b05021593145435f6b5aa596205657c367f60d67b708eebc124c",
)


def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _google_auth_configured() -> bool:
    """True if an [auth] section exists in secrets and st.login is available."""
    if not hasattr(st, "login"):
        return False
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def _password_login():
    """Fallback: render password form, block until authenticated."""
    if st.session_state.get("authenticated"):
        return
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align:center;margin-bottom:0'>📈 Nifty AI Trader</h1>"
            "<p style='text-align:center;color:#78909C;margin-top:4px'>Please sign in to continue</p>",
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", width="stretch")
        if submitted:
            if username == AUTH_USERNAME and _hash_pw(password) == AUTH_PASSWORD_SHA256:
                st.session_state["authenticated"] = True
                st.session_state["auth_user"] = username
                st.session_state.pop("_use_password_fallback", None)
                st.rerun()
            else:
                st.error("Invalid username or password.")
        if st.session_state.get("_use_password_fallback"):
            if st.button("← Back to Google sign-in", width="stretch"):
                st.session_state.pop("_use_password_fallback", None)
                st.rerun()
    st.stop()


def _google_login():
    """Google OIDC sign-in via native st.login / st.user, with email whitelist + password fallback."""
    user = getattr(st, "user", None)
    logged_in = bool(user and getattr(user, "is_logged_in", False))

    if not logged_in:
        # If user clicked "use password instead", drop to password form
        if st.session_state.get("_use_password_fallback"):
            _password_login()
            return

        _, mid, _ = st.columns([1, 1.4, 1])
        with mid:
            st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
            st.markdown(
                "<h1 style='text-align:center;margin-bottom:0'>📈 Nifty AI Trader</h1>"
                "<p style='text-align:center;color:#78909C;margin-top:4px'>"
                "Sign in with your Google account to continue</p>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if st.button("🔐  Sign in with Google", width="stretch", type="primary"):
                st.login()
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("🔑  Sign in with password instead", width="stretch"):
                st.session_state["_use_password_fallback"] = True
                st.rerun()
            st.markdown(
                "<p style='text-align:center;color:#B0BEC5;font-size:0.8rem;margin-top:1rem'>"
                "Private application — authorized accounts only</p>",
                unsafe_allow_html=True,
            )
        st.stop()

    # Logged in — enforce email whitelist
    email = (getattr(user, "email", "") or "").lower()
    if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
        _, mid, _ = st.columns([1, 1.4, 1])
        with mid:
            st.error(f"🚫 Access denied for **{email}**. This account is not authorized.")
            if st.button("Sign out and try another account", width="stretch"):
                st.logout()
        st.stop()

    # Authorized
    st.session_state["authenticated"] = True
    st.session_state["auth_user"] = getattr(user, "name", email) or email
    st.session_state["auth_email"] = email


def check_login():
    if _google_auth_configured():
        _google_login()
    elif _community_cloud_user():
        _community_cloud_login()
    else:
        _password_login()


def _community_cloud_user() -> bool:
    """
    True when running on Streamlit Community Cloud with an authenticated viewer.
    On Community Cloud, st.user.email is auto-populated for logged-in workspace
    members even without any [auth] secrets configured.
    """
    user = getattr(st, "user", None)
    if user is None:
        return False
    try:
        return bool(user.get("email"))
    except Exception:
        try:
            return bool(getattr(user, "email", None))
        except Exception:
            return False


def _community_cloud_login():
    """Use the email Community Cloud provides; enforce the allowlist."""
    user = getattr(st, "user", None)
    try:
        email = (user.get("email") or "").lower()
    except Exception:
        email = (getattr(user, "email", "") or "").lower()

    if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
        _, mid, _ = st.columns([1, 1.4, 1])
        with mid:
            st.error(
                f"🚫 Access denied for **{email}**.\n\n"
                "This account is not on the authorized list. "
                "Ask the app owner to add your email to ALLOWED_EMAILS."
            )
        st.stop()

    st.session_state["authenticated"] = True
    st.session_state["auth_user"]  = email
    st.session_state["auth_email"] = email


check_login()



# ── Helpers: load settings persisted to disk ──────────────────────────────────
SETTINGS_FILE = Path("settings.json")

def _secrets_defaults() -> dict:
    """Pull known keys from st.secrets (Streamlit Cloud secret manager) if present."""
    defaults = {}
    try:
        s = st.secrets
        mapping = {
            "api_key":        ("BREEZE_API_KEY",     None),
            "api_secret":     ("BREEZE_API_SECRET",  None),
            "session_token":  ("BREEZE_SESSION_TOKEN", None),
            "gnews_api_key":  ("GNEWS_API_KEY",      None),
            "capital":        ("CAPITAL",             None),
            "min_confidence": ("MIN_CONFIDENCE",      None),
            "target_pct":     ("TARGET_PCT",          None),
            "sl_pct":         ("SL_PCT",              None),
            "max_vix":        ("MAX_VIX",             None),
        }
        for json_key, (secret_key, _) in mapping.items():
            try:
                val = s[secret_key]
                if val not in (None, ""):
                    defaults[json_key] = val
            except Exception:
                pass
    except Exception:
        pass
    return defaults


def load_settings() -> dict:
    """Load settings: Streamlit secrets → settings.json → empty dict (priority order)."""
    base = _secrets_defaults()          # start with cloud secrets
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                on_disk = json.load(f)
            # Disk values override secrets only if non-empty
            for k, v in on_disk.items():
                if v not in (None, "", "YOUR_API_KEY_HERE"):
                    base[k] = v
        except Exception:
            pass
    return base

def save_settings(d: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(d, f, indent=2)


# ── Module imports (done inside try so app still loads if deps missing) ───────
@st.cache_resource
def import_modules():
    try:
        import settings as config
        import data_fetcher as df_mod
        import feature_engineering as fe
        import model_trainer as mt
        import options_engine as oe
        import tracker
        return config, df_mod, fe, mt, oe, tracker, None
    except Exception as e:
        return None, None, None, None, None, None, str(e)

cfg, df_mod, fe, mt, oe, tracker, import_err = import_modules()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<div style='padding-top:1.5rem'></div>", unsafe_allow_html=True)
col_title, col_date, col_logout = st.columns([3, 1, 0.6])
with col_title:
    st.markdown("## 📈 Nifty AI Options Trader")
with col_date:
    now = datetime.now()
    st.markdown(f"<div style='text-align:right;padding-top:10px;color:#666'>"
                f"<b>{now.strftime('%A, %d %b %Y')}</b><br>"
                f"<span style='font-size:0.85rem'>{now.strftime('%I:%M %p')}</span>"
                f"</div>", unsafe_allow_html=True)
with col_logout:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("🔒 Log out", width="stretch",
                 help=f"Signed in as {st.session_state.get('auth_user', '')}"):
        st.session_state["authenticated"] = False
        st.session_state.pop("auth_user", None)
        st.rerun()

if import_err:
    st.error(f"⚠️ Import error: {import_err}. Run `pip install -r requirements.txt`")
    st.stop()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Today's Signal",
    "📡  Live Monitor",
    "📈  Accuracy Tracker",
    "🧠  Model Health",
    "⚙️  Settings",
    "🔌  Data Sources",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TODAY'S SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    settings = load_settings()
    api_key  = settings.get("api_key", cfg.BREEZE_API_KEY)
    api_sec  = settings.get("api_secret", cfg.BREEZE_API_SECRET)
    ses_tok  = settings.get("session_token", "")
    capital  = settings.get("capital", cfg.CAPITAL_MIN)

    # ── Refresh button + model status ─────────────────────────────────────
    col_ref, col_mod, col_cap = st.columns([1, 2, 2])
    with col_ref:
        run_btn = st.button("🔄 Generate Today's Signal", type="primary", width="stretch")
    with col_mod:
        model_ready = mt.model_exists(str(cfg.MODEL_DIR))
        if model_ready:
            meta = mt.load_metadata(str(cfg.MODEL_DIR))
            trained_at = meta.get("trained_at", "Unknown")[:10]
            st.success(f"✅ Model ready — trained {trained_at}")
        else:
            st.warning("⚠️ Model not trained yet. Go to Model Health tab → Train Model.")
    with col_cap:
        st.info(f"💰 Capital: ₹{capital:,.0f}  |  Max loss/day: ₹{capital * cfg.MAX_LOSS_PCT:,.0f}")

    st.markdown("")

    # Use a flag instead of st.stop() — kills ALL tabs, not just this one
    _tab1_ready = model_ready

    if not model_ready:
        st.info("👆 Train the model first from the **Model Health** tab, then come back here.")

    if _tab1_ready:
        # ── Load/run signal ──────────────────────────────────────────────
        _force_news = st.session_state.pop("_force_news_refresh", False)
        if run_btn or _force_news or "suggestion" not in st.session_state:
            log_step(f"🔮 Generate signal (run_btn={run_btn})")
            with st.spinner("Fetching market data and running model…"):
                try:
                    breeze = None
                    if ses_tok and api_key and api_key != "YOUR_API_KEY_HERE":
                        try:
                            log_step("Step 1/6 — connecting to Breeze API…")
                            breeze = df_mod.init_breeze(api_key, api_sec, ses_tok)
                            log_step("Step 1/6 — Breeze API connected ✅")
                            st.session_state["breeze_obj"] = breeze
                            st.toast("✅ Breeze API connected", icon="✅")
                            # Start WebSocket live stream for GIFT + Nifty
                            import live_feeds
                            if not live_feeds.is_streaming():
                                live_feeds.start(breeze)
                                log_step("WebSocket live feed started (GIFT + Nifty)")
                        except Exception as e:
                            log_step(f"Breeze connection failed: {e}. Using cached/Stooq data.", "warning")
                            st.warning(f"Breeze connection failed: {e}. Using cached/Stooq data.")
                    else:
                        log_step("Step 1/6 — no Breeze credentials, using cached/Stooq data")

                    log_step("Step 2/6 — loading 5-min intraday data…")
                    from pathlib import Path as _Path
                    _cache = _Path("data/nifty_5min_2yr.csv")
                    nifty_5min = None
                    if _cache.exists():
                        nifty_5min = pd.read_csv(_cache, parse_dates=["date"])
                        log_step(f"Step 2/6 — loaded {len(nifty_5min)} cached 5-min candles")
                    if (nifty_5min is None or len(nifty_5min) < 100) and breeze:
                        from data_fetcher import fetch_intraday_chunked
                        nifty_5min = fetch_intraday_chunked(breeze, "NIFTY", total_days=60, chunk_days=12)
                        log_step(f"Step 2/6 — fetched {len(nifty_5min) if nifty_5min is not None else 0} candles from API")

                    nifty_df = df_mod.load_nifty_data(breeze, force_refresh=run_btn)
                    vix_df = df_mod.load_vix_data(breeze)

                    if nifty_5min is None or len(nifty_5min) < 100:
                        log_step("Step 2/6 — insufficient 5-min data", "error")
                        st.error("Not enough 5-min candle data. Train models first or check API.")
                        _tab1_ready = False
                    else:
                        log_step("Step 3/6 — loading supplementary context (VIX, FII)…")
                        daily_context = pd.DataFrame()
                        try:
                            fii_df = df_mod.load_fii_dii_data()
                            if vix_df is not None and "india_vix" in vix_df.columns:
                                daily_context = vix_df[["date", "india_vix"]].copy()
                            if fii_df is not None and "fii_net" in fii_df.columns:
                                if len(daily_context) > 0:
                                    daily_context = daily_context.merge(
                                        fii_df[["date", "fii_net"]], on="date", how="outer"
                                    )
                                else:
                                    daily_context = fii_df[["date", "fii_net"]].copy()
                        except Exception:
                            pass
                        log_step("Step 4/6 — building 5-min features…")
                        from intraday_predictor import predict_today_from_5min
                        log_step("Step 5/6 — running model inference (5-min pipeline)…")
                        preds = predict_today_from_5min(
                            nifty_5min, str(cfg.MODEL_DIR), daily_context=daily_context
                        )
                        direction  = preds.get("close_direction", preds.get("direction", 0))
                        confidence = preds.get("close_confidence", preds.get("confidence", 0.5))
                        atr_pct    = preds.get("atr_pct", 0.8)
                        vix        = preds.get("india_vix", 16.0)
                        spot = None
                        # Try WebSocket stream first (instant)
                        import live_feeds as _lf2
                        _nifty_tick = _lf2.get_latest("nifty")
                        if _nifty_tick and _nifty_tick.get("ltp", 0) > 0:
                            _n_age = _lf2.age_seconds("nifty")
                            if _n_age is not None and _n_age < 300:
                                spot = _nifty_tick["ltp"]
                        # Fallback to REST poll
                        if not spot and breeze:
                            live = df_mod.fetch_live_quote_breeze(breeze)
                            if live:
                                spot = live["ltp"]
                        if not spot:
                            spot = float(nifty_df["close"].iloc[-1])
                        opts_df = None
                        live_pcr = None
                        if breeze:
                            try:
                                expiry_str = oe.breeze_expiry_format(oe.next_expiry())
                                opts_df, live_pcr = df_mod.fetch_options_chain_breeze(
                                    breeze, expiry_str, spot
                                )
                            except Exception:
                                pass
                        # ── Gather all signal sources for consensus ─────────
                        import signal_aggregator as sa

                        # 1) Model vote (already computed)
                        model_vote = sa.vote_from_model(preds)
                        log_step(f"Model: {model_vote.reason}")

                        # 2) GIFT Nifty vote (stream → REST → gift_nifty.csv → 5min last resort)
                        gift_live    = None
                        gift_status  = "unavailable"
                        gift_gap_pct = 0.0
                        # Tier 0: WebSocket live stream tick (instant, no API call)
                        import live_feeds as _lf
                        _gift_tick = _lf.get_latest("gift")
                        if _gift_tick and _gift_tick.get("ltp", 0) > 0:
                            _age = _lf.age_seconds("gift")
                            if _age is not None and _age < 300:
                                gift_live = _gift_tick["ltp"]
                                _src = _gift_tick.get("stock", "")
                                gift_status = f"live stream ({_src})"
                                log_step(f"GIFT from WebSocket stream: ₹{gift_live:,.0f} ({_age:.0f}s ago)")
                        # Tier 1: REST poll from Breeze (if stream didn't have it)
                        if (gift_live is None or gift_live <= 0) and breeze:
                            try:
                                gift_live = df_mod.fetch_gift_nifty_breeze(breeze)
                                if gift_live and gift_live > 0:
                                    gift_status = "live (REST)"
                            except Exception:
                                pass
                        # Tier 2: gift_nifty.csv historical cache (try to populate if missing)
                        if gift_live is None or gift_live <= 0:
                            try:
                                from pathlib import Path as _PG
                                _gift_csv = _PG("data/gift_nifty.csv")
                                if breeze:
                                    try:
                                        _gift_df = df_mod.load_gift_data(breeze, force_refresh=not _gift_csv.exists())
                                    except Exception:
                                        _gift_df = None
                                    if _gift_df is not None and len(_gift_df) > 0:
                                        gift_live = float(_gift_df["gift_close"].iloc[-1])
                                        gift_status = "historical"
                                        log_step(f"GIFT live unavailable — using cached history: ₹{gift_live:,.0f}")
                                elif _gift_csv.exists():
                                    _gift_hist = pd.read_csv(_gift_csv, parse_dates=["date"])
                                    if len(_gift_hist) > 0:
                                        gift_live = float(_gift_hist["gift_close"].iloc[-1])
                                        gift_status = "historical"
                                        log_step(f"GIFT live unavailable — using cached: ₹{gift_live:,.0f}")
                            except Exception:
                                pass
                        # Tier 3: 5-min nifty cache ONLY if gift cache is truly empty
                        if gift_live is None or gift_live <= 0:
                            _gift_csv_exists = Path("data/gift_nifty.csv").exists()
                            _gift_csv_has_data = False
                            if _gift_csv_exists:
                                try:
                                    _gc = pd.read_csv("data/gift_nifty.csv")
                                    _gift_csv_has_data = len(_gc) > 0
                                except Exception:
                                    pass
                            if not _gift_csv_has_data:
                                try:
                                    _c5_gift = pd.read_csv("data/nifty_5min_2yr.csv", parse_dates=["date"])
                                    _dates_g = sorted(_c5_gift["date"].dt.date.unique())
                                    if len(_dates_g) >= 2:
                                        _last_day = _c5_gift[_c5_gift["date"].dt.date == _dates_g[-1]]
                                        _prev_day = _c5_gift[_c5_gift["date"].dt.date == _dates_g[-2]]
                                        gift_live = float(_last_day["open"].iloc[0])
                                        gift_status = "from_5min_cache (last resort)"
                                        log_step(f"GIFT proxy (last resort): today's open ₹{gift_live:,.0f} vs prev close ₹{float(_prev_day['close'].iloc[-1]):,.0f}")
                                except Exception:
                                    pass
                        prev_close = float(nifty_df["close"].iloc[-1]) if nifty_df is not None and len(nifty_df) > 0 else spot
                        gift_vote = sa.vote_from_gift(gift_live, prev_close)
                        if gift_live and gift_live > 0 and prev_close:
                            gift_gap_pct = (gift_live - prev_close) / prev_close * 100
                        log_step(f"GIFT ({gift_status}): {gift_vote.reason}")

                        st.session_state["gift_live"]    = gift_live
                        st.session_state["gift_gap_pct"] = gift_gap_pct
                        st.session_state["gift_status"]  = gift_status

                        # 3) OFI vote (Groww live only — no fallback)
                        ofi_data = {"ofi": 0.0, "available": False}
                        _groww_main = st.session_state.get("groww_obj")
                        if not _groww_main:
                            try:
                                import groww_connector as gc
                                _gk = settings.get("groww_api_key", "")
                                _gs = settings.get("groww_api_secret", "")
                                if _gk or settings.get("groww_access_token", ""):
                                    _groww_main = gc.init_groww(_gk, _gs)
                                    st.session_state["groww_obj"] = _groww_main
                                    log_step("Groww auto-connected via saved token")
                            except Exception as _gae:
                                log_step(f"Groww auto-connect failed: {_gae}", "warning")
                        if _groww_main:
                            try:
                                import groww_connector as gc
                                ofi_data = gc.get_live_ofi(_groww_main, "NIFTY")
                            except Exception:
                                pass
                        if False and not ofi_data.get("available") and opts_df is not None and len(opts_df) > 0:
                            try:
                                _ce_df = opts_df[opts_df["type"] == "CE"]
                                _pe_df = opts_df[opts_df["type"] == "PE"]
                                _total_ce_oi = _ce_df["oi"].sum()
                                _total_pe_oi = _pe_df["oi"].sum()
                                _pcr_val = round(_total_pe_oi / _total_ce_oi, 3) if _total_ce_oi > 0 else None

                                # 1) PCR signal: deviation from 1.0 (skip if OI was 0)
                                _pcr_score = 0.0
                                if _pcr_val is not None:
                                    _pcr_score = (_pcr_val - 1.0) * 0.4
                                    _pcr_score = max(-0.5, min(0.5, _pcr_score))

                                # 2) Max pain vs spot: if spot > max pain → bearish pull, spot < max pain → bullish pull
                                _strikes = sorted(opts_df["strike"].unique())
                                _max_pain = None
                                if _strikes:
                                    _pain = {}
                                    for _s in _strikes:
                                        _ce_itm = _ce_df[_ce_df["strike"] <= _s]
                                        _pe_itm = _pe_df[_pe_df["strike"] >= _s]
                                        _ce_p = (_ce_itm["oi"] * (_s - _ce_itm["strike"])).sum() if len(_ce_itm) > 0 else 0
                                        _pe_p = (_pe_itm["oi"] * (_pe_itm["strike"] - _s)).sum() if len(_pe_itm) > 0 else 0
                                        _pain[_s] = _ce_p + _pe_p
                                    if _pain:
                                        _max_pain = min(_pain, key=_pain.get)
                                _mp_score = 0.0
                                if _max_pain and spot:
                                    _mp_dist_pct = (spot - _max_pain) / spot * 100
                                    _mp_score = -_mp_dist_pct * 0.15
                                    _mp_score = max(-0.3, min(0.3, _mp_score))

                                # 3) OI concentration: where is the wall?
                                _max_ce_strike = int(_ce_df.loc[_ce_df["oi"].idxmax()]["strike"]) if len(_ce_df) > 0 else 0
                                _max_pe_strike = int(_pe_df.loc[_pe_df["oi"].idxmax()]["strike"]) if len(_pe_df) > 0 else 0
                                _wall_score = 0.0
                                if spot and _max_ce_strike and _max_pe_strike:
                                    _dist_to_resist = (_max_ce_strike - spot) / spot * 100
                                    _dist_to_support = (spot - _max_pe_strike) / spot * 100
                                    if _dist_to_resist < _dist_to_support:
                                        _wall_score = -0.15
                                    elif _dist_to_support < _dist_to_resist:
                                        _wall_score = 0.15

                                # 4) IV skew: PE IV > CE IV = fear premium
                                _atm_idx = len(_strikes) // 2 if _strikes else 0
                                _atm_strike = _strikes[_atm_idx] if _strikes else None
                                _iv_score = 0.0
                                if _atm_strike:
                                    _atm_ce = _ce_df[_ce_df["strike"] == _atm_strike]
                                    _atm_pe = _pe_df[_pe_df["strike"] == _atm_strike]
                                    _ce_iv = float(_atm_ce["iv"].iloc[0]) if len(_atm_ce) > 0 and _atm_ce["iv"].iloc[0] > 0 else 0
                                    _pe_iv = float(_atm_pe["iv"].iloc[0]) if len(_atm_pe) > 0 and _atm_pe["iv"].iloc[0] > 0 else 0
                                    if _ce_iv > 0 and _pe_iv > 0:
                                        _iv_skew = _pe_iv - _ce_iv
                                        _iv_score = -_iv_skew * 0.01
                                        _iv_score = max(-0.2, min(0.2, _iv_score))

                                # Combine all sub-signals
                                _combined_ofi = _pcr_score + _mp_score + _wall_score + _iv_score
                                _combined_ofi = max(-1.0, min(1.0, _combined_ofi))

                                _parts = []
                                if _pcr_val is not None:
                                    _parts.append(f"PCR={_pcr_val:.2f}({_pcr_score:+.2f})")
                                else:
                                    _parts.append("PCR=N/A(OI=0)")
                                if _max_pain:
                                    _parts.append(f"MaxPain={_max_pain}({_mp_score:+.2f})")
                                if _max_ce_strike:
                                    _parts.append(f"Resist={_max_ce_strike}")
                                if _max_pe_strike:
                                    _parts.append(f"Support={_max_pe_strike}")
                                if _iv_score != 0:
                                    _parts.append(f"IVskew({_iv_score:+.2f})")

                                _bias = "bearish" if _combined_ofi > 0.05 else "bullish" if _combined_ofi < -0.05 else "neutral"
                                ofi_data = {
                                    "ofi": round(_combined_ofi, 3),
                                    "available": True,
                                    "signal": " | ".join(_parts),
                                    "bias": _bias,
                                    "source": "options_chain",
                                    "raw_pcr": _pcr_val,
                                    "max_pain": _max_pain,
                                    "max_ce_strike": _max_ce_strike,
                                    "max_pe_strike": _max_pe_strike,
                                }
                                log_step(f"OFI from options chain: {_combined_ofi:+.3f} ({_bias}) — {' | '.join(_parts)}")
                            except Exception as _ofi_err:
                                log_step(f"Options chain OFI calc failed: {_ofi_err}", "warning")
                        ofi_vote = sa.vote_from_ofi(ofi_data)
                        log_step(f"OFI: {ofi_vote.reason}")

                        # 4) News sentiment vote
                        log_step("Step 6/6 — fetching news sentiment…")
                        news = {"n_articles": 0}
                        try:
                            import news_sentiment as ns
                            gnews_key = settings.get("gnews_api_key",
                                                    getattr(cfg, "GNEWS_API_KEY", ""))
                            news = ns.get_market_sentiment(gnews_key,
                                                           force_refresh=_force_news)
                        except Exception as ne:
                            log_step(f"News sentiment skipped: {ne}", "warning")
                            news = {"error": str(ne), "n_articles": 0}
                        news_vote = sa.vote_from_news(news)
                        log_step(f"News: {news_vote.reason}")

                        # ── CONSENSUS: LLM-powered signal aggregation (Groq) ──
                        import llm_signal
                        _market_ctx = llm_signal.build_market_context(
                            preds=preds, spot=spot, prev_close=prev_close,
                            vix=vix, atr_pct=atr_pct,
                            gift_live=gift_live, gift_gap_pct=gift_gap_pct,
                            gift_status=gift_status, ofi_data=ofi_data,
                            live_pcr=live_pcr, news=news, opts_df=opts_df,
                        )
                        _groq_key = settings.get("groq_api_key", "")
                        consensus = llm_signal.aggregate_with_llm(
                            [model_vote, gift_vote, ofi_vote, news_vote],
                            _market_ctx, _groq_key,
                        )
                        direction  = consensus.direction
                        confidence = consensus.confidence
                        _used_llm = "LLM" in consensus.summary
                        log_step(f"{'LLM' if _used_llm else 'Math'} Consensus: "
                                 f"{'BULLISH' if direction == 1 else 'BEARISH'} "
                                 f"{confidence:.0%} (agreement: {consensus.agreement_ratio:.0%})")

                        # ── Detailed signal breakdown ──────────────────────
                        _method = "LLM (Groq Llama-3.3-70B)" if _used_llm else "Weighted math"
                        _dir_emoji = "🟢 BULLISH" if direction == 1 else "🔴 BEARISH"
                        st.success(f"**Final Verdict ({_method})**: {_dir_emoji} — Confidence: **{confidence:.0%}** — Agreement: {consensus.agreement_ratio:.0%}")

                        # Build signal detail rows
                        _sig_rows = []

                        # Model signal
                        _m_dir = preds.get("close_direction", preds.get("direction"))
                        _m_conf = preds.get("close_confidence", preds.get("confidence", 0.5))
                        _m_agree = preds.get("ensemble_agree", preds.get("close_agree", False))
                        _m_pct = preds.get("close_pred_pct", 0)
                        _m_icon = "🟢" if _m_dir == 1 else "🔴" if _m_dir == 0 else "⚪"
                        _sig_rows.append({
                            "Signal": "ML Model (XGB+LGB)",
                            "Status": f"{_m_icon} {'Bullish' if _m_dir == 1 else 'Bearish' if _m_dir == 0 else 'N/A'}",
                            "Value": f"Move: {_m_pct:+.2f}%",
                            "Confidence": f"{_m_conf:.0%}",
                            "Strength": f"{model_vote.strength:.0%}",
                            "Detail": f"{'XGB+LGB agree' if _m_agree else 'XGB/LGB disagree'} | Pred close: {preds.get('predicted_close', 'N/A')}",
                        })

                        # GIFT signal (open direction only — not used for close consensus)
                        _g_icon = "🟢" if gift_vote.direction == 1 else "🔴" if gift_vote.direction == 0 else "⚪"
                        _g_dir_str = "Bullish" if gift_vote.direction == 1 else "Bearish" if gift_vote.direction == 0 else "Neutral/N/A"
                        _sig_rows.append({
                            "Signal": f"GIFT Nifty ({gift_status}) ⓘ Open only",
                            "Status": f"{_g_icon} {_g_dir_str}",
                            "Value": f"Gap: {gift_gap_pct:+.2f}%" if gift_live else "N/A",
                            "Confidence": f"{gift_vote.strength:.0%}" if gift_vote.available else "—",
                            "Strength": f"{gift_vote.strength:.0%}",
                            "Detail": f"Open direction only | GIFT: {f'₹{gift_live:,.0f}' if gift_live else 'N/A'} vs Prev: ₹{prev_close:,.0f}" if prev_close else gift_vote.reason,
                        })

                        # OFI / Order Flow signal
                        _o_icon = "🟢" if ofi_vote.direction == 1 else "🔴" if ofi_vote.direction == 0 else "⚪"
                        _o_dir_str = "Bullish" if ofi_vote.direction == 1 else "Bearish" if ofi_vote.direction == 0 else "Neutral"
                        _ofi_val = ofi_data.get("ofi", 0)
                        _ofi_bias = ofi_data.get("bias", "neutral")
                        _ofi_signal = ofi_data.get("signal", "")
                        _ofi_label = "Groww OFI" if ofi_data.get("available") else "Order Flow"
                        _sig_rows.append({
                            "Signal": _ofi_label,
                            "Status": f"{_o_icon} {_o_dir_str}",
                            "Value": f"OFI: {_ofi_val:+.3f}" if ofi_data.get("available") else "N/A",
                            "Confidence": f"{ofi_vote.strength:.0%}" if ofi_vote.available else "---",
                            "Strength": f"{ofi_vote.strength:.0%}",
                            "Detail": _ofi_signal if ofi_data.get("available") else "Groww not connected",
                        })

                        # News signal
                        _n_icon = "🟢" if news_vote.direction == 1 else "🔴" if news_vote.direction == 0 else "⚪"
                        _n_dir_str = "Bullish" if news_vote.direction == 1 else "Bearish" if news_vote.direction == 0 else "Neutral/N/A"
                        _n_score = news.get("score", 0)
                        _n_count = news.get("n_articles", 0)
                        _n_macro = news.get("n_macro", 0)
                        _n_mkt = news.get("n_market_events", 0)
                        _macro_s = news.get("macro_sentiment", 0)
                        _sig_rows.append({
                            "Signal": f"News ({news.get('backend', 'N/A')})",
                            "Status": f"{_n_icon} {_n_dir_str}",
                            "Value": f"Score: {_n_score:+.3f} ({_n_count} articles)",
                            "Confidence": f"{news_vote.strength:.0%}" if news_vote.available else "—",
                            "Strength": f"{news_vote.strength:.0%}",
                            "Detail": f"{news.get('label', 'N/A')} | +{news.get('n_positive', 0)} / -{news.get('n_negative', 0)} / ~{news.get('n_neutral', 0)} | Macro: {_n_macro} ({_macro_s:+.2f}) | Events: {_n_mkt}",
                        })

                        # Options chain (info only, not a vote)
                        if opts_df is not None and len(opts_df) > 0:
                            _opts_p = _market_ctx.get("options_params", {})
                            if _opts_p.get("available"):
                                _pcr_v = _opts_p.get("pcr")
                                if _pcr_v is not None:
                                    _pcr_icon = "🔴" if _pcr_v > 1.2 else "🟢" if _pcr_v < 0.8 else "⚪"
                                    _pcr_bias = "Bearish (high puts)" if _pcr_v > 1.2 else "Bullish (high calls)" if _pcr_v < 0.8 else "Balanced"
                                    _pcr_str = f"PCR: {_pcr_v:.2f}"
                                else:
                                    _pcr_icon, _pcr_bias, _pcr_str = "⚪", "PCR unavailable (OI=0)", "PCR: N/A"
                                _sig_rows.append({
                                    "Signal": "Options Chain",
                                    "Status": f"{_pcr_icon} {_pcr_bias}",
                                    "Value": f"{_pcr_str} | MaxPain: {_opts_p.get('max_pain', 'N/A')}",
                                    "Confidence": "—",
                                    "Strength": "info",
                                    "Detail": f"Support (PE OI): {_opts_p.get('max_pe_oi_strike', 'N/A')} | Resist (CE OI): {_opts_p.get('max_ce_oi_strike', 'N/A')} | IV skew: {_opts_p.get('iv_skew', 'N/A')}",
                                })

                        # Market context row
                        _vix_icon = "🔴" if vix > 20 else "🟡" if vix > 15 else "🟢"
                        _sig_rows.append({
                            "Signal": "Market Context",
                            "Status": f"{_vix_icon} VIX: {vix:.1f}",
                            "Value": f"Spot: ₹{spot:,.0f} | ATR: {atr_pct:.2f}%",
                            "Confidence": "—",
                            "Strength": "info",
                            "Detail": f"{'High volatility regime' if vix > 20 else 'Normal volatility' if vix > 14 else 'Low volatility / complacency'}",
                        })

                        st.dataframe(
                            pd.DataFrame(_sig_rows),
                            width="stretch", hide_index=True,
                            column_config={
                                "Signal": st.column_config.TextColumn("Signal Source", width="medium"),
                                "Status": st.column_config.TextColumn("Verdict", width="small"),
                                "Value": st.column_config.TextColumn("Raw Value", width="medium"),
                                "Confidence": st.column_config.TextColumn("Conf.", width="small"),
                                "Strength": st.column_config.TextColumn("Wt.", width="small"),
                                "Detail": st.column_config.TextColumn("Details", width="large"),
                            },
                        )

                        # LLM reasoning below the table
                        if _used_llm and "LLM Consensus:" in consensus.summary:
                            _llm_parts = consensus.summary.split("—", 1)
                            if len(_llm_parts) > 1:
                                _reasoning_and_rest = _llm_parts[1]
                                _risk_idx = _reasoning_and_rest.find("| Risks:")
                                _llm_reason = _reasoning_and_rest[:_risk_idx].strip() if _risk_idx > 0 else _reasoning_and_rest.split("|")[0].strip()
                                if _llm_reason:
                                    st.info(f"**LLM Analysis**: {_llm_reason}")
                                if _risk_idx > 0:
                                    _risks = _reasoning_and_rest[_risk_idx+9:].split("|")[0].strip()
                                    if _risks:
                                        st.warning(f"**Risk Factors**: {_risks}")

                        # Generate suggestion using CONSENSUS direction + confidence
                        log_step("Step 6/6 — generating trade suggestion…")
                        suggestion = oe.generate_suggestion(
                            direction, confidence, spot, atr_pct, vix, capital, opts_df
                        )
                        suggestion["news"] = news
                        suggestion["consensus"] = {
                            "direction": direction,
                            "confidence": confidence,
                            "agreement_ratio": consensus.agreement_ratio,
                            "summary": consensus.summary,
                            "method": "llm" if _used_llm else "weighted_math",
                            "market_context": _market_ctx if _used_llm else {},
                            "votes": {v.source: {"dir": v.direction, "strength": v.strength,
                                                  "reason": v.reason, "available": v.available}
                                      for v in consensus.votes},
                        }
                        log_step(f"Step 6/6 — suggestion: {suggestion.get('signal', '?')} "
                                 f"(confidence={suggestion.get('confidence', 0):.2%})")

                        feat_df = preds.pop("_feat_row", pd.DataFrame())
                        reasoning  = mt.reasoning_for_prediction(feat_df, str(cfg.MODEL_DIR))
                        st.session_state["suggestion"] = suggestion
                        st.session_state["reasoning"]  = reasoning
                        st.session_state["preds"]      = preds
                        st.session_state["feat_df"]    = feat_df
                        st.session_state["spot"]       = spot
                        st.session_state["live_pcr"]   = live_pcr
                        trades_df = tracker.load_trades()
                        tracker.log_suggestion(suggestion, trades_df)
                        log_step("✅ Signal generated successfully")

                except FileNotFoundError:
                    log.exception("❌ Model files not found")
                    st.error("Model files not found. Train the model first (Model Health tab).")
                    _tab1_ready = False
                except Exception as e:
                    log.exception("❌ Error generating signal")
                    st.error(f"Error generating signal: {e}")
                    st.exception(e)
                    _tab1_ready = False

    suggestion = st.session_state.get("suggestion", {}) if _tab1_ready else {}
    spot       = st.session_state.get("spot", 0)

    if _tab1_ready and not suggestion:
        st.info("Click 'Generate Today's Signal' to run the model.")

    signal = suggestion.get("signal", "NO_TRADE")

    # ── Signal card ────────────────────────────────────────────────────────
    if signal == "BUY_CE":
        box_cls  = "signal-buy-ce"
        emoji    = "🟢"
        sig_text = "BUY CALL (CE)"
        sig_col  = "#2e7d32"
    elif signal == "BUY_PE":
        box_cls  = "signal-buy-pe"
        emoji    = "🔴"
        sig_text = "BUY PUT (PE)"
        sig_col  = "#b71c1c"
    else:
        box_cls  = "signal-no-trade"
        emoji    = "🟡"
        sig_text = "NO TRADE TODAY"
        sig_col  = "#f57f17"

    conf_pct = suggestion.get("confidence", 0) * 100
    reason   = suggestion.get("reason", "")

    st.markdown(f"""
    <div class="signal-box {box_cls}">
        <div style="display:flex;align-items:center;gap:14px">
            <span style="font-size:2.8rem">{emoji}</span>
            <div>
                <div style="font-size:1.6rem;font-weight:700;color:{sig_col}">{sig_text}</div>
                <div style="font-size:0.95rem;color:#555;margin-top:4px">{reason}</div>
            </div>
            <div style="margin-left:auto;text-align:center">
                <div style="font-size:2rem;font-weight:700;color:{sig_col}">{conf_pct:.0f}%</div>
                <div style="font-size:0.8rem;color:#888">Model confidence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Spot / last-day close metric strip ─────────────────────────────────
    nifty_df_state = st.session_state.get("feat_df")
    last_close_val = None
    last_close_date = None
    prev_close_val = None
    if nifty_df_state is not None and len(nifty_df_state) >= 1 and "close" in nifty_df_state.columns:
        try:
            last_close_val  = float(nifty_df_state["close"].iloc[-1])
            last_close_date = pd.to_datetime(nifty_df_state["date"].iloc[-1]).strftime("%d %b %Y")
            if len(nifty_df_state) >= 2:
                prev_close_val = float(nifty_df_state["close"].iloc[-2])
        except Exception:
            pass

    cur_spot = st.session_state.get("spot")
    mc1, mc2, mc3 = st.columns(3)
    if last_close_val is not None:
        mc1.metric(
            f"Last close ({last_close_date})",
            f"₹{last_close_val:,.2f}",
            f"{(last_close_val - prev_close_val):+.2f}" if prev_close_val else None,
        )
    if cur_spot:
        delta_vs_close = (cur_spot - last_close_val) if last_close_val else None
        mc2.metric(
            "Live spot" if delta_vs_close is not None and abs(delta_vs_close) > 0.01 else "Spot used",
            f"₹{cur_spot:,.2f}",
            f"{delta_vs_close:+.2f} vs last close" if delta_vs_close is not None else None,
        )
    if suggestion.get("strike"):
        mc3.metric("Suggested strike", f"{suggestion['strike']} {suggestion.get('option_type', '')}")

    # ── News sentiment panel ───────────────────────────────────────────────
    news_data = suggestion.get("news") or {}
    if news_data.get("n_articles", 0) > 0:
        score    = news_data.get("score", 0.0)
        label    = news_data.get("label", "neutral")
        n_arts   = news_data.get("n_articles", 0)
        backend  = news_data.get("backend", "?")
        adjust   = suggestion.get("news_adjustment", "")
        orig_c   = suggestion.get("confidence_original")

        from_cache   = news_data.get("from_cache", False)
        cache_age_m  = news_data.get("cache_age_minutes")
        cache_age_h  = news_data.get("cache_age_hours")
        if from_cache and cache_age_m is not None:
            cache_label = (f"🕒 cached {cache_age_m:.0f} min ago" if cache_age_m < 90
                           else f"🕒 cached {cache_age_m/60:.1f}h ago")
        elif from_cache and cache_age_h is not None:
            cache_label = f"🕒 cached {cache_age_h:.1f}h ago"
        else:
            cache_label = "🆕 freshly fetched (realtime)"
        try:
            _ttl_min = int(getattr(cfg, "GNEWS_CACHE_MINUTES", 5))
        except Exception:
            _ttl_min = 5
        _ttl_txt = "no cache" if _ttl_min <= 0 else f"refreshes every {_ttl_min} min"

        # News coverage window (e.g. "29–31 May")
        try:
            _look_days = int(getattr(cfg, "GNEWS_LOOKBACK_DAYS", 3))
        except Exception:
            _look_days = 3
        _today = datetime.now()
        _start = _today - timedelta(days=max(0, _look_days - 1))
        if _start.month == _today.month:
            _range_txt = f"{_start.day}–{_today.day} {_today.strftime('%b')}"
        else:
            _range_txt = f"{_start.strftime('%d %b')} – {_today.strftime('%d %b')}"

        nh_col, nb_col = st.columns([4, 1])
        nh_col.caption(
            f"📅 Analysing last {_look_days} days of news ({_range_txt}) · "
            f"{cache_label} · {_ttl_txt}"
        )
        if nb_col.button("🔄 Fetch fresh news", width="stretch",
                         help="Pull the latest headlines now (uses GNews quota) and regenerate the signal"):
            st.session_state["_force_news_refresh"] = True
            st.rerun()

        # Show backend=none warning prominently so user knows to install vaderSentiment
        if backend == "none" and n_arts > 0:
            st.warning(
                f"📰 News fetched ({n_arts} articles) but **sentiment scoring unavailable** — "
                f"no NLP model installed. All scores = 0.0. "
                f"**Fix:** `pip install vaderSentiment` then restart the app."
            )
        with st.expander(f"📰 News sentiment: **{label}** ({score:+.2f}) — {n_arts} articles via {backend}", expanded=False):
            cN1, cN2, cN3 = st.columns(3)
            cN1.metric("Sentiment score", f"{score:+.2f}", label)
            cN2.metric("Positive / Negative",
                       f"{news_data.get('n_positive', 0)} / {news_data.get('n_negative', 0)}")
            if orig_c is not None:
                delta = (suggestion["confidence"] - orig_c) * 100
                cN3.metric("Confidence adjustment",
                           f"{suggestion['confidence']*100:.0f}%",
                           f"{delta:+.1f}% vs model",
                           delta_color="normal" if delta >= 0 else "inverse")
            if adjust:
                st.caption(f"_{adjust}_")
            top    = news_data.get("top_headlines", [])
            latest = news_data.get("latest_headlines", [])
            if top or latest:
                col_top, col_latest = st.columns(2)

                def _render(h):
                    icon = "🟢" if h["sentiment"] > 0.15 else "🔴" if h["sentiment"] < -0.15 else "⚪"
                    # Prefer IST display string; fall back to raw UTC
                    when = h.get("publishedIST") or (h.get("publishedAt", "") or "")[:16].replace("T", " ")
                    st.markdown(
                        f"{icon} **{h['sentiment']:+.2f}** — [{h['title']}]({h['url']}) "
                        f"<span style='color:#888;font-size:0.85em'>· {h['source']}"
                        f"{(' · ' + when) if when else ''}</span>",
                        unsafe_allow_html=True,
                    )

                with col_top:
                    st.markdown("**Top headlines driving sentiment:**")
                    for h in top:
                        _render(h)
                with col_latest:
                    st.markdown("**Last fetched articles:**")
                    for h in latest:
                        _render(h)
    elif news_data.get("error"):
        st.caption(f"📰 News sentiment unavailable: {news_data['error']}")

    # ── Trade details (only for actual trade signals) ──────────────────────

    # ── Open / close prediction cards + reasoning ──────────────────────────
    preds     = st.session_state.get("preds", {})
    reasoning = st.session_state.get("reasoning", {})

    if preds:
        # ── Flat prediction warning ───────────────────────────────────
        if preds and preds.get("flat_prediction"):
            st.warning(
                f"⚠️ **Low-conviction prediction**: predicted intraday move is only "
                f"{abs(preds.get('close_pred_pct',0)):.2f}% — essentially flat. "
                "Classifier and regressor are near boundary. Consider skipping."
            )
        st.markdown("#### 📐 Open & Close predictions")

        o_dir     = preds.get("open_direction",  0)
        o_conf    = preds.get("open_confidence", 0.5)
        o_pct     = preds.get("open_pred_pct",   0.0)
        o_range   = preds.get("open_range",      (0,0))
        o_agree   = preds.get("open_agree",      False)

        c_dir     = preds.get("close_direction",  0)
        c_conf    = preds.get("close_confidence", 0.5)
        c_pct     = preds.get("close_pred_pct",   0.0)
        c_range   = preds.get("close_range",      (0,0))
        c_agree   = preds.get("close_agree",      False)
        ens_agree = preds.get("ensemble_agree",   False)

        atr_val   = preds.get("atr_pct",   0.8)
        vix_val   = preds.get("india_vix", 16.0)
        pred_open = preds.get("predicted_open",  0)
        pred_close= preds.get("predicted_close", 0)

        # Row 1 — Open prediction
        st.markdown("**🔔 Opening prediction (9:15 AM)**")
        po1, po2, po3, po4 = st.columns(4)
        po1.metric("Open direction",
                   "Gap-up ↑" if o_dir == 1 else "Gap-down / flat ↓",
                   delta=f"{o_conf:.0%} confidence",
                   delta_color="normal" if o_dir == 1 else "inverse",
                   help="XGB + LGB ensemble prediction for tomorrow's open gap")
        po2.metric("Predicted open",
                   f"₹{pred_open:,.0f}",
                   delta=f"{o_pct:+.2f}% vs today close",
                   delta_color="normal" if o_pct >= 0 else "inverse")
        po3.metric("Open range (low–high)",
                   f"₹{o_range[0]:,} – ₹{o_range[1]:,}",
                   help="±0.25 ATR band around predicted open")
        po4.metric("Open model agreement",
                   "✅ Both agree" if o_agree else "⚠️ Models differ",
                   delta_color="off")

        st.markdown("")
        st.markdown("**📍 Closing prediction (3:15 PM)**")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Close direction",
                   "Bullish ↑" if c_dir == 1 else "Bearish ↓",
                   delta=f"{c_conf:.0%} confidence",
                   delta_color="normal" if c_dir == 1 else "inverse",
                   help="Will the market close above today's open?")
        pc2.metric("Predicted close",
                   f"₹{pred_close:,.0f}",
                   delta=f"{c_pct:+.2f}% intraday",
                   delta_color="normal" if c_pct >= 0 else "inverse")
        pc3.metric("Close range (low–high)",
                   f"₹{c_range[0]:,} – ₹{c_range[1]:,}",
                   help="±0.35 ATR band around predicted close")
        pc4.metric("Ensemble agreement",
                   "✅ Full consensus" if ens_agree else "⚠️ Partial / none",
                   delta=f"VIX {vix_val:.1f}  |  ATR {atr_val:.2f}%",
                   delta_color="off")

        # Row 3 — High / Low prediction (chained, influenced by open & close)
        h_pct      = preds.get("high_pred_pct", 0.0)
        l_pct      = preds.get("low_pred_pct",  0.0)
        pred_high  = preds.get("predicted_high", 0)
        pred_low   = preds.get("predicted_low",  0)
        day_rng    = preds.get("daily_range", (pred_low, pred_high))

        st.markdown("")
        st.markdown("**📊 Predicted day range (High / Low)**")
        ph1, ph2, ph3, ph4 = st.columns(4)
        ph1.metric("Predicted high",
                   f"₹{pred_high:,.0f}",
                   delta=f"{h_pct:+.2f}% vs open",
                   delta_color="normal",
                   help="Chained model — uses predicted open & close as inputs")
        ph2.metric("Predicted low",
                   f"₹{pred_low:,.0f}",
                   delta=f"{l_pct:+.2f}% vs open",
                   delta_color="inverse",
                   help="Chained model — uses predicted open, close & high as inputs")
        ph3.metric("Expected day range",
                   f"₹{day_rng[0]:,} – ₹{day_rng[1]:,}",
                   delta=f"{(pred_high - pred_low):,.0f} pts wide",
                   delta_color="off")
        ph4.metric("Range vs ATR",
                   f"{((pred_high - pred_low) / (pred_open or 1) * 100):.2f}%",
                   delta=f"ATR {atr_val:.2f}%",
                   delta_color="off",
                   help="Predicted high-low spread as % of open vs historical ATR")

        st.markdown("")

    # ── Reasoning panel ────────────────────────────────────────────────────
    if reasoning and (reasoning.get("bullish_factors") or reasoning.get("bearish_factors")):
        st.markdown("#### 🔍 Why is the model saying this?")
        st.caption(reasoning.get("summary_text", ""))
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.markdown("**Bullish signals**")
            bulls = reasoning.get("bullish_factors", [])
            if bulls:
                for label, score, val_str in bulls:
                    bar_w = min(int(abs(score) * 2000), 100)
                    st.markdown(
                        f"""<div style='display:flex;align-items:center;gap:8px;
                        padding:6px 0;border-bottom:0.5px solid var(--color-border-tertiary)'>
                        <div style='width:{bar_w}px;height:6px;background:var(--color-text-success);
                        border-radius:3px;flex-shrink:0;min-width:4px'></div>
                        <span style='font-size:12px;color:var(--color-text-primary);flex:1'>{label}</span>
                        <span style='font-size:12px;color:var(--color-text-secondary);white-space:nowrap'>{val_str}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No strong bullish signals today.")

        with rcol2:
            st.markdown("**Bearish signals**")
            bears = reasoning.get("bearish_factors", [])
            if bears:
                for label, score, val_str in bears:
                    bar_w = min(int(abs(score) * 2000), 100)
                    st.markdown(
                        f"""<div style='display:flex;align-items:center;gap:8px;
                        padding:6px 0;border-bottom:0.5px solid var(--color-border-tertiary)'>
                        <div style='width:{bar_w}px;height:6px;background:var(--color-text-danger);
                        border-radius:3px;flex-shrink:0;min-width:4px'></div>
                        <span style='font-size:12px;color:var(--color-text-primary);flex:1'>{label}</span>
                        <span style='font-size:12px;color:var(--color-text-secondary);white-space:nowrap'>{val_str}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No strong bearish signals today.")
        st.markdown("")

    # ── LIVE-SPOT AUTO-CALIBRATION (in line with Live Monitor) ────────────
    # Once the market is open, the predicted close re-anchors to the LIVE spot
    # automatically — same anchoring the Live Monitor uses — so both tabs agree.
    if preds:
        import pytz as _pytz_la
        from datetime import time as _dtime_la
        _ist_la = _pytz_la.timezone("Asia/Kolkata")
        _now_la = datetime.now(_ist_la)
        _mkt_open_la = (_now_la.weekday() < 5 and
                        _dtime_la(9, 15) <= _now_la.time() <= _dtime_la(15, 30))
        if _mkt_open_la:
            _spot_la = None
            _b_la = st.session_state.get("breeze_obj")
            if _b_la:
                try:
                    _q_la = df_mod.fetch_live_quote_breeze(_b_la)
                    if _q_la:
                        _spot_la = _q_la.get("ltp") or _q_la.get("open")
                except Exception:
                    pass
            if _spot_la and _spot_la > 15000:
                _cpct_la = float(preds.get("close_pred_pct", 0.0))
                _cdir_la = int(preds.get("close_direction", 0))
                _pred_close_static = float(preds.get("predicted_close", 0))
                _live_target_la = round(_spot_la * (1 + _cpct_la / 100))
                st.markdown("#### 🎯 Live-calibrated close target")
                st.caption("Same anchoring as the Live Monitor: the model's intraday % "
                           "applied to the current live spot instead of the pre-market estimate.")
                _la1, _la2, _la3 = st.columns(3)
                _la1.metric("Live spot now", f"₹{_spot_la:,.2f}")
                _la2.metric("Live-anchored close target", f"₹{_live_target_la:,}",
                            delta=f"{_cpct_la:+.2f}% from live spot",
                            delta_color="normal" if _cdir_la == 1 else "inverse")
                _la3.metric("Pre-market close estimate", f"₹{_pred_close_static:,.0f}",
                            delta=f"{_live_target_la - _pred_close_static:+,.0f} pts shift",
                            delta_color="off",
                            help="How much the target moved after anchoring to live price")
                st.markdown("")

    # ── 9:15 AM recalibration panel — always visible ──────────────────────
    if preds:
        import pytz
        from datetime import time as _dtime
        _ist     = pytz.timezone("Asia/Kolkata")
        _now_ist = datetime.now(_ist).time()

        if _dtime(9, 15) <= _now_ist <= _dtime(9, 40):
            _panel_label   = "#### 🔄 9:15 AM — enter actual open to recalibrate"
            _panel_caption = (
                "Market just opened. Enter the actual Nifty open price to "
                "recompute the close target using the real anchor price."
            )
        elif _now_ist < _dtime(9, 15):
            _panel_label   = "#### 🕘 Pre-market — recalibrate at 9:15 AM"
            _panel_caption = (
                "Market has not opened yet. At 9:15 AM, enter the actual open "
                "price here to update the close target and re-evaluate the signal."
            )
        else:
            _panel_label   = "#### 📝 Post-open recalibration"
            _panel_caption = (
                "Enter the actual open price to see the corrected close target. "
                "Useful for reviewing today's signal or accuracy tracking."
            )

        st.divider()
        st.markdown(_panel_label)
        st.caption(_panel_caption)

        # Auto-fetch actual open from Breeze live spot (if market open & Breeze connected)
        _breeze_live_open = None
        _auto_open_note   = ""
        _breeze_obj = st.session_state.get("breeze_obj")
        if _breeze_obj:
            try:
                _lq = df_mod.fetch_live_quote_breeze(_breeze_obj)
                if _lq and _lq.get("open", 0) > 0:
                    _breeze_live_open = _lq["open"]
                    _auto_open_note   = f"Auto-filled from Breeze live feed (₹{_breeze_live_open:,.2f})"
                elif _lq and _lq.get("ltp", 0) > 0:
                    _breeze_live_open = _lq["ltp"]
                    _auto_open_note   = f"Auto-filled from Breeze LTP (₹{_breeze_live_open:,.2f})"
            except Exception:
                pass

        _default_open = (
            _breeze_live_open if _breeze_live_open and _breeze_live_open > 15000
            else float(spot or preds.get("predicted_open", 23000))
        )

        with st.form("recal_form"):
            _rc1, _rc2 = st.columns([2, 1])
            with _rc1:
                _actual_open = st.number_input(
                    "Actual Nifty 50 open price (9:15 AM)",
                    min_value=15000.0, max_value=99999.0,
                    value=float(_default_open),
                    step=1.0,
                    help=(_auto_open_note if _auto_open_note
                          else "Enter the actual Nifty index price at 9:15 AM open")
                )
                if _auto_open_note:
                    st.caption(f"📡 {_auto_open_note}")
                elif not _breeze_obj:
                    st.caption("ℹ️ Add Breeze session token in Settings to auto-fill this field.")
            with _rc2:
                st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                _recal_btn = st.form_submit_button("↻ Recalibrate signal", type="primary",
                                                    width="stretch")

        if _recal_btn:
            from settings import MIN_CONFIDENCE as _MIN_CONF
            _pred_open  = float(preds.get("predicted_open",  23000))
            _close_pct  = float(preds.get("close_pred_pct",  0.0))
            _c_dir      = int(preds.get("close_direction",   0))
            _c_conf     = float(preds.get("close_confidence",0.5))
            _ens        = bool(preds.get("ensemble_agree",   False))
            _atr_v      = float(preds.get("atr_pct",         0.8))
            _vix_v      = float(preds.get("india_vix",       16.0))

            # KEY FIX: anchor close target to ACTUAL open, not predicted open
            _close_target = round(_actual_open * (1 + _close_pct / 100))
            _gap_pts      = _actual_open - _pred_open
            _gap_pct_v    = _gap_pts / _pred_open * 100 if _pred_open else 0
            _large_div    = abs(_gap_pct_v) > 0.4

            _gap_opposes = ((_c_dir == 0 and _gap_pts > 50) or
                            (_c_dir == 1 and _gap_pts < -50))
            _gap_confirms= ((_c_dir == 0 and _gap_pts < -30) or
                            (_c_dir == 1 and _gap_pts > 30))

            _conf_adj = _c_conf
            if _large_div:         _conf_adj *= 0.80
            if _gap_opposes:       _conf_adj *= 0.70
            if _gap_confirms:      _conf_adj = min(_conf_adj * 1.05, 0.95)

            _rec_signal = "NO_TRADE"
            _rec_reason = ""
            if not _ens:
                _rec_reason = "Ensemble models disagree — no trade."
            elif _conf_adj < _MIN_CONF:
                _rec_reason = (f"Confidence dropped to {_conf_adj:.0%} after open divergence "
                               f"({_gap_pts:+.0f} pts). Below {_MIN_CONF:.0%} threshold.")
            elif _gap_opposes and _large_div:
                _rec_reason = (f"Open moved {_gap_pts:+.0f} pts opposite to signal. "
                               f"Wait for 9:20 AM 5-min candle to confirm.")
                _rec_signal = "WAIT"
            elif _vix_v > 25:
                _rec_reason = f"VIX {_vix_v:.1f} — premiums too expensive."
            else:
                _rec_signal = "BUY_CE" if _c_dir == 1 else "BUY_PE"

            if abs(_gap_pts) > 20:
                _msg = (f"Open diverged {_gap_pts:+.0f} pts ({_gap_pct_v:+.2f}%) "
                        f"from predicted ₹{_pred_open:,.0f}. ")
                if _gap_opposes:   _msg += "Gap CONTRADICTS signal — confidence reduced."
                elif _gap_confirms:_msg += "Gap CONFIRMS signal — partial move already done."
                else:              _msg += "Neutral gap — close target re-anchored to actual open."
                if _large_div: st.warning(_msg)
                else:          st.info(_msg)

            _m1, _m2, _m3, _m4 = st.columns(4)
            _m1.metric("Actual open",   f"₹{_actual_open:,.0f}",
                       delta=f"{_gap_pts:+.0f} vs predicted",
                       delta_color="inverse" if abs(_gap_pts) > 50 else "off")
            _m2.metric("Close target",  f"₹{_close_target:,.0f}",
                       delta=f"{_close_pct:+.2f}% from actual open",
                       delta_color="normal" if _c_dir == 1 else "inverse",
                       help="close_pct applied to actual open — not predicted open")
            _m3.metric("Adj. confidence", f"{_conf_adj:.0%}",
                       delta=f"{(_conf_adj - _c_conf)*100:+.0f}pp",
                       delta_color="inverse" if _conf_adj < _c_conf else "normal")
            _m4.metric("Signal", _rec_signal, delta_color="off")

            st.markdown("")
            _dir_str = "CALL (CE) ↑" if _c_dir == 1 else "PUT (PE) ↓"
            _wait_note = (" Confirm with first 5-min candle before entering."
                         if _large_div else "")
            if _rec_signal in ("BUY_CE", "BUY_PE"):
                st.success(f"Signal: **BUY {_dir_str}** | Close target ₹{_close_target:,} | "
                           f"Confidence {_conf_adj:.0%}.{_wait_note}")
            elif _rec_signal == "WAIT":
                st.warning(f"⏳ WAIT — {_rec_reason}")
            else:
                st.error(f"❌ NO TRADE — {_rec_reason}")

            st.caption("📌 Rule: Never enter before 9:20 AM. "
                       "If the 9:15–9:20 candle contradicts the signal — skip the trade.")

    # Ensure gift session state vars exist
    if "gift_live"    not in st.session_state: st.session_state["gift_live"]    = None
    if "gift_gap_pct" not in st.session_state: st.session_state["gift_gap_pct"] = 0.0
    if "gift_status"  not in st.session_state: st.session_state["gift_status"]  = "unavailable"

    if signal in ("BUY_CE", "BUY_PE"):
        st.markdown("#### 📋 Trade Parameters")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strike",        f"{suggestion['strike']} {suggestion['option_type']}")
        c2.metric("Expiry",        suggestion["expiry"])
        c3.metric("Lots",          suggestion["lots"])
        c4.metric("Capital used",  f"₹{suggestion['capital_used']:,}")

        st.markdown("")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Entry premium", f"₹{suggestion['premium_entry']}")
        c6.metric("Target premium",f"₹{suggestion['target_premium']}",
                  delta=f"+{suggestion['target_premium'] - suggestion['premium_entry']:.1f}")
        c7.metric("Stop loss",     f"₹{suggestion['sl_premium']}",
                  delta=f"-{suggestion['premium_entry'] - suggestion['sl_premium']:.1f}",
                  delta_color="inverse")
        c8.metric("Risk:Reward",   f"1 : {suggestion['risk_reward']}")

        st.markdown("")

        # Entry / exit steps
        st.markdown("#### ⏱ How to trade this")
        steps = [
            ("1", "Before 9:15 AM", "Review this signal. Confirm you're comfortable with the risk."),
            ("2", f"9:30 – 9:45 AM", f"Open Nifty 50 options chain. Buy **{suggestion['strike']} {suggestion['option_type']} {suggestion['expiry']}**. "
             f"Target entry premium ≈ **₹{suggestion['premium_entry']}** ({suggestion['lots']} lot{'s' if suggestion['lots']>1 else ''})"),
            ("3", "After entry",   f"Set target at **₹{suggestion['target_premium']}** and stop loss at **₹{suggestion['sl_premium']}**. "
             f"Max P&L: **+₹{suggestion['max_profit_inr']:,}** / **-₹{suggestion['max_loss_inr']:,}**"),
            ("4", f"By {suggestion['time_exit']}",
             "If neither target nor SL is hit, exit at market price. Never hold intraday options overnight."),
        ]
        for num, time_lbl, desc in steps:
            st.markdown(f"""
            <div class="step-box">
                <b>Step {num} — {time_lbl}</b><br>{desc}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Max P&L gauge
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("**Expected P&L range**")
            fig = go.Figure(go.Bar(
                x=[suggestion["max_loss_inr"], suggestion["max_profit_inr"]],
                y=["Max loss", "Max profit"],
                orientation="h",
                marker_color=["#ef5350", "#66bb6a"],
                text=[f"−₹{suggestion['max_loss_inr']:,}", f"+₹{suggestion['max_profit_inr']:,}"],
                textposition="outside",
            ))
            fig.update_layout(
                height=150, margin=dict(l=0, r=60, t=10, b=10),
                showlegend=False, plot_bgcolor="white",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with col_g2:
            st.markdown("**Market context**")
            ctx_data = {
                "Nifty spot":   f"₹{spot:,.2f}",
                "ATM strike":   str(suggestion["atm_strike"]),
                "India VIX":    str(suggestion["india_vix"]),
                "Expiry in":    f"{(suggestion['expiry_date'] - date.today()).days} days",
            }
            for k, v in ctx_data.items():
                st.markdown(f"<div style='display:flex;justify-content:space-between;"
                            f"padding:4px 0;border-bottom:1px solid #eee'>"
                            f"<span style='color:#666'>{k}</span>"
                            f"<b>{v}</b></div>", unsafe_allow_html=True)

    # ── End of day outcome form ────────────────────────────────────────────
    with st.expander("📝 Record today's outcome (fill after market close)"):
        st.markdown("Fill this in after 3:30 PM to track your P&L and model accuracy.")
        with st.form("outcome_form"):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                _spot_default = float(spot) if float(spot) >= 1000.0 else 22000.0
                actual_close = st.number_input("Nifty actual close price", min_value=1000.0,
                                               max_value=99999.0, value=_spot_default, step=1.0)
                exit_prem    = st.number_input("Your exit premium (₹)", min_value=0.0,
                                               max_value=9999.0, value=0.0, step=0.5)
            with col_f2:
                exit_reason = st.selectbox("Exit reason",
                                           ["TARGET", "STOP_LOSS", "TIME_EXIT", "MANUAL", "DID_NOT_TRADE"])
                notes = st.text_input("Notes (optional)", placeholder="e.g. slippage on entry")
            submitted = st.form_submit_button("💾 Save outcome", type="primary")
            if submitted and exit_reason != "DID_NOT_TRADE":
                tracker.update_outcome(date.today(), actual_close, exit_prem, exit_reason, notes)
                st.success("✅ Outcome saved! Check the Accuracy Tracker tab.")
                st.cache_data.clear()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 (Live Monitor) — shown as tab2 in the bar
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        import intraday_predictor as ip
        import live_engine as le
        IP_OK = True
    except Exception as _ie:
        IP_OK = False
        st.error(f"Live engine not available: {_ie}")

    if IP_OK:
        _settings_lm = load_settings()
        _breeze_lm   = st.session_state.get("breeze_obj")

        _hz_labels = {"5min":"5 min","15min":"15 min","30min":"30 min",
                      "60min":"1 hour","120min":"2 hours","180min":"3 hours",
                      "close":"Day close"}

        # ── Market status banner ───────────────────────────────────────────
        _status = le.market_status_text()
        _mkt_open = le.is_market_open()

        st.markdown("### 📡 Live multi-horizon predictions")
        if "🟢" in _status:
            st.success(_status)
        elif "🟡" in _status:
            st.info(_status)
        else:
            st.warning(_status)

        # ── AUTO-REFRESH (built-in fragment, no external package needed) ────
        _refresh_secs = le.next_refresh_seconds() if _mkt_open else 300
        if _mkt_open:
            st.caption(f"⏱️ Auto-refreshing every {_refresh_secs//60}m {_refresh_secs%60}s "
                       f"during market hours. You can also refresh manually below.")

        _manual_refresh = st.button("🔄 Refresh now", key="lm_manual_refresh")

        # ── Train models if needed ──────────────────────────────────────────
        if not ip.intraday_models_exist(str(cfg.MODEL_DIR)):
            st.warning("⚠️ Intraday models not trained yet.")
            if st.button("🚀 Train intraday models now", type="primary"):
                with st.spinner("Loading 2-year 5-min data (incremental) + training 7 calibrated models..."):
                    try:
                        from pathlib import Path as _P
                        _c5 = _P("data/nifty_5min_2yr.csv")
                        _df5 = None
                        if _c5.exists():
                            _df5 = pd.read_csv(_c5, parse_dates=["date"])
                            # Incremental: fetch gap days and append
                            if _breeze_lm is not None:
                                import pytz as _pz
                                _ist2 = _pz.timezone("Asia/Kolkata")
                                _ld = _df5["date"].max()
                                if hasattr(_ld, 'to_pydatetime'):
                                    _ld = _ld.to_pydatetime()
                                if _ld.tzinfo is None:
                                    _ld = _ist2.localize(_ld)
                                _gd = (datetime.now(_ist2) - _ld).days
                                if _gd > 1:
                                    _nd = df_mod.fetch_intraday_chunked(_breeze_lm, "NIFTY", total_days=_gd + 2, chunk_days=12)
                                    if _nd is not None and len(_nd) > 0:
                                        _df5 = pd.concat([_df5, _nd], ignore_index=True)
                                        _df5 = _df5.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
                                        _c5.parent.mkdir(parents=True, exist_ok=True)
                                        _df5.to_csv(_c5, index=False)
                        elif _breeze_lm is not None:
                            _df5 = df_mod.fetch_intraday_chunked(_breeze_lm, "NIFTY", total_days=730, chunk_days=12)
                            if _df5 is not None and len(_df5) > 0:
                                _c5.parent.mkdir(parents=True, exist_ok=True)
                                _df5.to_csv(_c5, index=False)
                        else:
                            _df5 = df_mod.load_intraday_data(_breeze_lm, force_refresh=True)

                        if _df5 is None or len(_df5) < 200:
                            st.error("Not enough 5-min data. Connect Breeze session token and retry.")
                        else:
                            from intraday_predictor import train_daily_models_from_5min
                            train_daily_models_from_5min(_df5, str(cfg.MODEL_DIR), verbose=False)
                            _res = ip.train_intraday_models(_df5, str(cfg.MODEL_DIR), verbose=False)
                            st.success(f"Trained daily + {len(_res)} intraday models on {len(_df5):,} candles (unified 5-min pipeline)! Refresh to see predictions.")
                            st.rerun()
                    except Exception as _te:
                        st.error(f"Training failed: {_te}")
        else:
            # ════════════════════════════════════════════════════════════════
            # LIVE PREDICTION FRAGMENT — auto-reruns every _refresh_secs
            # ════════════════════════════════════════════════════════════════
            def _render_live_predictions():
                _breeze_f = st.session_state.get("breeze_obj")

                # ── NON-TRADING DAY RECAP ─────────────────────────────────────
                if not le.is_trading_day() or (not le.is_market_open() and datetime.now(le.IST).time() > le.MARKET_CLOSE):
                    _hist_dates_recap = le.list_history_dates()
                    _last_td = None
                    for _hd in _hist_dates_recap:
                        _last_td = _hd
                        break
                    if _last_td:
                        _recap_df = le.get_daily_history(_last_td)
                        if not _recap_df.empty:
                            _verified = _recap_df[_recap_df["status"] != "Pending"]
                            _n_correct = len(_verified[_verified["status"].str.contains("Correct", na=False)])
                            _n_wrong = len(_verified[_verified["status"].str.contains("Wrong", na=False)])
                            _n_flat = len(_verified[_verified["status"].str.contains("Flat", na=False)])
                            _n_total = _n_correct + _n_wrong + _n_flat
                            _acc_pct = (_n_correct / max(_n_correct + _n_wrong, 1)) * 100

                            st.markdown(f"### Last trading day recap ({_last_td})")

                            _rc1, _rc2, _rc3, _rc4 = st.columns(4)
                            with _rc1:
                                st.metric("Predictions", _n_total)
                            with _rc2:
                                st.metric("Correct", _n_correct)
                            with _rc3:
                                st.metric("Wrong", _n_wrong)
                            with _rc4:
                                st.metric("Accuracy", f"{_acc_pct:.0f}%" if _n_total > 0 else "--")

                            st.markdown("#### How each prediction performed")
                            _hz_order_recap = ["5min","15min","30min","60min","120min","180min","close"]
                            for _hz_r in _hz_order_recap:
                                _hz_rows = _recap_df[_recap_df.get("horizon", pd.Series(dtype=str)) == _hz_r] if "horizon" in _recap_df.columns else pd.DataFrame()
                                if _hz_rows.empty:
                                    continue
                                for _, _rr in _hz_rows.iterrows():
                                    _pred_dir = _rr.get("pred_dir", "?")
                                    _status_r = _rr.get("status", "?")
                                    _entry_r = _rr.get("entry_price", 0)
                                    _target_r = _rr.get("target_price", 0)
                                    _actual_r = _rr.get("actual_price", 0)
                                    _conf_r = _rr.get("confidence", 0)
                                    _time_r = _rr.get("time", "")
                                    _move_r = (_actual_r - _entry_r) if _actual_r and _entry_r else 0

                                    _is_correct = "Correct" in str(_status_r)
                                    _is_wrong = "Wrong" in str(_status_r)
                                    _border_clr = "#27500A" if _is_correct else "#A32D2D" if _is_wrong else "#888888"
                                    _bg_clr = "#EAF3DE" if _is_correct else "#FCEBEB" if _is_wrong else "#F5F5F5"

                                    _hz_label = _hz_labels.get(_hz_r, _hz_r)
                                    st.markdown(
                                        f"<div style='border:1px solid var(--color-border-tertiary);border-left:4px solid {_border_clr};"
                                        f"border-radius:10px;padding:12px 16px;margin-bottom:8px;background:var(--color-background-primary)'>"
                                        f"<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>"
                                        f"<span style='font-size:14px;font-weight:600;min-width:80px'>{_hz_label}</span>"
                                        f"<span style='font-size:13px'>{_pred_dir}</span>"
                                        f"<span style='font-size:13px;color:var(--color-text-secondary)'>{_conf_r:.0%} conf</span>"
                                        f"<span style='font-size:13px;font-weight:600;color:{_border_clr}'>{_status_r}</span>"
                                        f"<span style='margin-left:auto;font-size:12px;color:var(--color-text-secondary)'>at {_time_r}</span>"
                                        f"</div>"
                                        f"<div style='display:flex;gap:20px;margin-top:8px;font-size:12px;color:var(--color-text-secondary)'>"
                                        f"<span>Entry: <b>₹{_entry_r:,.0f}</b></span>"
                                        f"<span>Target: <b>₹{_target_r:,.0f}</b></span>"
                                        f"<span>Actual: <b style='color:{_border_clr}'>₹{_actual_r:,.0f}</b> ({_move_r:+.0f} pts)</span>"
                                        f"</div>"
                                        f"</div>", unsafe_allow_html=True)

                            if _n_flat > 0:
                                st.caption(f"{_n_flat} prediction(s) were flat (move < 5 pts, not counted in accuracy).")

                    # ── Show predictions on last available cached data ────────
                    st.markdown("---")
                    st.markdown("### 🔮 Model predictions (last available data)")
                    from pathlib import Path as _PLM
                    _c5_closed = _PLM("data/nifty_5min_2yr.csv")
                    if _c5_closed.exists() and ip.intraday_models_exist(str(cfg.MODEL_DIR)):
                        _df5_closed = pd.read_csv(_c5_closed, parse_dates=["date"])
                        _last_date_closed = _df5_closed["date"].dt.date.max()
                        _preds_closed = ip.predict_all_horizons(_df5_closed, str(cfg.MODEL_DIR))
                        _err_closed = _preds_closed.pop("error", None)
                        _anchor_closed = _preds_closed.pop("_anchor_price", 0)
                        _preds_closed.pop("_last_candle_time", None)
                        _preds_closed.pop("_last_candle_date", None)
                        _preds_closed.pop("_minutes_elapsed", None)
                        _preds_closed.pop("_minutes_remaining", None)
                        _preds_closed.pop("_candle_close", None)
                        _preds_closed.pop("_stale_data", None)

                        if _err_closed:
                            st.warning(f"Could not generate predictions: {_err_closed}")
                        else:
                            st.caption(f"Based on cached data up to **{_last_date_closed}** | "
                                       f"Anchor price: ₹{_anchor_closed:,.2f}")
                            _hz_order_closed = ["5min","15min","30min","60min","120min","180min","close"]
                            for _hz_c in _hz_order_closed:
                                if _hz_c not in _preds_closed:
                                    continue
                                _p = _preds_closed[_hz_c]
                                _dir_c = _p.get("direction", 0)
                                _conf_c = _p.get("confidence", 0)
                                _target_c = _p.get("target_price", 0)
                                _move_c = _p.get("predicted_move_pct", 0)
                                _dir_icon = "🟢 Bullish" if _dir_c == 1 else "🔴 Bearish"
                                _border = "#27500A" if _dir_c == 1 else "#A32D2D"
                                _hz_lbl = _hz_labels.get(_hz_c, _hz_c)
                                st.markdown(
                                    f"<div style='border:1px solid var(--color-border-tertiary);border-left:4px solid {_border};"
                                    f"border-radius:10px;padding:12px 16px;margin-bottom:8px;background:var(--color-background-primary)'>"
                                    f"<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>"
                                    f"<span style='font-size:14px;font-weight:600;min-width:80px'>{_hz_lbl}</span>"
                                    f"<span style='font-size:13px'>{_dir_icon}</span>"
                                    f"<span style='font-size:13px;color:var(--color-text-secondary)'>{_conf_c:.0%} conf</span>"
                                    f"<span style='font-size:13px'>Target: ₹{_target_c:,.0f}</span>"
                                    f"<span style='margin-left:auto;font-size:12px;color:var(--color-text-secondary)'>{_move_c:+.2f}%</span>"
                                    f"</div></div>", unsafe_allow_html=True)
                    else:
                        st.info("No cached data or models not trained yet. Train the model to see predictions.")
                    return

                # ── LIVE MARKET FLOW ──────────────────────────────────────────
                # Live spot
                _live_spot = None
                _lq = None
                if _breeze_f:
                    try:
                        _lq = df_mod.fetch_live_quote_breeze(_breeze_f)
                        if _lq:
                            _live_spot = _lq.get("ltp") or _lq.get("open")
                    except Exception:
                        pass

                # Header metrics row
                _h1, _h2, _h3 = st.columns([1.3, 1, 1])
                with _h1:
                    if _live_spot:
                        _prev = _lq.get("prev_close", 0) if _lq else 0
                        _chg  = (_live_spot - _prev) if _prev else 0
                        _chgp = (_chg / _prev * 100) if _prev else 0
                        st.metric("Live Nifty", f"₹{_live_spot:,.2f}",
                                  delta=f"{_chg:+.2f} ({_chgp:+.2f}%)" if _prev else None)
                    else:
                        st.metric("Live Nifty", "--", help="Connect Breeze for live price")
                with _h2:
                    st.metric("Updated", datetime.now(le.IST).strftime("%I:%M:%S %p"))
                with _h3:
                    _mins_left = int((datetime.now(le.IST).replace(hour=15,minute=30,second=0) -
                                      datetime.now(le.IST)).total_seconds() / 60)
                    st.metric("Min to close", max(_mins_left, 0) if le.is_market_open() else "--")

                # Fetch candles + predict
                _df5_live = df_mod.load_intraday_data(_breeze_f,
                                                       force_refresh=le.is_market_open())

                # Optional Groww OFI
                _ofi_data = {"ofi": 0.0, "available": False, "signal": "", "bias": "neutral"}
                _groww_client = st.session_state.get("groww_obj")
                if not _groww_client:
                    try:
                        import groww_connector as gc
                        _gk = _settings_lm.get("groww_api_key", "")
                        _gs = _settings_lm.get("groww_api_secret", "")
                        if _gk or _settings_lm.get("groww_access_token", ""):
                            _groww_client = gc.init_groww(_gk, _gs)
                            st.session_state["groww_obj"] = _groww_client
                    except Exception:
                        pass
                if _groww_client:
                    try:
                        import groww_connector as gc
                        _ofi_data = gc.get_live_ofi(_groww_client, "NIFTY")
                    except Exception:
                        pass

                # Auto-verify due predictions
                if _live_spot:
                    _nv = le.verify_due_predictions(_live_spot)
                    if _nv:
                        _nc = sum(1 for r in _nv if r.get("correct") == 1)
                        _nf = sum(1 for r in _nv if r.get("flat"))
                        _nw = len(_nv) - _nc - _nf
                        _parts = [f"{_nc} correct", f"{_nw} wrong"]
                        if _nf:
                            _parts.append(f"{_nf} flat")
                        st.success(f"✅ Auto-verified {len(_nv)} prediction(s): "
                                   f"{', '.join(_parts)}.")

                if _df5_live is None or len(_df5_live) < 20:
                    st.warning("No 5-min candle data. Connect Breeze session token in Settings.")
                    return

                # Pass live spot as the anchor so targets aren't computed off stale candles
                _preds_live = ip.predict_all_horizons(_df5_live, str(cfg.MODEL_DIR),
                                                       live_price=_live_spot)
                _last_candle  = _preds_live.pop("_last_candle_time", "?")
                _last_cdate   = _preds_live.pop("_last_candle_date", "?")
                _mins_elap    = _preds_live.pop("_minutes_elapsed", 0)
                _mins_rem     = _preds_live.pop("_minutes_remaining", 0)
                _anchor_px    = _preds_live.pop("_anchor_price", _live_spot or 0)
                _candle_close = _preds_live.pop("_candle_close", 0)
                _stale        = _preds_live.pop("_stale_data", False)
                _err          = _preds_live.pop("error", None)

                if _err:
                    st.error(_err)
                    return

                # Anchor = live spot (authoritative); log predictions for verification
                _entry_log = _anchor_px
                if le.is_market_open() and _entry_log:
                    le.log_predictions_batch(_preds_live, _entry_log)

                # Warn if candle data is stale (Breeze hasn't streamed today's bars)
                if _stale:
                    st.warning(
                        f"⚠️ Intraday candle data is stale (last candle dated {_last_cdate}, "
                        f"{_last_candle}). Predictions are anchored to the LIVE spot "
                        f"(₹{_anchor_px:,.2f}) instead. Click 'Refresh now' — if it stays stale, "
                        f"your Breeze session may need a fresh token, or today's intraday feed "
                        f"hasn't started. Directional signals are still valid; price targets "
                        f"are estimates."
                    )
                    st.caption(f"Live spot ₹{_anchor_px:,.2f} | stale candle close ₹{_candle_close:,.2f} | "
                               f"{_mins_elap} min since open | {_mins_rem} min to close")
                else:
                    st.caption(f"Last candle: {_last_candle} | {_mins_elap} min since open | "
                               f"{_mins_rem} min to close | anchored at ₹{_entry_log:,.2f}")

                # OFI panel
                if _ofi_data["available"]:
                    _ofi = _ofi_data["ofi"]
                    _oc = ("var(--color-text-success)" if _ofi > 0.15
                           else "var(--color-text-danger)" if _ofi < -0.15
                           else "var(--color-text-secondary)")
                    st.markdown(
                        f"<div style='display:flex;gap:12px;align-items:center;padding:10px 14px;"
                        f"background:var(--color-background-secondary);border-radius:8px;margin:8px 0'>"
                        f"<span style='font-size:13px;font-weight:500'>📊 Order Flow Imbalance</span>"
                        f"<span style='font-size:18px;font-weight:600;color:{_oc}'>{_ofi:+.2f}</span>"
                        f"<span style='font-size:12px;color:var(--color-text-secondary)'>{_ofi_data['signal']}</span>"
                        f"</div>", unsafe_allow_html=True)
                elif _groww_client:
                    st.caption("📊 Groww connected — OFI activates during market hours (live depth needed).")

                # ── NEWS SENTIMENT (cached — respects GNews 5-min TTL) ────
                _news_lm = None
                _gnews_key_lm = _settings_lm.get("gnews_api_key", "")
                if _gnews_key_lm:
                    try:
                        import news_sentiment as _ns_lm
                        _news_lm = _ns_lm.get_market_sentiment(_gnews_key_lm)
                    except Exception:
                        _news_lm = None
                if _news_lm and _news_lm.get("n_articles", 0) > 0:
                    _nsc  = _news_lm.get("score", 0.0)
                    _nlbl = _news_lm.get("label", "neutral")
                    _nn   = _news_lm.get("n_articles", 0)
                    _nbk  = _news_lm.get("backend", "none")
                    _nclr = ("var(--color-text-success)" if _nsc > 0.10
                             else "var(--color-text-danger)" if _nsc < -0.10
                             else "var(--color-text-secondary)")
                    _nemoji = "📈" if _nsc > 0.10 else "📉" if _nsc < -0.10 else "📰"
                    st.markdown(
                        f"<div style='display:flex;gap:12px;align-items:center;padding:10px 14px;"
                        f"background:var(--color-background-secondary);border-radius:8px;margin:8px 0'>"
                        f"<span style='font-size:13px;font-weight:500'>{_nemoji} News sentiment</span>"
                        f"<span style='font-size:16px;font-weight:600;color:{_nclr};"
                        f"text-transform:capitalize'>{_nlbl} ({_nsc:+.2f})</span>"
                        f"<span style='font-size:12px;color:var(--color-text-secondary)'>"
                        f"{_nn} articles via {_nbk} · refreshes every 5 min</span>"
                        f"</div>", unsafe_allow_html=True)
                    if _nbk == "none":
                        st.caption("⚠️ Articles fetched but unscored — run `pip install vaderSentiment` and restart.")

                # ── SEPARATE CARD PER HORIZON ─────────────────────────────
                st.markdown("#### Predictions by horizon")
                _hz_order = ["5min","15min","30min","60min","120min","180min","close"]
                _available = [h for h in _hz_order if h in _preds_live]

                # Get live accuracy + magnitude per horizon for display on cards
                _calib_today = le.get_calibration_summary()
                _ph = _calib_today.get("per_horizon", {})
                _mag = _calib_today.get("magnitude", {})

                for _hz in _available:
                    _p = _preds_live[_hz]
                    _dir, _conf = _p["direction"], _p["confidence"]
                    _agree = _p["ensemble_agree"]
                    _tt = _p["target_time"]
                    _tgt_price = _p.get("target_price", 0)
                    _entry_price = _p.get("entry_price", _entry_log)

                    _dir_word = "UP ↑" if _dir == 1 else "DOWN ↓"
                    _dir_color = "#27500A" if _dir == 1 else "#A32D2D"
                    _dir_bg = "#EAF3DE" if _dir == 1 else "#FCEBEB"
                    _agree_badge = "✅ Both models agree" if _agree else "⚠️ Models disagree"
                    _move_pts = (_tgt_price - _entry_price) if _tgt_price else 0

                    # Historical accuracy for this horizon
                    _hz_acc = _ph.get(_hz, {}).get("accuracy")
                    _hz_mae = _mag.get(_hz, {}).get("mae_pts")
                    _acc_str = f"{_hz_acc}% dir. accuracy" if _hz_acc is not None else "no history yet"
                    _mae_str = f" · avg ±{_hz_mae} pts price error" if _hz_mae is not None else ""

                    st.markdown(
                        f"<div style='border:1px solid var(--color-border-tertiary);border-left:4px solid {_dir_color};"
                        f"border-radius:10px;padding:14px 16px;margin-bottom:10px;background:var(--color-background-primary)'>"
                        f"<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>"
                        f"<span style='font-size:15px;font-weight:600;min-width:80px'>{_hz_labels.get(_hz,_hz)}</span>"
                        f"<span style='background:{_dir_bg};color:{_dir_color};font-size:15px;font-weight:600;"
                        f"padding:3px 12px;border-radius:20px'>{_dir_word}</span>"
                        f"<span style='font-size:14px;color:var(--color-text-primary)'>{_conf:.0%} confidence</span>"
                        f"<span style='font-size:12px;color:var(--color-text-secondary)'>{_agree_badge}</span>"
                        f"<span style='margin-left:auto;font-size:12px;color:var(--color-text-secondary)'>by {_tt}</span>"
                        f"</div>"
                        f"<div style='display:flex;align-items:center;gap:20px;margin-top:10px;font-size:13px;color:var(--color-text-secondary)'>"
                        f"<span>Entry: <b style='color:var(--color-text-primary)'>₹{_entry_price:,.0f}</b></span>"
                        f"<span>Target: <b style='color:{_dir_color}'>₹{_tgt_price:,.0f}</b> ({_move_pts:+.0f} pts)</span>"
                        f"<span style='margin-left:auto'>{_acc_str}{_mae_str}</span>"
                        f"</div>"
                        f"</div>", unsafe_allow_html=True)

                # ── Recent verified predictions (direction + magnitude) ───
                _recent = le.get_recent_verifications(limit=10)
                if _recent:
                    st.markdown("#### 🔁 Recent results (direction + price accuracy)")
                    for _r in _recent:
                        _icon = "✅" if _r["correct"] else "❌"
                        _ds = "UP ↑" if _r["direction"] == 1 else "DOWN ↓"
                        _made = _r["ts"][11:16]
                        _entry = _r.get("entry_price", 0)
                        _actual = _r.get("actual_price", 0)
                        _tgt = _r.get("target_price", 0)
                        _perr = _r.get("price_error_pts")
                        _hzl = _hz_labels.get(_r["horizon"], _r["horizon"])
                        _col = "#27500A" if _r["correct"] else "#A32D2D"

                        # Price accuracy badge
                        _price_badge = ""
                        if _perr is not None and _tgt:
                            _pcolor = ("#27500A" if _perr <= 15 else
                                       "#BA7517" if _perr <= 40 else "#A32D2D")
                            _price_badge = (f"<span style='color:{_pcolor};font-size:12px'>"
                                            f"target ₹{_tgt:,.0f}, off by {_perr:.0f} pts</span>")

                        st.markdown(
                            f"<div style='display:flex;gap:10px;align-items:center;padding:7px 0;"
                            f"border-bottom:0.5px solid var(--color-border-tertiary);font-size:13px'>"
                            f"<span style='font-size:15px'>{_icon}</span>"
                            f"<span style='min-width:50px;color:var(--color-text-secondary)'>{_made}</span>"
                            f"<span style='min-width:60px;font-weight:500'>{_hzl}</span>"
                            f"<span style='min-width:70px'>{_ds}</span>"
                            f"<span style='min-width:130px;color:var(--color-text-secondary)'>"
                            f"₹{_entry:,.0f} → ₹{_actual:,.0f}</span>"
                            f"{_price_badge}"
                            f"</div>", unsafe_allow_html=True)

                # 5-min chart
                _today = date.today().isoformat()
                _tdf = _df5_live[_df5_live["date"].astype(str).str[:10] == _today].tail(75)
                if len(_tdf) > 2:
                    import plotly.graph_objects as go
                    _fig = go.Figure()
                    _fig.add_trace(go.Candlestick(
                        x=pd.to_datetime(_tdf["date"]),
                        open=_tdf["open"], high=_tdf["high"],
                        low=_tdf["low"], close=_tdf["close"],
                        increasing_line_color="#66bb6a", decreasing_line_color="#ef5350"))
                    if _live_spot:
                        _fig.add_hline(y=_live_spot, line_dash="dash", line_color="#378ADD",
                                       annotation_text=f"Live ₹{_live_spot:,.0f}")
                    _fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=20),
                                       plot_bgcolor="white", paper_bgcolor="white",
                                       xaxis_rangeslider_visible=False,
                                       xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"))
                    st.plotly_chart(_fig, width="stretch", config={"displayModeBar": False})

            # ── Run the live prediction display ─────────────────────────────
            # Auto-refresh strategy (in priority order):
            #   1. st.fragment(run_every) — native, reruns ONLY this section
            #      every 60s without a full page reload. Streamlit >= 1.33.
            #   2. streamlit_autorefresh — if installed, triggers a full rerun.
            #   3. Manual "Refresh now" button — always available.
            # The fragment updates the live spot + predictions every 60 seconds
            # during market hours (new 5-min candles arrive every 5 min, but
            # the live spot and countdown update every minute).
            _refreshed = False
            if _mkt_open:
                try:
                    _frag = st.fragment(run_every="60s")(_render_live_predictions)
                    _frag()
                    _refreshed = True
                except Exception:
                    _refreshed = False

                # Secondary: streamlit_autorefresh full-page trigger if available
                if not _refreshed:
                    try:
                        from streamlit_autorefresh import st_autorefresh
                        st_autorefresh(interval=60_000, key="lm_autorefresh")
                        _render_live_predictions()
                        _refreshed = True
                    except ImportError:
                        pass

            if not _refreshed:
                _render_live_predictions()
                if _mkt_open:
                    st.info("💡 For hands-free updates every minute, install the helper: "
                            "`pip install streamlit-autorefresh` and restart. "
                            "Otherwise use the Refresh button above.")

            # ── Daily history + calibration + magnitude ────────────────────
            st.divider()
            st.markdown("#### 📋 Prediction history (direction + magnitude accuracy)")
            _hist_dates = le.list_history_dates()
            _sel_date = date.today().isoformat()
            if _hist_dates:
                _sel_date = st.selectbox("View date", _hist_dates, index=0, key="hist_date_sel")
            _hist_df = le.get_daily_history(_sel_date)
            if len(_hist_df) > 0:
                _cols_show = ["time","horizon","pred_dir","confidence","entry_price","actual_price","status"]
                if "price_error_pts" in _hist_df.columns:
                    _cols_show.insert(6, "price_error_pts")
                _disp = _hist_df[[c for c in _cols_show if c in _hist_df.columns]].copy()
                _rename = {"time":"Time","horizon":"Horizon","pred_dir":"Predicted",
                           "confidence":"Conf","entry_price":"Entry ₹","actual_price":"Actual ₹",
                           "price_error_pts":"Price err (pts)","status":"Result"}
                _disp = _disp.rename(columns=_rename)
                if "Conf" in _disp.columns:
                    _disp["Conf"] = (pd.to_numeric(_disp["Conf"],errors="coerce")*100).round(0).fillna(0).astype(int).astype(str)+"%"
                if "Horizon" in _disp.columns:
                    _disp["Horizon"] = _disp["Horizon"].map(_hz_labels).fillna(_disp["Horizon"])
                st.dataframe(_disp, width="stretch", hide_index=True, height=260)

                _calib = le.get_calibration_summary(_sel_date)
                if _calib.get("n_verified", 0) > 0:
                    _cc = st.columns(4)
                    _cc[0].metric("Predictions", _calib["n_total"])
                    _cc[1].metric("Verified", _calib["n_verified"])
                    _cc[2].metric("Direction accuracy", f"{_calib['overall_acc']}%")
                    _mag_all = _calib.get("magnitude", {})
                    if _mag_all:
                        _avg_mae = round(np.mean([m["mae_pts"] for m in _mag_all.values()]), 1)
                        _cc[3].metric("Avg price error", f"±{_avg_mae} pts",
                                      help="How far predicted target prices were from actual")

                    # Magnitude accuracy table per horizon
                    if _mag_all:
                        st.markdown("**Price magnitude accuracy by horizon**")
                        st.caption("Direction can be right while the predicted price is far off. "
                                   "This shows how close the predicted target was to the actual price.")
                        _mag_rows = []
                        for _hz, _m in _mag_all.items():
                            _quality = ("🟢 Tight" if _m["mae_pts"] <= 15 else
                                        "🟡 Moderate" if _m["mae_pts"] <= 40 else "🔴 Wide")
                            _mag_rows.append({
                                "Horizon": _hz_labels.get(_hz, _hz),
                                "Predictions": _m["n"],
                                "Avg price error": f"±{_m['mae_pts']} pts",
                                "As % of price": f"{_m['mae_pct']:.2f}%",
                                "Magnitude quality": _quality,
                            })
                        st.dataframe(pd.DataFrame(_mag_rows), width="stretch", hide_index=True)

                    # Calibration buckets
                    if _calib.get("calibration"):
                        st.markdown("**Confidence calibration**")
                        _cal_rows = []
                        for _b, _cd in _calib["calibration"].items():
                            _gap = _cd["actual_acc"] - _cd["expected_acc"]
                            _cal_rows.append({
                                "Confidence": _b, "Predictions": _cd["n"],
                                "Expected": f"{_cd['expected_acc']:.0f}%",
                                "Actual": f"{_cd['actual_acc']:.1f}%",
                                "Status": ("✅ Calibrated" if abs(_gap)<=8 else
                                           "⚠️ Overconfident" if _gap<0 else "📈 Underconfident"),
                            })
                        st.dataframe(pd.DataFrame(_cal_rows), width="stretch", hide_index=True)

                _csv = _hist_df.to_csv(index=False).encode()
                _dcol1, _dcol2, _dcol3 = st.columns([1.2, 1.4, 1.4])
                with _dcol1:
                    st.download_button("⬇️ Export (CSV)", _csv,
                                       f"predictions_{_sel_date}.csv", "text/csv", key="hist_dl")
                with _dcol2:
                    if st.button("🗑 Reset this day's log", key="clear_day",
                                 help="Remove this day's predictions (e.g. if they were logged with a stale price anchor)"):
                        _n = le.clear_predictions_for_date(_sel_date)
                        st.success(f"Cleared {_n} prediction(s) for {_sel_date}. Stats will rebuild from fresh predictions.")
                        st.rerun()
                with _dcol3:
                    if st.button("🗑 Reset ALL prediction history", key="clear_all",
                                 help="Wipe the entire prediction log and start the accuracy stats fresh"):
                        _n = le.clear_all_predictions()
                        st.success(f"Cleared all {_n} logged predictions. Accuracy stats reset.")
                        st.rerun()
            else:
                st.caption("No predictions logged for this date yet.")

            # EOD retrain
            if le.is_eod_retrain_window() and le.should_retrain_today(str(cfg.MODEL_DIR)):
                st.info("📚 Market closed — retraining intraday models with today's data…")
                try:
                    _eod = le.run_eod_retrain(st.session_state.get("breeze_obj"), str(cfg.MODEL_DIR))
                    if "error" not in _eod:
                        st.success(f"Retrained {len(_eod)} models with today's candles.")
                except Exception as _re:
                    st.warning(f"EOD retrain failed: {_re}")

        # Model info
        _intra_meta = ip.load_intraday_metadata(str(cfg.MODEL_DIR))
        if _intra_meta:
            with st.expander("ℹ️ Intraday model training info"):
                _hzr = _intra_meta.get("horizons", {})
                if _hzr:
                    _mdf = pd.DataFrame([
                        {"Horizon": _hz_labels.get(h,h), "CV Accuracy": f"{v['cv_accuracy']*100:.1f}%",
                         "Training samples": f"{v['n_samples']:,}",
                         "Calibrated": "✅" if v.get("calibrated") else "—"}
                        for h, v in _hzr.items()])
                    st.dataframe(_mdf, width="stretch", hide_index=True)
                st.caption(f"Last trained: {_intra_meta.get('trained_at','—')[:16]}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ACCURACY TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    trades_df = tracker.load_trades()
    stats     = tracker.compute_stats(trades_df)

    # ── Pending vs completed split ─────────────────────────────────────────
    # Even before outcomes are recorded, we can show useful info about
    # logged suggestions (CE/PE mix, avg confidence, projected risk, etc.)
    pending_df = pd.DataFrame()
    if trades_df is not None and len(trades_df) > 0:
        # A trade is "pending" if direction_correct is empty/NaN
        mask_pend = (trades_df["direction_correct"].isna()) | (trades_df["direction_correct"] == "")
        pending_df = trades_df[mask_pend].copy()

    # ── Callout when there's pending work ──────────────────────────────────
    if stats["total_trades"] > 0 and stats["completed"] == 0:
        st.info(
            f"📋 **{stats['total_trades']} suggestion(s) logged, none completed yet.** "
            "The metrics below will populate once you record actual outcomes. "
            "Scroll down to **✏️ Edit a past trade outcome** to update."
        )
    elif len(pending_df) > 0 and stats["completed"] > 0:
        st.info(f"📋 {len(pending_df)} suggestion(s) waiting for outcomes. Scroll down to record them.")

    # ── Summary metrics ────────────────────────────────────────────────────
    st.markdown("#### 📊 Performance summary")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total trades",     stats["total_trades"])
    m2.metric("Completed",        stats["completed"])
    m3.metric("Direction accuracy", f"{stats['accuracy_all']}%" if stats["completed"] else "—",
              help="% of times model predicted the correct market direction. Computed only on completed trades.")
    m4.metric("Win rate",         f"{stats['win_rate']}%" if stats["completed"] else "—",
              help="% of trades that were profitable (P&L > 0). Computed only on completed trades.")
    m5.metric("Total P&L",        f"₹{stats['total_pnl']:,}" if stats["completed"] else "—",
              delta=f"₹{stats['total_pnl']:,}" if stats["total_pnl"] != 0 else None)
    m6.metric("Realized R:R",     f"1 : {stats['risk_reward']}" if stats["completed"] else "—",
              help="Average winner vs average loser ratio. Computed only on completed trades.")

    st.markdown("")
    r1, r2, r3 = st.columns(3)
    r1.metric("7-day accuracy",  f"{stats['accuracy_7d']}%" if stats["completed"] else "—",
              delta=f"{stats['accuracy_7d'] - stats['accuracy_all']:.1f}% vs overall" if stats["completed"] else None)
    r2.metric("30-day accuracy", f"{stats['accuracy_30d']}%" if stats["completed"] else "—")
    r3.metric("Max drawdown",    f"₹{stats['max_drawdown']:,}" if stats["completed"] else "—",
              delta_color="inverse",
              delta=f"−₹{stats['max_drawdown']:,}" if stats["max_drawdown"] > 0 else None)

    # ── Pending-trade insights (useful before any completions exist) ───────
    if len(pending_df) > 0:
        st.divider()
        st.markdown("#### 📥 Logged suggestions (not yet completed)")
        try:
            # CE/PE split
            opt_types = pending_df["option_type"].astype(str).str.upper()
            n_ce = int((opt_types == "CE").sum())
            n_pe = int((opt_types == "PE").sum())
            n_no = int(pending_df["signal"].astype(str).str.contains("NO_TRADE", na=False).sum())

            # Confidence stats — only on rows with a numeric confidence
            conf = pd.to_numeric(pending_df.get("confidence"), errors="coerce").dropna()
            avg_conf = float(conf.mean()) * 100 if len(conf) > 0 else 0

            # Helper: safely extract a numeric Series from a DataFrame column
            def _col_series(df, col):
                raw = df[col] if col in df.columns else pd.Series(dtype=float)
                return pd.to_numeric(pd.Series(raw) if not isinstance(raw, pd.Series) else raw,
                                     errors="coerce")

            # Capital staged across pending trades
            cap_used        = _col_series(pending_df, "capital_used").fillna(0)
            total_cap       = int(cap_used.sum())

            # Projected max-loss across pending trades
            max_loss        = _col_series(pending_df, "max_loss_inr").fillna(0)
            total_proj_loss = int(max_loss.sum())

            # Projected max-profit across pending trades
            max_profit        = _col_series(pending_df, "max_profit_inr").fillna(0)
            total_proj_profit = int(max_profit.sum())

            # Average projected R:R
            rr_proj = _col_series(pending_df, "risk_reward").dropna()
            avg_rr  = float(rr_proj.mean()) if len(rr_proj) > 0 else 0

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("CE / PE / NO_TRADE",  f"{n_ce} / {n_pe} / {n_no}",
                      help="Breakdown of logged signal types")
            p2.metric("Avg suggested confidence", f"{avg_conf:.0f}%" if avg_conf else "—")
            p3.metric("Capital staged (₹)", f"₹{total_cap:,}",
                      help="Total ₹ committed across all logged trades (sum of capital_used)")
            p4.metric("Avg projected R:R", f"1 : {avg_rr:.2f}" if avg_rr else "—")

            p5, p6 = st.columns(2)
            p5.metric("Projected max profit (₹)", f"₹{total_proj_profit:,}",
                      help="If every logged trade hit target. Not a prediction — just the sum.")
            p6.metric("Projected max loss (₹)",   f"₹{total_proj_loss:,}",
                      help="If every logged trade hit stop loss. Bounds your downside.")

            st.caption(
                "_These numbers reflect what the model **suggested**, not actual outcomes. "
                "Real accuracy, win-rate, and P&L will appear above once you record outcomes._"
            )
        except Exception as e:
            st.caption(f"_Pending-trade summary skipped: {e}_")

    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────
    if stats["completed"] > 0:
        done   = stats["completed_df"]
        cum_pnl = stats["cum_pnl"]
        dates_done = done["date"].tolist()

        col_ch1, col_ch2 = st.columns(2)

        with col_ch1:
            st.markdown("**Cumulative P&L (₹)**")
            color = "#66bb6a" if cum_pnl[-1] >= 0 else "#ef5350"
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=list(range(1, len(cum_pnl) + 1)),
                y=cum_pnl,
                fill="tozeroy",
                fillcolor="rgba(102,187,106,0.12)" if cum_pnl[-1] >= 0 else "rgba(239,83,80,0.12)",
                line=dict(color=color, width=2),
                mode="lines+markers",
                marker=dict(size=5),
            ))
            fig1.add_hline(y=0, line_dash="dash", line_color="#aaa", line_width=1)
            fig1.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=30),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Trade #", gridcolor="#f0f0f0"),
                yaxis=dict(title="₹", gridcolor="#f0f0f0"),
            )
            st.plotly_chart(fig1, width="stretch", config={"displayModeBar": False})

        with col_ch2:
            st.markdown("**Rolling 10-trade direction accuracy (%)**")
            roll_acc = (done["direction_correct"]
                        .rolling(10, min_periods=3)
                        .mean() * 100).tolist()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(range(1, len(roll_acc) + 1)),
                y=roll_acc,
                line=dict(color="#5c6bc0", width=2),
                mode="lines+markers",
                marker=dict(size=5),
            ))
            fig2.add_hline(y=65, line_dash="dot", line_color="#66bb6a",
                           annotation_text="Target 65%", annotation_position="right")
            fig2.add_hline(y=50, line_dash="dot", line_color="#ef5350",
                           annotation_text="Retrain alert", annotation_position="right")
            fig2.update_layout(
                height=280, margin=dict(l=0, r=60, t=10, b=30),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Trade #", gridcolor="#f0f0f0"),
                yaxis=dict(title="%", range=[0, 105], gridcolor="#f0f0f0"),
            )
            st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

        # ── P&L by month ──────────────────────────────────────────────────
        st.markdown("**Monthly P&L breakdown**")
        done["month"] = pd.to_datetime(done["date"]).dt.to_period("M").astype(str)
        monthly = done.groupby("month")["pnl"].sum().reset_index()
        monthly.columns = ["Month", "P&L"]
        fig3 = px.bar(
            monthly, x="Month", y="P&L",
            color="P&L",
            color_continuous_scale=["#ef5350", "#ffffff", "#66bb6a"],
            color_continuous_midpoint=0,
            text=monthly["P&L"].apply(lambda x: f"₹{int(x):,}"),
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=30),
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

    else:
        st.info("📭 No completed trades yet. Make trades and record outcomes to see stats here.")

    # ── Trade log table ────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Full trade log**")
    if len(trades_df) > 0:
        display_cols = ["date", "signal", "strike", "confidence", "premium_entry",
                        "exit_premium", "exit_reason", "pnl", "direction_correct", "notes"]
        disp = trades_df[[c for c in display_cols if c in trades_df.columns]].copy()
        disp["confidence"] = pd.to_numeric(disp["confidence"], errors="coerce").map(
            lambda x: f"{x:.0%}" if pd.notna(x) else "")
        disp["pnl"] = pd.to_numeric(disp["pnl"], errors="coerce").map(
            lambda x: f"₹{int(x):,}" if pd.notna(x) else "")
        disp["direction_correct"] = disp["direction_correct"].map(
            lambda x: "✅" if str(x) == "1" else ("❌" if str(x) == "0" else ""))
        st.dataframe(disp.sort_values("date", ascending=False), width="stretch", height=320)
    else:
        st.info("No trades logged yet.")

    # ── Manual outcome editor ──────────────────────────────────────────────
    with st.expander("✏️ Edit a past trade outcome"):
        if len(trades_df) > 0:
            trade_dates = trades_df["date"].astype(str).tolist()
            sel_date = st.selectbox("Select trade date", trade_dates)
            c1, c2, c3 = st.columns(3)
            with c1:
                act_close = st.number_input("Nifty actual close", min_value=1000.0, value=22000.0)
            with c2:
                exit_p = st.number_input("Exit premium (₹)", min_value=0.0, value=0.0, step=0.5)
            with c3:
                ex_reason = st.selectbox("Exit reason", ["TARGET", "STOP_LOSS", "TIME_EXIT", "MANUAL"])
            notes2 = st.text_input("Notes")
            if st.button("Update outcome"):
                tracker.update_outcome(sel_date, act_close, exit_p, ex_reason, notes2)
                st.success("✅ Updated successfully.")
                st.rerun()
        else:
            st.write("No trades to edit yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    try:
        model_dir_str = str(cfg.MODEL_DIR)
    except Exception:
        model_dir_str = "models"

    # ── Model status ──────────────────────────────────────────────────────
    st.markdown("#### 🧠 Model status")

    try:
        meta = mt.load_metadata(model_dir_str)
    except Exception as e:
        meta = {}
        st.warning(f"Could not read model metadata: {e}")

    if meta:
        # New trainer writes cv_open + cv_close (two separate models).
        # Older trainer wrote a single cv_accuracy. Support both for backward compat.
        cv_open  = float(meta.get("cv_open",     meta.get("cv_accuracy", 0)))
        cv_close = float(meta.get("cv_close",    meta.get("cv_accuracy", 0)))
        c1, c2, c3, c4 = st.columns(4)
        if "cv_open" in meta or "cv_close" in meta:
            _o_sk = meta.get("open_skill")
            _c_sk = meta.get("close_skill")
            _help = None
            if _o_sk is not None and _c_sk is not None:
                _help = (
                    "Skill = accuracy above the majority-class baseline (always guessing "
                    "the more common outcome).\n\n"
                    f"OPEN: {cv_open*100:.1f}% vs baseline {meta.get('open_baseline_acc',0)*100:.1f}% "
                    f"(gap-up base rate {meta.get('open_base_rate',0)*100:.1f}%) -> "
                    f"skill {_o_sk*100:+.1f}%\n\n"
                    f"CLOSE: {cv_close*100:.1f}% vs baseline {meta.get('close_baseline_acc',0)*100:.1f}% "
                    f"(bull base rate {meta.get('close_base_rate',0)*100:.1f}%) -> "
                    f"skill {_c_sk*100:+.1f}%"
                )
            _delta = (f"skill {_o_sk*100:+.1f}% / {_c_sk*100:+.1f}%"
                      if (_o_sk is not None and _c_sk is not None) else None)
            c1.metric("ML Model (open/close)",
                     f"{cv_open*100:.1f}% / {cv_close*100:.1f}%",
                     delta=_delta, delta_color="off", help=_help)
        else:
            c1.metric("ML Model accuracy", f"{cv_open*100:.1f}%")
        total_candles = meta.get("total_candles", meta.get("n_samples", 0))
        n_days = meta.get("n_days", meta.get("n_samples", 0))
        c2.metric("Training candles", f"{total_candles:,}", delta=f"{n_days} days")
        c3.metric("Features",      meta.get("n_features", 0))
        c4.metric("Last trained",  str(meta.get("trained_at", "---"))[:10])
        st.caption(
            "ML model accuracy is from pure technical indicators only. "
            "The **combined system** (Model + GIFT + OFI + News + LLM) "
            "targets 60-65% by aggregating multiple independent signal sources."
        )
    else:
        st.info("ℹ️ Model has not been trained yet. Click **Train model now** below to get started.")

    # ── Retrain alert ──────────────────────────────────────────────────────
    try:
        trades_df2 = tracker.load_trades()
        stats2     = tracker.compute_stats(trades_df2)
        retrain_threshold = getattr(cfg, "RETRAIN_THRESHOLD", 0.50) * 100
        if stats2["accuracy_7d"] > 0 and stats2["accuracy_7d"] < retrain_threshold:
            st.warning(
                f"⚠️ 7-day accuracy ({stats2['accuracy_7d']}%) fell below "
                f"{retrain_threshold:.0f}%. Consider retraining the model."
            )
    except Exception:
        pass

    st.divider()

    # ── Train model section ────────────────────────────────────────────────
    st.markdown("#### 🔄 Train / retrain model")
    st.markdown(
        "Training builds the prediction model from historical Nifty data. "
        "Takes about **1–2 minutes**. Do this once on first setup, then monthly "
        "or whenever 7-day accuracy drops below 50%."
    )

    # Show what data sources will be used
    with st.expander("ℹ️ What happens when you click Train"):
        st.markdown("""
        1. **Fetches ~2 years of 5-min intraday candles** from Breeze API (incremental, cached data preserved)
        2. **Fetches VIX & FII** context data
        3. **Builds 45+ features from 5-min candles** — multi-scale EMAs, MACD, RSI, VWAP, Bollinger, ATR, volume profiles, session patterns, calendar
        4. **Trains daily Open/Close/High/Low models** from end-of-day 5-min feature snapshots (XGB + LGB with calibration)
        5. **Trains 7 intraday horizon models** (5m/15m/30m/1h/2h/3h/close) on raw 5-min candles
        6. The dashboard auto-refreshes once training is complete
        """)

    col_train, col_info = st.columns([1, 2])
    with col_train:
        train_btn = st.button("🚀 Train model now", type="primary", width="stretch")
    with col_info:
        if meta:
            fold_scores = meta.get("fold_scores", [])
            if fold_scores:
                st.markdown(
                    "**Last CV fold scores:** " +
                    " | ".join(f"{s*100:.1f}%" for s in fold_scores)
                )
            period = f"{str(meta.get('train_start',''))[:10]}  →  {str(meta.get('train_end',''))[:10]}"
            st.caption(f"Training period: {period}")

    # ── Training execution ─────────────────────────────────────────────────
    if train_btn:
        log_step("Train model -- clicked (2-year 5-min intraday pipeline)")
        progress_bar = st.progress(0, text="Starting...")
        status_box   = st.empty()

        try:
            # Step 1: Connect to Breeze
            status_box.info("Step 1/7 -- Reading credentials...")
            progress_bar.progress(5, text="Reading credentials...")
            log_step("Step 1/7 -- reading credentials...")

            saved_s   = load_settings()
            api_key3  = saved_s.get("api_key", getattr(cfg, "BREEZE_API_KEY", ""))
            api_sec3  = saved_s.get("api_secret", getattr(cfg, "BREEZE_API_SECRET", ""))
            ses_tok3  = saved_s.get("session_token", "")

            breeze3 = None
            if ses_tok3 and api_key3 and api_key3 not in ("", "YOUR_API_KEY_HERE"):
                try:
                    breeze3 = df_mod.init_breeze(api_key3, api_sec3, ses_tok3)
                    status_box.info("Step 1/7 -- Breeze API connected")
                    log_step("Step 1/7 -- Breeze connected")
                except Exception as be:
                    progress_bar.empty()
                    status_box.error(f"Breeze connection failed: {be}. A valid session token is required for 2-year intraday training.")
                    st.stop()
            else:
                progress_bar.empty()
                status_box.error("Breeze API key and session token are required for 2-year intraday training. Configure them in Settings.")
                st.stop()

            # Step 2: Incremental 5-min data fetch (2 years)
            from pathlib import Path as _Path
            _cache_file = _Path("data/nifty_5min_2yr.csv")
            _cache_file.parent.mkdir(parents=True, exist_ok=True)

            progress_bar.progress(10, text="Fetching 2-year 5-min intraday data...")
            status_box.info("Step 2/7 -- Fetching 2-year 5-min intraday candles (incremental, cached data preserved)...")
            log_step("Step 2/7 -- incremental 5-min fetch starting...")

            import pytz as _pytz
            _ist = _pytz.timezone("Asia/Kolkata")
            _now_ist = datetime.now(_ist)
            _existing_5min = None

            if _cache_file.exists():
                _existing_5min = pd.read_csv(_cache_file, parse_dates=["date"])
                _last_dt = _existing_5min["date"].max()
                if hasattr(_last_dt, 'to_pydatetime'):
                    _last_dt = _last_dt.to_pydatetime()
                if _last_dt.tzinfo is None:
                    _last_dt = _ist.localize(_last_dt)
                _gap = (_now_ist - _last_dt).days
                if _gap <= 1:
                    status_box.info(f"Step 2/7 -- Cache up to date ({len(_existing_5min):,} candles). Skipping fetch.")
                    log_step(f"Step 2/7 -- cache up to date, {len(_existing_5min)} candles")
                else:
                    status_box.info(f"Step 2/7 -- Cache has {len(_existing_5min):,} candles, gap: {_gap} days. Fetching missing data...")
                    _new_5min = df_mod.fetch_intraday_chunked(breeze3, "NIFTY", total_days=_gap + 2, chunk_days=12)
                    if _new_5min is not None and len(_new_5min) > 0:
                        _existing_5min = pd.concat([_existing_5min, _new_5min], ignore_index=True)
                        _existing_5min = _existing_5min.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
                        _existing_5min.to_csv(_cache_file, index=False)
                        log_step(f"Step 2/7 -- appended {len(_new_5min)} candles, total {len(_existing_5min)}")
            else:
                status_box.info("Step 2/7 -- No cache found. Full 3-year fetch (this takes a few minutes on first run)...")
                _existing_5min = df_mod.fetch_intraday_chunked(breeze3, "NIFTY", total_days=1095, chunk_days=12)
                if _existing_5min is not None and len(_existing_5min) > 0:
                    _existing_5min.to_csv(_cache_file, index=False)
                    log_step(f"Step 2/7 -- fetched {len(_existing_5min)} candles")

            if _existing_5min is None or len(_existing_5min) < 200:
                progress_bar.empty()
                status_box.error("Not enough 5-min data. Check Breeze session token and retry.")
                st.stop()

            status_box.info(f"Step 2/5 -- {len(_existing_5min):,} five-minute candles ready")

            # Step 3: Fetch daily context (VIX, FII)
            progress_bar.progress(30, text="Fetching VIX & FII context...")
            status_box.info("Step 3/5 -- Downloading VIX, FII context data...")
            log_step("Step 3/5 -- supplementary context...")
            _daily_ctx = pd.DataFrame()
            try:
                _vix3 = df_mod.load_vix_data(breeze3, force_refresh=True)
                _fii3 = df_mod.load_fii_dii_data(force_refresh=True)
                if _vix3 is not None and "india_vix" in _vix3.columns:
                    _daily_ctx = _vix3[["date", "india_vix"]].copy()
                if _fii3 is not None and "fii_net" in _fii3.columns:
                    if len(_daily_ctx) > 0:
                        _daily_ctx = _daily_ctx.merge(_fii3[["date", "fii_net"]], on="date", how="outer")
                    else:
                        _daily_ctx = _fii3[["date", "fii_net"]].copy()
            except Exception as _ctx_e:
                log_step(f"Step 3/5 -- context data partial: {_ctx_e}", "warning")

            # Step 4: Train daily models from 5-min EOD snapshots
            progress_bar.progress(50, text="Training daily models (5-min features)...")
            status_box.info("Step 4/5 -- Training daily Open/Close/High/Low from 5-min EOD features...")
            log_step("Step 4/5 -- training daily models (unified 5-min pipeline)...")
            from intraday_predictor import train_daily_models_from_5min
            results = train_daily_models_from_5min(
                _existing_5min, model_dir_str, daily_context=_daily_ctx, verbose=True
            )
            log_step("Step 4/5 -- daily models trained")

            # Step 5: Train intraday horizon models
            progress_bar.progress(80, text="Training 7 intraday horizon models...")
            status_box.info("Step 5/5 -- Training 7 intraday horizon models (5m/15m/30m/1h/2h/3h/close)...")
            log_step("Step 5/5 -- training intraday horizon models...")
            try:
                _intra_res = ip.train_intraday_models(_existing_5min, str(cfg.MODEL_DIR), verbose=False)
                _n_intra = len(_intra_res) if _intra_res else 0
                log_step(f"Step 5/5 -- {_n_intra} intraday models trained")
            except Exception as _ie:
                _n_intra = 0
                log_step(f"Step 5/5 -- intraday models skipped: {_ie}", "warning")

            progress_bar.progress(100, text="Done!")

            _n_days = _existing_5min["date"].dt.date.nunique()
            _yrs = round(_n_days / 252, 1)
            _cv_o = results.get("open_cv", 0)
            _cv_c = results.get("close_cv", 0)
            _candles = len(_existing_5min)
            status_box.success(
                f"Trained on **{_candles:,} five-min candles** ({_n_days:,} days / {_yrs} yrs) -- "
                f"Open CV: **{_cv_o*100:.1f}%** | Close CV: **{_cv_c*100:.1f}%** | "
                f"Intraday models: {_n_intra} | Pipeline: unified 5-min"
            )

            _best = max(_cv_o, _cv_c)
            if _best < 0.52:
                st.warning(
                    "Accuracy below 52%. Markets may be in a choppy regime. "
                    "Paper-trade first and monitor the 7-day rolling accuracy."
                )
            elif _best >= 0.60:
                st.success("Accuracy above 60% -- model is ready for paper trading!")

            st.cache_resource.clear()
            st.rerun()

        except Exception as e:
            log.exception("Training failed")
            progress_bar.empty()
            status_box.error(f"Training failed: {e}")
            st.exception(e)

    # ── Feature importance chart ───────────────────────────────────────────
    try:
        imp = mt.load_importance(model_dir_str)
        if not imp.empty:
            st.divider()
            st.markdown("#### 🏆 Top 20 most influential features")
            st.caption(
                "These are the indicators the model relies on most. "
                "Higher score = stronger influence on the prediction."
            )
            top20 = imp.head(20).sort_values("importance")
            fig_imp = px.bar(
                top20, x="importance", y="feature", orientation="h",
                color="importance",
                color_continuous_scale=["#bbdefb", "#1565c0"],
                labels={"importance": "Importance score", "feature": ""},
            )
            fig_imp.update_layout(
                height=520, margin=dict(l=0, r=20, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#f0f0f0"),
            )
            st.plotly_chart(fig_imp, width="stretch", config={"displayModeBar": False})
    except Exception:
        pass   # Feature importance chart is optional; don't crash if unavailable


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### ⚙️ Configuration")

    saved = load_settings()

    st.markdown("##### 🔑 ICICI Breeze API credentials")
    st.markdown(
        "**How to get your API credentials:**\n"
        "1. Login to [api.icicidirect.com](https://api.icicidirect.com)\n"
        "2. Create an app → you'll get an API Key and Secret\n"
        "3. Every morning, visit: `https://api.icicidirect.com/apiuser/login?api_key=YOUR_KEY`\n"
        "4. Login and copy the `apisession` value from the URL into Session Token below"
    )

    with st.form("settings_form"):
        api_k  = st.text_input("API Key",     value=saved.get("api_key", ""),    type="password",
                               placeholder="Paste your Breeze API Key")
        api_s  = st.text_input("API Secret",  value=saved.get("api_secret", ""), type="password",
                               placeholder="Paste your Breeze API Secret")
        ses_t  = st.text_input("Session Token (refresh daily)",
                               value=saved.get("session_token", ""),
                               placeholder="Paste today's session token from the login URL")

        st.divider()
        st.markdown("##### 📰 News sentiment (optional)")
        st.caption(
            "Get a free GNews API key at gnews.io (100 requests/day). "
            "News sentiment is scored with FinBERT (or VADER fallback) and adjusts "
            "model confidence — small boost when news agrees, larger penalty when it disagrees."
        )
        gnews_k = st.text_input("GNews API Key", value=saved.get("gnews_api_key", ""),
                                type="password",
                                placeholder="Paste your free GNews API key")

        st.divider()
        st.markdown("##### 📊 Groww API — Order Flow Imbalance (optional)")
        st.caption(
            "Groww provides tick-level order book depth that Breeze doesn't. "
            "This unlocks Order Flow Imbalance (OFI) — buy vs sell pressure — "
            "which improves 5-min and 15-min prediction accuracy by 4-6%. "
            "Get API access at groww.in/trade-api (TOTP-based)."
        )
        groww_k = st.text_input("Groww API Key", value=saved.get("groww_api_key", ""),
                                type="password", placeholder="Paste your Groww API key")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            groww_s = st.text_input("Groww API Secret", value=saved.get("groww_api_secret", ""),
                                    type="password", placeholder="Groww API secret")
        with gcol2:
            groww_totp = st.text_input("Groww TOTP (current code)", value="",
                                       placeholder="6-digit code from authenticator")

        st.divider()
        st.markdown("##### 💰 Capital & risk settings")
        cap_min = st.number_input("Capital per trade (₹)",    min_value=5000,  max_value=500000,
                                   value=int(saved.get("capital", cfg.CAPITAL_MIN)), step=5000)
        min_conf_pct = st.slider("Minimum confidence to trade (%)", 50, 100,
                                  int(saved.get("min_confidence", cfg.MIN_CONFIDENCE) * 100))

        st.divider()
        st.markdown("##### 📊 Trade parameters")
        c1s, c2s = st.columns(2)
        with c1s:
            target_pct  = st.number_input("Target (% premium gain)",   value=int(saved.get("target_pct", 80)), step=5)
            sl_pct      = st.number_input("Stop loss (% premium loss)", value=int(saved.get("sl_pct", 30)),    step=5)
        with c2s:
            max_vix     = st.number_input("Max India VIX to trade at",  value=int(saved.get("max_vix", 25)),   step=1)

        save_btn = st.form_submit_button("💾 Save settings", type="primary")
        if save_btn:
            save_settings({
                "api_key":          api_k,
                "api_secret":       api_s,
                "session_token":    ses_t,
                "gnews_api_key":    gnews_k,
                "groww_api_key":    groww_k,
                "groww_api_secret": groww_s,
                "capital":          cap_min,
                "min_confidence":   min_conf_pct / 100,
                "target_pct":       target_pct,
                "sl_pct":           sl_pct,
                "max_vix":          max_vix,
            })
            st.success("✅ Settings saved.")

            # Connect Groww if credentials or saved token available
            if groww_k or saved.get("groww_access_token", ""):
                try:
                    import groww_connector as gc
                    _gclient = gc.init_groww(groww_k, groww_s, groww_totp if groww_totp else None)
                    st.session_state["groww_obj"] = _gclient
                    st.success("✅ Groww API connected — OFI will appear in Live Monitor.")
                except Exception as _ge:
                    st.warning(f"Groww connection failed: {_ge}")

    st.divider()
    st.markdown("##### 🗄 Data management")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        data_files = list(Path("data").glob("*.csv")) if Path("data").exists() else []
        st.info(f"Cached data files: {len(data_files)}")
    with col_d2:
        if st.button("🗑 Clear data cache (force re-download)"):
            for f in data_files:
                f.unlink()
            st.success("Cache cleared.")
    with col_d3:
        trades_file = Path("trades/trades.csv")
        if trades_file.exists():
            with open(trades_file, "rb") as f:
                st.download_button("⬇️ Download trade log CSV", f, file_name="nifty_trades.csv",
                                   mime="text/csv", width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATA SOURCES STATUS
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🔌 Data sources — live availability")
    st.caption("Every data feed the system uses, its purpose, and whether it's working right now.")

    if st.button("🔍 Check all sources now", type="primary", key="ds_check"):
        st.session_state["ds_checked"] = True

    if st.session_state.get("ds_checked"):
        _ds_settings = load_settings()
        _ds_breeze   = st.session_state.get("breeze_obj")
        _ds_rows = []

        def _cache_age_str(pattern):
            """Age of the newest cache file matching the glob pattern (handles
            filename differences across versions, e.g. intraday_nifty.csv vs
            nifty_intraday_5min.csv)."""
            _matches = list(cfg.DATA_DIR.glob(pattern))
            if not _matches:
                return None
            _newest = max(_matches, key=lambda p: p.stat().st_mtime)
            _age_h = (datetime.now().timestamp() - _newest.stat().st_mtime) / 3600
            if _age_h < 1:    return f"{int(_age_h*60)} min ago ({_newest.name})"
            elif _age_h < 48: return f"{_age_h:.1f} hours ago ({_newest.name})"
            else:             return f"{_age_h/24:.0f} days ago ({_newest.name})"

        # ── 1. Breeze connection ───────────────────────────────────────────
        _breeze_live = False
        if _ds_breeze:
            try:
                _q = df_mod.fetch_live_quote_breeze(_ds_breeze)
                _breeze_live = bool(_q and (_q.get("ltp") or _q.get("open")))
            except Exception:
                _breeze_live = False
        _ds_rows.append({
            "Source": "ICICI Breeze API", "Used for": "All market data + live quotes",
            "Status": "🟢 Connected & live" if _breeze_live else
                      ("🟡 Session set, quote failed (token expired?)" if _ds_breeze else "🔴 Not connected"),
            "Detail": "Session token refreshes daily" if _breeze_live else "Paste today's session token in Settings",
        })

        # ── 2. Nifty daily OHLCV ───────────────────────────────────────────
        _age = _cache_age_str("*nifty*ohlcv*.csv")
        _ds_rows.append({
            "Source": "Nifty 50 daily OHLCV", "Used for": "Daily open/close models, trend features",
            "Status": "🟢 Cached" if _age else "🔴 No data",
            "Detail": f"Last fetched {_age}" if _age else "Generate a signal to fetch",
        })

        # ── 3. Intraday 5-min candles ──────────────────────────────────────
        _age = _cache_age_str("*intraday*.csv")
        _ds_rows.append({
            "Source": "Nifty 5-min candles", "Used for": "Live Monitor multi-horizon models",
            "Status": "🟢 Cached" if _age else "🔴 No data",
            "Detail": f"Last fetched {_age}. Note: Breeze can lag today's candles a few minutes." if _age
                      else "Train intraday models in Live Monitor to fetch",
        })

        # ── 4. India VIX ───────────────────────────────────────────────────
        _age = _cache_age_str("*vix*.csv")
        _ds_rows.append({
            "Source": "India VIX", "Used for": "Volatility regime, high-VIX trade block",
            "Status": "🟢 Cached" if _age else "🔴 No data",
            "Detail": f"Last fetched {_age}" if _age else
                      ("Fetch error: " + df_mod.LAST_FETCH_ERRORS.get("vix", "")[:90]
                       if getattr(df_mod, "LAST_FETCH_ERRORS", {}).get("vix")
                       else "Generate a signal in Today's Signal tab — VIX is fetched with it"),
        })

        # ── 5. GIFT Nifty ──────────────────────────────────────────────────
        _gift_live = None
        _gift_src = ""
        # Check WebSocket stream first
        import live_feeds as _lf_ds
        _gift_tick_ds = _lf_ds.get_latest("gift")
        if _gift_tick_ds and _gift_tick_ds.get("ltp", 0) > 0:
            _gift_live = _gift_tick_ds["ltp"]
            _g_age = _lf_ds.age_seconds("gift")
            _gift_src = f"WebSocket stream ({_gift_tick_ds.get('stock', '')}), {_g_age:.0f}s ago" if _g_age else "WebSocket stream"
        # Fallback to REST
        if not _gift_live and _ds_breeze:
            try:
                _gift_live = df_mod.fetch_gift_nifty_breeze(_ds_breeze)
                if _gift_live:
                    _gift_src = "REST poll"
            except Exception:
                pass
        _age = _cache_age_str("*gift*.csv")
        try:
            import live_engine as _le_ds
            _mkt_open_ds = _le_ds.is_market_open()
        except Exception:
            _mkt_open_ds = False
        _5min_exists = Path("data/nifty_5min_2yr.csv").exists()
        _is_streaming = _lf_ds.is_streaming()
        _ds_rows.append({
            "Source": "GIFT Nifty (live stream + history)", "Used for": "Pre-market direction signal",
            "Status": ("🟢 Live stream" if _gift_live and "stream" in _gift_src else
                       ("🟢 Live REST" if _gift_live else
                        ("🟡 Stream connected, no ticks yet" if _is_streaming else
                         ("🟡 History cached" if _age else
                          ("🟡 5-min cache fallback" if _5min_exists else "🔴 Unavailable"))))),
            "Detail": ((f"₹{_gift_live:,.0f} via {_gift_src}") if _gift_live else
                       ("WebSocket connected, waiting for market ticks" if _is_streaming
                        else (f"History cached {_age}" if _age else "Start Breeze session to enable live stream"))),
        })

        # ── 6. FII/DII flows ───────────────────────────────────────────────
        _age = _cache_age_str("*fii*.csv")
        _ds_rows.append({
            "Source": "FII/DII flows (NSDL)", "Used for": "Institutional flow features",
            "Status": "🟢 Cached" if _age else "🟡 Using zero-fallback",
            "Detail": f"Last fetched {_age}" if _age else "NSDL fetch happens with daily signal; zeros used if blocked",
        })

        # ── 7. Options chain / PCR ─────────────────────────────────────────
        _pcr_ok = False
        if _ds_breeze:
            try:
                import options_engine as _oe_ds
                _exp = _oe_ds.next_expiry()
                _spot_ds = _q.get("ltp") if _breeze_live and _q else None
                if _spot_ds:
                    _odf, _pcr_val = df_mod.fetch_options_chain_breeze(
                        _ds_breeze, _oe_ds.breeze_expiry_format(_exp), _spot_ds)
                    _pcr_ok = _odf is not None and len(_odf) > 0
            except Exception:
                _pcr_ok = False
        _ds_rows.append({
            "Source": "Options chain + PCR", "Used for": "Live premiums for trade params, Put-Call Ratio",
            "Status": "🟢 Live" if _pcr_ok else "🟡 Using VIX-based premium estimates",
            "Detail": "Live premiums from Breeze" if _pcr_ok else "Falls back to Black-Scholes estimate when unavailable",
        })

        # ── 8. News (GNews) ────────────────────────────────────────────────
        _gnews_key = _ds_settings.get("gnews_api_key", "")
        _news_ok, _news_n, _news_backend = False, 0, "none"
        if _gnews_key:
            try:
                import news_sentiment as _ns_ds
                _nres = _ns_ds.get_market_sentiment(_gnews_key)
                _news_n  = _nres.get("n_articles", 0)
                _news_backend = _nres.get("backend", "none")
                _news_ok = _news_n > 0
            except Exception:
                _news_ok = False
        _ds_rows.append({
            "Source": "GNews headlines", "Used for": "News sentiment confidence adjustment",
            "Status": "🟢 Working" if _news_ok else ("🟡 Key set, fetch failed" if _gnews_key else "🔴 No API key"),
            "Detail": f"{_news_n} articles fetched" if _news_ok else "Add free key from gnews.io in Settings",
        })

        # ── 9. Sentiment scoring backend ───────────────────────────────────
        _ds_rows.append({
            "Source": "Sentiment NLP model", "Used for": "Scoring fetched headlines",
            "Status": "🟢 " + _news_backend.upper() if _news_backend != "none" else "🔴 Not installed",
            "Detail": "FinBERT (best) or VADER (fallback)" if _news_backend != "none"
                      else "Run: pip install vaderSentiment",
        })

        # ── 10. Groww OFI ──────────────────────────────────────────────────
        _groww_c = st.session_state.get("groww_obj")
        if not _groww_c:
            try:
                import groww_connector as _gc_auto
                _gk = _ds_settings.get("groww_api_key", "")
                _gs = _ds_settings.get("groww_api_secret", "")
                if _gk or _ds_settings.get("groww_access_token", ""):
                    _groww_c = _gc_auto.init_groww(_gk, _gs)
                    st.session_state["groww_obj"] = _groww_c
            except Exception:
                pass
        _ofi_ok = False
        if _groww_c:
            try:
                import groww_connector as _gc_ds
                _ofi_res = _gc_ds.get_live_ofi(_groww_c, "NIFTY")
                _ofi_ok = _ofi_res.get("available", False)
            except Exception:
                pass
        _ds_rows.append({
            "Source": "Order Flow / OFI", "Used for": "Order Flow Imbalance — buy/sell pressure",
            "Status": ("🟢 Live OFI" if _ofi_ok else
                       ("🟡 Connected, no depth" if _groww_c else "⚪ Not connected")),
            "Detail": ("OFI feeding Live Monitor" if _ofi_ok else
                       ("Groww connected but no depth data" if _groww_c
                        else "Connect Groww for live OFI")),
        })

        # ── 11. Intraday models ────────────────────────────────────────────
        try:
            import intraday_predictor as _ip_ds
            _models_ok = _ip_ds.intraday_models_exist(str(cfg.MODEL_DIR))
            _meta_ds = _ip_ds.load_intraday_metadata(str(cfg.MODEL_DIR))
            _trained_at = _meta_ds.get("trained_at", "")[:16] if _meta_ds else ""
        except Exception:
            _models_ok, _trained_at = False, ""
        _ds_rows.append({
            "Source": "Intraday models (7 horizons)", "Used for": "Live Monitor predictions",
            "Status": "🟢 Trained" if _models_ok else "🔴 Not trained",
            "Detail": f"Last trained {_trained_at}" if _models_ok else "Train from Live Monitor tab",
        })

        # ── 12. Daily models ───────────────────────────────────────────────
        _daily_ok = (cfg.MODEL_DIR / "xgb_close_clf.pkl").exists() or \
                    (cfg.MODEL_DIR / "xgb_close.pkl").exists() or \
                    any(cfg.MODEL_DIR.glob("*close*.pkl"))
        _ds_rows.append({
            "Source": "Daily open/close models", "Used for": "Today's Signal tab",
            "Status": "🟢 Trained" if _daily_ok else "🔴 Not trained",
            "Detail": "Retrain monthly from Model Health" if _daily_ok else "Train from Model Health tab",
        })

        _ds_df = pd.DataFrame(_ds_rows)
        st.dataframe(_ds_df, width="stretch", hide_index=True, height=480)

        _n_green = sum(1 for r in _ds_rows if "🟢" in r["Status"])
        _n_amber = sum(1 for r in _ds_rows if "🟡" in r["Status"])
        _n_red   = sum(1 for r in _ds_rows if "🔴" in r["Status"])
        _sc1, _sc2, _sc3 = st.columns(3)
        _sc1.metric("🟢 Working", _n_green)
        _sc2.metric("🟡 Degraded / fallback", _n_amber)
        _sc3.metric("🔴 Unavailable", _n_red)
        st.caption("⚪ = optional source. Degraded sources have automatic fallbacks — "
                   "the system keeps working but with reduced signal quality.")
    else:
        st.info("Click **Check all sources now** to test every data feed. "
                "Checks live Breeze quotes, GIFT, news, options chain, Groww, and model files.")
