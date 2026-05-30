"""
app.py — Nifty AI Options Trader Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime
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
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            if username == AUTH_USERNAME and _hash_pw(password) == AUTH_PASSWORD_SHA256:
                st.session_state["authenticated"] = True
                st.session_state["auth_user"] = username
                st.session_state.pop("_use_password_fallback", None)
                st.rerun()
            else:
                st.error("Invalid username or password.")
        if st.session_state.get("_use_password_fallback"):
            if st.button("← Back to Google sign-in", use_container_width=True):
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
            if st.button("🔐  Sign in with Google", use_container_width=True, type="primary"):
                st.login()
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("🔑  Sign in with password instead", use_container_width=True):
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
            if st.button("Sign out and try another account", use_container_width=True):
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
    if st.button("🔒 Log out", use_container_width=True,
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Today's Signal",
    "📈  Accuracy Tracker",
    "🧠  Model Health",
    "⚙️  Settings",
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
        run_btn = st.button("🔄 Generate Today's Signal", type="primary", use_container_width=True)
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
                            st.toast("✅ Breeze API connected", icon="✅")
                        except Exception as e:
                            log_step(f"Breeze connection failed: {e}. Using cached/Stooq data.", "warning")
                            st.warning(f"Breeze connection failed: {e}. Using cached/Stooq data.")
                    else:
                        log_step("Step 1/6 — no Breeze credentials, using cached/Stooq data")

                    log_step("Step 2/6 — loading Nifty / VIX / global data…")
                    nifty_df  = df_mod.load_nifty_data(breeze, force_refresh=run_btn)
                    vix_df    = df_mod.load_vix_data(breeze)
                    global_df = df_mod.load_global_data()

                    if nifty_df is None or len(nifty_df) < 50:
                        log_step("Step 2/6 — insufficient Nifty data", "error")
                        st.error("Not enough market data. Check your internet connection.")
                        _tab1_ready = False
                    else:
                        log_step(f"Step 2/6 — Nifty data loaded ({len(nifty_df)} rows)")
                        log_step("Step 3/6 — loading FII/DII, GIFT, PCR, intraday, correlated data…")
                        fii_df     = df_mod.load_fii_dii_data()
                        gift_df    = df_mod.load_gift_data(breeze)
                        pcr_df     = df_mod.load_pcr_data()
                        intra_df   = df_mod.load_intraday_data(breeze)
                        corr_dict  = df_mod.load_correlated_data(breeze)
                        log_step("Step 4/6 — building features…")
                        feat_df = fe.build_features(
                            nifty_df, vix_df, global_df, fii_df, gift_df, pcr_df,
                            intraday_df=intra_df, corr_dict=corr_dict
                        )
                        log_step(f"Step 4/6 — features built ({feat_df.shape[0]}x{feat_df.shape[1]})")
                        log_step("Step 5/6 — running model inference…")
                        preds = mt.predict_today(feat_df, str(cfg.MODEL_DIR))
                        direction  = preds.get("close_direction", preds.get("direction", 0))
                        confidence = preds.get("close_confidence", preds.get("confidence", 0.5))
                        atr_pct    = preds.get("atr_pct", 0.8)
                        vix        = preds.get("india_vix", 16.0)
                        spot = None
                        if breeze:
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
                        # ── News sentiment FIRST — so it can gate the trade ──
                        log_step("Step 6/6 — fetching news sentiment…")
                        news = {"n_articles": 0}
                        news_conf = confidence
                        news_reason = ""
                        try:
                            import news_sentiment as ns
                            gnews_key = settings.get("gnews_api_key",
                                                    getattr(cfg, "GNEWS_API_KEY", ""))
                            news = ns.get_market_sentiment(gnews_key, days=2,
                                                           force_refresh=_force_news)
                            if news.get("n_articles", 0) >= 3:
                                news_conf, news_reason = ns.adjust_confidence(
                                    direction,
                                    confidence,
                                    news,
                                    max_boost   = getattr(cfg, "NEWS_MAX_BOOST",   0.08),
                                    max_penalty = getattr(cfg, "NEWS_MAX_PENALTY", 0.15),
                                )
                                log_step(f"News {news.get('label','?')} ({news.get('score',0):+.2f}) "
                                         f"→ confidence {confidence:.0%} → {news_conf:.0%}")
                        except Exception as ne:
                            log_step(f"News sentiment skipped: {ne}", "warning")
                            news = {"error": str(ne), "n_articles": 0}

                        # Generate suggestion using the NEWS-ADJUSTED confidence,
                        # so strong adverse news can pull it below threshold
                        # (BUY → NO_TRADE) and agreement can keep a trade alive.
                        log_step("Step 6/6 — generating trade suggestion…")
                        suggestion = oe.generate_suggestion(
                            direction, news_conf, spot, atr_pct, vix, capital, opts_df
                        )
                        suggestion["news"] = news
                        if news.get("n_articles", 0) >= 3:
                            suggestion["confidence_original"] = confidence
                            suggestion["news_adjustment"]     = news_reason
                            if news_conf < confidence and suggestion.get("signal") == "NO_TRADE" \
                                    and confidence >= getattr(cfg, "MIN_CONFIDENCE", 0.70):
                                suggestion["reason"] = (
                                    f"Model was {confidence:.0%} confident, but {news.get('label','adverse')} "
                                    f"news ({news.get('score',0):+.2f}) cut it to {news_conf:.0%} — below "
                                    f"threshold. Standing down. ({news_reason})"
                                )
                        log_step(f"Step 6/6 — suggestion: {suggestion.get('signal', '?')} "
                                 f"(confidence={suggestion.get('confidence', 0):.2%})")

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
        cache_age    = news_data.get("cache_age_hours")
        cache_label  = (f"🕒 cached {cache_age:.1f}h ago" if (from_cache and cache_age is not None)
                        else "🆕 freshly fetched")

        nh_col, nb_col = st.columns([4, 1])
        nh_col.caption(f"News status: {cache_label} · auto-refreshes every 4h")
        if nb_col.button("🔄 Fetch fresh news", use_container_width=True,
                         help="Pull the latest headlines now (uses GNews quota) and regenerate the signal"):
            st.session_state["_force_news_refresh"] = True
            st.rerun()

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
            top = news_data.get("top_headlines", [])
            if top:
                st.markdown("**Top headlines driving sentiment:**")
                for h in top:
                    icon = "🟢" if h["sentiment"] > 0.15 else "🔴" if h["sentiment"] < -0.15 else "⚪"
                    st.markdown(
                        f"{icon} **{h['sentiment']:+.2f}** — [{h['title']}]({h['url']}) "
                        f"<span style='color:#888;font-size:0.85em'>· {h['source']}</span>",
                        unsafe_allow_html=True,
                    )
    elif news_data.get("error"):
        st.caption(f"📰 News sentiment unavailable: {news_data['error']}")

    # ── Trade details (only for actual trade signals) ──────────────────────

    # ── Open / close prediction cards + reasoning ──────────────────────────
    preds     = st.session_state.get("preds", {})
    reasoning = st.session_state.get("reasoning", {})

    if preds:
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
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
# TAB 2 — ACCURACY TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
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

            # Capital staged across pending trades
            cap_used = pd.to_numeric(pending_df.get("capital_used"), errors="coerce").fillna(0)
            total_cap = int(cap_used.sum())

            # Projected max-loss across pending trades
            max_loss = pd.to_numeric(pending_df.get("max_loss_inr"), errors="coerce").fillna(0)
            total_proj_loss = int(max_loss.sum())

            # Projected max-profit across pending trades
            max_profit = pd.to_numeric(pending_df.get("max_profit_inr"), errors="coerce").fillna(0)
            total_proj_profit = int(max_profit.sum())

            # Average projected R:R
            rr_proj = pd.to_numeric(pending_df.get("risk_reward"), errors="coerce").dropna()
            avg_rr = float(rr_proj.mean()) if len(rr_proj) > 0 else 0

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
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

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
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

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
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

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
        st.dataframe(disp.sort_values("date", ascending=False), use_container_width=True, height=320)
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
with tab3:
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
            c1.metric("CV open / close",
                     f"{cv_open*100:.1f}% / {cv_close*100:.1f}%")
        else:
            c1.metric("CV accuracy", f"{cv_open*100:.1f}%")
        c2.metric("Training rows", f"{meta.get('n_samples', 0):,}")
        c3.metric("Features",      meta.get("n_features", 0))
        c4.metric("Last trained",  str(meta.get("trained_at", "—"))[:10])
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
        1. **Downloads ~2 years of Nifty OHLCV** from Breeze API (or Stooq as backup)
        2. **Downloads India VIX** and global cues (Dow, S&P 500, Dollar Index)
        3. **Computes 35+ technical indicators** — RSI, MACD, Bollinger Bands, ATR, etc.
        4. **Trains XGBoost** using 5-fold walk-forward cross-validation (no data leakage)
        5. **Saves the model** to the `models/` folder
        6. The dashboard auto-refreshes once training is complete
        """)

    col_train, col_info = st.columns([1, 2])
    with col_train:
        train_btn = st.button("🚀 Train model now", type="primary", use_container_width=True)
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
        log_step("🚂 Train model — clicked")
        progress_bar = st.progress(0, text="Starting…")
        status_box   = st.empty()

        try:
            status_box.info("Step 1/5 — Reading credentials…")
            progress_bar.progress(10, text="Reading credentials…")
            log_step("Step 1/5 — reading credentials…")

            saved_s   = load_settings()
            api_key3  = saved_s.get("api_key", getattr(cfg, "BREEZE_API_KEY", ""))
            api_sec3  = saved_s.get("api_secret", getattr(cfg, "BREEZE_API_SECRET", ""))
            ses_tok3  = saved_s.get("session_token", "")

            breeze3 = None
            if ses_tok3 and api_key3 and api_key3 not in ("", "YOUR_API_KEY_HERE"):
                try:
                    log_step("Step 1/5 — connecting to Breeze API…")
                    breeze3 = df_mod.init_breeze(api_key3, api_sec3, ses_tok3)
                    log_step("Step 1/5 — Breeze API connected ✅")
                    status_box.info("Step 1/5 — Breeze API connected ✅")
                except Exception as be:
                    log_step(f"Breeze unavailable ({be}), using Stooq backup", "warning")
                    status_box.warning(f"Breeze unavailable ({be}), using Stooq backup…")
            else:
                log_step("Step 1/5 — no Breeze token, using Stooq for data")
                status_box.info("Step 1/5 — No Breeze session token. Using Stooq for data…")

            progress_bar.progress(20, text="Downloading Nifty data…")
            status_box.info("Step 2/5 — Downloading Nifty OHLCV data…")
            log_step("Step 2/5 — downloading Nifty OHLCV data…")
            nifty3 = df_mod.load_nifty_data(breeze3, force_refresh=True)  # uses TRAINING_DAYS from settings

            if nifty3 is None or len(nifty3) < 100:
                log_step("Step 2/5 — could not load Nifty data", "error")
                progress_bar.empty()
                status_box.error(
                    "❌ Could not load Nifty data. "
                    "Check your internet connection. "
                    "If Breeze is not configured, Stooq is used as fallback — "
                    "make sure you have an internet connection."
                )
            else:
                log_step(f"Step 2/5 — Nifty data loaded ({len(nifty3)} rows)")
                progress_bar.progress(40, text="Downloading VIX & global data…")
                status_box.info("Step 3/5 — Downloading India VIX and global cues…")
                log_step("Step 3/5 — downloading VIX, global, FII/DII, GIFT, PCR…")
                vix3    = df_mod.load_vix_data(breeze3, force_refresh=True)
                global3 = df_mod.load_global_data(force_refresh=True)
                fii3    = df_mod.load_fii_dii_data(force_refresh=True)
                gift3   = df_mod.load_gift_data(force_refresh=True)
                pcr3    = df_mod.load_pcr_data(force_refresh=True)

                progress_bar.progress(60, text="Building features...")
                status_box.info("Step 4/5 — Computing 50+ technical indicators...")
                log_step("Step 4/5 — building features…")
                feat3 = fe.build_features(nifty3, vix3, global3, fii3, gift3, pcr3)
                log_step(f"Step 4/5 — features built ({feat3.shape[0]}x{feat3.shape[1]})")

                progress_bar.progress(75, text="Training XGBoost model…")
                status_box.info(
                    f"Step 5/5 — Training XGBoost on {len(feat3)} days of data… "
                    f"(5-fold walk-forward CV)"
                )
                log_step(f"Step 5/5 — training models on {len(feat3)} rows (Optuna + walk-forward CV)…")
                _, _, _, _, results = mt.train_model(feat3, model_dir_str, verbose=True, use_optuna=True)
                log_step("Step 5/5 — training complete")

                progress_bar.progress(100, text="Done!")
                # results is the metadata dict returned by train_model
                _td   = results.get("n_samples",  0)
                _yrs  = round(_td / 252, 1)
                _cv_o = results.get("cv_open",    results.get("cv_accuracy", 0))
                _cv_c = results.get("cv_close",   results.get("cv_accuracy", 0))
                _ens  = "XGB + LGB" if results.get("lgb_available") else "XGB only"
                _opt  = results.get("optuna_trials", 0)
                status_box.success(
                    f"✅ Trained on **{_td:,} days ({_yrs} yrs)** — "
                    f"Open CV: **{_cv_o*100:.1f}%** | Close CV: **{_cv_c*100:.1f}%** | "
                    f"Ensemble: {_ens} | Optuna: {_opt} trials"
                )

                _best = max(_cv_o, _cv_c)
                if _best < 0.52:
                    st.warning(
                        "⚠️ Accuracy below 52%. Markets may be in a choppy regime. "
                        "Paper-trade first and monitor the 7-day rolling accuracy."
                    )
                elif _best >= 0.60:
                    st.success("🎯 Accuracy above 60% — model is ready for paper trading!")

                st.cache_resource.clear()
                st.rerun()

        except Exception as e:
            log.exception("❌ Training failed")
            progress_bar.empty()
            status_box.error(f"❌ Training failed: {e}")
            st.exception(e)   # shows full traceback to help debug

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
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        pass   # Feature importance chart is optional; don't crash if unavailable


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
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
                "api_key":        api_k,
                "api_secret":     api_s,
                "session_token":  ses_t,
                "gnews_api_key":  gnews_k,
                "capital":        cap_min,
                "min_confidence": min_conf_pct / 100,
                "target_pct":     target_pct,
                "sl_pct":         sl_pct,
                "max_vix":        max_vix,
            })
            st.success("✅ Settings saved.")

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
                                   mime="text/csv", use_container_width=True)
