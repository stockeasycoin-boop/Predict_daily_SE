"""
auth_breeze.py — One-shot ICICI Breeze session-token helper.

Run once each morning (Breeze session tokens expire daily, same as Fyers):
    python auth_breeze.py

Flow:
  1. Reads api_key (+ optional api_secret) from settings.json / env / Streamlit secrets.
  2. Prints the Breeze login URL — open it in your browser, log in.
  3. After login Breeze redirects to a URL containing `apisession=XXXXXX`.
  4. Paste either the full URL or just the token back here.
  5. Script saves it to settings.json under "session_token".
  6. Restart Streamlit (or rerun the app) to pick up the new token.

For Streamlit Cloud:
  - Run this locally, get the token, then paste BREEZE_SESSION_TOKEN into
    Streamlit Cloud Secrets each morning.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

SETTINGS_PATH = Path(__file__).parent / "settings.json"
LOGIN_URL_TEMPLATE = "https://api.icicidirect.com/apiuser/login?api_key={key}"


# ── Credential discovery (env → Streamlit secrets → settings.json) ───────────
def _from_streamlit_secrets(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        return ""


def _from_json(key: str) -> str:
    try:
        return json.load(open(SETTINGS_PATH)).get(key, "")
    except Exception:
        return ""


def _resolve(*names) -> str:
    for n in names:
        v = os.getenv(n) or _from_streamlit_secrets(n) or _from_json(n.lower())
        if v:
            return v
    return ""


def get_breeze_credentials() -> tuple[str, str, str]:
    api_key       = _resolve("BREEZE_API_KEY",    "api_key")
    api_secret    = _resolve("BREEZE_API_SECRET", "api_secret")
    session_token = _resolve("BREEZE_SESSION_TOKEN", "session_token")
    return api_key, api_secret, session_token


# ── Persistence ──────────────────────────────────────────────────────────────
def _persist_token(token: str) -> None:
    """Write session_token to settings.json without clobbering other keys."""
    data = {}
    if SETTINGS_PATH.exists():
        try:
            data = json.load(open(SETTINGS_PATH))
        except Exception:
            data = {}
    data["session_token"] = token
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))
    print(f"\nSaved session_token to {SETTINGS_PATH}")


def _extract_token(raw: str) -> str:
    """Accept either a pasted URL (`?apisession=XXX`) or the raw token."""
    raw = raw.strip()
    if not raw:
        return ""
    if "apisession=" in raw or "API_Session=" in raw or "api_session=" in raw:
        q = parse_qs(urlparse(raw).query)
        for key in ("apisession", "API_Session", "api_session"):
            if q.get(key):
                return q[key][0]
    return raw


# ── Optional verification — confirms token actually works ────────────────────
def _verify(api_key: str, api_secret: str, token: str) -> bool:
    try:
        from breeze_connect import BreezeConnect
    except ImportError:
        print("[verify] breeze-connect not installed; skipping check.")
        return True
    try:
        bz = BreezeConnect(api_key=api_key)
        bz.generate_session(api_secret=api_secret, session_token=token)
        # Light probe — get_customer_details requires an active session
        try:
            resp = bz.get_customer_details(api_session=token)
            ok = (isinstance(resp, dict) and resp.get("Status") == 200)
        except Exception:
            # Some setups don't expose this; treat session creation as success
            ok = True
        print("Token verification:", "OK ✅" if ok else "FAILED ❌")
        return ok
    except Exception as e:
        print(f"[verify] failed: {e}")
        return False


def main() -> int:
    api_key, api_secret, _ = get_breeze_credentials()
    if not api_key:
        print("ERROR: BREEZE_API_KEY / api_key not configured.")
        print("Set it in settings.json (api_key) or as the BREEZE_API_KEY env var.")
        return 1
    if not api_secret:
        print("WARNING: BREEZE_API_SECRET / api_secret not configured.")
        print("The login URL will still work, but token verification will be skipped.")

    login_url = LOGIN_URL_TEMPLATE.format(key=quote(api_key, safe=""))
    print(f"api_key (URL-encoded) preview: {api_key[:8]}…")
    print()
    print("STEP 1 — open this URL in your browser and log in:")
    print(f"\n  {login_url}\n")
    print("STEP 2 — after login, ICICI will redirect you to a URL like:")
    print("  https://api.icicidirect.com/...?apisession=XXXXXXXX&...")
    print("  COPY THE FULL URL FROM THE BROWSER ADDRESS BAR (or just the token).")
    print()
    raw = input("Paste the redirected URL (or just the apisession token): ").strip()

    token = _extract_token(raw)
    if not token:
        print("ERROR: could not parse session token from input.")
        return 2

    print(f"\nExtracted session_token (first 8 chars): {token[:8]}…")

    if api_secret:
        _verify(api_key, api_secret, token)

    _persist_token(token)
    print("\nDone. The app will pick up the new session token on next start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
