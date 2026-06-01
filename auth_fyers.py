"""
auth_fyers.py — One-shot Fyers OAuth helper.

Run once each morning (tokens expire daily):
    python auth_fyers.py

Flow:
  1. Reads CLIENT_ID + SECRET_ID from env / Streamlit secrets / settings.json.
  2. Prints a login URL — open it in your browser, log in, approve.
  3. Fyers redirects to REDIRECT_URI with `?auth_code=...` in the query string.
  4. Paste the full redirected URL (or just the auth_code) back here.
  5. Script exchanges code for an access_token and writes it to settings.json
     under "fyers_access_token". Restart the app to pick it up.

For Streamlit Cloud:
  - Run this locally, get the token, then paste FYERS_ACCESS_TOKEN into
    Streamlit Cloud Secrets each morning.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

try:
    from fyers_apiv3 import fyersModel
except ImportError:
    print("ERROR: fyers-apiv3 not installed. Run: pip install fyers-apiv3")
    sys.exit(1)

from fyers_data import get_fyers_credentials

DEFAULT_REDIRECT = "http://127.0.0.1:8080/"
SETTINGS_PATH = Path(__file__).parent / "settings.json"


def _redirect_uri() -> str:
    return (
        os.getenv("FYERS_REDIRECT_URI")
        or _from_streamlit_secrets("FYERS_REDIRECT_URI")
        or _from_json("fyers_redirect_uri")
        or DEFAULT_REDIRECT
    )


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


def _persist_token(token: str) -> None:
    """Write access_token to settings.json without clobbering other keys."""
    data = {}
    if SETTINGS_PATH.exists():
        try:
            data = json.load(open(SETTINGS_PATH))
        except Exception:
            data = {}
    data["fyers_access_token"] = token
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))
    print(f"\nSaved access_token to {SETTINGS_PATH}")


def _extract_auth_code(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    if "auth_code=" in raw:
        q = parse_qs(urlparse(raw).query)
        return (q.get("auth_code") or [""])[0]
    return raw


def main() -> int:
    client_id, secret_id, _ = get_fyers_credentials()
    if not client_id or not secret_id:
        print("ERROR: FYERS_CLIENT_ID / FYERS_SECRET_ID not configured.")
        print("Set them in settings.json (fyers_client_id, fyers_secret_id) or as env vars.")
        return 1

    redirect = _redirect_uri()
    print(f"client_id    = {client_id}")
    print(f"redirect_uri = {redirect}")
    print()

    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_id,
        redirect_uri=redirect,
        response_type="code",
        grant_type="authorization_code",
        state="local-cli",
    )

    auth_url = session.generate_authcode()
    print("STEP 1 — open this URL in your browser and log in:")
    print(f"\n  {auth_url}\n")
    print("STEP 2 — after login, you'll be redirected to a URL like:")
    print(f"  {redirect}?s=ok&code=200&auth_code=AAA...&state=local-cli")
    print()
    raw = input("Paste the redirected URL (or just the auth_code): ").strip()

    code = _extract_auth_code(raw)
    if not code:
        print("ERROR: could not parse auth_code from input.")
        return 2

    session.set_token(code)
    resp = session.generate_token()
    token = (resp or {}).get("access_token")
    if not token:
        print(f"ERROR: token exchange failed: {resp}")
        return 3

    print("\nToken generated successfully (first 16 chars):")
    print(f"  {token[:16]}…")
    _persist_token(token)
    print("\nDone. The app will pick up the new token on next start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
