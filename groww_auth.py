"""
groww_auth.py — Groww Trade API authentication helper.

Two auth flows supported:
  1. TOTP flow:     API key + TOTP code (6-digit from authenticator app)
  2. Approval flow: API key + secret (from Groww developer portal)

Usage:
  python groww_auth.py

The script will:
  1. Read your Groww API key and secret from settings.json
  2. Generate TOTP (if secret is a TOTP seed) or use approval flow
  3. Get a fresh access token from Groww
  4. Save the token back to settings.json
  5. Test the connection with a basic API call
"""

import json
import sys
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent / "settings.json"


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {}


def save_settings(settings: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def get_groww_token(api_key: str, api_secret: str = "", totp_code: str = "") -> str:
    """
    Get a fresh Groww access token.

    Tries in order:
      1. TOTP flow (if totp_code provided or secret is a TOTP seed)
      2. Approval/secret flow (if api_secret provided)
      3. Direct token (if api_key is already an access token)
    """
    from growwapi import GrowwAPI

    # Flow 1: Explicit TOTP code provided
    if totp_code:
        print(f"[Groww Auth] Trying TOTP flow with code: {totp_code[:2]}****")
        try:
            token = GrowwAPI.get_access_token(api_key, totp=totp_code)
            print("[Groww Auth] TOTP flow succeeded!")
            return token
        except Exception as e:
            print(f"[Groww Auth] TOTP flow failed: {e}")

    # Flow 2: Auto-generate TOTP from secret (if it looks like a TOTP seed)
    if api_secret and len(api_secret) in (16, 32, 64):
        try:
            import pyotp
            totp_auto = pyotp.TOTP(api_secret).now()
            print(f"[Groww Auth] Auto-generated TOTP: {totp_auto}")
            token = GrowwAPI.get_access_token(api_key, totp=totp_auto)
            print("[Groww Auth] Auto-TOTP flow succeeded!")
            return token
        except ImportError:
            print("[Groww Auth] pyotp not installed. Run: pip install pyotp")
        except Exception as e:
            print(f"[Groww Auth] Auto-TOTP flow failed: {e}")

    # Flow 3: Approval flow with secret
    if api_secret:
        print("[Groww Auth] Trying approval/secret flow...")
        try:
            token = GrowwAPI.get_access_token(api_key, secret=api_secret)
            print("[Groww Auth] Secret flow succeeded!")
            return token
        except Exception as e:
            print(f"[Groww Auth] Secret flow failed: {e}")

    # Flow 4: Maybe api_key is already an access token
    print("[Groww Auth] Trying api_key as direct access token...")
    try:
        client = GrowwAPI(api_key)
        client.get_ltp(exchange_trading_symbols=("NIFTY 50",), segment="CASH")
        print("[Groww Auth] Direct token works!")
        return api_key
    except Exception as e:
        print(f"[Groww Auth] Direct token failed: {e}")

    return ""


def test_connection(token: str) -> bool:
    """Test the access token with basic API calls."""
    from growwapi import GrowwAPI

    client = GrowwAPI(token)
    print("\n--- Testing Groww Connection ---")

    # Test 1: LTP
    try:
        r = client.get_ltp(exchange_trading_symbols=("NIFTY 50",), segment="CASH")
        print(f"  get_ltp(NIFTY 50): {r}")
    except Exception as e:
        print(f"  get_ltp: FAILED - {e}")
        return False

    # Test 2: Quote
    try:
        r = client.get_quote(trading_symbol="NIFTY 50", exchange="NSE", segment="CASH")
        keys = list(r.keys()) if isinstance(r, dict) else str(type(r))
        print(f"  get_quote(NIFTY 50): keys={keys}")
    except Exception as e:
        print(f"  get_quote: FAILED - {e}")

    # Test 3: Option chain
    try:
        from datetime import date, timedelta
        today = date.today()
        # Find next Thursday (weekly expiry)
        days_ahead = (3 - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        exp = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        r = client.get_option_chain(exchange="NSE", underlying="NIFTY", expiry_date=exp)
        n = len(r) if isinstance(r, (list, dict)) else 0
        print(f"  get_option_chain(NIFTY, {exp}): {n} entries")
    except Exception as e:
        print(f"  get_option_chain: FAILED - {e}")

    print("--- Connection test complete ---\n")
    return True


def main():
    print("=" * 60)
    print("  Groww Trade API — Authentication Helper")
    print("=" * 60)

    settings = load_settings()
    api_key = settings.get("groww_api_key", "")
    api_secret = settings.get("groww_api_secret", "")

    if not api_key:
        print("\nNo Groww API key found in settings.json.")
        print("Get your API key from: https://groww.in/trade-api")
        api_key = input("Enter Groww API key: ").strip()
        if not api_key:
            print("Aborted.")
            return

    print(f"\nAPI Key: {api_key[:20]}...{api_key[-10:]} ({len(api_key)} chars)")
    print(f"Secret:  {'set' if api_secret else 'not set'} ({len(api_secret)} chars)")

    # Ask for TOTP if needed
    totp_code = ""
    if api_secret:
        choice = input("\nAuth method — [1] Auto-TOTP from secret  [2] Enter TOTP manually  [3] Approval flow: ").strip()
        if choice == "2":
            totp_code = input("Enter 6-digit TOTP from your authenticator app: ").strip()
    else:
        totp_code = input("\nEnter 6-digit TOTP (or press Enter to try direct token): ").strip()

    # Get token
    token = get_groww_token(api_key, api_secret, totp_code)

    if not token:
        print("\nFailed to get access token. Check your credentials.")
        print("Tips:")
        print("  - Generate a fresh API key from https://groww.in/trade-api")
        print("  - Make sure Market Data permissions are enabled")
        print("  - If using TOTP, ensure your authenticator app is synced")
        return

    print(f"\nAccess token obtained: {token[:20]}...{token[-10:]} ({len(token)} chars)")

    # Save token
    settings["groww_access_token"] = token
    save_settings(settings)
    print("Token saved to settings.json (key: groww_access_token)")

    # Test
    ok = test_connection(token)
    if ok:
        print("Groww is ready! The app will use this token for live OFI data.")
    else:
        print("Token obtained but API calls failed.")
        print("Your Groww plan may not include market data access.")


if __name__ == "__main__":
    main()
