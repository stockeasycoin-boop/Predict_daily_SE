#!/usr/bin/env bash
# start.sh — install deps, generate model artifacts, launch the app.
# Args after the script are forwarded to bootstrap.py, e.g.:
#   ./start.sh --cache-only      # skip Breeze fetch (use existing data cache)
#   ./start.sh --force           # rebuild models even if present
set -e

echo "[1/3] Installing dependencies…"
python -m pip install -r requirements.txt

echo "[2/3] Generating model artifacts (idempotent — skips if already built)…"
python bootstrap.py "$@"

echo "[3/3] Launching app…"
streamlit run app.py
