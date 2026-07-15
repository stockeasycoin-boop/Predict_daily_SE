@echo off
REM start.bat - install deps, generate model artifacts, launch the app (Windows).
REM Forward args to bootstrap.py, e.g.:  start.bat --cache-only   /   start.bat --force

echo [1/3] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [2/3] Generating model artifacts (idempotent - skips if already built)...
python bootstrap.py %*
if errorlevel 1 exit /b 1

echo [3/3] Launching app...
streamlit run app.py
