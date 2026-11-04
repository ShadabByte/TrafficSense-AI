@echo off
setlocal
title TrafficSense AI

cd /d "%~dp0"

echo ========================================
echo           TRAFFICSENSE AI
echo ========================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Python environment missing.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERROR: app.py missing.
    pause
    exit /b 1
)

"venv\Scripts\python.exe" -c "import streamlit" >nul 2>&1

if errorlevel 1 (
    echo Streamlit is missing from the virtual environment.
    echo Install it using: pip install streamlit
    pause
    exit /b 1
)

powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue; if ($c) { exit 1 } else { exit 0 }"

if errorlevel 1 (
    echo TrafficSense AI may already be running on port 8501.
    start "" "http://localhost:8501"
    exit /b 0
)

echo Starting TrafficSense AI...
start "" "http://localhost:8501"

"venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501

echo.
echo TrafficSense AI has stopped.
pause
