@echo off
title ACMGS Desktop Launcher
echo ========================================================
echo Starting ACMGS Cyber-Physical Platform
echo FastAPI Backend (Port 8000) + Streamlit Cockpit (Port 8501)
echo ========================================================

start "ACMGS FastAPI Server" cmd /k "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul
start "ACMGS Streamlit Cockpit" cmd /k "streamlit run src/dashboard/app.py --server.port 8501"

echo Both services launched successfully.
pause
