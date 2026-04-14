@echo off
setlocal
title Telecom Customer Churn - System Launcher

:: Clean screen and display banner
cls
echo ============================================================
echo   📊 TELECOM CUSTOMER CHURN: AGENTIC MLOPS SYSTEM
echo ============================================================
echo.
echo [SYSTEM] Initializing Backend Services and Dashboard...
echo.

:: Step 1: Check/Sync Dependencies
echo [1/5] Verifying dependencies with UV...
uv sync --quiet
if "%ERRORLEVEL%" NEQ "0" (
    echo.
    echo 🚨 Error: Failed to sync dependencies. Verify 'uv' is installed.
    pause
    exit /b %ERRORLEVEL%
)
echo      Done.
echo.

:: Step 2: Launch MLflow Tracking Server
echo [2/5] Launching MLflow Tracking Server...
echo      URL: http://127.0.0.1:5000
echo      Storage: local ./mlruns
start "CHURN-MLFLOW" /min cmd /k "title CHURN-MLFLOW && uv run mlflow server --host 127.0.0.1 --port 5000"

:: Step 3: Launch Embedding Service (Port 8001)
echo [3/5] Launching NLP Embedding Service...
echo      URL: http://127.0.0.1:8001
start "CHURN-EMBED" /min cmd /k "title CHURN-EMBED && uv run uvicorn src.api.embedding_service.main:app --host 127.0.0.1 --port 8001"

:: Step 4: Launch Prediction API (Port 8000)
echo [4/5] Launching Late-Fusion Prediction API...
echo      URL: http://127.0.0.1:8000
start "CHURN-PRED-API" /min cmd /k "title CHURN-PRED-API && uv run uvicorn src.api.prediction_service.main:app --host 127.0.0.1 --port 8000"

:: Wait for warmup
echo.
echo [WAIT] Stalling for service initialization (10s)...
echo        (SentenceTransformer takes ~5-8s to load)
timeout /t 10 >nul

:: Step 5: Launch Gradio UI in the foreground (Port 7860)
echo.
echo [5/5] Launching Interactive Dashboard...
echo      URL: http://127.0.0.1:7860
echo.
echo ------------------------------------------------------------
echo 💡 TIP: MLflow, Embedding, and Prediction services are 
echo    running in the background (minimized).
echo.
echo    To stop EVERYTHING:
echo    1. Close the background command windows.
echo    2. Press Ctrl+C in this window.
echo ------------------------------------------------------------
echo.

:: Launch the browser automatically
echo [INFO] Opening Gradio Dashboard in your browser...
start http://localhost:7860

:: Run Gradio UI in foreground so user sees primary logs
uv run python src/ui/app.py

:: If the user stops the app
echo.
echo [SYSTEM] System Services Terminated.
pause
