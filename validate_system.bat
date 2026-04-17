@echo off
setlocal
title Telecom Customer Churn - Multi-Point System Validation

:: Clean screen and display banner
cls
echo ============================================================
echo   🛠 TELECOM CHURN: SYSTEM HEALTH CHECK
echo ============================================================
echo.
echo [SYSTEM] Starting full architecture health check...
echo.

:: Pillar 0: Sync Dependencies
echo [0/4] Pillar 0: Syncing all dependencies...
call uv sync --all-extras --quiet
if %ERRORLEVEL% NEQ 0 goto :FAILED

:: Pillar 1: Static Code Quality
echo [1/4] Pillar 1: Static Code Quality (Pyright ^& Ruff)...
echo      - Running Pyright (Static Type Checking)...
call uv run pyright src/
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo.
echo      - Running Ruff (Linting)...
call uv run ruff check .
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo.
echo      - Running Ruff (Formatting Check)...
call uv run ruff format --check .
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo      Done.
echo.

:: Pillar 2: Functional Logic ^& Coverage
echo [2/4] Pillar 2: Functional Logic ^& Coverage...
echo      - Running Pytest with Coverage (Gate: 65%%)...
call uv run pytest tests/ --cov=src --cov-fail-under=65 --tb=short
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo      Done.
echo.

:: Pillar 3: Pipeline Synchronization
echo [3/4] Pillar 3: Pipeline Synchronization (DVC)...
call uv run dvc status
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo      Done.
echo.

:: Pillar 4: App Service Health
echo [4/4] Pillar 4: App Service Health...
:: Check Prediction API (8000)
powershell -Command "try { $c = New-Object System.Net.Sockets.TcpClient('localhost', 8000); if ($c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% NEQ 0 (
    echo      Prediction API is OFFLINE on port 8000.
) else (
    echo      Prediction API is ONLINE on port 8000.
)

:: Check Embedding Service (8001)
powershell -Command "try { $c = New-Object System.Net.Sockets.TcpClient('localhost', 8001); if ($c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% NEQ 0 (
    echo      Embedding Service is OFFLINE on port 8001.
) else (
    echo      Embedding Service is ONLINE on port 8001.
)

:: Check Gradio UI (7860)
powershell -Command "try { $c = New-Object System.Net.Sockets.TcpClient('localhost', 7860); if ($c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% NEQ 0 (
    echo      Gradio UI is OFFLINE on port 7860.
) else (
    echo      Gradio UI is ONLINE on port 7860.
)

echo      Done.
echo.

:SUCCESS
echo ============================================================
echo   ✅ SYSTEM HEALTH: 100%% (ALL GATES PASSED)
echo ============================================================
echo.
echo Your Hardened Telecom Churn architecture is validated.
pause
exit /b 0

:FAILED
echo.
echo ============================================================
echo   ❌ VALIDATION FAILED
echo ============================================================
echo.
echo Please review the logs above and correct the issues.
pause
exit /b 1
