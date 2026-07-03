@echo off
setlocal

cd /d "%~dp0\.."

if "%MODEL_PATH%"=="" set MODEL_PATH=models\gemma-4-E2B-it.litertlm
if "%SERVER_PORT%"=="" set SERVER_PORT=8005

if not exist "%MODEL_PATH%" (
  echo Model file not found: %MODEL_PATH%
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    python -m venv .venv
    if errorlevel 1 exit /b 1
  )
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo LiteRT Session Server
echo Local:   http://127.0.0.1:%SERVER_PORT%/v1
echo LAN:     http://^<YOUR_WINDOWS_IP^>:%SERVER_PORT%/v1
echo Health:  http://127.0.0.1:%SERVER_PORT%/healthz
echo.

".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port %SERVER_PORT%
