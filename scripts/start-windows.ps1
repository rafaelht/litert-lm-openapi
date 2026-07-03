$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

$ModelPath = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "models/gemma-4-E2B-it.litertlm" }
$ServerPort = if ($env:SERVER_PORT) { $env:SERVER_PORT } else { "8005" }

if (-not (Test-Path $ModelPath)) {
    Write-Error "Model file not found: $ModelPath"
    exit 1
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $created = $false

    try {
        py -3.12 -m venv .venv
        $created = $true
    } catch {
        Write-Host "Python 3.12 launcher not available, trying python..."
    }

    if (-not $created) {
        python -m venv .venv
    }
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "LiteRT Session Server"
Write-Host "Local:   http://127.0.0.1:$ServerPort/v1"
Write-Host "LAN:     http://<YOUR_WINDOWS_IP>:$ServerPort/v1"
Write-Host "Health:  http://127.0.0.1:$ServerPort/healthz"
Write-Host ""

$env:MODEL_PATH = $ModelPath
$env:SERVER_PORT = $ServerPort
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port $ServerPort
