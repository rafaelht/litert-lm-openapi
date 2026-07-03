#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH_VALUE="${MODEL_PATH:-models/gemma-4-E2B-it.litertlm}"
SERVER_PORT_VALUE="${SERVER_PORT:-8005}"

if [[ ! -f "$MODEL_PATH_VALUE" ]]; then
  echo "Model file not found: $MODEL_PATH_VALUE"
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  elif [[ -x "/opt/homebrew/bin/python3.12" ]]; then
    /opt/homebrew/bin/python3.12 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi

.venv/bin/python -m pip install -r requirements.txt

echo
echo "LiteRT Session Server"
echo "Local:   http://127.0.0.1:${SERVER_PORT_VALUE}/v1"
echo "LAN:     http://<YOUR_MAC_IP>:${SERVER_PORT_VALUE}/v1"
echo "Health:  http://127.0.0.1:${SERVER_PORT_VALUE}/healthz"
echo

MODEL_PATH="$MODEL_PATH_VALUE" \
SERVER_PORT="$SERVER_PORT_VALUE" \
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "$SERVER_PORT_VALUE"
