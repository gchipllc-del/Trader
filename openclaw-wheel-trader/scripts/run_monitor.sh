#!/bin/bash
# OpenClaw monitor runner — called by launchd every 3 minutes.
# Uses strict mode so any failure is visible in the launchd exit code.
set -euo pipefail

export PATH="/Users/jesse/anaconda3/bin:$PATH"
PROJECT_ROOT="/Users/jesse/Desktop/projects/traderbot/openclaw-wheel-trader"
cd "$PROJECT_ROOT"

# Load .env so ALPACA_API_KEY / _SECRET_KEY are available under launchd,
# which doesn't inherit a login shell's env. Without this, python raises
# EnvironmentError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set").
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

python -c "
from lib.alpaca_client import AlpacaClient
from lib.monitor import run_monitoring_check
client = AlpacaClient()
run_monitoring_check(client)
" >> logs/cron_monitor.log 2>&1
