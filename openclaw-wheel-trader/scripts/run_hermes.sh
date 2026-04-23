#!/bin/bash
# Hermes self-optimization — runs daily at 15:15 via launchd.
set -euo pipefail

export PATH="/Users/jesse/anaconda3/bin:$PATH"
PROJECT_ROOT="/Users/jesse/Desktop/projects/traderbot/openclaw-wheel-trader"
cd "$PROJECT_ROOT"

# Load .env so ALPACA credentials are available (launchd doesn't inherit
# a login shell's env).
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

python main.py hermes >> logs/cron_hermes.log 2>&1
