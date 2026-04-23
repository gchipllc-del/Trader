#!/bin/bash
# OpenClaw scan runner — called by launchd.
set -euo pipefail

export PATH="/Users/jesse/anaconda3/bin:$PATH"
PROJECT_ROOT="/Users/jesse/Desktop/projects/traderbot/openclaw-wheel-trader"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

python main.py scan >> logs/cron_scan.log 2>&1
