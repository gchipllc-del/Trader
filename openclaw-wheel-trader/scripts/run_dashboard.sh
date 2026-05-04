#!/bin/bash
# Traderbot dashboard runner — called by ai.openclaw.dashboard LaunchAgent.
# Launches the Flask dashboard on port 5051 (localhost only). Long-running;
# KeepAlive auto-restarts on any exit (graceful or crash) since the
# dashboard isn't trade-sensitive — losing visibility is worse than
# auto-restart loops.
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

mkdir -p "$PROJECT_ROOT/logs"
exec python main.py dashboard --port 5051
