#!/bin/bash
# OpenClaw monitor runner — called by cron every 3 minutes
export PATH="/Users/jesse/anaconda3/bin:$PATH"
cd /Users/jesse/Desktop/projects/traderbot/openclaw-wheel-trader
python -c "
from lib.alpaca_client import AlpacaClient
from lib.monitor import run_monitoring_check
client = AlpacaClient()
run_monitoring_check(client)
" >> logs/cron_monitor.log 2>&1
