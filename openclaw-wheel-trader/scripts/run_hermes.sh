#!/bin/bash
# Hermes self-optimization — runs after market close
export PATH="/Users/jesse/anaconda3/bin:$PATH"
cd /Users/jesse/Desktop/projects/traderbot/openclaw-wheel-trader
python main.py hermes >> logs/cron_hermes.log 2>&1
