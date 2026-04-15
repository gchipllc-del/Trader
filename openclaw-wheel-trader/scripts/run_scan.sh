#!/bin/bash
# OpenClaw scan runner — called by cron
export PATH="/Users/jesse/anaconda3/bin:$PATH"
cd /Users/jesse/Desktop/projects/traderbot/openclaw-wheel-trader
python main.py scan >> logs/cron_scan.log 2>&1
