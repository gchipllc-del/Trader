"""Traderbot-local thin wrapper around tradingcore.hermes_ledger.

Pins this bot's experiments to data/hermes_experiments.jsonl. The shared
implementation lives in tradingcore so polybot can use the same logic
with its own ledger path.
"""
from __future__ import annotations

from pathlib import Path

from tradingcore.hermes_ledger import (
    open_experiment as _open_experiment,
    list_open_experiments as _list_open_experiments,
    close_experiment as _close_experiment,
    recently_rolled_back as _recently_rolled_back,
    history as _history,
    stats as _stats,
)

LEDGER_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "hermes_experiments.jsonl"
)


def open_experiment(**kwargs):
    kwargs.setdefault("ledger_path", LEDGER_PATH)
    return _open_experiment(**kwargs)


def list_open_experiments(**kwargs):
    kwargs.setdefault("ledger_path", LEDGER_PATH)
    return _list_open_experiments(**kwargs)


def close_experiment(experiment_id, **kwargs):
    kwargs.setdefault("ledger_path", LEDGER_PATH)
    return _close_experiment(experiment_id, **kwargs)


def recently_rolled_back(param, lookback_hours=72.0, **kwargs):
    kwargs.setdefault("ledger_path", LEDGER_PATH)
    return _recently_rolled_back(param, lookback_hours=lookback_hours, **kwargs)


def history(limit=50, **kwargs):
    kwargs.setdefault("ledger_path", LEDGER_PATH)
    return _history(limit=limit, **kwargs)


def stats(**kwargs):
    kwargs.setdefault("ledger_path", LEDGER_PATH)
    return _stats(**kwargs)


__all__ = [
    "LEDGER_PATH",
    "open_experiment",
    "list_open_experiments",
    "close_experiment",
    "recently_rolled_back",
    "history",
    "stats",
]
