"""
Defense-in-depth test isolation.

Why this exists: on 2026-05-01 a refactor of the positions store left a
window where tests using `monkeypatch.setattr(stock_engine, "POSITIONS_PATH", ...)`
patched the engine's alias but not the shared store's path, and tests
silently wrote into the live data/positions.json. Took until verification
to catch it. The audit log also got polluted by the same gap.

This conftest auto-redirects the WRITE-CRITICAL data files to per-session
tmp paths for EVERY test, regardless of what the test author remembered to
patch. A test that genuinely wants to exercise file-write behavior should
override these fixtures or accept the per-session redirect.

Tests should still patch their own engine's `POSITIONS_PATH` if they want
their writes to survive into a known temp file (e.g. for assertions). The
conftest just guarantees that if they FORGET, the live files stay clean.
"""

import os
from pathlib import Path

import pytest


# Resolved once per session — every test in this run shares the same tmp tree.
_ISOLATED_DATA_DIR: Path | None = None
_ISOLATED_LOG_DIR: Path | None = None


@pytest.fixture(scope="session")
def _isolated_dirs(tmp_path_factory):
    """Per-session tmp dirs for positions store + audit log."""
    global _ISOLATED_DATA_DIR, _ISOLATED_LOG_DIR
    _ISOLATED_DATA_DIR = tmp_path_factory.mktemp("isolated_data")
    _ISOLATED_LOG_DIR = tmp_path_factory.mktemp("isolated_logs")
    return {"data": _ISOLATED_DATA_DIR, "logs": _ISOLATED_LOG_DIR}


@pytest.fixture(autouse=True)
def _isolate_live_writes(_isolated_dirs, monkeypatch):
    """
    Auto-applied: redirect every test's positions.json + audit_log.jsonl
    writes to the per-session tmp dirs. Prevents the 2026-05-01 leak.
    """
    data_dir = _isolated_dirs["data"]
    logs_dir = _isolated_dirs["logs"]

    # Positions store
    try:
        from lib import positions_store
        monkeypatch.setattr(positions_store, "POSITIONS_PATH",
                            data_dir / "positions.json", raising=False)
    except ImportError:
        pass

    # Each engine module re-exports POSITIONS_PATH; patch them too so
    # their module-local lookups also see the tmp path. (Tests that
    # explicitly patch one of these will override us — that's fine.)
    for mod_name in ("lib.stock_engine", "lib.crypto_engine",
                     "lib.csp_engine", "lib.cc_engine", "lib.monitor"):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            monkeypatch.setattr(mod, "POSITIONS_PATH",
                                data_dir / "positions.json", raising=False)
        except ImportError:
            pass

    # Audit log
    try:
        from lib import audit
        monkeypatch.setattr(audit, "AUDIT_FILE",
                            logs_dir / "audit_log.jsonl", raising=False)
        monkeypatch.setattr(audit, "LOG_DIR", logs_dir, raising=False)
    except ImportError:
        pass

    # Order-dedup hash store
    try:
        from lib import order_dedup
        monkeypatch.setattr(order_dedup, "DEDUP_FILE",
                            data_dir / "intent_hashes.json", raising=False)
    except ImportError:
        pass

    # Trade history (used by stock_engine.execute_stock_sell + Hermes)
    for mod_name in ("lib.stock_engine", "lib.cc_engine"):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "TRADE_HISTORY_PATH"):
                monkeypatch.setattr(mod, "TRADE_HISTORY_PATH",
                                    data_dir / "trade_history.json",
                                    raising=False)
        except ImportError:
            pass

    yield


def pytest_addoption(parser):
    """Escape hatch: --no-isolation lets a test author opt out for genuine
    live-fs tests. Not currently used; kept as future option."""
    parser.addoption("--no-isolation", action="store_true",
                     default=False,
                     help="Disable the auto positions/audit redirect.")
