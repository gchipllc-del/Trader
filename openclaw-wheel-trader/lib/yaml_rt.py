"""Comment-preserving YAML I/O for files that Hermes mutates in-place.

PyYAML's ``safe_load`` discards comments and ``safe_dump`` writes only
the data — so a load→mutate→dump cycle silently destroys every inline
explanation in config/settings.yaml and config/wheel_strategy.yaml.
This has caused multiple manual restores during Hermes runs.

ruamel.yaml's round-trip ("rt") loader keeps comments, blank lines,
key ordering, and indentation style as part of the in-memory tree.
The same library's dumper writes them back unchanged when ONLY the
specific values Hermes targeted have been mutated.

USAGE
    from lib.yaml_rt import rt_load, rt_dump

    data = rt_load(STRATEGY_PATH)
    data["stock_params"]["max_position_pct"] = 0.30   # mutate in place
    rt_dump(data, STRATEGY_PATH)
    # → all surrounding comments + ordering preserved

The returned object is a ruamel ``CommentedMap``, which behaves like
a regular dict for read/write but carries comment metadata. Pass it
back to ``rt_dump`` to keep that metadata in the output.

FALLBACK
    If ruamel.yaml isn't installed (unlikely — it's a transitive dep
    of several pinned packages), fall back silently to the old behavior.
    Comments will be lost in that case but the bot still functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from ruamel.yaml import YAML

    _RT_AVAILABLE = True
    _yaml = YAML(typ="rt")
    _yaml.preserve_quotes = True
    _yaml.indent(mapping=2, sequence=4, offset=2)
    _yaml.width = 200
except ImportError:
    _RT_AVAILABLE = False
    _yaml = None
    import yaml as _pyyaml


def rt_load(path: str | Path) -> Any:
    """Load YAML preserving comments / structure for round-trip writes.

    Falls back to ``yaml.safe_load`` if ruamel.yaml isn't installed.
    """
    path = Path(path)
    if not path.exists():
        return {}
    if _RT_AVAILABLE:
        with open(path) as f:
            data = _yaml.load(f)
        return data if data is not None else {}
    # PyYAML fallback — comments lost on write
    with open(path) as f:
        return _pyyaml.safe_load(f) or {}


def rt_dump(data: Any, path: str | Path) -> None:
    """Dump YAML preserving comments / structure of the loaded tree.

    The ``data`` should be the same object returned by ``rt_load`` (or
    a sub-tree of it) — that's what carries the comment metadata. New
    keys inserted by the caller will appear without comments, which is
    the right behavior.

    Falls back to ``yaml.safe_dump(default_flow_style=False,
    sort_keys=False)`` to match Hermes' historical write style.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _RT_AVAILABLE:
        with open(path, "w") as f:
            _yaml.dump(data, f)
        return
    # PyYAML fallback — same flags Hermes was using
    with open(path, "w") as f:
        _pyyaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


__all__ = ["rt_load", "rt_dump"]
