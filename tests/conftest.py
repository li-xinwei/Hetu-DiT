"""Shared pytest helpers.

``hetu_dit/__init__.py`` eagerly imports diffusers / torch — too heavy for
a CPU-only CI gate. Tests that target lightweight modules (cstrace, the
parse script, entrypoint utils, the ILP solver) use ``load_module`` below
to import the file directly via ``importlib.util.spec_from_file_location``,
sidestepping the package ``__init__``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def load_module(rel_path: str, module_name: str | None = None):
    """Load a Python file at ``REPO_ROOT / rel_path`` as a fresh module.

    The loaded module bypasses ``hetu_dit/__init__.py`` so it cannot
    reference symbols from the rest of the package via ``from hetu_dit.*``;
    only modules that import stdlib + lightweight third-party libs work.
    """
    full = REPO_ROOT / rel_path
    name = module_name or full.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
