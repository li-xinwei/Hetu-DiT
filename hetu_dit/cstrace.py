"""Cold-start tracing helper.

Emits ``[CSTRACE] {timestamp} {stage} key=value ...`` lines to stdout when the
``HETU_COLDSTART_TRACE`` environment variable is set to a truthy value
(``1``/``true``/``yes``). When unset, ``cst_print`` is a cheap no-op so it is
safe to leave the calls in production code paths.

Parse the resulting log lines with ``scripts/parse-coldstart-trace.py``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

_TRACE_ENABLED: bool = os.environ.get("HETU_COLDSTART_TRACE", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def is_enabled() -> bool:
    return _TRACE_ENABLED


def cst_print(stage: str, **fields: Any) -> None:
    """Emit a single CSTRACE line if tracing is enabled."""
    if not _TRACE_ENABLED:
        return
    parts = [f"[CSTRACE] {time.time():.6f} {stage}"]
    for k, v in fields.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True, file=sys.stdout)
