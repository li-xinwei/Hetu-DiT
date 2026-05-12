"""Lightweight in-process metrics for the eval harness.

Exposes counters incremented by the dispatcher and a latency ring buffer
appended on request completion. The ``/metrics`` HTTP endpoint in
``hetu_dit/entrypoint/api_server.py`` reads this dict and returns JSON.

The hit-type taxonomy is the dispatcher's classification:
  - ready: ``_find_ready_executor`` returned non-None (same executor, already bound, fastest)
  - warm:  ``_find_warm_l2_executor`` matched a warm-tagged + L2-parked executor (true pre-warm pool hit)
  - l2:    ``_find_warm_l2_executor`` matched any L2-parked executor (single-pod L2 mechanism)
  - cold:  fell through to switch_parallel_env (legacy reconfigure path)

All operations are cheap and lock-free; CPython int += and list.append are
atomic against the GIL.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

_LATENCY_RING_MAX = 1000

_received_ts: Dict[str, float] = {}
_dispatched_ts: Dict[str, float] = {}

_metrics: Dict[str, Any] = {
    "requests_total": 0,
    "requests_hit_ready": 0,
    "requests_hit_warm": 0,
    "requests_hit_l2": 0,
    "requests_hit_cold": 0,
    "request_latency_ms": [],
    "request_queue_ms": [],
    "process_start_ts": time.time(),
}


def record_received(task_id: str) -> None:
    """Called by the API layer when /generate accepts a request."""
    _received_ts[task_id] = time.time()


def record_dispatch(task_id: str, hit_type: str) -> None:
    """Called by the dispatcher when an executor is assigned to a task."""
    if hit_type not in ("ready", "warm", "l2", "cold"):
        hit_type = "cold"
    _metrics[f"requests_hit_{hit_type}"] = _metrics.get(f"requests_hit_{hit_type}", 0) + 1
    _metrics["requests_total"] += 1
    _dispatched_ts[task_id] = time.time()


def _append_bounded(key: str, value: float) -> None:
    ring: List[float] = _metrics[key]
    ring.append(value)
    if len(ring) > _LATENCY_RING_MAX:
        del ring[: len(ring) - _LATENCY_RING_MAX]


def record_done(task_id: str) -> None:
    """Called when the engine completes a task (after execute_model_done)."""
    now = time.time()
    received = _received_ts.pop(task_id, None)
    dispatched = _dispatched_ts.pop(task_id, None)
    if received is not None:
        _append_bounded("request_latency_ms", (now - received) * 1000.0)
    if received is not None and dispatched is not None:
        _append_bounded("request_queue_ms", (dispatched - received) * 1000.0)


def _percentile(samples: List[float], pct: float) -> float:
    if not samples:
        return 0.0
    sorted_s = sorted(samples)
    idx = int(round((pct / 100.0) * (len(sorted_s) - 1)))
    return sorted_s[idx]


def snapshot() -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of current metric state."""
    lat = list(_metrics["request_latency_ms"])
    queue = list(_metrics["request_queue_ms"])
    return {
        "requests_total": _metrics["requests_total"],
        "requests_hit_ready": _metrics["requests_hit_ready"],
        "requests_hit_warm": _metrics["requests_hit_warm"],
        "requests_hit_l2": _metrics["requests_hit_l2"],
        "requests_hit_cold": _metrics["requests_hit_cold"],
        "request_latency_ms_count": len(lat),
        "p50_ms": _percentile(lat, 50),
        "p95_ms": _percentile(lat, 95),
        "p99_ms": _percentile(lat, 99),
        "queue_p50_ms": _percentile(queue, 50),
        "queue_p95_ms": _percentile(queue, 95),
        "process_start_ts": _metrics["process_start_ts"],
        "uptime_s": time.time() - _metrics["process_start_ts"],
    }
