#!/usr/bin/env python3
"""Cost + SLO helpers for the warm-pool eval harness.

Pure-Python, no IO. Imported by ``scripts/aggregate_runs.py`` and
``scripts/gen_report.py`` to convert raw timings into dollar / SLO metrics.

Cost model
----------
For a run of duration ``wall_clock_s`` on a pool of ``pool_size`` pods (head +
warm), each pod with ``gpu_per_pod`` GPUs, the GPU-hour bill is

    gpu_seconds      = pool_size * gpu_per_pod * wall_clock_s
    cost_usd         = gpu_seconds / 3600 * rate_usd_per_gpu_hour
    cost_per_request = cost_usd / max(requests_served, 1)

``observed_active_gpu_seconds`` is the time pods spent in ``active-busy`` —
used to compute the idle fraction. If the caller doesn't have it, pass 0.0.

SLO definition (matches TridentServe arxiv 2510.02838): a request meets the
SLO when ``latency_ms <= multiplier * reference_latency_ms`` where the
reference is the optimal-parallelism single-request latency on a hot cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class CostInputs:
    pool_size: int
    wall_clock_s: float
    gpu_per_pod: int = 8       # H100 SXM 8x default
    rate_usd_per_gpu_hour: float = 4.0  # RunPod H100 SXM 2026-05 quote
    requests_served: int = 0


@dataclass(frozen=True)
class CostOutputs:
    gpu_seconds: float
    cost_usd: float
    cost_per_request_usd: float
    idle_gpu_seconds: float
    idle_fraction: float


def compute(
    ci: CostInputs, *, observed_active_gpu_seconds: float = 0.0
) -> CostOutputs:
    gpu_seconds = max(0.0, ci.pool_size * ci.gpu_per_pod * ci.wall_clock_s)
    cost_usd = gpu_seconds / 3600.0 * ci.rate_usd_per_gpu_hour
    if ci.requests_served > 0:
        cost_per_request = cost_usd / ci.requests_served
    else:
        cost_per_request = float("inf")
    idle_gpu_seconds = max(0.0, gpu_seconds - observed_active_gpu_seconds)
    idle_fraction = (
        idle_gpu_seconds / gpu_seconds if gpu_seconds > 0 else 0.0
    )
    return CostOutputs(
        gpu_seconds=gpu_seconds,
        cost_usd=cost_usd,
        cost_per_request_usd=cost_per_request,
        idle_gpu_seconds=idle_gpu_seconds,
        idle_fraction=idle_fraction,
    )


def slo_target_ms(
    model: str,
    height: int,
    width: int,
    reference_latency_ms_lookup: Dict[Tuple[str, int, int], float],
    multiplier: float = 2.5,
) -> float:
    """Return the SLO threshold (ms) for a given (model, h, w) request."""
    key = (model, height, width)
    if key not in reference_latency_ms_lookup:
        raise KeyError(
            f"no reference latency for {key}; "
            "calibrate data/optimal_latencies.json first"
        )
    return multiplier * reference_latency_ms_lookup[key]


def slo_attainment(
    latencies_ms: Iterable[float], threshold_ms: float
) -> Tuple[int, int, float]:
    """Return (n_violations, n_total, attainment_fraction) over the sample."""
    total = 0
    violations = 0
    for lat in latencies_ms:
        total += 1
        if lat > threshold_ms:
            violations += 1
    attainment = (total - violations) / total if total else 0.0
    return violations, total, attainment
