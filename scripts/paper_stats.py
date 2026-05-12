#!/usr/bin/env python3
"""Bootstrap CI + small-sample statistics helpers for the paper-rigor report.

Pure-Python; no numpy/scipy dependency. Used by aggregate_runs.py and
gen_report.py to attach 95% confidence intervals to percentile estimates and
to support the latency CDF / sensitivity analysis sections of the report.
"""

from __future__ import annotations

import random
from statistics import mean, median, pstdev
from typing import Dict, List, Sequence, Tuple


def percentile(samples: Sequence[float], pct: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    idx = int(round((pct / 100.0) * (len(s) - 1)))
    return s[idx]


def bootstrap_ci(
    samples: Sequence[float],
    statistic,
    *,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) via percentile bootstrap.

    `statistic` is a callable taking a sequence and returning a float (e.g.
    ``lambda xs: percentile(xs, 95)``). For samples < 2, CI collapses to the
    point estimate.
    """
    if not samples:
        return float("nan"), float("nan"), float("nan")
    point = statistic(samples)
    if len(samples) < 2:
        return point, point, point

    rng = random.Random(seed)
    n = len(samples)
    samples_list = list(samples)
    boot_stats = []
    for _ in range(n_boot):
        resample = [samples_list[rng.randrange(n)] for _ in range(n)]
        boot_stats.append(statistic(resample))
    boot_stats.sort()
    alpha = (1 - confidence) / 2
    lo_idx = int(alpha * n_boot)
    hi_idx = int((1 - alpha) * n_boot) - 1
    return point, boot_stats[lo_idx], boot_stats[max(hi_idx, lo_idx)]


def mean_ci(samples: Sequence[float], **kwargs) -> Tuple[float, float, float]:
    """Bootstrap CI for the sample mean."""
    return bootstrap_ci(samples, mean, **kwargs)


def percentile_ci(samples: Sequence[float], pct: float, **kwargs) -> Tuple[float, float, float]:
    return bootstrap_ci(samples, lambda xs: percentile(xs, pct), **kwargs)


def cdf_points(samples: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Return (xs, ys) where xs are sorted samples and ys are cumulative fraction."""
    if not samples:
        return [], []
    s = sorted(samples)
    n = len(s)
    ys = [(i + 1) / n for i in range(n)]
    return s, ys


def slo_attainment_at_thresholds(
    samples: Sequence[float], thresholds_ms: Sequence[float]
) -> Dict[float, float]:
    """Fraction of samples <= each threshold, for sensitivity analysis."""
    out: Dict[float, float] = {}
    total = len(samples)
    if total == 0:
        return {t: float("nan") for t in thresholds_ms}
    s = sorted(samples)
    for t in thresholds_ms:
        # binary search for count <= t
        lo, hi = 0, total
        while lo < hi:
            mid = (lo + hi) // 2
            if s[mid] <= t:
                lo = mid + 1
            else:
                hi = mid
        out[t] = lo / total
    return out
