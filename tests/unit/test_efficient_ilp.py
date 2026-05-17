"""Unit tests for ``hetu_dit/core/request_manager/efficient_ilp.py``.

Needs ``pulp`` (small ~1 MB install). Loaded via ``conftest.load_module``
to bypass the package ``__init__``. No torch / ray.
"""

from __future__ import annotations

import pytest

pulp = pytest.importorskip("pulp")

from tests.conftest import load_module

efficient_ilp = load_module(
    "hetu_dit/core/request_manager/efficient_ilp.py",
    module_name="hetu_dit_efficient_ilp",
)


def test_constants_match_paper():
    """{1,2,4,8} K-set + 0.8 efficiency threshold are in the TridentServe
    paper — guard against silent edits."""
    assert efficient_ilp.K_SET == (1, 2, 4, 8)
    assert efficient_ilp.EFF_TH == 0.8


def test_select_tasks_empty_returns_empty():
    out = efficient_ilp.select_tasks(now=0.0, m_free=8, busy_eta=None, tasks=[])
    assert out == []


def test_select_tasks_no_free_gpus_returns_empty():
    """With free_gpus=0 and no busy ETAs, even a single ready task can't start."""
    task = {
        "id": "T1",
        "t": {1: 1.0, 2: 0.6, 4: 0.4, 8: 0.3},
        "ddl": 100.0,
        "input_config": None,
    }
    out = efficient_ilp.select_tasks(now=0.0, m_free=0, busy_eta=None, tasks=[task])
    assert out == []


def test_passes_efficiency_filter_degree_one_always_passes():
    """degree=1 ⇒ ratio = t1/runtime / 1 ≥ 0.8 (when runtime≤t1)."""
    assert efficient_ilp._passes_efficiency_filter(10.0, 10.0, 1, None) is True


def test_passes_efficiency_filter_low_efficiency_rejected():
    """t1=10, runtime=8 at degree=4: efficiency = (10/8)/4 = 0.3125."""
    assert efficient_ilp._passes_efficiency_filter(10.0, 8.0, 4, None) is False


def test_passes_efficiency_filter_high_efficiency_passes():
    """Near-linear at degree=2: t1=10, runtime=5.5 → eff = (10/5.5)/2 ≈ 0.909."""
    assert efficient_ilp._passes_efficiency_filter(10.0, 5.5, 2, None) is True


def test_build_time_windows_no_busy_eta():
    starts, ends, capacities = efficient_ilp._build_time_windows(
        now=100.0, free_gpus=8, busy_eta=None
    )
    assert starts[0] == 100.0
    assert capacities[0] == 8


def test_build_time_windows_with_busy_eta():
    starts, ends, capacities = efficient_ilp._build_time_windows(
        now=0.0, free_gpus=2, busy_eta=[5.0, 10.0]
    )
    assert starts[0] == 0.0
    assert capacities[0] == 2
    # capacity grows as busy workers free
    assert max(capacities) >= 4
