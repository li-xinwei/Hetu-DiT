"""Unit tests for ``hetu_dit/cstrace.py``.

Loads the file via ``conftest.load_module`` to bypass ``hetu_dit/__init__.py``
(which eagerly imports diffusers / torch). Pure stdlib otherwise.
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import load_module

CSTRACE_LINE = re.compile(r"^\[CSTRACE\] (\d+\.\d+) (\S+)(.*)$")


def _fresh_cstrace(monkeypatch, env_value):
    """Reload cstrace under a controlled HETU_COLDSTART_TRACE value."""
    if env_value is None:
        monkeypatch.delenv("HETU_COLDSTART_TRACE", raising=False)
    else:
        monkeypatch.setenv("HETU_COLDSTART_TRACE", env_value)
    return load_module("hetu_dit/cstrace.py", module_name="hetu_dit_cstrace")


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "YES", "on", "On"])
def test_enabled_when_env_set_truthy(monkeypatch, val):
    mod = _fresh_cstrace(monkeypatch, val)
    assert mod.is_enabled() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "anything-else"])
def test_disabled_when_env_unset_or_falsy(monkeypatch, val):
    mod = _fresh_cstrace(monkeypatch, val)
    assert mod.is_enabled() is False


def test_disabled_is_silent(monkeypatch, capsys):
    mod = _fresh_cstrace(monkeypatch, "")
    mod.cst_print("stage_x", rank=0)
    out = capsys.readouterr().out
    assert out == ""


def test_enabled_emits_well_formed_line(monkeypatch, capsys):
    mod = _fresh_cstrace(monkeypatch, "1")
    mod.cst_print("weight_load_start", rank=3, strategy="nixl_pipelined")
    out = capsys.readouterr().out.strip()
    m = CSTRACE_LINE.match(out)
    assert m is not None, f"line did not match expected format: {out!r}"
    ts, stage, rest = m.groups()
    assert float(ts) > 0
    assert stage == "weight_load_start"
    assert "rank=3" in rest
    assert "strategy=nixl_pipelined" in rest


def test_enabled_no_kwargs(monkeypatch, capsys):
    mod = _fresh_cstrace(monkeypatch, "1")
    mod.cst_print("process_start")
    out = capsys.readouterr().out
    assert "[CSTRACE]" in out
    assert "process_start" in out
