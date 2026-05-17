"""Unit tests for ``hetu_dit/entrypoint/utils.py``.

Loaded via ``conftest.load_module`` to bypass ``hetu_dit/__init__.py``.
Pure stdlib (socket + os).
"""

from __future__ import annotations

import socket

import pytest

from tests.conftest import load_module

utils = load_module("hetu_dit/entrypoint/utils.py", module_name="hetu_dit_entrypoint_utils")


def test_get_loopback_host_returns_v6_when_supported(monkeypatch):
    monkeypatch.setattr(socket, "inet_pton", lambda fam, addr: b"")
    assert utils.get_loopback_host() == "::1"


def test_get_loopback_host_falls_back_to_v4(monkeypatch):
    def boom(fam, addr):
        raise OSError("no IPv6")

    monkeypatch.setattr(socket, "inet_pton", boom)
    assert utils.get_loopback_host() == "127.0.0.1"


def test_get_bind_host_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("HETUDIT_HOST", "1.2.3.4")
    assert utils.get_bind_host("0.0.0.0") == "0.0.0.0"


def test_get_bind_host_env_when_no_arg(monkeypatch):
    monkeypatch.setenv("HETUDIT_HOST", "10.0.0.1")
    assert utils.get_bind_host(None) == "10.0.0.1"


def test_get_bind_host_default(monkeypatch):
    monkeypatch.delenv("HETUDIT_HOST", raising=False)
    assert utils.get_bind_host(None) == "0.0.0.0"


@pytest.mark.parametrize(
    "model_class,output_type,expected",
    [
        ("sd3", "pil", "stable_diffusion_3_result_T1.png"),
        ("sd3.5", "pil", "stable_diffusion_3_result_T1.png"),
        ("flux", "pil", "flux_result_T1.png"),
        ("hunyuandit", "pil", "hunyuandit_result_T1.png"),
        ("cogvideox", "anything", "cogvideox_T1.mp4"),
        ("hunyuanvideo", "anything", "hunyuan_video_T1.mp4"),
    ],
)
def test_build_output_filename_known(model_class, output_type, expected):
    assert utils.build_output_filename(model_class, "T1", output_type) == expected


def test_build_output_filename_unknown_returns_none():
    assert utils.build_output_filename("unknown_model", "T1") is None


def test_build_output_filename_sd3_non_pil_returns_none():
    assert utils.build_output_filename("sd3", "T1", "tensor") is None
