"""Unit tests for ``scripts/parse-coldstart-trace.py``.

Pure-Python parser, no torch / ray / cuda — runs anywhere pytest does.
Covers: line parsing, phase timeline extraction, per-rank weight-load
durations, block-level histograms, and a no-markers-found degraded path.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

# Load the script as a module — it isn't on sys.path normally because it
# lives under scripts/ with a hyphenated name (Python disallows ``import``
# of hyphens). importlib.util gets us a clean handle.
_PARSER_PATH = Path(__file__).parents[2] / "scripts" / "parse-coldstart-trace.py"


@pytest.fixture(scope="module")
def parser_module():
    spec = importlib.util.spec_from_file_location("parse_coldstart_trace", _PARSER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SAMPLE_LOG = """\
some unrelated log line here
[CSTRACE] 1000.000000 process_start
[CSTRACE] 1001.200000 imports_done
[CSTRACE] 1003.500000 create_engine_start
[CSTRACE] 1050.000000 engine_created
[CSTRACE] 1050.500000 uvicorn_start
[CSTRACE] 1051.000000 startup_hook_entered
[CSTRACE] 1051.100000 init_executors_start
[CSTRACE] 1051.500000 weight_load_start rank=0 strategy=default
[CSTRACE] 1051.500000 weight_load_start rank=1 strategy=default
[CSTRACE] 1051.700000 block_load_start rank=1 block=transformer idx=0
[CSTRACE] 1052.500000 block_load_done rank=1 block=transformer idx=0
[CSTRACE] 1052.500000 block_load_start rank=1 block=text_encoder idx=1
[CSTRACE] 1052.900000 block_load_done rank=1 block=text_encoder idx=1
[CSTRACE] 1071.500000 weight_load_done rank=0 strategy=default
[CSTRACE] 1072.000000 weight_load_done rank=1 strategy=default
[CSTRACE] 1075.000000 init_executors_done
[CSTRACE] 1075.100000 init_monitor_done
[CSTRACE] 1075.500000 startup_complete
"""


def test_parse_lines_extracts_all_markers(parser_module):
    events = parser_module.parse_lines(SAMPLE_LOG.splitlines())
    assert len(events) == 18  # 18 [CSTRACE] lines in SAMPLE_LOG; 1 noise line filtered
    assert events[0] == (1000.0, "process_start", {})
    rank0_done = next(e for e in events if e[1] == "weight_load_done" and e[2].get("rank") == "0")
    assert rank0_done[0] == 1071.5
    assert rank0_done[2] == {"rank": "0", "strategy": "default"}


def test_parse_lines_skips_non_marker_lines(parser_module):
    log = "noise line\n[CSTRACE] 100.0 stage_a\nmore noise\n[CSTRACE] 101.0 stage_b\n"
    events = parser_module.parse_lines(log.splitlines())
    assert [stage for _, stage, _ in events] == ["stage_a", "stage_b"]


def test_parse_lines_sorts_by_timestamp(parser_module):
    # Ingest deliberately out-of-order lines.
    lines = [
        "[CSTRACE] 200.0 b",
        "[CSTRACE] 100.0 a",
        "[CSTRACE] 300.0 c",
    ]
    events = parser_module.parse_lines(lines)
    assert [stage for _, stage, _ in events] == ["a", "b", "c"]


def test_phase_timeline_offsets_relative_to_process_start(parser_module, capsys):
    events = parser_module.parse_lines(SAMPLE_LOG.splitlines())
    parser_module.report_phase_timeline(events)
    out = capsys.readouterr().out
    assert "process_start" in out
    assert "imports_done" in out
    assert "75.500 s" in out  # startup_complete relative to process_start


def test_weight_load_durations(parser_module, capsys):
    events = parser_module.parse_lines(SAMPLE_LOG.splitlines())
    parser_module.report_weight_load(events)
    out = capsys.readouterr().out
    # rank 0: 1071.5 - 1051.5 = 20.0 s; rank 1: 1072.0 - 1051.5 = 20.5 s
    assert "20.000 s" in out
    assert "20.500 s" in out
    assert "strategy=default" in out
    # max-across-ranks line
    assert "wall-clock contribution" in out


def test_block_level_histogram_when_markers_present(parser_module, capsys):
    events = parser_module.parse_lines(SAMPLE_LOG.splitlines())
    parser_module.report_block_transfers(events)
    out = capsys.readouterr().out
    assert "rank   1" in out
    assert "blocks=  2" in out


def test_block_histogram_silent_without_block_markers(parser_module, capsys):
    log = "\n".join(
        [
            "[CSTRACE] 100.0 weight_load_start rank=0 strategy=default",
            "[CSTRACE] 120.0 weight_load_done rank=0 strategy=default",
        ]
    )
    events = parser_module.parse_lines(log.splitlines())
    parser_module.report_block_transfers(events)
    out = capsys.readouterr().out
    assert out == ""  # no block_load_* markers → no section


def test_main_returns_1_when_no_markers(parser_module, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("nothing useful here\n"))
    rc = parser_module.main(["parse-coldstart-trace.py"])
    assert rc == 1


def test_main_returns_0_with_valid_input_via_stdin(parser_module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(SAMPLE_LOG))
    rc = parser_module.main(["parse-coldstart-trace.py"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "phase timeline" in out
    assert "weight load" in out


def test_main_returns_0_with_file_argument(parser_module, tmp_path, capsys):
    log_file = tmp_path / "server.log"
    log_file.write_text(SAMPLE_LOG)
    rc = parser_module.main(["parse-coldstart-trace.py", str(log_file)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "phase timeline" in out
