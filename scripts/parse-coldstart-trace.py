#!/usr/bin/env python3
"""Parse [CSTRACE] markers emitted by ``hetu_dit/cstrace.py``.

Reads a server log from stdin (or a file passed as argv[1]) and prints:

* phase timeline (process_start through startup_complete) on the head node
* per-rank weight load duration (weight_load_start -> weight_load_done)
* per-rank block transfer latency histogram (if block_load_* markers present)
* cold-start wall-clock contribution from weight load (max across ranks)

Usage::

    HETU_COLDSTART_TRACE=1 python -m hetu_dit.entrypoint.api_server ...   2>&1 | tee server.log
    python scripts/parse-coldstart-trace.py server.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, List

SCHEMA_VERSION = "eval-v1"

LINE_RE = re.compile(r"\[CSTRACE\]\s+(?P<ts>\d+\.\d+)\s+(?P<stage>\S+)(?P<rest>.*)")
KV_RE = re.compile(r"(\w+)=(\S+)")


def parse_lines(lines: Iterable[str]):
    events = []
    for line in lines:
        m = LINE_RE.search(line)
        if not m:
            continue
        ts = float(m.group("ts"))
        stage = m.group("stage")
        fields = dict(KV_RE.findall(m.group("rest")))
        events.append((ts, stage, fields))
    events.sort(key=lambda e: e[0])
    return events


def fmt_duration(seconds: float) -> str:
    return f"{seconds:7.3f} s"


def report_phase_timeline(events):
    headline_stages = (
        "process_start",
        "imports_done",
        "create_engine_start",
        "engine_created",
        "uvicorn_start",
        "startup_hook_entered",
        "init_executors_start",
        "init_executors_done",
        "init_monitor_done",
        "startup_complete",
    )
    seen = {}
    for ts, stage, _ in events:
        if stage in headline_stages and stage not in seen:
            seen[stage] = ts
    if not seen:
        return
    base = seen.get("process_start", min(seen.values()))
    print("== phase timeline (offset from process_start) ==")
    for stage in headline_stages:
        if stage in seen:
            print(f"  {stage:<24} {fmt_duration(seen[stage] - base)}")
    print()


def report_weight_load(events):
    starts: dict[str, float] = {}
    durations: dict[str, float] = {}
    strategy: str | None = None
    for ts, stage, fields in events:
        rank = fields.get("rank")
        if rank is None:
            continue
        if stage == "weight_load_start":
            starts[rank] = ts
            strategy = strategy or fields.get("strategy")
        elif stage == "weight_load_done" and rank in starts:
            durations[rank] = ts - starts[rank]
    if not durations:
        return
    print(f"== weight load (strategy={strategy}) ==")
    for rank in sorted(durations, key=lambda r: int(r)):
        print(f"  rank {rank:>3}  {fmt_duration(durations[rank])}")
    vals = list(durations.values())
    print(
        f"  -> per-rank mean={fmt_duration(mean(vals))} "
        f"median={fmt_duration(median(vals))} max={fmt_duration(max(vals))}"
    )
    print(f"  -> wall-clock contribution (max across ranks) = {fmt_duration(max(vals))}")
    print()


def report_block_transfers(events):
    """Per-rank histogram of block_load durations, if present."""
    starts: dict[tuple[str, str], float] = {}
    durations_by_rank: dict[str, list[float]] = defaultdict(list)
    for ts, stage, fields in events:
        rank = fields.get("rank")
        block = fields.get("block")
        if rank is None or block is None:
            continue
        key = (rank, block)
        if stage == "block_load_start":
            starts[key] = ts
        elif stage == "block_load_done" and key in starts:
            durations_by_rank[rank].append(ts - starts[key])
    if not durations_by_rank:
        return
    print("== block-level transfers ==")
    for rank in sorted(durations_by_rank, key=lambda r: int(r)):
        vals = durations_by_rank[rank]
        print(
            f"  rank {rank:>3}  blocks={len(vals):>3}  "
            f"mean={mean(vals):.3f}s median={median(vals):.3f}s max={max(vals):.3f}s"
        )
    print()


def build_phase_timeline(events) -> Dict[str, float]:
    headline_stages = (
        "process_start",
        "imports_done",
        "create_engine_start",
        "engine_created",
        "uvicorn_start",
        "startup_hook_entered",
        "init_executors_start",
        "init_executors_done",
        "init_monitor_done",
        "startup_complete",
    )
    seen: Dict[str, float] = {}
    for ts, stage, _ in events:
        if stage in headline_stages and stage not in seen:
            seen[stage] = ts
    if not seen:
        return {}
    base = seen.get("process_start", min(seen.values()))
    return {stage: seen[stage] - base for stage in headline_stages if stage in seen}


def build_weight_load(events) -> Dict[str, Any]:
    starts: Dict[str, float] = {}
    durations: Dict[str, float] = {}
    strategy = None
    for ts, stage, fields in events:
        rank = fields.get("rank")
        if rank is None:
            continue
        if stage == "weight_load_start":
            starts[rank] = ts
            strategy = strategy or fields.get("strategy")
        elif stage == "weight_load_done" and rank in starts:
            durations[rank] = ts - starts[rank]
    if not durations:
        return {}
    vals = list(durations.values())
    return {
        "strategy": strategy,
        "per_rank_seconds": {r: durations[r] for r in sorted(durations, key=lambda x: int(x))},
        "mean_seconds": mean(vals),
        "median_seconds": median(vals),
        "wall_clock_max_seconds": max(vals),
    }


def build_requests(events) -> List[Dict[str, Any]]:
    """Group per-request markers into structured records.

    Looks for request_start / request_dispatched / request_done emitted by
    cst_request. Falls back to generate_received / execute_model_done pairs
    for legacy traces predating the request_* markers.
    """
    by_req: Dict[str, Dict[str, Any]] = {}
    for ts, stage, fields in events:
        req_id = fields.get("req_id") or fields.get("task_id")
        if req_id is None:
            continue
        rec = by_req.setdefault(req_id, {"req_id": req_id})
        if stage == "request_start" or stage == "generate_received":
            rec.setdefault("start_ts", ts)
        elif stage == "request_dispatched":
            rec["dispatched_ts"] = ts
            rec["hit_type"] = fields.get("hit_type", "unknown")
            if "executor_id" in fields:
                rec["executor_id"] = fields["executor_id"]
        elif stage == "request_done" or stage == "execute_model_done":
            rec.setdefault("done_ts", ts)

    out: List[Dict[str, Any]] = []
    for req_id, rec in by_req.items():
        start = rec.get("start_ts")
        done = rec.get("done_ts")
        dispatched = rec.get("dispatched_ts")
        if start is not None and done is not None:
            rec["latency_ms"] = (done - start) * 1000.0
        if start is not None and dispatched is not None:
            rec["queue_ms"] = (dispatched - start) * 1000.0
        rec.setdefault("hit_type", "unknown")
        out.append(rec)
    out.sort(key=lambda r: r.get("start_ts", 0))
    return out


def emit_json(events) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_timeline": build_phase_timeline(events),
        "weight_load": build_weight_load(events),
        "requests": build_requests(events),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="-",
                        help="path to log file, or '-' for stdin (default)")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of text tables")
    args = parser.parse_args(argv[1:])

    if args.path != "-":
        with open(args.path, "r", errors="replace") as fh:
            events = parse_lines(fh)
    else:
        events = parse_lines(sys.stdin)

    if not events:
        if args.json:
            json.dump({"schema_version": SCHEMA_VERSION, "phase_timeline": {},
                       "weight_load": {}, "requests": []}, sys.stdout)
            print()
            return 0
        print("no [CSTRACE] markers found", file=sys.stderr)
        return 1

    if args.json:
        json.dump(emit_json(events), sys.stdout, indent=2, default=str)
        print()
        return 0

    report_phase_timeline(events)
    report_weight_load(events)
    report_block_transfers(events)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
