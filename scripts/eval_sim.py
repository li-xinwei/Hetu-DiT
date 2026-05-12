#!/usr/bin/env python3
"""Event-driven simulator for the warm-pool eval harness.

Replays a workload trace (Request lines, see scripts/workload_gen.py) against
a parametric pool model and emits a synthetic CSTRACE log + summary.txt
identical in shape to a real run subdir. This lets the aggregator and report
generator be mode-agnostic: a sim run and a real run produce the same
artifacts, only the parameters differ.

Pool model
----------
* head_size head pods + warm_replicas warm pods (default head_size=1).
* Each pod starts at t=0, reaches state="l2" after T_BOOT_TO_L2 seconds.
* Per request at time t:
    1. ready    — pool has an "active-idle" pod for the same parallel config
                  (already bound, just finished previous request).
    2. warm     — pool has a warm-tagged pod in state="l2". Cost T_BIND_WARM.
    3. l2       — pool has any (non-warm-tagged) pod in state="l2". Cost
                  T_BIND_COLD_L2 (head pod first bind).
    4. cold     — none of the above. Either wait for a busy pod to free, or
                  spawn from scratch. Cost = max(remaining bind, T_BIND_COLD).
* After bind + T_INFER, pod returns to "active-idle".
* No eviction modeled (a pod stays bound to its last config until process exit).

Calibration constants are taken from the 2026-05-06 H100 SXM session
(`results/runpod-h100-2026-05-06/`). They are CLI-tunable for sweeps.
"""

from __future__ import annotations

import argparse
import ast
import heapq
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default timing constants (seconds). Tunable per-run via CLI.
T_BOOT_TO_L2_DEFAULT = 25.0  # process_start → prepare_for_l2_done
T_BIND_WARM_DEFAULT = 3.14   # observed warm-hit total in path-2inst-warm-042402
T_BIND_COLD_L2_DEFAULT = 12.0  # single-pod L2 (path4) bind time (no warm tag)
T_BIND_COLD_DEFAULT = 35.0   # path1 default cold start total time
T_INFER_SD3_1024 = 3.0       # SD3 1024x1024 20 steps inference time


@dataclass
class Pod:
    pod_id: int
    is_warm_tagged: bool
    boot_done_at: float  # wall-clock when reaches state="l2"
    busy_until: float = 0.0  # active-busy until this timestamp
    last_config: Optional[Tuple[int, int]] = None
    has_been_bound: bool = False
    has_been_l2: bool = False  # has reached at least "l2" state once

    def state_at(self, t: float) -> str:
        if t < self.boot_done_at:
            return "booting"
        if self.busy_until > t:
            return "active-busy"
        if self.has_been_bound:
            return "active-idle"
        return "l2"


@dataclass
class TraceRequest:
    req_id: str
    height: int
    width: int
    timestamp: float
    infer_seconds: float


@dataclass
class DispatchRecord:
    req_id: str
    received_ts: float
    dispatched_ts: float
    done_ts: float
    pod_id: int
    hit_type: str

    @property
    def latency_ms(self) -> float:
        return (self.done_ts - self.received_ts) * 1000.0

    @property
    def queue_ms(self) -> float:
        return (self.dispatched_ts - self.received_ts) * 1000.0


def parse_trace(path: Path) -> List[TraceRequest]:
    out: List[TraceRequest] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if not text.startswith("Request(") or not text.endswith(")"):
                continue
            try:
                node = ast.parse(text, mode="eval").body
            except SyntaxError:
                continue
            if not isinstance(node, ast.Call):
                continue
            rec = {}
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant):
                    rec[kw.arg] = kw.value.value
            req_id = str(
                rec.get("request_id_local") or rec.get("request_id") or len(out)
            )
            try:
                ts = float(rec["timestamp"])
                h = int(rec["height"])
                w = int(rec["width"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append(
                TraceRequest(
                    req_id=req_id,
                    height=h,
                    width=w,
                    timestamp=max(0.0, ts),
                    infer_seconds=T_INFER_SD3_1024,
                )
            )
    out.sort(key=lambda r: r.timestamp)
    return out


def simulate(
    requests: List[TraceRequest],
    head_size: int,
    warm_replicas: int,
    t_boot_l2: float,
    t_bind_warm: float,
    t_bind_cold_l2: float,
    t_bind_cold: float,
) -> Tuple[List[DispatchRecord], List[Pod], float]:
    pods: List[Pod] = []
    pod_id = 0
    # Head pods boot WITH the workload (they receive the first cold request).
    for _ in range(head_size):
        pods.append(
            Pod(pod_id=pod_id, is_warm_tagged=False, boot_done_at=t_boot_l2)
        )
        pod_id += 1
    # Warm pods are pre-deployed: assumed already at L2 by t=0.
    for _ in range(warm_replicas):
        pods.append(
            Pod(pod_id=pod_id, is_warm_tagged=True, boot_done_at=0.0,
                has_been_l2=True)
        )
        pod_id += 1

    if not pods:
        raise ValueError("simulate: pool must have >=1 pod (head_size + warm_replicas > 0)")

    records: List[DispatchRecord] = []
    pending: List[Tuple[float, int]] = []  # (free_at, pod_id) heap

    for req in requests:
        received = req.timestamp
        config = (req.height, req.width)

        # Drain busy pods that freed before `received`.
        while pending and pending[0][0] <= received:
            _, pid = heapq.heappop(pending)
            pods[pid].busy_until = max(pods[pid].busy_until, pending and pending[0][0] or 0)

        # Decide hit_type: prefer ready (same config, active-idle) > warm > l2 > cold.
        chosen: Optional[Pod] = None
        hit_type = "cold"
        bind_cost = 0.0

        # 1. ready hit — any pod active-idle with matching last_config
        for pod in pods:
            if pod.state_at(received) == "active-idle" and pod.last_config == config:
                chosen = pod
                hit_type = "ready"
                bind_cost = 0.0
                break

        # 2. warm hit — warm-tagged pod at l2
        if chosen is None:
            for pod in pods:
                if pod.is_warm_tagged and pod.state_at(received) == "l2":
                    chosen = pod
                    hit_type = "warm"
                    bind_cost = t_bind_warm
                    break

        # 3. l2 hit — any pod at l2 (head pod first time)
        if chosen is None:
            for pod in pods:
                if pod.state_at(received) == "l2":
                    chosen = pod
                    hit_type = "l2"
                    bind_cost = t_bind_cold_l2
                    break

        # 4. config mismatch on active-idle pod (need rebind = cold-like)
        if chosen is None:
            for pod in pods:
                if pod.state_at(received) == "active-idle":
                    chosen = pod
                    hit_type = "cold"
                    bind_cost = t_bind_cold_l2  # rebind for new config
                    break

        # 5. all busy or booting — pick the soonest-available pod (queue)
        if chosen is None:
            # Wait for first pod to free OR finish booting, whichever is sooner.
            earliest_free = min(
                max(pod.boot_done_at, pod.busy_until) for pod in pods
            )
            for pod in pods:
                if max(pod.boot_done_at, pod.busy_until) == earliest_free:
                    chosen = pod
                    break
            assert chosen is not None
            # If pod hadn't yet reached l2, this is a structural cold (boot+bind).
            if not chosen.has_been_l2:
                hit_type = "cold"
                bind_cost = t_bind_cold  # full cold-start
            else:
                hit_type = "ready" if chosen.last_config == config else "cold"
                bind_cost = 0.0 if hit_type == "ready" else t_bind_cold_l2

        # Compute dispatched timestamp: must wait for pod to be free + bound.
        wait_for_pod = max(received, chosen.boot_done_at, chosen.busy_until)
        dispatched = wait_for_pod + bind_cost
        done = dispatched + req.infer_seconds

        chosen.busy_until = done
        chosen.last_config = config
        chosen.has_been_bound = True
        chosen.has_been_l2 = True
        heapq.heappush(pending, (done, chosen.pod_id))

        records.append(
            DispatchRecord(
                req_id=req.req_id,
                received_ts=received,
                dispatched_ts=dispatched,
                done_ts=done,
                pod_id=chosen.pod_id,
                hit_type=hit_type,
            )
        )

    wall_clock_s = max(r.done_ts for r in records) if records else t_boot_l2
    return records, pods, wall_clock_s


def emit_cstrace(records: List[DispatchRecord], out_path: Path, t_boot_l2: float, head: int, warm: int) -> None:
    """Write synthetic cstrace.log. Wall-clock origin set to time.time() for realism."""
    epoch = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        def emit(stage: str, t_offset: float, **kv: object) -> None:
            parts = [f"[CSTRACE] {epoch + t_offset:.6f} {stage}"]
            for k, v in kv.items():
                parts.append(f"{k}={v}")
            f.write(" ".join(parts) + "\n")

        emit("process_start", 0.0)
        emit("imports_done", 1.0)
        emit("create_engine_start", 1.1)
        emit("engine_created", 2.0)
        emit("uvicorn_start", 2.1)
        emit("startup_hook_entered", 2.2)
        emit("init_executors_start", 2.3)
        emit("init_executors_done", t_boot_l2 - 0.5)
        emit("init_monitor_done", t_boot_l2 - 0.4)
        emit("startup_complete", t_boot_l2)
        emit("simulator_pool_ready", t_boot_l2, head=head, warm=warm)
        for rec in records:
            emit("generate_received", rec.received_ts, req_id=rec.req_id)
            emit("request_start", rec.received_ts, req_id=rec.req_id)
            emit("request_dispatched", rec.dispatched_ts,
                 req_id=rec.req_id, hit_type=rec.hit_type, executor_id=rec.pod_id)
            emit("execute_model_done", rec.done_ts, task_id=rec.req_id)
            emit("request_done", rec.done_ts, req_id=rec.req_id)


def emit_summary(
    out_path: Path,
    *,
    pattern: str,
    pool_size: int,
    head: int,
    warm: int,
    model: str,
    n_requests_planned: int,
    n_requests_served: int,
    seed: int,
    wall_clock_s: float,
    t_boot_l2: float,
    t_bind_warm: float,
    t_bind_cold_l2: float,
    t_bind_cold: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f"workload_pattern={pattern}\n"
        f"pool_size={pool_size}\n"
        f"head_size={head}\n"
        f"warm_replicas={warm}\n"
        f"model={model}\n"
        f"n_requests_planned={n_requests_planned}\n"
        f"n_requests_served={n_requests_served}\n"
        f"seed={seed}\n"
        f"wall_clock_s={wall_clock_s:.3f}\n"
        f"t_boot_to_l2={t_boot_l2}\n"
        f"t_bind_warm={t_bind_warm}\n"
        f"t_bind_cold_l2={t_bind_cold_l2}\n"
        f"t_bind_cold={t_bind_cold}\n"
        f"mode=sim\n"
    )
    out_path.write_text(body, encoding="utf-8")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, type=Path,
                        help="trace file produced by workload_gen.py")
    parser.add_argument("--out", required=True, type=Path,
                        help="output run directory; writes cstrace.log + summary.txt")
    parser.add_argument("--head-size", type=int, default=1)
    parser.add_argument("--warm-replicas", type=int, default=0)
    parser.add_argument("--pattern", default="sim",
                        help="workload_pattern tag written into summary.txt")
    parser.add_argument("--model", default="sd3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-boot-l2", type=float, default=T_BOOT_TO_L2_DEFAULT)
    parser.add_argument("--t-bind-warm", type=float, default=T_BIND_WARM_DEFAULT)
    parser.add_argument("--t-bind-cold-l2", type=float, default=T_BIND_COLD_L2_DEFAULT)
    parser.add_argument("--t-bind-cold", type=float, default=T_BIND_COLD_DEFAULT)
    args = parser.parse_args(argv[1:])

    requests = parse_trace(args.workload)
    if not requests:
        print(f"no requests parsed from {args.workload}", file=sys.stderr)
        return 1

    records, pods, wall_clock_s = simulate(
        requests,
        head_size=args.head_size,
        warm_replicas=args.warm_replicas,
        t_boot_l2=args.t_boot_l2,
        t_bind_warm=args.t_bind_warm,
        t_bind_cold_l2=args.t_bind_cold_l2,
        t_bind_cold=args.t_bind_cold,
    )

    emit_cstrace(records, args.out / "cstrace.log",
                 t_boot_l2=args.t_boot_l2,
                 head=args.head_size, warm=args.warm_replicas)
    emit_summary(
        args.out / "summary.txt",
        pattern=args.pattern,
        pool_size=args.head_size + args.warm_replicas,
        head=args.head_size,
        warm=args.warm_replicas,
        model=args.model,
        n_requests_planned=len(requests),
        n_requests_served=len(records),
        seed=args.seed,
        wall_clock_s=wall_clock_s,
        t_boot_l2=args.t_boot_l2,
        t_bind_warm=args.t_bind_warm,
        t_bind_cold_l2=args.t_bind_cold_l2,
        t_bind_cold=args.t_bind_cold,
    )

    # Brief stdout breakdown
    by_hit: Dict[str, int] = {}
    for r in records:
        by_hit[r.hit_type] = by_hit.get(r.hit_type, 0) + 1
    print(
        f"sim done: {len(records)} reqs, wall_clock={wall_clock_s:.1f}s, "
        f"hits={by_hit}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
