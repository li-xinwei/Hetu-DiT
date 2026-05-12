#!/usr/bin/env python3
"""Real-mode bench driver for Hetu-DiT bare-process eval on RunPod.

We can't run K3s on a community RunPod pod (nested overlayfs prohibited),
so we run the api_server as a plain Python process. The pre-warm pool
mechanism still tests at the Ray-actor layer (`l2_pool_enabled` flag,
`Worker.l2_state` state machine, `_find_warm_l2_executor` dispatch logic).

What we measure per CELL:
  - apply_to_ready_s: time from `python3 -m hetu_dit.entrypoint.api_server`
    to /readyz green. Captures cold-start cost.
  - per-request latency: time from /generate POST to status==completed,
    for both a single sequential request AND a burst of N concurrent requests.

Cells:
  cold_seq:    no l2 pool, 1 request (baseline cold-start)
  cold_burst4: no l2 pool, 4 concurrent (baseline serialized)
  l2_seq:      --l2_pool_enabled, 1 request (warm-bind path)
  l2_burst4:   --l2_pool_enabled, 4 concurrent (warm-pool parallel)

The 4-GPU community RTX 3090 pod runs sp=1 cfg=1 per instance, so up to 4
instances can run in parallel. With l2_pool_enabled, idle workers are
pre-L2'd; dispatcher routes concurrent requests to separate L2 executors.

Inputs to the script via env vars:
  MODEL_PATH=/root/models/stable-diffusion-3-medium-diffusers
  RESULTS_DIR=/root/eval-results

Output: one subdir per cell containing api.log, cstrace.log, request_log.json,
summary.txt — matching the format scripts/aggregate_runs.py consumes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def wait_for_readyz(port: int, timeout_s: float) -> Optional[float]:
    """Poll /readyz on localhost:PORT. Return seconds-to-ready or None on timeout."""
    import urllib.request
    start = time.perf_counter()
    deadline = start + timeout_s
    while time.perf_counter() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=5)
            return time.perf_counter() - start
        except Exception:
            time.sleep(2)
    return None


def send_generate(port: int, req_id: str, prompt: str, height: int = 1024, width: int = 1024,
                  steps: int = 20, seed: int = 42) -> dict:
    """Send /generate, poll /status until completed. Return timings."""
    import urllib.request
    import urllib.error
    body = json.dumps({
        "req_id": req_id,
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "num_frames": 0,
        "seed": seed,
    }).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        body, {"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=600).read()
    t_ack = time.perf_counter() - t0
    j = json.loads(resp.decode())
    task_id = j.get("task_id", req_id)
    # Poll status
    final = None
    for _ in range(600):
        try:
            r = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/status/{task_id}", timeout=10,
            ).read()
            d = json.loads(r.decode())
            if d.get("status") in ("completed", "failed"):
                final = d
                break
        except Exception:
            pass
        time.sleep(1)
    t_done = time.perf_counter() - t0
    return {
        "req_id": req_id,
        "task_id": task_id,
        "ack_s": t_ack,
        "total_s": t_done,
        "final": final,
    }


def burst_generate(port: int, n: int, prompt_prefix: str = "a serene mountain landscape") -> List[dict]:
    """Fire N requests concurrently via threads."""
    import concurrent.futures
    results = [None] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futures = {
            ex.submit(
                send_generate, port, f"{prompt_prefix.replace(' ', '_')}_{i}",
                f"{prompt_prefix}, variant {i}", 1024, 1024, 20, 42 + i,
            ): i for i in range(n)
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"req_id": f"req_{i}", "error": str(e)}
    return results


def start_api_server(
    out_dir: Path,
    model_path: str,
    n_gpus: int,
    l2_pool: bool,
    port: int = 8000,
    cstrace: bool = True,
) -> Tuple[subprocess.Popen, Path]:
    """Spawn api_server with given config. Return (process, log_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "api.log"
    env = os.environ.copy()
    env["HETU_COLDSTART_TRACE"] = "1" if cstrace else ""
    env["HETUDIT_MODEL_CLASS"] = "sd3"
    env["HETUDIT_MODEL"] = model_path
    env["HETUDIT_MACHINE_NUMS"] = "1"
    env["HETUDIT_EXPECTED_GPUS"] = str(n_gpus)
    cmd = [
        "python3", "-m", "hetu_dit.entrypoint.api_server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--stage_level",
    ]
    if l2_pool:
        cmd.append("--l2_pool_enabled")
    f = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                            preexec_fn=os.setsid)
    return proc, log_path


def stop_api_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def extract_cstrace(api_log: Path, out_path: Path) -> int:
    """Pull [CSTRACE] lines from api.log into a separate file. Return count."""
    count = 0
    with api_log.open("r", encoding="utf-8", errors="replace") as f, out_path.open("w") as o:
        for line in f:
            if line.startswith("[CSTRACE]"):
                o.write(line)
                count += 1
    return count


def run_cell(cell_name: str, out_root: Path, model_path: str, n_gpus: int,
             l2_pool: bool, burst_n: int, timeout_ready_s: float = 600) -> dict:
    cell_dir = out_root / cell_name
    cell_dir.mkdir(parents=True, exist_ok=True)
    port = 8000

    print(f"\n[{cell_name}] starting api_server (l2_pool={l2_pool}, gpus={n_gpus})", flush=True)
    proc, api_log = start_api_server(cell_dir, model_path, n_gpus, l2_pool, port=port)

    try:
        ready_s = wait_for_readyz(port, timeout_ready_s)
        if ready_s is None:
            print(f"[{cell_name}] /readyz timeout after {timeout_ready_s}s", flush=True)
            return {"cell": cell_name, "error": "readyz_timeout"}
        print(f"[{cell_name}] /readyz green after {ready_s:.2f}s", flush=True)

        # Single sequential request first.
        seq = send_generate(port, f"{cell_name}_seq", "a photo of a cat in a hat")
        print(f"[{cell_name}] seq request: total={seq['total_s']:.2f}s", flush=True)

        # Then burst.
        if burst_n > 0:
            print(f"[{cell_name}] burst of {burst_n} concurrent...", flush=True)
            burst = burst_generate(port, burst_n, "a forest landscape")
            for i, r in enumerate(burst):
                if r.get("total_s") is not None:
                    print(f"[{cell_name}]   req {i}: total={r['total_s']:.2f}s", flush=True)
        else:
            burst = []

        # Snapshot /metrics.
        try:
            import urllib.request
            metrics_json = json.loads(
                urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
            )
        except Exception as e:
            metrics_json = {"error": str(e)}

        result = {
            "cell": cell_name,
            "l2_pool": l2_pool,
            "n_gpus": n_gpus,
            "apply_to_ready_s": ready_s,
            "sequential": seq,
            "burst_n": burst_n,
            "burst_results": burst,
            "final_metrics": metrics_json,
        }
        (cell_dir / "request_log.json").write_text(json.dumps(result, indent=2, default=str))

        # Extract cstrace separately for the aggregator.
        n_cstrace = extract_cstrace(api_log, cell_dir / "cstrace.log")
        print(f"[{cell_name}] cstrace lines: {n_cstrace}", flush=True)

        # Write extended summary.txt for aggregator compatibility.
        seq_total = seq.get("total_s", 0.0)
        burst_p95 = sorted([r["total_s"] for r in burst if "total_s" in r])[
            int(0.95 * len(burst)) - 1
        ] if burst else 0.0
        summary = (
            f"cell={cell_name}\n"
            f"workload_pattern={'burst-n' + str(burst_n) if burst_n else 'cold'}\n"
            f"pool_size={n_gpus}\n"
            f"head_size=1\n"
            f"warm_replicas={n_gpus - 1 if l2_pool else 0}\n"
            f"model=sd3\n"
            f"n_requests_planned={1 + burst_n}\n"
            f"n_requests_served={1 + len([r for r in burst if 'total_s' in r])}\n"
            f"seed=0\n"
            f"wall_clock_s={ready_s + seq_total + (burst_p95 if burst else 0)}\n"
            f"mode=real\n"
            f"l2_pool_enabled={'true' if l2_pool else 'false'}\n"
            f"apply_to_ready_s={ready_s:.3f}\n"
            f"sequential_request_total_s={seq_total:.3f}\n"
            f"burst_p95_total_s={burst_p95:.3f}\n"
        )
        (cell_dir / "summary.txt").write_text(summary)
        return result

    finally:
        print(f"[{cell_name}] stopping api_server", flush=True)
        stop_api_server(proc)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/root/models/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--out", default="/root/eval-results")
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--burst", type=int, default=4)
    parser.add_argument("--timeout-ready", type=float, default=600)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cells = [
        ("01_cold_baseline_no_l2",  False),
        ("02_l2_pool_enabled",      True),
    ]
    all_results = []
    for name, l2 in cells:
        r = run_cell(name, out_root, args.model_path, args.n_gpus,
                     l2, args.burst, args.timeout_ready)
        all_results.append(r)

    (out_root / "ALL_CELLS.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nALL DONE -> {out_root}/ALL_CELLS.json")


if __name__ == "__main__":
    main()
