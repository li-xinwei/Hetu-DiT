#!/usr/bin/env python3
"""Open-loop baseline-pathology benchmark for multi-model serving.

The closed-loop sibling (coldstart_slo_bench.py) deliberately removed
queueing to isolate the *pure* cold-start tax (result: ~0 in steady state,
absorbed by the CPU SingletonModelManager + OS page-cache).

This script does the opposite on purpose: it replays the mentor's merged
multi-model trace **open-loop** (fire requests at their scaled arrival
times without waiting for completion). Concurrency emerges naturally and
per-model queues form. It exists to characterize the *baseline pathology*
of the current PR-1/PR-2 dispatcher under Yifei's scenario:

    "different models have different workload sizes -- some models'
     requests queue up while other models sit idle"

The current dispatcher (get_ready_executor_or_reconfigure) has no
load-aware reallocation: a hot model's requests pile into
waiting_reconfigure_tasks while an idle model's executor stays "ready"
holding its worker/GPU slot, and nothing moves that slot to the hot
model. We measure exactly that: per-model outstanding-request count,
per-model end-to-end latency / SLO over time.

Metrics emitted (the motivation-figure data)
--------------------------------------------
- per-request: scheduled_ts, send_ts, done_ts, latency, ok
- per-model time series @1s: sent / done / outstanding (queue proxy)
- summary: per-model p50/p95/p99 latency, SLO pass %, max outstanding
Reload events are correlated post-hoc from the server log
("Loading pipeline components" timestamps) by the caller.
"""
import argparse
import json
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

_RE = re.compile(
    r"request_id=(\d+), timestamp=([\d.]+), height=(\d+), width=(\d+), "
    r"num_frames=(\d+).*?num_inference_steps=(\d+)"
)


def load_trace(model, path):
    out = []
    with open(path) as fh:
        for line in fh:
            m = _RE.search(line)
            if not m:
                continue
            rid, ts, h, w, nf, steps = m.groups()
            out.append(
                {
                    "model": model,
                    "ts": float(ts),
                    "height": int(h),
                    "width": int(w),
                    "num_frames": int(nf),
                    "num_inference_steps": int(steps),
                }
            )
    return out


def build_timeline(trace_dir, models, rate_scale, duration, max_requests):
    merged = []
    for mdl in models:
        merged += load_trace(mdl, f"{trace_dir}/{mdl}_trace.txt")
    merged.sort(key=lambda r: r["ts"])
    out = []
    for i, r in enumerate(merged):
        sched = r["ts"] / rate_scale  # seconds-from-start to fire this req
        if duration and sched > duration:
            break
        if max_requests and i >= max_requests:
            break
        r["seq"] = i
        r["sched"] = sched
        out.append(r)
    return out


class Bench:
    def __init__(self, base, slo_interactive, slo_relaxed):
        self.base = base
        self.slo_i = slo_interactive
        self.slo_r = slo_relaxed
        self.lock = threading.Lock()
        self.records = {}  # seq -> record dict
        self.timeseries = []
        self._stop = threading.Event()

    def _send(self, req):
        seq = req["seq"]
        body = json.dumps(
            {
                "model": req["model"],
                "prompt": "A futuristic cityscape",
                "negative_prompt": "low quality, blurry",
                "height": req["height"],
                "width": req["width"],
                "num_frames": req["num_frames"],
                "num_inference_steps": req["num_inference_steps"],
                "seed": 42,
                "req_id": f"ol{seq}",
            }
        ).encode()
        rec = {
            "seq": seq,
            "model": req["model"],
            "res": f"{req['width']}x{req['height']}",
            "steps": req["num_inference_steps"],
            "sched": round(req["sched"], 3),
            "send_ts": round(time.time() - self.t0, 3),
            "done_ts": None,
            "latency_s": None,
            "ok": False,
            "task_id": None,
        }
        try:
            r = urllib.request.Request(
                f"{self.base}/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(r, timeout=30) as resp:
                d = json.loads(resp.read())
            rec["task_id"] = d.get("task_id")
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"send:{e}"
        with self.lock:
            self.records[seq] = rec

    def _poller(self, poll_interval):
        while not self._stop.is_set():
            with self.lock:
                pending = [
                    r
                    for r in self.records.values()
                    if r["task_id"] and r["done_ts"] is None
                ]
            for r in pending:
                try:
                    with urllib.request.urlopen(
                        f"{self.base}/status/{r['task_id']}", timeout=10
                    ) as resp:
                        d = json.loads(resp.read())
                    if d.get("image_url"):
                        now = time.time() - self.t0
                        with self.lock:
                            r["done_ts"] = round(now, 3)
                            r["latency_s"] = round(now - r["send_ts"], 3)
                            r["ok"] = True
                except urllib.error.URLError:
                    pass
            time.sleep(poll_interval)

    def _sampler(self, models, sample_interval):
        while not self._stop.is_set():
            t = round(time.time() - self.t0, 2)
            row = {"t": t}
            with self.lock:
                recs = list(self.records.values())
            for m in models:
                mr = [r for r in recs if r["model"] == m]
                sent = len(mr)
                done = sum(1 for r in mr if r["ok"])
                row[m] = {"sent": sent, "done": done, "outstanding": sent - done}
            self.timeseries.append(row)
            time.sleep(sample_interval)

    def run(self, timeline, models, poll_interval, sample_interval, drain_s):
        self.t0 = time.time()
        pol = threading.Thread(target=self._poller, args=(poll_interval,), daemon=True)
        smp = threading.Thread(
            target=self._sampler, args=(models, sample_interval), daemon=True
        )
        pol.start()
        smp.start()
        # open-loop producer: fire each request at its scheduled wall time
        with ThreadPoolExecutor(max_workers=64) as pool:
            for req in timeline:
                wait = req["sched"] - (time.time() - self.t0)
                if wait > 0:
                    time.sleep(wait)
                pool.submit(self._send, req)
                print(
                    f"  fire seq={req['seq']:4d} {req['model']:5s} "
                    f"{req['width']}x{req['height']} @sched={req['sched']:.2f}s",
                    flush=True,
                )
        # drain: keep polling until all done or drain timeout
        deadline = time.time() + drain_s
        while time.time() < deadline:
            with self.lock:
                left = sum(
                    1
                    for r in self.records.values()
                    if r["task_id"] and r["done_ts"] is None
                )
            if left == 0:
                break
            time.sleep(2)
        self._stop.set()
        time.sleep(0.5)


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return round(xs[k], 3)


def summarize(records, models, slo_i, slo_r):
    out = {"per_model": {}, "global": {}}
    allok = [r for r in records if r["ok"]]
    for m in models:
        mr = [r for r in records if r["model"] == m]
        ok = [r for r in mr if r["ok"]]
        lats = [r["latency_s"] for r in ok]
        out["per_model"][m] = {
            "n": len(mr),
            "completed": len(ok),
            "failed": len(mr) - len(ok),
            "p50_s": _pct(lats, 50),
            "p95_s": _pct(lats, 95),
            "p99_s": _pct(lats, 99),
            "max_s": round(max(lats), 3) if lats else None,
            "slo_interactive_pass_pct": (
                round(100.0 * sum(1 for x in lats if x <= slo_i) / len(lats), 1)
                if lats
                else None
            ),
            "slo_relaxed_pass_pct": (
                round(100.0 * sum(1 for x in lats if x <= slo_r) / len(lats), 1)
                if lats
                else None
            ),
        }
    glats = [r["latency_s"] for r in allok]
    out["global"] = {
        "n": len(records),
        "completed": len(allok),
        "p50_s": _pct(glats, 50),
        "p95_s": _pct(glats, 95),
        "p99_s": _pct(glats, 99),
        "slo_interactive_threshold_s": slo_i,
        "slo_relaxed_threshold_s": slo_r,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--trace-dir", default="data")
    ap.add_argument("--models", default="sd3,flux")
    ap.add_argument(
        "--rate-scale",
        type=float,
        default=0.1,
        help="send req at ts/rate_scale; <1 slows arrivals (runnable), "
        "sweep upward to find SLO-collapse / thrashing onset",
    )
    ap.add_argument("--duration", type=float, default=120.0,
                    help="cap replay to first N (scaled) seconds")
    ap.add_argument("--max-requests", type=int, default=400)
    ap.add_argument("--poll-interval", type=float, default=0.5)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--drain-s", type=float, default=600.0,
                    help="after last send, keep polling up to this long")
    ap.add_argument("--slo-interactive", type=float, default=10.0)
    ap.add_argument("--slo-relaxed", type=float, default=30.0)
    ap.add_argument("--out", default="openloop_result.json")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    tl = build_timeline(
        a.trace_dir, models, a.rate_scale, a.duration, a.max_requests
    )
    by_model = defaultdict(int)
    for r in tl:
        by_model[r["model"]] += 1
    span = tl[-1]["sched"] if tl else 0
    print(
        f"open-loop timeline: {len(tl)} reqs over {span:.1f}s "
        f"(rate_scale={a.rate_scale}) per-model={dict(by_model)}",
        flush=True,
    )
    b = Bench(a.base_url, a.slo_interactive, a.slo_relaxed)
    b.run(tl, models, a.poll_interval, a.sample_interval, a.drain_s)
    recs = [b.records[k] for k in sorted(b.records)]
    summary = summarize(recs, models, a.slo_interactive, a.slo_relaxed)
    with open(a.out, "w") as fh:
        json.dump(
            {"summary": summary, "timeseries": b.timeseries, "records": recs},
            fh,
            indent=2,
        )
    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))
    # quick pathology readout: peak outstanding per model
    peak = {m: 0 for m in models}
    for row in b.timeseries:
        for m in models:
            peak[m] = max(peak[m], row.get(m, {}).get("outstanding", 0))
    print("peak outstanding per model:", peak)
    print(f"full result -> {a.out}")


if __name__ == "__main__":
    main()
