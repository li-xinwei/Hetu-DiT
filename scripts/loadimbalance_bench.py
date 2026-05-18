#!/usr/bin/env python3
"""Industrial load-imbalance latency-bottleneck probe (open-loop).

Goal (context.md §8.41): under high concurrency + load imbalance, find
*where* multimodel serving latency actually goes. Open-loop = arrivals
fire on a fixed schedule regardless of completion (the industrial case
Yifei specified: some models queue while others sit idle).

Three scenarios:
  S1 alternate-stress : sd3/flux switch on EVERY request (worst-case swap
                        pressure) — validates fix#4 (no cumulative OOM)
                        and yields clean per-switch bind cost.
  S2 coldburst        : heavy steady sd3 background; flux silent then a
                        sudden burst — measures the cold/rare model's
                        head-of-line wait behind the busy model.
  S3 gpuhog           : sd3 saturates the executor (big, back-to-back);
                        flux trickles — measures rare-model starvation.

Per request we record submit->done wall (e2e). The server prints
`SWITCHCOST task=<id> ... bind_s=<> infer_s=<>`; we join on task_id so
each request decomposes into:  e2e = wait(queue+switch-by-others)
                                     + bind_s(own switch) + infer_s.
That decomposition is the bottleneck answer.

Pure stdlib + requests. Usage:
  python3 scripts/loadimbalance_bench.py --base-url http://localhost:8000 \
     --scenario coldburst --out ~/S2.json
"""
import argparse, json, threading, time, sys
import requests

PROMPT = "A futuristic cityscape at golden hour, highly detailed"


def gen(base, model, w, h, steps, seed, req_id):
    # MUST send a unique req_id: the server builds
    # task_id = f"task-{req_id}_{model}_{w}x{h}"; without it req_id=None and
    # all same-(model,res) requests collide on one task_id (the §8.41
    # harness bug that collapsed 40 arrivals to ~2 records).
    t = requests.post(
        f"{base}/generate",
        json={
            "model": model,
            "prompt": PROMPT,
            "negative_prompt": "low quality, blurry",
            "width": w,
            "height": h,
            "num_inference_steps": steps,
            "seed": seed,
            "req_id": req_id,
        },
        timeout=30,
    )
    t.raise_for_status()
    return t.json()["task_id"]


def schedule(scenario, duration):
    """Return list of (fire_offset_s, model, w, h, steps) open-loop arrivals."""
    arr = []
    if scenario == "alternate":
        # one every 1.5s, strict sd3/flux alternation -> switch every req
        n = max(8, int(duration / 1.5))
        for i in range(n):
            arr.append((i * 1.5, "sd3" if i % 2 == 0 else "flux", 512, 512, 20))
    elif scenario == "coldburst":
        # sd3 steady 1/2s for the whole window; flux silent until t=25s
        # then 8 requests within 1s (the cold-model burst).
        t = 0.0
        while t < duration:
            arr.append((t, "sd3", 768, 768, 20))
            t += 2.0
        for k in range(8):
            arr.append((25.0 + k * 0.12, "flux", 512, 512, 20))
    elif scenario == "gpuhog":
        # sd3 saturating: back-to-back big jobs (1024^2, 28 steps) every
        # 1s for the window; flux rare: 1 every 15s (the starved model).
        t = 0.0
        while t < duration:
            arr.append((t, "sd3", 1024, 1024, 28))
            t += 1.0
        t = 8.0
        while t < duration:
            arr.append((t, "flux", 512, 512, 20))
            t += 15.0
    else:
        raise SystemExit(f"unknown scenario {scenario}")
    arr.sort(key=lambda x: x[0])
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument(
        "--scenario", required=True, choices=["alternate", "coldburst", "gpuhog"]
    )
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--drain-s", type=float, default=240.0)
    ap.add_argument("--poll-interval", type=float, default=0.5)
    ap.add_argument("--out", default="/tmp/loadimb.json")
    a = ap.parse_args()

    arrivals = schedule(a.scenario, a.duration)
    recs = {}  # task_id -> rec
    lock = threading.Lock()
    t0 = time.time()
    print(
        f"scenario={a.scenario} arrivals={len(arrivals)} "
        f"duration={a.duration}s drain={a.drain_s}s",
        flush=True,
    )

    def fire(idx, off, model, w, h, steps):
        while time.time() - t0 < off:
            time.sleep(0.01)
        st = round(time.time() - t0, 3)
        try:
            tid = gen(a.base_url, model, w, h, steps, 42 + idx, f"li{idx}")
        except Exception as e:  # noqa: BLE001
            print(f"SEND_FAIL idx={idx} model={model}: {e}", flush=True)
            return
        with lock:
            recs[tid] = {
                "task_id": tid,
                "model": model,
                "res": f"{w}x{h}",
                "steps": steps,
                "submit_s": st,
                "done_s": None,
                "e2e_s": None,
            }

    threads = [
        threading.Thread(target=fire, args=(i, off, m, w, h, s), daemon=True)
        for i, (off, m, w, h, s) in enumerate(arrivals)
    ]
    for t in threads:
        t.start()

    deadline = t0 + a.duration + a.drain_s
    while time.time() < deadline:
        with lock:
            pending = [r for r in recs.values() if r["done_s"] is None]
            total, done = len(recs), sum(
                1 for r in recs.values() if r["done_s"] is not None
            )
        if recs and done == len(arrivals):
            break
        for r in list(pending):
            try:
                s = requests.get(
                    f"{a.base_url}/status/{r['task_id']}", timeout=10
                ).json()
            except Exception:  # noqa: BLE001
                continue
            if s.get("status") in ("completed", "done", "finished", "success"):
                now = round(time.time() - t0, 3)
                with lock:
                    r["done_s"] = now
                    r["e2e_s"] = round(now - r["submit_s"], 3)
        print(
            f"[t={round(time.time()-t0,1)}s] sent={total} done={done}"
            f"/{len(arrivals)} pending={len(pending)}",
            flush=True,
        )
        time.sleep(a.poll_interval)

    json.dump(
        {"scenario": a.scenario, "arrivals": len(arrivals), "recs": recs},
        open(a.out, "w"),
        indent=2,
    )
    fin = [r for r in recs.values() if r["e2e_s"] is not None]
    print(f"\n=== {a.scenario}: {len(fin)}/{len(arrivals)} completed ===")
    for m in sorted({r["model"] for r in recs.values()}):
        e = sorted(r["e2e_s"] for r in fin if r["model"] == m)
        if not e:
            print(f"  {m}: 0 completed (STARVED/never finished in window)")
            continue
        p = lambda q: e[min(len(e) - 1, int(q * len(e)))]  # noqa: E731
        print(
            f"  {m}: n={len(e)} e2e p50={p(.5):.2f}s p95={p(.95):.2f}s "
            f"max={e[-1]:.2f}s min={e[0]:.2f}s"
        )
    print(f"JSON -> {a.out}\nBENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
