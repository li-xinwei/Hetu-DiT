#!/usr/bin/env python3
"""Idle-GPU-waste vs warm-start tradeoff probe (Hetu Benchmark add-on).

The core residency tradeoff in multimodel serving on a FIXED GPU set:

  POLE A  per-model DEDICATED — each model owns a GPU, always resident.
          latency = pure queue+infer (ZERO model-switch tax) BUT a
          model's card sits IDLE (wasted) whenever it has no traffic.
  POLE B  SHARED / time-multiplexed swap (Hetu-DiT's serial dispatcher,
          opt#1) — ZERO idle (one card, always working) BUT every
          cross-model request pays the ~6-9s warm-start swap (§8.31:
          weights are page-cache-warm so it's a RAM→GPU PCIe H2D, not a
          from-disk cold start).

This script drives the SAME industrial traffic at both poles and
quantifies BOTH arms so the tradeoff is explicit:
  • latency: e2e p50/p95/p99 + switch-tax share (authoritative, via
    each server's /task_timeline — no /status, no log scrape).
  • idle waste: GPU-idle fraction over the run, from a server-side
    `nvidia-smi` utilisation sampler (idle_vs_warmstart_gpusample.sh),
    joined to the measured run window. Pole A's wasted GPU-seconds is
    the price of its zero-switch latency; Pole B's is ~0.

POLE A topology (needs >=2 GPU): two single-model servers, one per
GPU/model:
  CUDA_VISIBLE_DEVICES=0 ... --models sd3=...  --port 8000
  CUDA_VISIBLE_DEVICES=1 ... --models flux=... --port 8001
  python3 idle_vs_warmstart.py --pole A --sd3-url :8000 --flux-url :8001
POLE B: the normal opt#1 server (one process), --pole B --url :8000.

Same Gamma-burst industrial workload + refpoint conditions as
hetu_benchmark v2 so results are directly comparable.
"""
import argparse, json, os, threading, time, random
import requests

PROMPT = "A futuristic cityscape at golden hour, highly detailed"
SD = ("sd3", 512, 512, 20)
FX = ("flux", 512, 512, 20)
SDb = ("sd3", 768, 768, 20)


def _gamma(rate, alpha, skew, dur, seed):
    rng = random.Random(seed)
    out, t, scale = [], 0.0, (1.0 / rate) / alpha
    while t < dur:
        t += rng.gammavariate(alpha, scale)
        if t >= dur:
            break
        out.append((round(t, 3), *(SD if rng.random() < skew else FX)))
    return out


def _steady(m, period, dur, t0=0.0):
    out, t = [], t0
    while t < dur:
        out.append((round(t, 3), *m))
        t += period
    return out


def workloads():
    # the industrial conditions where the tradeoff bites (skew + the
    # rare model whose dedicated card would idle).
    return {
        "steady_skew_18to1": _steady(SD, 1.0, 60) + _steady(FX, 20.0, 60),
        "cold_burst": _steady(SDb, 2.0, 60)
        + [(round(25 + k * 0.12, 3), *FX) for k in range(8)],
        "gamma_industrial": _gamma(0.3, 0.4, 0.85, 60, 42),
    }


def gen(url, m, w, h, st, seed, rid):
    r = requests.post(f"{url}/generate", json={
        "model": m, "prompt": PROMPT, "negative_prompt": "low quality",
        "width": w, "height": h, "num_inference_steps": st,
        "seed": seed, "req_id": rid}, timeout=30)
    r.raise_for_status()
    return r.json()["task_id"]


def _pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 3)


def _route(pole, m, urls):
    # Pole A: sd3->sd3 server, flux->flux server. Pole B: single url.
    if pole == "A":
        return urls["sd3"] if m == "sd3" else urls["flux"]
    return urls["one"]


def drain(url, max_wait):
    t = time.time()
    while time.time() - t < max_wait:
        s = requests.get(f"{url}/dispatch_stats", timeout=10).json() \
            if True else {}
        try:
            if s.get("ready") and s.get("queue_depth", 1) == 0:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)
    return False


def run_workload(pole, name, arr, urls, drain_s):
    sub, lock, t0 = {}, threading.Lock(), time.time()
    print(f"\n=== POLE {pole} :: {name} (n={len(arr)}) ===", flush=True)

    def fire(i, off, m, w, h, st):
        while time.time() - t0 < off:
            time.sleep(0.01)
        u = _route(pole, m, urls)
        try:
            tid = gen(u, m, w, h, st, 42 + i, f"{name}-{i}")
            with lock:
                sub[tid] = (u, m)
        except Exception as e:  # noqa: BLE001
            print(f"SEND_FAIL {name}-{i}: {e}", flush=True)

    ths = [threading.Thread(target=fire, args=(i, a[0], a[1], a[2], a[3],
                                               a[4]), daemon=True)
           for i, a in enumerate(sorted(arr))]
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(0.5)
    for u in set(urls.values()):
        drain(u, drain_s)
    wall = time.time() - t0

    # authoritative per-request timeline from each server involved
    tl = {}
    for u in set(urls.values()):
        try:
            for h in requests.get(f"{u}/task_timeline",
                                  timeout=10).json().get("handles", []):
                tl[h["task_id"]] = h
        except Exception:  # noqa: BLE001
            pass
    e2e, sw, q, per = [], [], [], {}
    for tid, (u, m) in sub.items():
        h = tl.get(tid)
        if not h or h.get("done_ts") is None or not h.get("ok"):
            continue
        _e = h["done_ts"] - h["submit_ts"]
        _q = (h["start_ts"] - h["submit_ts"]) if h.get("start_ts") else 0.0
        _s = h.get("bind_s") or 0.0
        e2e.append(_e)
        q.append(_q)
        sw.append(_s)
        per.setdefault(m, []).append(_e)
    return {
        "pole": pole, "workload": name, "wall_s": round(wall, 1),
        "n": len(arr), "completed": len(e2e),
        "e2e_p50": _pct(e2e, .5), "e2e_p95": _pct(e2e, .95),
        "e2e_p99": _pct(e2e, .99),
        "switch_share_pct": round(100 * sum(sw) / sum(e2e), 1)
        if e2e and sum(e2e) else 0.0,
        "switch_p95_s": _pct(sw, .95),
        "per_model_e2e_p95": {m: _pct(v, .95) for m, v in per.items()},
    }


def gpu_idle_from_sampler(path, t_start_epoch, t_end_epoch):
    """idle fraction per GPU from the server-side nvidia-smi sampler
    (lines: 'epoch,gpu_index,util%'). Pole A's idle = the wasted-card
    price; Pole B's ≈ 0."""
    if not path or not os.path.exists(path):
        return {}
    perg = {}
    for ln in open(path, errors="ignore"):
        p = ln.strip().split(",")
        if len(p) < 3:
            continue
        try:
            ep, gi, ut = float(p[0]), int(p[1]), float(p[2])
        except ValueError:
            continue
        if t_start_epoch <= ep <= t_end_epoch:
            d = perg.setdefault(gi, [0, 0])
            d[0] += 1
            d[1] += 1 if ut < 5.0 else 0      # <5% util == idle
    return {gi: round(v[1] / v[0], 3) for gi, v in perg.items() if v[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pole", required=True, choices=["A", "B"])
    ap.add_argument("--url", default="http://localhost:8000",
                    help="Pole B single server")
    ap.add_argument("--sd3-url", default="http://localhost:8000")
    ap.add_argument("--flux-url", default="http://localhost:8001")
    ap.add_argument("--drain-s", type=float, default=180.0)
    ap.add_argument("--gpu-sample", default="",
                    help="server-side nvidia-smi sample file (epoch,idx,util)")
    ap.add_argument("--out", default="/tmp/idle_vs_warm.json")
    a = ap.parse_args()
    urls = ({"sd3": a.sd3_url, "flux": a.flux_url} if a.pole == "A"
            else {"one": a.url})
    res, t0 = [], time.time()
    for name, arr in workloads().items():
        res.append(run_workload(a.pole, name, arr, urls, a.drain_s))
    t1 = time.time()
    idle = gpu_idle_from_sampler(a.gpu_sample, t0, t1)
    rep = {"pole": a.pole, "results": res,
           "gpu_idle_fraction": idle,
           "wasted_gpu_seconds_est": round(
               sum(idle.values()) * (t1 - t0), 1) if idle else None}
    json.dump(rep, open(a.out, "w"), indent=2)
    print(f"\n#### IDLE-vs-WARMSTART  POLE {a.pole} ####")
    for r in res:
        print(f"  {r['workload']}: e2e p50={r['e2e_p50']}s "
              f"p95={r['e2e_p95']}s p99={r['e2e_p99']}s | "
              f"switch_share={r['switch_share_pct']}% "
              f"sw_p95={r['switch_p95_s']}s | "
              f"per-model p95={r['per_model_e2e_p95']}")
    print(f"  GPU idle fraction = {idle}  "
          f"(wasted GPU-s ≈ {rep['wasted_gpu_seconds_est']})")
    print(f"REPORT -> {a.out}\nIDLE_VS_WARM_DONE")


if __name__ == "__main__":
    main()
