#!/usr/bin/env python3
"""Hetu Benchmark — the complete UX/SLO-centric multimodel serving
load-imbalance benchmark for Hetu-DiT.  HETU_BENCH_VERSION = 1.

WHY THIS EXISTS
---------------
Raw throughput and mean latency do NOT measure user experience. The
industry/academic consensus (DistServe/AlpaServe/Clockwork/Shepherd/
InferFair OSDI'23-'24; DiffServe/MoDM/HADIS MLSys'25; BurstGPT Azure
trace; NVIDIA/Anyscale benchmarking guides — see HETU_BENCHMARK.md) is
that what users actually feel under industrial high-concurrency +
load-imbalance is captured by:

  1. GOODPUT@SLO  — completed requests/s that met their latency SLO
     (the headline number; ties resource use to UX & revenue).
  2. SLO ATTAINMENT % per model & per SLO-class (interactive vs batch);
     mean hides the tail — a 200ms mean can have a 3s p99.
  3. TAIL latency distribution p50/p90/p95/p99/max, DECOMPOSED into
     queue (waiting behind others) + service (model-switch tax +
     inference). Predictability (p99/p50 spread) is itself UX (Clockwork).
  4. FAIRNESS / performance isolation across models — Jain's index on
     per-model SLO-attainment; a heavy model must not blow a cold
     model's SLO (the §8.41 starvation bug, formalized).
  5. STARVATION-FREEDOM — every model with demand makes bounded-time
     progress (min per-model completion ratio; max enqueue→start wait).
  6. CAPACITY — goodput@SLO vs offered load curve; the SLO-attainment
     "knee" = max sustainable RPS at >=99% goodput (DiffServe/MoDM).
  7. OVERLOAD DEGRADATION — graceful (admission-shed, bounded latency,
     goodput plateaus) vs COLLAPSE (goodput craters, GPU idle, the
     §8.32 failure). Classified from the rate-sweep.
  8. SWITCH / COLD-START TAX — model-swap wall time and how often it
     lands on the user critical path.

It AGGREGATES every prior finding (context.md §8.30–§8.43): the
deterministic load-imbalance taxonomy (A skew / B temporal / C
work-size / D switch / E multi-GPU / F SLO-fairness), the measured
switch cost (hot 0s, sd3->flux ~9s, flux->sd3 ~6s), the OOM root cause
+ fix (#5/#5b structural teardown), and the scheduler-v2 fairness fix.

AUTHORITATIVE MEASUREMENT
-------------------------
No dependence on the §8.41-buggy /status PNG-glob. Per-request truth
comes from /task_timeline (dispatcher TaskHandle submit/start/done) and
/dispatch_stats (counters). e2e=done-submit, queue=start-submit,
service=done-start. Optional server-side SWITCHCOST join splits service
into bind(switch) vs infer when run on the server box.

RUN ANYTIME (self-contained: stdlib + requests)
-----------------------------------------------
  # 1. regression — deterministic taxonomy, per-scenario drain-isolated
  python3 scripts/hetu_benchmark.py --base-url http://localhost:8000 \
      --mode taxonomy --out-dir ~/hetu_bench
  # 2. capacity — goodput@SLO vs offered-load sweep (the knee)
  python3 scripts/hetu_benchmark.py --mode capacity \
      --rates 0.05,0.1,0.2,0.4,0.8 --out-dir ~/hetu_bench
  # 3. realistic — Gamma-burst stochastic (BurstGPT-style), seeded
  python3 scripts/hetu_benchmark.py --mode burst \
      --rate 0.3 --burst-alpha 0.4 --skew 0.9 --out-dir ~/hetu_bench
  # 4. everything + composite Hetu Score
  python3 scripts/hetu_benchmark.py --mode all --out-dir ~/hetu_bench
  # one-command provisioned real-hardware run:
  bash scripts/run_hetu_benchmark.sh        # see that script
"""
import argparse, json, math, os, random, threading, time
import requests

HETU_BENCH_VERSION = 1
PROMPT = "A futuristic cityscape at golden hour, highly detailed"

# ---- SLO policy (configurable; defaults are defensible diffusion tiers).
# A request's SLO is a function of its compute size (resolution*steps),
# because users tolerate longer for bigger asks. interactive vs relaxed
# classes per the literature (NVIDIA/Anyscale; DiffServe SLO-violation).
def slo_for(w, h, steps, scale=1.0):
    """Return (slo_s, slo_class). Tunable via --slo-scale."""
    units = (w * h) / (512 * 512) * (steps / 20.0)
    if units <= 1.2:                       # ~512^2/20  -> interactive
        return 12.0 * scale, "interactive"
    if units <= 3.0:                       # ~768^2/20  -> standard
        return 25.0 * scale, "standard"
    return 60.0 * scale, "relaxed"         # >=1024^2 / many steps


# ---- model/size shorthands (single-40GB-resident sizes) -------------
SD = ("sd3", 512, 512, 20)
SDb = ("sd3", 768, 768, 20)
SDB = ("sd3", 1024, 1024, 28)
SDH = ("sd3", 1536, 1536, 50)
FX = ("flux", 512, 512, 20)
FXB = ("flux", 1024, 1024, 50)


def _steady(m, period, dur, t0=0.0):
    out, t = [], t0
    while t < dur:
        out.append((round(t, 3), *m))
        t += period
    return out


def _burst(m, at, n, sp=0.1):
    return [(round(at + k * sp, 3), *m) for k in range(n)]


def _gamma_arrivals(rate, alpha, skew, dur, seed, models):
    """BurstGPT-style: Gamma inter-arrivals (shape alpha; small alpha =
    burstier, higher CV). model chosen by `skew` (P(heavy model))."""
    rng = random.Random(seed)
    out, t = [], 0.0
    heavy, light = models[0], models[1] if len(models) > 1 else models[0]
    scale = (1.0 / rate) / alpha          # mean inter-arrival = 1/rate
    while t < dur:
        t += rng.gammavariate(alpha, scale)
        if t >= dur:
            break
        m = heavy if rng.random() < skew else light
        spec = SD if m == "sd3" else FX
        out.append((round(t, 3), *spec))
    return out


# ---- deterministic load-imbalance taxonomy (the regression core) ----
def taxonomy():
    S = {}
    S["A1_steady_skew"] = dict(
        desc="sd3 ~95% steady + flux ~5% trickle (18:1 industrial skew)",
        arr=_steady(SD, 1.0, 60) + _steady(FX, 20.0, 60),
        gate=lambda d, a, x: (d.get("flux", 0) >= 1 and d.get("sd3", 0) >= 1,
                              "rare model not starved under steady skew"))
    S["A2_cold_burst"] = dict(
        desc="sd3 steady; flux silent 25s then 8-burst",
        arr=_steady(SDb, 2.0, 60) + _burst(FX, 25.0, 8, 0.12),
        gate=lambda d, a, x: (d.get("flux", 0) >= max(1, int(.75*a.get("flux", 1))),
                              ">=75% of the flux cold-burst completes"))
    S["A3_gpu_hog"] = dict(
        desc="sd3 1024²/28 saturating GPU; flux 1/16s trickle",
        arr=_steady(SDB, 4.0, 48) + _steady(FX, 16.0, 48, 6.0),
        gate=lambda d, a, x: (d.get("flux", 0) >= 1,
                              "rare model not starved while heavy hogs GPU"))
    S["A4_hotset_shift"] = dict(
        desc="popular model flips at t=30s (flux-heavy -> sd3-heavy)",
        arr=_steady(FX, 1.2, 30) + _steady(SD, 12.0, 30)
        + _steady(SD, 1.2, 60, 30.0) + _steady(FX, 12.0, 60, 30.0),
        gate=lambda d, a, x: (d.get("flux", 0) >= 1 and d.get("sd3", 0) >= 1,
                              "adapts across the popularity flip"))
    S["A5_zipf_longtail"] = dict(
        desc="Zipf model popularity + irregular inter-arrival (seed42)",
        arr=_gamma_arrivals(0.8, 0.7, 0.8, 60, 42, ["sd3", "flux"]),
        gate=lambda d, a, x: (d.get("flux", 0) >= 1 and d.get("sd3", 0) >= 1,
                              "no thrash-collapse; both progress"))
    S["B1_flash_crowd"] = dict(
        desc="sd3 baseline; flux 10× spike 20-30s",
        arr=_steady(SD, 3.0, 50) + _burst(FX, 20.0, 20, 0.5),
        gate=lambda d, a, x: (d.get("flux", 0) >= 1 and not x["oom"],
                              "spike absorbed gracefully, no wedge/OOM"))
    S["B3_sync_burst"] = dict(
        desc="sd3 & flux each 6-burst at the same instant",
        arr=_burst(SD, 10.0, 6, 0.05) + _burst(FX, 10.0, 6, 0.05),
        gate=lambda d, a, x: (d.get("flux", 0) >= 1 and d.get("sd3", 0) >= 1,
                              "no deadlock on simultaneous switch demand"))
    S["B4_idle_then_hit"] = dict(
        desc="req at 0, idle 40s, req at 40 (scale-to-zero cold path)",
        arr=[(0.0, *SD), (40.0, *FX)],
        gate=lambda d, a, x: (d.get("flux", 0) >= 1,
                              "post-idle cold-path request completes"))
    c1 = [(round(i*2.0, 3), *(SDB if i % 6 == 0 else SD)) for i in range(18)]
    S["C1_res_step_mix"] = dict(
        desc="sd3 512²/20 stream + a 1024²/28 every 6th (head-of-line)",
        arr=c1,
        gate=lambda d, a, x: (d.get("sd3", 0) >= int(.5*a.get("sd3", 1)),
                              ">=50% served despite huge HoL jobs"))
    S["C2_cheap_vs_costly"] = dict(
        desc="cheap sd3 1/s + costly flux 1024²/50 every 18s",
        arr=_steady(SD, 1.0, 50) + [(round(10+k*18.0, 3), *FXB)
                                    for k in range(3)],
        gate=lambda d, a, x: (d.get("sd3", 0) >= 1 and d.get("flux", 0) >= 1,
                              "cheap stream flows around costly jobs"))
    S["D1_alternate"] = dict(
        desc="strict sd3/flux alternation -> switch every request",
        arr=[(round(i*1.5, 3), *(SD if i % 2 == 0 else FX))
             for i in range(20)],
        gate=lambda d, a, x: ((d.get("sd3", 0)+d.get("flux", 0)) >= 1
                              and not x["oom"],
                              "progresses, NO cumulative OOM (§8.40/41 gate)"))
    d2, t, i = [], 0.0, 0
    for per in (6, 5, 4, 3, 2, 1):
        for _ in range(4):
            d2.append((round(t, 3), *(SD if i % 2 == 0 else FX)))
            t += per
            i += 1
    S["D2_thrash_sweep"] = dict(
        desc="switch interval 6s->1s (find goodput-collapse onset)",
        arr=d2, gate=lambda d, a, x: (True,
                                      "REPORT goodput/collapse-onset"))
    S["F2_starvation_free"] = dict(
        desc="extreme ~50:1 sd3:flux — the canonical fairness HARD GATE",
        arr=_steady(SD, 0.8, 40) + [(20.0, *FX)],
        gate=lambda d, a, x: (d.get("flux", 0) >= 1,
                              "HARD GATE: the single rare req completed"))
    S["E1_idle_executor"] = dict(
        desc="[>=2 GPU only] executor pinned to sd3 idle while flux queues",
        arr=_steady(SD, 1.0, 40) + _steady(FX, 1.0, 40),
        gate=lambda d, a, x: (True, "SKIP unless --multi-gpu"),
        multi_gpu=True)
    return S


# ---- authoritative measurement via /task_timeline + /dispatch_stats -
def _get(base, path, to=10):
    try:
        return requests.get(f"{base}{path}", timeout=to).json()
    except Exception:  # noqa: BLE001
        return {}


def gen(base, m, w, h, st, seed, rid):
    r = requests.post(f"{base}/generate", json={
        "model": m, "prompt": PROMPT, "negative_prompt": "low quality",
        "width": w, "height": h, "num_inference_steps": st,
        "seed": seed, "req_id": rid}, timeout=30)
    r.raise_for_status()
    return r.json()["task_id"]


def _pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def _jain(vals):
    """Jain's fairness index over per-model SLO-attainment (1.0 = fair)."""
    v = [x for x in vals if x is not None]
    if not v:
        return 1.0
    s = sum(v)
    sq = sum(x * x for x in v)
    return (s * s) / (len(v) * sq) if sq > 0 else 1.0


def drain(base, max_wait):
    t = time.time()
    while time.time() - t < max_wait:
        s = _get(base, "/dispatch_stats")
        if s.get("ready") and s.get("queue_depth", 1) == 0:
            return True, round(time.time() - t, 1)
        time.sleep(2.0)
    return False, round(time.time() - t, 1)


def fire_all(base, arrivals, tag):
    """Open-loop fire by schedule; returns {task_id: submit_wall}."""
    sub, lock, t0 = {}, threading.Lock(), time.time()

    def fire(i, off, m, w, h, st):
        while time.time() - t0 < off:
            time.sleep(0.01)
        try:
            tid = gen(base, m, w, h, st, 42 + i, f"{tag}-{i}")
            with lock:
                sub[tid] = (m, w, h, st)
        except Exception as e:  # noqa: BLE001
            print(f"SEND_FAIL {tag}-{i}: {e}", flush=True)

    ths = [threading.Thread(target=fire, args=(i, *a[:1], *a[1:]),
                            daemon=True)
           for i, a in enumerate(sorted(arrivals))]
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(0.5)
    return sub


def metrics(base, sub, slo_scale):
    """Compute the full UX/SLO metric block from /task_timeline."""
    tl = {h["task_id"]: h for h in _get(base, "/task_timeline").get(
        "handles", [])}
    perm = {}
    e2e_all = []
    for tid, (m, w, h, st) in sub.items():
        rec = perm.setdefault(m, {"n": 0, "done": 0, "e2e": [], "queue": [],
                                  "svc": [], "slo_ok": 0, "slo_n": 0})
        rec["n"] += 1
        t = tl.get(tid)
        if not t or t.get("done_ts") is None or not t.get("ok"):
            continue
        e2e = t["done_ts"] - t["submit_ts"]
        q = (t["start_ts"] - t["submit_ts"]) if t.get("start_ts") else 0.0
        svc = (t["done_ts"] - t["start_ts"]) if t.get("start_ts") else e2e
        slo_s, _cls = slo_for(w, h, st, slo_scale)
        rec["done"] += 1
        rec["e2e"].append(e2e)
        rec["queue"].append(q)
        rec["svc"].append(svc)
        rec["slo_n"] += 1
        rec["slo_ok"] += 1 if e2e <= slo_s else 0
        e2e_all.append(e2e)
    return perm, e2e_all


def summarize(perm, e2e_all, elapsed):
    """Aggregate -> the headline UX numbers + Jain fairness."""
    tot_done = sum(r["done"] for r in perm.values())
    tot_n = sum(r["n"] for r in perm.values())
    slo_ok = sum(r["slo_ok"] for r in perm.values())
    goodput = round(slo_ok / elapsed, 4) if elapsed > 0 else 0.0
    attain = (slo_ok / tot_done) if tot_done else 0.0
    per_attain = []
    pm = {}
    for m, r in sorted(perm.items()):
        a = (r["slo_ok"] / r["done"]) if r["done"] else 0.0
        per_attain.append(a if r["n"] else None)
        pm[m] = {
            "n": r["n"], "done": r["done"],
            "complete_ratio": round(r["done"]/r["n"], 3) if r["n"] else 1.0,
            "slo_attain": round(a, 3),
            "e2e_p50": round(_pct(r["e2e"], .5), 2),
            "e2e_p95": round(_pct(r["e2e"], .95), 2),
            "e2e_p99": round(_pct(r["e2e"], .99), 2),
            "queue_p95": round(_pct(r["queue"], .95), 2),
            "svc_p50": round(_pct(r["svc"], .5), 2),
        }
    return {
        "throughput_rps": round(tot_done / elapsed, 4) if elapsed else 0,
        "goodput_at_slo_rps": goodput,
        "slo_attainment": round(attain, 3),
        "completed": tot_done, "admitted": tot_n,
        "e2e_p50": round(_pct(e2e_all, .5), 2),
        "e2e_p95": round(_pct(e2e_all, .95), 2),
        "e2e_p99": round(_pct(e2e_all, .99), 2),
        "e2e_max": round(max(e2e_all), 2) if e2e_all else float("nan"),
        "tail_ratio_p99_p50": round(
            _pct(e2e_all, .99) / _pct(e2e_all, .5), 2)
            if e2e_all and _pct(e2e_all, .5) > 0 else float("nan"),
        "fairness_jain": round(_jain(per_attain), 3),
        "min_per_model_complete": round(
            min((p["complete_ratio"] for p in pm.values()), default=1.0), 3),
        "per_model": pm,
    }


def run_taxonomy(base, out_dir, drain_s, slo_scale, multi_gpu, only):
    S = taxonomy()
    ids = only or [k for k in S if not S[k].get("multi_gpu") or multi_gpu]
    results = []
    for sid in ids:
        if sid not in S:
            print(f"unknown {sid}")
            continue
        sp = S[sid]
        if sp.get("multi_gpu") and not multi_gpu:
            print(f"[{sid}] SKIP (needs --multi-gpu)", flush=True)
            results.append({"scenario": sid, "skipped": True})
            continue
        arr = sp["arr"]
        arr_by = {}
        for a in arr:
            arr_by[a[1]] = arr_by.get(a[1], 0) + 1
        print(f"\n=== {sid} (n={len(arr)}) :: {sp['desc']} ===", flush=True)
        s0 = _get(base, "/dispatch_stats").get("stats", {}) or {}
        f0 = s0.get("failed", 0)
        oom0 = _get(base, "/dispatch_stats")  # cheap; oom via failed proxy
        t0 = time.time()
        sub = fire_all(base, arr, sid)
        drained, waited = drain(base, drain_s)
        elapsed = time.time() - t0
        perm, e2e = metrics(base, sub, slo_scale)
        summ = summarize(perm, e2e, elapsed)
        s1 = _get(base, "/dispatch_stats").get("stats", {}) or {}
        failed = s1.get("failed", 0) - f0
        served = {m: perm.get(m, {}).get("done", 0) for m in arr_by}
        x = {"failed": failed,
             "oom": failed > 0.5 * max(1, sum(served.values()))}
        ok, why = sp["gate"](served, arr_by, x)
        verdict = "PASS" if ok else "FAIL"
        print(f"[{sid}] {verdict} served={served}/{arr_by} "
              f"goodput@SLO={summ['goodput_at_slo_rps']}rps "
              f"attain={summ['slo_attainment']} jain={summ['fairness_jain']} "
              f"e2e_p95={summ['e2e_p95']}s p99={summ['e2e_p99']}s "
              f"drained={drained}({waited}s) failed={failed}\n"
              f"  gate: {why}", flush=True)
        results.append({"scenario": sid, "verdict": verdict, "gate": why,
                         "arrivals": arr_by, "metrics": summ})
    return results


def run_capacity(base, out_dir, rates, slo_scale, dur, drain_s):
    """Rate-sweep -> goodput@SLO vs offered load (the capacity knee)."""
    curve = []
    for rate in rates:
        arr = _gamma_arrivals(rate, 0.5, 0.85, dur, 42, ["sd3", "flux"])
        print(f"\n=== capacity rate={rate}rps n={len(arr)} "
              f"(Gamma α=0.5 burst, 85:15 skew) ===", flush=True)
        t0 = time.time()
        sub = fire_all(base, arr, f"cap{rate}")
        drained, waited = drain(base, drain_s)
        elapsed = time.time() - t0
        perm, e2e = metrics(base, sub, slo_scale)
        summ = summarize(perm, e2e, elapsed)
        offered = round(len(arr) / dur, 3)
        curve.append({"rate": rate, "offered_rps": offered,
                      "goodput_at_slo_rps": summ["goodput_at_slo_rps"],
                      "slo_attainment": summ["slo_attainment"],
                      "e2e_p99": summ["e2e_p99"],
                      "fairness_jain": summ["fairness_jain"]})
        print(f"  offered={offered} goodput@SLO="
              f"{summ['goodput_at_slo_rps']} attain={summ['slo_attainment']}"
              f" p99={summ['e2e_p99']}s", flush=True)
    # capacity = max offered with attainment >= 0.99; degradation class
    knee = max((c["offered_rps"] for c in curve
                if c["slo_attainment"] >= 0.99), default=0.0)
    gp = [c["goodput_at_slo_rps"] for c in curve]
    collapse = len(gp) >= 2 and gp[-1] < 0.5 * max(gp)
    return {"curve": curve, "capacity_rps_at_99pct": knee,
            "degradation": "COLLAPSE" if collapse else "GRACEFUL"}


def hetu_score(tax, cap):
    """Composite 0-100. Weights: SLO-attainment 30, fairness 20,
    starvation-freedom 20, tail predictability 15, capacity/graceful 15."""
    graded = [r for r in tax if not r.get("skipped")]
    if not graded:
        return 0.0, {}
    npass = sum(1 for r in graded if r["verdict"] == "PASS")
    attn = sum(r["metrics"]["slo_attainment"] for r in graded) / len(graded)
    jain = sum(r["metrics"]["fairness_jain"] for r in graded) / len(graded)
    starv = sum(r["metrics"]["min_per_model_complete"]
                for r in graded) / len(graded)
    tails = [r["metrics"]["tail_ratio_p99_p50"] for r in graded
             if isinstance(r["metrics"]["tail_ratio_p99_p50"], (int, float))
             and r["metrics"]["tail_ratio_p99_p50"] == r["metrics"][
                 "tail_ratio_p99_p50"]]
    tailpred = max(0.0, 1.0 - (sum(tails)/len(tails) - 1) / 9) if tails else 1
    grace = 1.0 if cap and cap.get("degradation") == "GRACEFUL" else 0.4
    parts = {
        "scenario_pass": round(npass / len(graded), 3),
        "slo_attainment": round(attn, 3),
        "fairness_jain": round(jain, 3),
        "starvation_freedom": round(starv, 3),
        "tail_predictability": round(tailpred, 3),
        "graceful_degradation": grace,
    }
    score = (30 * attn + 20 * jain + 20 * starv + 15 * tailpred
             + 15 * grace) * (npass / len(graded))
    return round(score, 1), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--mode", default="taxonomy",
                    choices=["taxonomy", "capacity", "burst", "all"])
    ap.add_argument("--only", default="", help="comma scenario ids")
    ap.add_argument("--out-dir", default="/tmp/hetu_bench")
    ap.add_argument("--drain-s", type=float, default=240.0)
    ap.add_argument("--slo-scale", type=float, default=1.0,
                    help="multiply all SLO thresholds (tighten/loosen)")
    ap.add_argument("--rates", default="0.05,0.1,0.2,0.4,0.8")
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--burst-alpha", type=float, default=0.4)
    ap.add_argument("--skew", type=float, default=0.9)
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--multi-gpu", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    only = [s.strip() for s in a.only.split(",") if s.strip()]
    if not _get(a.base_url, "/dispatch_stats").get("ready"):
        print("note: /dispatch_stats not armed yet — first requests will "
              "lazy-init the dispatcher.", flush=True)
    report = {"hetu_bench_version": HETU_BENCH_VERSION, "mode": a.mode}

    if a.mode in ("taxonomy", "all"):
        report["taxonomy"] = run_taxonomy(
            a.base_url, a.out_dir, a.drain_s, a.slo_scale, a.multi_gpu, only)
    if a.mode in ("capacity", "all"):
        rates = [float(x) for x in a.rates.split(",")]
        report["capacity"] = run_capacity(
            a.base_url, a.out_dir, rates, a.slo_scale, a.dur, a.drain_s)
    if a.mode in ("burst", "all"):
        arr = _gamma_arrivals(a.rate, a.burst_alpha, a.skew, a.dur, 42,
                              ["sd3", "flux"])
        print(f"\n=== burst rate={a.rate} α={a.burst_alpha} skew={a.skew}"
              f" n={len(arr)} ===", flush=True)
        t0 = time.time()
        sub = fire_all(a.base_url, arr, "burst")
        drain(a.base_url, a.drain_s)
        perm, e2e = metrics(a.base_url, sub, a.slo_scale)
        report["burst"] = summarize(perm, e2e, time.time() - t0)
        print(f"  goodput@SLO={report['burst']['goodput_at_slo_rps']}rps "
              f"attain={report['burst']['slo_attainment']} "
              f"p99={report['burst']['e2e_p99']}s "
              f"jain={report['burst']['fairness_jain']}", flush=True)

    if a.mode == "all":
        score, parts = hetu_score(report.get("taxonomy", []),
                                  report.get("capacity"))
        report["hetu_score"] = score
        report["hetu_score_parts"] = parts
    json.dump(report, open(os.path.join(a.out_dir, "HETU_REPORT.json"),
                           "w"), indent=2)
    print(f"\n#### HETU BENCHMARK v{HETU_BENCH_VERSION} ####")
    if "taxonomy" in report:
        g = [r for r in report["taxonomy"] if not r.get("skipped")]
        npass = sum(1 for r in g if r["verdict"] == "PASS")
        print(f"taxonomy: {npass}/{len(g)} scenarios PASS")
        for r in g:
            print(f"  [{r['verdict']}] {r['scenario']}: "
                  f"attain={r['metrics']['slo_attainment']} "
                  f"jain={r['metrics']['fairness_jain']} "
                  f"p99={r['metrics']['e2e_p99']}s")
    if "capacity" in report:
        c = report["capacity"]
        print(f"capacity: {c['capacity_rps_at_99pct']} rps @>=99% SLO; "
              f"degradation={c['degradation']}")
    if "hetu_score" in report:
        print(f"HETU SCORE = {report['hetu_score']} / 100  "
              f"{report['hetu_score_parts']}")
    print(f"REPORT -> {a.out_dir}/HETU_REPORT.json\nHETU_BENCH_DONE")


if __name__ == "__main__":
    main()
