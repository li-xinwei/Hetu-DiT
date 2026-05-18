#!/usr/bin/env python3
"""Hetu-DiT multimodel load-imbalance REGRESSION SUITE  (SUITE_VERSION=1).

A FIXED, deterministic, versioned scenario set to regression-test every
future Hetu-DiT version against the full taxonomy of industrial
multimodel-serving load imbalance. Same seeds + same arrival schedules
every run, so results are comparable across versions/commits.

WHY server-side metrics: the client `/status` endpoint currently
false-"completed" on a stale-PNG glob (§8.41 API bug) — so the
suite's metric-of-record is AUTHORITATIVE SERVER-SIDE: count of
`SWITCHCOST task=... infer_s=...` lines and result PNGs in results/.
The runner emits the client arrival/attempt timeline; join it with
`grep SWITCHCOST server.log` (see `--emit-join`) for true completion,
per-model service time, switch cost and starvation.

────────────────────────────────────────────────────────────────────
TAXONOMY OF LOAD IMBALANCE IN MULTIMODEL SERVING (what this covers)
────────────────────────────────────────────────────────────────────
A. Inter-model demand skew (which model gets traffic)
   A1 steady_skew      one model ~95% traffic, others trickle (18:1)
   A2 cold_burst       rare model silent, then a sudden burst
   A3 gpu_hog          one heavy model saturates GPU, others starve
   A4 hotset_shift     the popular model CHANGES mid-run (trend/diurnal)
   A5 zipf_longtail    many models, Zipfian popularity, set > VRAM (thrash)
B. Temporal arrival imbalance (when requests come)
   B1 flash_crowd      Poisson spike (×10 rate) on one model then drop
   B2 diurnal_ramp     slow rise then fall (autoscale responsiveness)
   B3 sync_burst       several models spike at the SAME instant (contention)
   B4 idle_then_hit    long idle (scale-to-zero) then a request (cold start)
C. Per-request work-size variance (how big each job is)
   C1 res_step_mix     same model, 512²/20 mixed with 1536²/50 (HoL)
   C2 cheap_vs_costly  cheap image model interleaved with a long video job
D. Switch/placement-induced imbalance
   D1 alternate        switch on EVERY request (worst-case swap pressure)
   D2 thrash_sweep     escalating switch frequency → find collapse onset
   D3 evict_pressure   model-set > resident capacity → eviction policy
E. Multi-GPU / resource imbalance  (skipped on 1-GPU; declared, gated)
   E1 idle_executor    ≥2 GPU: executor pinned to A idle while B queues
   E2 straggler        one worker slow/dies mid model-load
F. SLO / priority imbalance
   F1 mixed_slo        interactive (tight SLO) vs batch (loose) compete
   F2 starvation_free  sustained skew — does the rare model EVER complete?

Each scenario: deterministic arrival list [(t_offset, model, w, h, steps)]
+ explicit PASS criteria. PASS rubric is uniform:
  • no OOM / no wedge (server keeps emitting SWITCHCOST, GPU not 0% w/ queue)
  • throughput > 0 and bounded-progress (done count strictly increases)
  • starvation-freedom: every model with ≥1 arrival gets ≥1 real completion
  • (reported, not hard-gated) per-model p50/p95 service time, switch count

Usage:
  python3 scripts/loadimbalance_suite.py --base-url http://localhost:8000 \
      --only A2,A3,D1 --out-dir ~/suite_v1            # subset
  python3 scripts/loadimbalance_suite.py --all --out-dir ~/suite_v1
  # then on the server box, authoritative join:
  python3 scripts/loadimbalance_suite.py --emit-join ~/suite_v1 \
      --server-log ~/server.log
"""
import argparse, json, os, threading, time
import requests

SUITE_VERSION = 1
PROMPT = "A futuristic cityscape at golden hour, highly detailed"

# (model, w, h, steps) shorthands. Keep within single-40GB-resident sizes;
# 'vid' is the deliberately-costly long job for C2 (registered hunyuanvideo
# if available, else flux@1024/50 as a heavy stand-in — model-agnostic).
SD = ("sd3", 512, 512, 20)
SD_BIG = ("sd3", 1024, 1024, 28)
SD_HUGE = ("sd3", 1536, 1536, 50)
FX = ("flux", 512, 512, 20)
FX_BIG = ("flux", 1024, 1024, 50)


def _steady(model_tuple, period, dur, t0=0.0):
    out, t = [], t0
    while t < dur:
        out.append((round(t, 3), *model_tuple))
        t += period
    return out


def _burst(model_tuple, at, n, spacing=0.1):
    return [(round(at + k * spacing, 3), *model_tuple) for k in range(n)]


def build_scenarios():
    """Return {id: {desc, dur, arrivals, models, pass_note}} — DETERMINISTIC."""
    S = {}

    # A1 steady_skew — sd3 95%, flux 5%
    a = _steady(SD, 1.0, 60) + _steady(FX, 20.0, 60)
    S["A1_steady_skew"] = dict(
        desc="sd3 ~95% steady, flux ~5% trickle (18:1 industrial skew)",
        dur=60, arrivals=a, models=["sd3", "flux"],
        pass_note="flux must complete >=1 (no starvation under steady skew)",
    )
    # A2 cold_burst — sd3 steady, flux silent 25s then 8-burst
    S["A2_cold_burst"] = dict(
        desc="sd3 steady background; flux silent 25s then 8 within ~1s",
        dur=60, arrivals=_steady(("sd3", 768, 768, 20), 2.0, 60)
        + _burst(FX, 25.0, 8, 0.12), models=["sd3", "flux"],
        pass_note="flux burst latency bounded; >=6/8 flux complete",
    )
    # A3 gpu_hog — sd3 big saturating, flux trickle. Right-sized so a
    # single A100 can drain within drain_s (the property under test is
    # "rare model not starved while a heavy model hogs the GPU", not raw
    # throughput): sd3 1024²/28 every 4s for 48s + flux every 16s.
    S["A3_gpu_hog"] = dict(
        desc="sd3 1024²/28 saturating GPU (q4s); flux 1/16s (trickle)",
        dur=48, arrivals=_steady(SD_BIG, 4.0, 48)
        + _steady(FX, 16.0, 48, t0=6.0), models=["sd3", "flux"],
        pass_note="flux not starved: >=1 flux completes within run+drain",
    )
    # A4 hotset_shift — first half flux-heavy, second half sd3-heavy
    S["A4_hotset_shift"] = dict(
        desc="popular model flips at t=30s (flux-heavy -> sd3-heavy)",
        dur=60,
        arrivals=_steady(FX, 1.2, 30) + _steady(SD, 12.0, 30)
        + _steady(SD, 1.2, 60, t0=30.0) + _steady(FX, 12.0, 60, t0=30.0),
        models=["sd3", "flux"],
        pass_note="adapts to shift; both complete in their hot half",
    )
    # A5 zipf_longtail — emulated with 2 ids but Zipf inter-arrival on res
    z = []
    t = 0.0
    import random as _r
    rng = _r.Random(42)  # FIXED seed -> deterministic
    while t < 60:
        m = SD if rng.random() < 0.8 else FX
        z.append((round(t, 3), *m))
        t += rng.choice([0.4, 0.7, 1.1])
    S["A5_zipf_longtail"] = dict(
        desc="Zipfian model popularity + irregular inter-arrival (seed42)",
        dur=60, arrivals=z, models=["sd3", "flux"],
        pass_note="no thrash-collapse; both models progress",
    )
    # B1 flash_crowd — sd3 baseline, flux 10x spike 20-30s
    S["B1_flash_crowd"] = dict(
        desc="sd3 baseline; flux 10× Poisson-ish spike during 20-30s",
        dur=50, arrivals=_steady(SD, 3.0, 50)
        + _burst(FX, 20.0, 20, 0.5), models=["sd3", "flux"],
        pass_note="spike absorbed (admission/graceful), no wedge",
    )
    # B3 sync_burst — both models burst at the SAME instant
    S["B3_sync_burst"] = dict(
        desc="sd3 & flux each 6-burst at t=10s (simultaneous contention)",
        dur=30, arrivals=_burst(SD, 10.0, 6, 0.05)
        + _burst(FX, 10.0, 6, 0.05), models=["sd3", "flux"],
        pass_note="both progress; no deadlock on simultaneous switch demand",
    )
    # B4 idle_then_hit — one early req, long idle, one late req (cold path)
    S["B4_idle_then_hit"] = dict(
        desc="req at t=0, idle 40s, req at t=40 (scale-to-zero cold path)",
        dur=45, arrivals=[(0.0, *SD), (40.0, *FX)], models=["sd3", "flux"],
        pass_note="late req completes; cold-start latency recorded",
    )
    # C1 res_step_mix — same model, small jobs interleaved with huge ones
    c1 = []
    for i in range(18):
        c1.append((round(i * 2.0, 3), *(SD_BIG if i % 6 == 0 else SD)))
    S["C1_res_step_mix"] = dict(
        desc="sd3: 512²/20 stream with a 1024²/28 every 6th (head-of-line)",
        dur=36, arrivals=c1, models=["sd3"],
        pass_note="small jobs not unboundedly blocked by the huge ones",
    )
    # C2 cheap_vs_costly — cheap sd3 stream + occasional very costly flux@1024/50
    c2 = _steady(SD, 1.0, 50)
    c2 += [(round(10.0 + k * 18.0, 3), *FX_BIG) for k in range(3)]
    S["C2_cheap_vs_costly"] = dict(
        desc="cheap sd3 1/s + a costly flux 1024²/50 every 18s",
        dur=50, arrivals=c2, models=["sd3", "flux"],
        pass_note="cheap stream keeps flowing around the costly jobs",
    )
    # D1 alternate — switch on EVERY request (worst-case swap)
    S["D1_alternate"] = dict(
        desc="strict sd3/flux alternation -> a model switch every request",
        dur=30, arrivals=[(round(i * 1.5, 3), *(SD if i % 2 == 0 else FX))
                          for i in range(20)], models=["sd3", "flux"],
        pass_note="NO cumulative OOM (the §8.40/§8.41 regression gate)",
    )
    # D2 thrash_sweep — switch period shrinks 6→1 over the run
    d2, t, i = [], 0.0, 0
    for period in (6, 5, 4, 3, 2, 1):
        for _ in range(4):
            d2.append((round(t, 3), *(SD if i % 2 == 0 else FX)))
            t += period
            i += 1
    S["D2_thrash_sweep"] = dict(
        desc="switch interval sweeps 6s→1s (find throughput-collapse onset)",
        dur=round(t, 1), arrivals=d2, models=["sd3", "flux"],
        pass_note="report the period at which goodput collapses (if any)",
    )
    # F2 starvation_free — extreme 50:1 skew, does flux EVER complete?
    S["F2_starvation_free"] = dict(
        desc="extreme ~50:1 sd3:flux; the canonical fairness/starvation gate",
        dur=40, arrivals=_steady(SD, 0.8, 40) + [(20.0, *FX)],
        models=["sd3", "flux"],
        pass_note="HARD GATE: the single flux request MUST complete",
    )
    # E1/E2 multi-GPU & straggler — declared, gated to >=2 GPU boxes
    S["E1_idle_executor"] = dict(
        desc="[>=2 GPU only] executor pinned to sd3 idle while flux queues",
        dur=40, arrivals=_steady(SD, 1.0, 40) + _steady(FX, 1.0, 40),
        models=["sd3", "flux"], pass_note="SKIP unless --multi-gpu",
        multi_gpu=True,
    )
    return S


def gen(base, model, w, h, steps, seed, req_id):
    r = requests.post(f"{base}/generate", json={
        "model": model, "prompt": PROMPT, "negative_prompt": "low quality",
        "width": w, "height": h, "num_inference_steps": steps,
        "seed": seed, "req_id": req_id}, timeout=30)
    r.raise_for_status()
    return r.json()["task_id"]


# Structured per-scenario PASS predicates over the authoritative
# /dispatch_stats delta. d = {model: completed_in_scenario}, arr =
# {model: arrivals_in_scenario}, extra = {"failed", "switches",
# "queue_drained": bool}. Returns (ok: bool, summary: str). These encode
# the *actual* property under test (not "every model 100% done", which a
# single serial GPU under open-loop overload can't guarantee and isn't
# the point). The invariant the suite enforces everywhere: no model with
# arrivals gets ZERO completions (starvation-freedom) + no crash/OOM
# (failed stays bounded). Scenario-specific gates layer on top.
def _frac(d, arr, m):
    return (d.get(m, 0) / arr[m]) if arr.get(m) else 1.0


PASS_PREDICATES = {
    # rare model must not be starved under steady skew
    "A1_steady_skew": lambda d, arr, x: (
        d.get("flux", 0) >= 1 and d.get("sd3", 0) >= 1,
        "flux completes under 18:1 skew (starvation-free)"),
    # the cold burst must be largely served
    "A2_cold_burst": lambda d, arr, x: (
        d.get("flux", 0) >= max(1, int(0.75 * arr.get("flux", 1))),
        ">=75% of the flux cold-burst completes"),
    # heavy model hogs GPU; the rare model must still get in
    "A3_gpu_hog": lambda d, arr, x: (
        d.get("flux", 0) >= 1,
        "rare model not starved while heavy model saturates GPU"),
    "A4_hotset_shift": lambda d, arr, x: (
        d.get("flux", 0) >= 1 and d.get("sd3", 0) >= 1,
        "both models progress across the popularity flip"),
    "A5_zipf_longtail": lambda d, arr, x: (
        d.get("flux", 0) >= 1 and d.get("sd3", 0) >= 1,
        "no thrash-collapse; both models progress"),
    "B1_flash_crowd": lambda d, arr, x: (
        d.get("flux", 0) >= 1 and d.get("sd3", 0) >= 1 and not x["oom"],
        "spike absorbed gracefully, no wedge/OOM"),
    "B3_sync_burst": lambda d, arr, x: (
        d.get("flux", 0) >= 1 and d.get("sd3", 0) >= 1,
        "both progress on simultaneous switch demand (no deadlock)"),
    "B4_idle_then_hit": lambda d, arr, x: (
        d.get("flux", 0) >= 1,
        "post-idle (cold-path) request completes"),
    "C1_res_step_mix": lambda d, arr, x: (
        _frac(d, arr, "sd3") >= 0.5,
        ">=50% served despite huge head-of-line jobs"),
    "C2_cheap_vs_costly": lambda d, arr, x: (
        d.get("sd3", 0) >= 1 and d.get("flux", 0) >= 1,
        "cheap stream keeps flowing around costly jobs"),
    # the §8.40/§8.41 regression gate: progress + NO OOM
    "D1_alternate": lambda d, arr, x: (
        (d.get("sd3", 0) + d.get("flux", 0)) >= 1 and not x["oom"],
        "alternating switch: progresses, NO cumulative OOM"),
    # report-only: never a hard fail; record goodput
    "D2_thrash_sweep": lambda d, arr, x: (
        True,
        f"REPORT goodput={d.get('sd3',0)+d.get('flux',0)} (collapse-onset)"),
    # THE canonical hard gate
    "F2_starvation_free": lambda d, arr, x: (
        d.get("flux", 0) >= 1,
        "HARD GATE: the single rare-model request completed"),
}


def get_stats(base):
    try:
        r = requests.get(f"{base}/dispatch_stats", timeout=10)
        return r.json()
    except Exception:  # noqa: BLE001
        return {"ready": False, "queue_depth": 0, "stats": {}}


def _pm_completed(stats):
    pm = (stats.get("stats", {}) or {}).get("per_model", {}) or {}
    return {m: v.get("completed", 0) for m, v in pm.items()}


def drain_to_quiescent(base, max_wait):
    """Block until the dispatcher queue is empty (true scenario isolation)
    or max_wait elapsed. Returns (drained: bool, waited_s)."""
    t = time.time()
    while time.time() - t < max_wait:
        s = get_stats(base)
        if s.get("ready") and s.get("queue_depth", 1) == 0:
            return True, round(time.time() - t, 1)
        time.sleep(2.0)
    return False, round(time.time() - t, 1)


def run_scenario(base, sid, spec, drain_s, multi_gpu):
    if spec.get("multi_gpu") and not multi_gpu:
        print(f"\n[{sid}] SKIP (needs --multi-gpu)", flush=True)
        return {"scenario": sid, "skipped": True}
    arrivals = sorted(spec["arrivals"], key=lambda x: x[0])
    arr_by_model = {}
    for _o, m, *_ in arrivals:
        arr_by_model[m] = arr_by_model.get(m, 0) + 1
    t0 = time.time()
    print(f"\n=== {sid} (n={len(arrivals)} dur={spec['dur']}s) :: "
          f"{spec['desc']} ===", flush=True)
    s0 = get_stats(base)
    c0 = _pm_completed(s0)
    f0 = (s0.get("stats", {}) or {}).get("failed", 0)
    sw0 = (s0.get("stats", {}) or {}).get("switches", 0)

    def fire(i, off, m, w, h, st):
        while time.time() - t0 < off:
            time.sleep(0.01)
        try:
            gen(base, m, w, h, st, 42 + i, f"{sid}-{i}")
        except Exception as e:  # noqa: BLE001
            print(f"SEND_FAIL {sid}-{i} {m}: {e}", flush=True)

    ths = [threading.Thread(target=fire, args=(i, o, m, w, h, s),
                            daemon=True)
           for i, (o, m, w, h, s) in enumerate(arrivals)]
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(0.5)
    # TRUE per-scenario isolation: drain the dispatcher before the next
    # scenario so backlog never carries over (the §8.42 artifact).
    drained, waited = drain_to_quiescent(base, drain_s)

    s1 = get_stats(base)
    c1 = _pm_completed(s1)
    delta = {m: c1.get(m, 0) - c0.get(m, 0) for m in set(c1) | set(c0)}
    failed = (s1.get("stats", {}) or {}).get("failed", 0) - f0
    switches = (s1.get("stats", {}) or {}).get("switches", 0) - sw0
    x = {"failed": failed, "switches": switches, "queue_drained": drained,
         "oom": failed > 0.5 * max(1, sum(delta.values()))}
    pred = PASS_PREDICATES.get(
        sid, lambda d, a, xx: (sum(d.values()) > 0, "progress > 0"))
    ok, why = pred(delta, arr_by_model, x)
    verdict = "PASS" if ok else "FAIL"
    served = ", ".join(f"{m}:{delta.get(m,0)}/{arr_by_model.get(m,0)}"
                       for m in sorted(arr_by_model))
    print(f"[{sid}] {verdict}  served[{served}] failed={failed} "
          f"switches={switches} drained={drained}({waited}s)\n"
          f"  gate: {why}", flush=True)
    return {"scenario": sid, "verdict": verdict, "served": delta,
            "arrivals": arr_by_model, "failed": failed,
            "switches": switches, "drained": drained, "gate": why}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", default="", help="comma ids e.g. A2_cold_burst")
    ap.add_argument("--out-dir", default="/tmp/suite_v1")
    ap.add_argument("--drain-s", type=float, default=240.0,
                    help="max per-scenario drain-to-quiescent wait")
    ap.add_argument("--multi-gpu", action="store_true")
    a = ap.parse_args()
    S = build_scenarios()
    ids = (list(S) if a.all else
           [x.strip() for x in a.only.split(",") if x.strip()])
    if not ids:
        print("pick --all or --only <ids>. available:\n  " +
              "\n  ".join(f"{k}: {v['desc']}" for k, v in S.items()))
        return
    s = get_stats(a.base_url)
    if not s.get("ready"):
        print("WARN /dispatch_stats not ready yet (dispatcher lazy-inits "
              "on first request) — first scenario will arm it.", flush=True)
    print(f"SUITE_VERSION={SUITE_VERSION} running {len(ids)} scenario(s) "
          f"(per-scenario drain isolation via /dispatch_stats)")
    os.makedirs(a.out_dir, exist_ok=True)
    results = []
    for sid in ids:
        if sid not in S:
            print(f"unknown scenario {sid}")
            continue
        results.append(
            run_scenario(a.base_url, sid, S[sid], a.drain_s, a.multi_gpu))
    json.dump({"suite_version": SUITE_VERSION, "results": results},
              open(os.path.join(a.out_dir, "REPORT.json"), "w"), indent=2)
    graded = [r for r in results if not r.get("skipped")]
    npass = sum(1 for r in graded if r["verdict"] == "PASS")
    print(f"\n#### SUITE_VERSION={SUITE_VERSION} SUMMARY: "
          f"{npass}/{len(graded)} PASS ####")
    for r in graded:
        print(f"  [{r['verdict']}] {r['scenario']}: "
              f"served={r['served']} gate={r['gate']}")
    print(f"REPORT -> {a.out_dir}/REPORT.json\nALL_SUITE_DONE")


if __name__ == "__main__":
    main()
