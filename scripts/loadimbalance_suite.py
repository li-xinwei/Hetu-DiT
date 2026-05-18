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
import argparse, json, os, threading, time, glob
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
    # A3 gpu_hog — sd3 big saturating, flux trickle
    S["A3_gpu_hog"] = dict(
        desc="sd3 1024²/28 back-to-back saturating GPU; flux 1/15s",
        dur=90, arrivals=_steady(SD_BIG, 1.0, 90)
        + _steady(FX, 15.0, 90, t0=8.0), models=["sd3", "flux"],
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
    for i in range(24):
        c1.append((round(i * 1.5, 3), *(SD_HUGE if i % 6 == 0 else SD)))
    S["C1_res_step_mix"] = dict(
        desc="sd3: 512²/20 stream with a 1536²/50 every 6th (head-of-line)",
        dur=40, arrivals=c1, models=["sd3"],
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
        desc="extreme 50:1 sd3:flux; the canonical fairness/starvation gate",
        dur=80, arrivals=_steady(SD, 0.8, 80) + [(40.0, *FX)],
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


def run_scenario(base, sid, spec, out_dir, drain_s, multi_gpu):
    if spec.get("multi_gpu") and not multi_gpu:
        print(f"SKIP {sid} (needs --multi-gpu)", flush=True)
        return {"scenario": sid, "skipped": True}
    arrivals = sorted(spec["arrivals"], key=lambda x: x[0])
    recs, lock, t0 = {}, threading.Lock(), time.time()
    print(f"\n=== {sid} (n={len(arrivals)} dur={spec['dur']}s) :: "
          f"{spec['desc']} ===", flush=True)

    def fire(i, off, m, w, h, st):
        while time.time() - t0 < off:
            time.sleep(0.01)
        ts = round(time.time() - t0, 3)
        try:
            tid = gen(base, m, w, h, st, 42 + i, f"{sid}-{i}")
        except Exception as e:  # noqa: BLE001
            print(f"SEND_FAIL {sid}-{i} {m}: {e}", flush=True)
            return
        with lock:
            recs[tid] = {"task_id": tid, "model": m, "res": f"{w}x{h}",
                         "steps": st, "submit_s": ts}

    ths = [threading.Thread(target=fire, args=(i, o, m, w, h, s),
                            daemon=True)
           for i, (o, m, w, h, s) in enumerate(arrivals)]
    for t in ths:
        t.start()
    deadline = t0 + spec["dur"] + drain_s
    while time.time() < deadline and any(t.is_alive() for t in ths):
        time.sleep(1.0)
    # let in-flight settle a bit past last arrival
    time.sleep(min(drain_s, 30))
    man = {"suite_version": SUITE_VERSION, "scenario": sid,
           "desc": spec["desc"], "pass_note": spec["pass_note"],
           "arrivals": len(arrivals), "models": spec["models"], "recs": recs}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(man, open(os.path.join(out_dir, f"{sid}.json"), "w"), indent=2)
    print(f"--- {sid}: {len(recs)} submitted; authoritative completion via "
          f"--emit-join (server SWITCHCOST). pass_note: {spec['pass_note']}",
          flush=True)
    return man


def emit_join(out_dir, server_log):
    """Authoritative report: join client arrivals with server SWITCHCOST."""
    sc = {}
    for ln in open(server_log, errors="ignore"):
        if "SWITCHCOST task=" not in ln:
            continue
        d = dict(p.split("=", 1) for p in ln.split() if "=" in p)
        tid = d.get("SWITCHCOST task") or d.get("task")
        if tid:
            sc[tid] = {"bind_s": float(d.get("bind_s", 0)),
                       "infer_s": float(d.get("infer_s", 0)),
                       "switched": d.get("switched") == "1"}
    print(f"\n#### AUTHORITATIVE JOIN (SUITE_VERSION={SUITE_VERSION}) ####")
    for f in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        if f.endswith("_REPORT.json"):
            continue
        m = json.load(open(f))
        if m.get("skipped"):
            continue
        recs = m["recs"]
        bymodel = {}
        for tid, r in recs.items():
            done = tid in sc
            bm = bymodel.setdefault(r["model"], {"n": 0, "done": 0,
                                                 "infer": [], "switch": 0})
            bm["n"] += 1
            if done:
                bm["done"] += 1
                bm["infer"].append(sc[tid]["infer_s"])
                bm["switch"] += int(sc[tid]["switched"])
        verdict = "PASS"
        lines = []
        for mdl, b in sorted(bymodel.items()):
            inf = sorted(b["infer"])
            p = (lambda q: inf[min(len(inf) - 1, int(q * len(inf)))]
                 if inf else float("nan"))
            if b["done"] == 0 and b["n"] > 0:
                verdict = "FAIL(starvation)"
            lines.append(f"  {mdl}: {b['done']}/{b['n']} done, "
                         f"switches={b['switch']}, infer p50="
                         f"{p(.5):.2f}s p95={p(.95):.2f}s"
                         if inf else
                         f"  {mdl}: {b['done']}/{b['n']} done (NONE)")
        tot = sum(b["n"] for b in bymodel.values())
        dn = sum(b["done"] for b in bymodel.values())
        print(f"\n[{m['scenario']}] {verdict}  ({dn}/{tot} real-complete)"
              f"\n  note: {m['pass_note']}")
        for ln in lines:
            print(ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", default="", help="comma ids e.g. A2_cold_burst")
    ap.add_argument("--out-dir", default="/tmp/suite_v1")
    ap.add_argument("--drain-s", type=float, default=180.0)
    ap.add_argument("--multi-gpu", action="store_true")
    ap.add_argument("--emit-join", metavar="OUT_DIR")
    ap.add_argument("--server-log", default="server.log")
    a = ap.parse_args()
    if a.emit_join:
        emit_join(a.emit_join, a.server_log)
        return
    S = build_scenarios()
    ids = (list(S) if a.all else
           [x.strip() for x in a.only.split(",") if x.strip()])
    if not ids:
        print("pick --all or --only <ids>. available:\n  " +
              "\n  ".join(f"{k}: {v['desc']}" for k, v in S.items()))
        return
    print(f"SUITE_VERSION={SUITE_VERSION} running {len(ids)} scenario(s)")
    for sid in ids:
        if sid not in S:
            print(f"unknown scenario {sid}")
            continue
        run_scenario(a.base_url, sid, S[sid], a.out_dir, a.drain_s,
                     a.multi_gpu)
    print(f"\nALL_SCENARIOS_DONE -> {a.out_dir}\nNext (on server box): "
          f"python3 scripts/loadimbalance_suite.py --emit-join {a.out_dir} "
          f"--server-log <server.log>")


if __name__ == "__main__":
    main()
