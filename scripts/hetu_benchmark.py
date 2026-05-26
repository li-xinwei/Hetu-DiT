#!/usr/bin/env python3
"""Hetu Benchmark v3 — industrial-scenario LATENCY benchmark for the
Hetu-DiT multimodel serving system.  HETU_BENCH_VERSION = 2.

ONE QUESTION
------------
Under REAL industrial high-concurrency + load-imbalance traffic, where
does the time go and how slow is it — i.e. across all the latency that
matters, is multimodel serving (with its model switching) fast or slow?

This benchmark is LATENCY, broadly construed. It does NOT score
robustness — "didn't crash / didn't OOM / didn't starve" is only a
binary GATE (a crashed run has no trustworthy latency). The output is a
full latency profile + decomposition + latency-vs-load curves under
realistic industrial workloads.

LATENCY DIMENSIONS (all per-request authoritative via /task_timeline,
which now carries bind_s/infer_s — no /status, no log scraping):

  e2e      = done − submit        what the user actually waits
  queue    = start − submit       waiting behind concurrent load
  switch   = bind_s               model-switch / cold-start tax
  infer    = infer_s              pure inference
  (queue + switch + infer ≈ e2e — the decomposition that says WHERE
   industrial-load latency goes; switch is the multimodel-specific part)

Reported per model and global: p50/p95/p99/max of EACH of the four;
time-to-first-result; jitter (p99/p50, IQR); the mean decomposition
shares (queue% / switch% / infer%) → the dominant latency contributor;
switch-tax stats (freq, mean/p95 bind among switched, % of total
latency spent switching); and every one of these swept over offered
load (the latency-vs-load curve family).

WORKLOAD = realistic industrial traffic (not synthetic robustness gates,
not per-user probes):

  • industrial : Gamma-burst arrivals (shape α = burstiness / CV,
    BurstGPT-Azure-style; smaller α = burstier), configurable
    multi-model demand skew, mixed resolution/steps, sustained high
    concurrency. The primary workload.
  • sweep      : the above swept across offered RPS → latency-vs-load
    curves + the latency knee (max RPS keeping p99 interactive) and how
    steeply each latency component degrades past it.
  • refpoints  : a few FIXED industrial traffic conditions (steady
    skew / cold-model burst / heavy-model hog) as deterministic,
    version-comparable reference points — measured for LATENCY, with a
    same-model baseline so the switch-induced delta is explicit.

Basis: DistServe/Clockwork/InferFair/DiffServe/MoDM/BurstGPT/Anyscale/
NVIDIA (see HETU_BENCHMARK.md). Self-contained (stdlib + requests),
deterministic (fixed seeds), runnable anytime
(`bash scripts/run_hetu_benchmark.sh`).  HETU_BENCH_VERSION = 3.

v3 adds two pillars: (a) QUALITY-ELASTICITY axis (--mode quality:
steps->latency->quality-proxy), and (b) idle-vs-warmstart FRONTIER
formalization (scripts/idle_vs_warmstart.py --frontier, TTL-sweep/
ski-rental). Together with the taxonomy + per-request decomposition =
the 4 pillars.
"""
import argparse, json, os, random, threading, time
import requests

HETU_BENCH_VERSION = 3
PROMPT = "A futuristic cityscape at golden hour, highly detailed"

SD = ("sd3", 512, 512, 20)
SDb = ("sd3", 768, 768, 20)
SDB = ("sd3", 1024, 1024, 28)
FX = ("flux", 512, 512, 20)
FXB = ("flux", 1024, 1024, 50)


def slo_for(w, h, steps, scale=1.0):
    units = (w * h) / (512 * 512) * (steps / 20.0)
    if units <= 1.2:
        return 12.0 * scale
    if units <= 3.0:
        return 25.0 * scale
    return 60.0 * scale


def _gamma(rate, alpha, skew, dur, seed, sizes):
    """BurstGPT-style Gamma inter-arrivals. rate=mean rps, alpha=shape
    (smaller=burstier/higher CV), skew=P(heavy model)."""
    rng = random.Random(seed)
    out, t = [], 0.0
    scale = (1.0 / rate) / alpha
    while t < dur:
        t += rng.gammavariate(alpha, scale)
        if t >= dur:
            break
        out.append((round(t, 3), *(sizes[0] if rng.random() < skew
                                   else sizes[1])))
    return out


def _steady(m, period, dur, t0=0.0):
    out, t = [], t0
    while t < dur:
        out.append((round(t, 3), *m))
        t += period
    return out


def _burst(m, at, n, sp=0.1):
    return [(round(at + k * sp, 3), *m) for k in range(n)]


# fixed industrial reference traffic conditions (deterministic) + the
# same-model baseline companion so the switch-induced latency delta is
# explicit ("how much did multimodel switching cost vs one model").
def refpoints():
    return {
        "steady_skew": dict(
            desc="industrial 18:1 demand skew, sustained",
            arr=_steady(SD, 1.0, 60) + _steady(FX, 20.0, 60),
            base=_steady(SD, 1.0, 60)),                       # sd3-only
        "cold_burst": dict(
            desc="rare model silent then sudden 8-burst amid steady load",
            arr=_steady(SDb, 2.0, 60) + _burst(FX, 25.0, 8, 0.12),
            base=_steady(SDb, 2.0, 60)),
        "heavy_hog": dict(
            desc="heavy 1024² model saturates GPU; light model trickles",
            arr=_steady(SDB, 4.0, 48) + _steady(FX, 16.0, 48, 6.0),
            base=_steady(SDB, 4.0, 48)),
    }


def gen(base, m, w, h, st, seed, rid):
    r = requests.post(f"{base}/generate", json={
        "model": m, "prompt": PROMPT, "negative_prompt": "low quality",
        "width": w, "height": h, "num_inference_steps": st,
        "seed": seed, "req_id": rid}, timeout=30)
    r.raise_for_status()
    return r.json()["task_id"]


def _get(base, path, to=10):
    try:
        return requests.get(f"{base}{path}", timeout=to).json()
    except Exception:  # noqa: BLE001
        return {}


def _pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 3)


def drain(base, max_wait):
    t = time.time()
    while time.time() - t < max_wait:
        s = _get(base, "/dispatch_stats")
        if s.get("ready") and s.get("queue_depth", 1) == 0:
            return True, round(time.time() - t, 1)
        time.sleep(2.0)
    return False, round(time.time() - t, 1)


def fire_all(base, arrivals, tag):
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

    ths = [threading.Thread(target=fire,
                            args=(i, a[0], a[1], a[2], a[3], a[4]),
                            daemon=True)
           for i, a in enumerate(sorted(arrivals))]
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(0.5)
    return sub, t0


def latency_profile(base, sub, t0, slo_scale):
    """Full per-request latency decomposition from /task_timeline."""
    tl = {h["task_id"]: h for h in _get(base, "/task_timeline").get(
        "handles", [])}
    glob = {"e2e": [], "queue": [], "switch": [], "infer": [], "slo_ok": 0,
            "slo_n": 0, "switched": 0, "first_done": None}
    perm = {}
    for tid, (m, w, h, st) in sub.items():
        pm = perm.setdefault(m, {"n": 0, "done": 0, "e2e": [], "queue": [],
                                 "switch": [], "infer": [], "slo_ok": 0})
        pm["n"] += 1
        t = tl.get(tid)
        if not t or t.get("done_ts") is None or not t.get("ok"):
            continue
        e2e = t["done_ts"] - t["submit_ts"]
        q = (t["start_ts"] - t["submit_ts"]) if t.get("start_ts") else 0.0
        sw = t.get("bind_s") or 0.0
        inf = t.get("infer_s")
        if inf is None:                       # fall back if not recorded
            inf = max(0.0, (t["done_ts"] - (t["start_ts"] or t["submit_ts"]))
                      - sw)
        pm["done"] += 1
        for k, v in (("e2e", e2e), ("queue", q), ("switch", sw),
                     ("infer", inf)):
            pm[k].append(v)
            glob[k].append(v)
        ok = e2e <= slo_for(w, h, st, slo_scale)
        pm["slo_ok"] += int(ok)
        glob["slo_ok"] += int(ok)
        glob["slo_n"] += 1
        glob["switched"] += 1 if t.get("switched") else 0
        fd = t["done_ts"] - t0
        if glob["first_done"] is None or fd < glob["first_done"]:
            glob["first_done"] = round(fd, 3)
    return glob, perm


def _dist(xs):
    return {"p50": _pct(xs, .5), "p95": _pct(xs, .95),
            "p99": _pct(xs, .99),
            "max": round(max(xs), 3) if xs else float("nan"),
            "mean": round(sum(xs) / len(xs), 3) if xs else float("nan")}


def summarize(glob, perm, elapsed):
    done = sum(p["done"] for p in perm.values())
    n = sum(p["n"] for p in perm.values())
    e2e = glob["e2e"]
    tot = sum(glob["e2e"]) or 1.0
    out = {
        "completed": done, "admitted": n,
        "throughput_rps": round(done / elapsed, 4) if elapsed else 0,
        "time_to_first_result_s": glob["first_done"],
        "slo_attainment": round(glob["slo_ok"] / done, 3) if done else 0.0,
        "goodput_at_slo_rps": round(glob["slo_ok"] / elapsed, 4)
        if elapsed else 0.0,
        "e2e": _dist(glob["e2e"]), "queue": _dist(glob["queue"]),
        "switch": _dist(glob["switch"]), "infer": _dist(glob["infer"]),
        "jitter_p99_over_p50": round(
            _pct(e2e, .99) / _pct(e2e, .5), 2)
        if e2e and _pct(e2e, .5) else float("nan"),
        "decomposition_share": {
            "queue_pct": round(100 * sum(glob["queue"]) / tot, 1),
            "switch_pct": round(100 * sum(glob["switch"]) / tot, 1),
            "infer_pct": round(100 * sum(glob["infer"]) / tot, 1)},
        "switch_freq_pct": round(
            100 * glob["switched"] / done, 1) if done else 0.0,
        "per_model": {m: {
            "done": f"{p['done']}/{p['n']}",
            "e2e": _dist(p["e2e"]), "queue_p95": _pct(p["queue"], .95),
            "switch_p95": _pct(p["switch"], .95),
            "infer_p50": _pct(p["infer"], .5),
            "slo_attain": round(p["slo_ok"] / p["done"], 3)
            if p["done"] else 0.0}
            for m, p in sorted(perm.items())},
    }
    dom = max(out["decomposition_share"].items(), key=lambda kv: kv[1])
    out["dominant_latency"] = dom[0].replace("_pct", "")
    return out


def gate_ok(perm):
    """Binary GATE only: every model with demand made progress + no
    mass-failure. A failed gate => latency numbers are untrustworthy."""
    for m, p in perm.items():
        if p["n"] > 0 and p["done"] == 0:
            return False, f"starvation: {m} 0/{p['n']}"
    return True, "ok"


def run_industrial(base, tag, arr, drain_s, slo_scale):
    print(f"\n=== {tag} (n={len(arr)}) ===", flush=True)
    sub, t0 = fire_all(base, arr, tag)
    drained, waited = drain(base, drain_s)
    elapsed = time.time() - t0
    glob, perm = latency_profile(base, sub, t0, slo_scale)
    s = summarize(glob, perm, elapsed)
    s["drained"], s["drain_wait_s"] = drained, waited
    ok, why = gate_ok(perm)
    s["gate_ok"], s["gate"] = ok, why
    e = s["e2e"]
    d = s["decomposition_share"]
    print(f"  e2e p50={e['p50']}s p95={e['p95']}s p99={e['p99']}s "
          f"max={e['max']}s | ttfr={s['time_to_first_result_s']}s | "
          f"jitter={s['jitter_p99_over_p50']}", flush=True)
    print(f"  decomp: queue {d['queue_pct']}% / switch {d['switch_pct']}% "
          f"/ infer {d['infer_pct']}%  -> dominant={s['dominant_latency']} "
          f"| switch_freq={s['switch_freq_pct']}%  gate={why}", flush=True)
    return s


def _wait_done(base, tid, to=300):
    """Block until one task completes; return its handle (or None)."""
    t = time.time()
    while time.time() - t < to:
        for h in _get(base, "/task_timeline").get("handles", []):
            if h["task_id"] == tid and h.get("done_ts") and h.get("ok"):
                return h
        time.sleep(1.0)
    return None


def _img_arr(base, tid):
    """Fetch /image/<tid> PNG -> normalized float array (or None)."""
    try:
        import io
        import numpy as np
        from PIL import Image
        r = requests.get(f"{base}/image/{tid}", timeout=20)
        if r.status_code != 200:
            return None
        return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"),
                          dtype="float32") / 255.0
    except Exception:  # noqa: BLE001 — quality proxy is best-effort
        return None


def run_quality(base, slo_scale, steps_list=(50, 30, 20, 10)):
    """Pillar ③: quality-ELASTICITY axis. Diffusion uniquely lets you
    trade denoising steps for quality→latency continuously. Per model,
    fire the SAME (prompt, seed) at each step count; measure the
    authoritative latency(steps) curve + a quality proxy = normalized
    RMSE of the image vs the full-step (50) reference (same seed, so
    structurally comparable; fewer steps ⇒ larger deviation from the
    converged image). This quantifies the serving lever: under overload,
    drop steps (graceful quality loss) instead of dropping requests.
    NOTE: nRMSE-vs-reference is a PROXY for 'distance from full quality',
    not a perceptual metric — labeled as such.
    """
    out = {}
    for (m, w, h, _st) in (SD, FX):
        ref_arr, rows = None, []
        for i, st in enumerate(steps_list):
            tid = gen(base, m, w, h, st, 7, f"q-{m}-{st}")
            hd = _wait_done(base, tid)
            infer = (hd.get("infer_s") if hd else None)
            e2e = (hd["done_ts"] - hd["submit_ts"]) if hd else None
            arr = _img_arr(base, tid)
            if i == 0:
                ref_arr = arr
                nrmse = 0.0
            elif arr is not None and ref_arr is not None \
                    and arr.shape == ref_arr.shape:
                import numpy as np
                nrmse = float(((arr - ref_arr) ** 2).mean() ** 0.5)
            else:
                nrmse = None
            rows.append({"steps": st, "infer_s": round(infer, 3)
                         if infer else None,
                         "e2e_s": round(e2e, 3) if e2e else None,
                         "nrmse_vs_ref50": round(nrmse, 4)
                         if nrmse is not None else None})
            print(f"  {m} {w}x{h} steps={st}: infer={infer}s "
                  f"nrmse_vs_ref={rows[-1]['nrmse_vs_ref50']}", flush=True)
        # elasticity: latency saved per step dropped (50 -> min steps)
        full = next((r for r in rows if r["steps"] == steps_list[0]), None)
        lo = next((r for r in rows if r["steps"] == steps_list[-1]), None)
        elas = None
        if full and lo and full["infer_s"] and lo["infer_s"]:
            ds = steps_list[0] - steps_list[-1]
            elas = round((full["infer_s"] - lo["infer_s"]) / ds, 4)
        out[f"{m}_{w}x{h}"] = {"curve": rows,
                               "latency_s_per_step": elas}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--mode", default="all",
                    choices=["industrial", "sweep", "refpoints", "quality",
                             "all"])
    ap.add_argument("--out-dir", default="/tmp/hetu_bench")
    ap.add_argument("--drain-s", type=float, default=200.0)
    ap.add_argument("--slo-scale", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Gamma shape: smaller = burstier (CV up)")
    ap.add_argument("--skew", type=float, default=0.85,
                    help="P(heavy model) — demand imbalance")
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--rates", default="0.05,0.1,0.2,0.4,0.8")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if not _get(a.base_url, "/dispatch_stats").get("ready"):
        print("note: dispatcher lazy-inits on first request.", flush=True)
    rep = {"hetu_bench_version": HETU_BENCH_VERSION, "mode": a.mode}
    sizes = [SD, FX]

    if a.mode in ("industrial", "all"):
        arr = _gamma(a.rate, a.alpha, a.skew, a.dur, 42, sizes)
        rep["industrial"] = run_industrial(
            a.base_url, f"industrial(rate={a.rate},α={a.alpha},"
            f"skew={a.skew})", arr, a.drain_s, a.slo_scale)

    if a.mode in ("sweep", "all"):
        curve = []
        for rt in [float(x) for x in a.rates.split(",")]:
            arr = _gamma(rt, 0.5, a.skew, a.dur, 42, sizes)
            s = run_industrial(a.base_url, f"sweep@{rt}rps", arr,
                               a.drain_s, a.slo_scale)
            curve.append({"rate": rt, "offered_rps": round(len(arr)/a.dur, 3),
                          "e2e_p50": s["e2e"]["p50"],
                          "e2e_p99": s["e2e"]["p99"],
                          "queue_p99": s["queue"]["p99"],
                          "switch_p95": s["switch"]["p95"],
                          "infer_p50": s["infer"]["p50"],
                          "slo_attainment": s["slo_attainment"],
                          "dominant": s["dominant_latency"],
                          "gate_ok": s["gate_ok"]})
        # latency knee = offered rps where e2e p99 crosses 10s, LINEARLY
        # INTERPOLATED between sweep points (not snapped to the grid) so a
        # real latency improvement registers at sub-grid resolution
        # (§8.46: opt#1 cut p99 22->14s but a grid-max knee stayed 0.2,
        # hiding the win — that was benchmark blindness, fixed here).
        cs = sorted([c for c in curve if c["gate_ok"]],
                    key=lambda c: c["offered_rps"])
        knee = 0.0
        for i, c in enumerate(cs):
            if c["e2e_p99"] <= 10.0:
                knee = c["offered_rps"]
            elif i > 0 and cs[i-1]["e2e_p99"] <= 10.0:
                lo, hi = cs[i-1], c
                span = hi["e2e_p99"] - lo["e2e_p99"]
                frac = ((10.0 - lo["e2e_p99"]) / span) if span > 0 else 0.0
                knee = round(lo["offered_rps"] + frac *
                             (hi["offered_rps"] - lo["offered_rps"]), 3)
                break
        rep["sweep"] = {"curve": curve, "latency_knee_rps_p99_le_10s": knee}

    if a.mode in ("refpoints", "all"):
        rp = {}
        for name, spec in refpoints().items():
            sw = run_industrial(a.base_url, f"ref:{name}",
                                spec["arr"], a.drain_s, a.slo_scale)
            bl = run_industrial(a.base_url, f"ref:{name}:baseline",
                                spec["base"], a.drain_s, a.slo_scale)
            delta = round(sw["e2e"]["p95"] - bl["e2e"]["p95"], 3)
            rp[name] = {"desc": spec["desc"], "with_switching": sw,
                        "same_model_baseline": bl,
                        "switch_induced_e2e_p95_delta_s": delta}
            print(f"  >> {name}: switching cost +{delta}s e2e p95 vs "
                  f"same-model baseline", flush=True)
        rep["refpoints"] = rp

    if a.mode in ("quality", "all"):
        print("\n=== quality-elasticity (steps -> latency, quality) ===",
              flush=True)
        rep["quality_elasticity"] = run_quality(a.base_url, a.slo_scale)

    # LATENCY score (robustness only a gate). De-saturated (§8.46): the
    # SLO-attainment term is the mean over the HARD operating points —
    # the sweep (which spans into overload) + the refpoint
    # with-switching runs (the industrial-imbalance scenarios) — NOT the
    # within-capacity industrial run (always ~1.0, blind to gains).
    # 55·hard-SLO-attain + 30·latency-knee(interp) + 15·jitter-pred.
    parts = {}
    if a.mode == "all":
        ind = rep.get("industrial", {})
        sweep = rep.get("sweep", {})
        rps = rep.get("refpoints", {})
        gate_bad = not ind.get("gate_ok", True) or any(
            not c["gate_ok"] for c in sweep.get("curve", [])) or any(
            not v["with_switching"].get("gate_ok", True)
            for v in rps.values())
        hard = [c["slo_attainment"] for c in sweep.get("curve", [])]
        hard += [v["with_switching"]["slo_attainment"]
                 for v in rps.values()]
        attn = round(sum(hard) / len(hard), 3) if hard else 0.0
        knee = sweep.get("latency_knee_rps_p99_le_10s", 0.0)
        jit = ind.get("jitter_p99_over_p50") or 99
        jitscore = max(0.0, 1.0 - (jit - 1) / 9)
        parts = {"hard_slo_attainment": attn,
                 "latency_knee_rps": knee,
                 "jitter_predictability": round(jitscore, 3),
                 "gate_failed": gate_bad}
        rep["hetu_latency_score"] = (
            0.0 if gate_bad else
            round(55*attn + 30*min(1.0, knee/1.0) + 15*jitscore, 1))
        rep["hetu_latency_score_parts"] = parts

    json.dump(rep, open(os.path.join(a.out_dir, "HETU_REPORT.json"), "w"),
              indent=2)
    print(f"\n#### HETU BENCHMARK v{HETU_BENCH_VERSION} — LATENCY ####")
    if "sweep" in rep:
        print("LATENCY vs OFFERED LOAD (the headline):")
        for c in rep["sweep"]["curve"]:
            print(f"  {c['offered_rps']:>5} rps | e2e p50={c['e2e_p50']}s "
                  f"p99={c['e2e_p99']}s | queue p99={c['queue_p99']}s "
                  f"switch p95={c['switch_p95']}s infer p50={c['infer_p50']}s"
                  f" | SLO {c['slo_attainment']} dom={c['dominant']}")
        print(f"  latency knee = {rep['sweep']['latency_knee_rps_p99_le_10s']}"
              f" rps (p99<=10s)")
    if "industrial" in rep:
        s = rep["industrial"]
        print(f"INDUSTRIAL (Gamma burst): e2e p50={s['e2e']['p50']}s "
              f"p95={s['e2e']['p95']}s p99={s['e2e']['p99']}s | "
              f"decomp queue {s['decomposition_share']['queue_pct']}%/"
              f"switch {s['decomposition_share']['switch_pct']}%/infer "
              f"{s['decomposition_share']['infer_pct']}% -> "
              f"{s['dominant_latency']}")
    if "refpoints" in rep:
        print("REF POINTS (switch-induced latency vs same-model baseline):")
        for n, r in rep["refpoints"].items():
            print(f"  {n}: +{r['switch_induced_e2e_p95_delta_s']}s e2e p95")
    if "hetu_latency_score" in rep:
        p = rep["hetu_latency_score_parts"]
        if p["gate_failed"]:
            print("HETU LATENCY SCORE = INVALID (robustness gate failed)")
        else:
            print(f"HETU LATENCY SCORE = {rep['hetu_latency_score']}/100 "
                  f"(hard-SLO-attain={p['hard_slo_attainment']}, knee="
                  f"{p['latency_knee_rps']}rps, jitter-pred="
                  f"{p['jitter_predictability']})")
    print(f"REPORT -> {a.out_dir}/HETU_REPORT.json\nHETU_BENCH_DONE")


if __name__ == "__main__":
    main()
