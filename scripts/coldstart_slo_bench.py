#!/usr/bin/env python3
"""Cold-start SLO benchmark for the multi-model serving MVP.

Replays the mentor's per-model traces (data/<model>_trace.txt) as one
*interleaved* multi-model workload against a running api_server, measures
per-request end-to-end latency, isolates the model-switch cold-start tax
(no GPU LRU yet — PR-4), and scores the result against industrial
interactive image-generation SLOs.

Methodology
-----------
The raw traces arrive at ~30 req/s (sd3) which is impossible on a single
GPU and would bury cold-start latency under queueing delay. To measure the
*cold-start tax itself* we replay **closed-loop** (1 outstanding request):
send -> wait for completion -> send next. This yields clean service time.

We still preserve the mentor's industrial request *sequence* by merging the
per-model traces on their original timestamps (Zipf-skewed ~20:1 sd3:flux,
~9.4% model-switch rate, mean same-model run ~10). The arrival *rate* is
dropped on purpose; the arrival *pattern* (which model, what resolution,
how often it switches) is faithful.

cold-start tax = latency(first request after a model switch)
               - latency(matched same-model,same-res,same-steps warm request)

This controls for inference time so the delta is purely the
from_pretrained + .to("cuda") reload penalty.
"""
import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict

_RE = re.compile(
    r"request_id=(\d+), timestamp=([\d.]+), height=(\d+), width=(\d+), "
    r"num_frames=(\d+).*?num_inference_steps=(\d+)"
)


def load_trace(model, path):
    reqs = []
    with open(path) as fh:
        for line in fh:
            m = _RE.search(line)
            if not m:
                continue
            rid, ts, h, w, nf, steps = m.groups()
            reqs.append(
                {
                    "model": model,
                    "ts": float(ts),
                    "height": int(h),
                    "width": int(w),
                    "num_frames": int(nf),
                    "num_inference_steps": int(steps),
                }
            )
    return reqs


def _tag(window):
    prev = None
    for i, r in enumerate(window):
        r["seq"] = i
        r["cold"] = prev is not None and r["model"] != prev
        r["initial"] = prev is None
        prev = r["model"]
    return window


def build_workload(
    trace_dir, models, window_start, n, pattern, alt_steps, alt_res, alt_block
):
    """Two workload shapes.

    pattern='trace'     : merge per-model traces on original timestamps and
                          slice a contiguous window -> faithful industrial
                          skew (~20:1 sd3:flux, ~9% switch rate). Answers the
                          *production SLO* question.
    pattern='alternate' : ignore traces, emit alternating *blocks* of
                          `alt_block` same-model requests at fixed res/steps
                          (e.g. block=3 -> sd3,sd3,sd3,flux,flux,flux,...).
                          First request of each block is a guaranteed cold
                          switch; the rest are warm at identical res/steps ->
                          a perfectly matched baseline for a clean cold-start
                          tax. Answers the *cold-start magnitude* question.
    """
    if pattern == "alternate":
        w, h = (int(x) for x in alt_res.split("x"))
        window = []
        i = 0
        while len(window) < n:
            mdl = models[i % len(models)]
            for _ in range(alt_block):
                if len(window) >= n:
                    break
                window.append(
                    {
                        "model": mdl,
                        "ts": float(len(window)),
                        "height": h,
                        "width": w,
                        "num_frames": 1,
                        "num_inference_steps": alt_steps,
                    }
                )
            i += 1
        return _tag(window)
    merged = []
    for mdl in models:
        merged += load_trace(mdl, f"{trace_dir}/{mdl}_trace.txt")
    merged.sort(key=lambda r: r["ts"])
    return _tag(merged[window_start : window_start + n])


def _post_generate(base, req, rid):
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
            "req_id": f"slo{rid}",
        }
    ).encode()
    r = urllib.request.Request(
        f"{base}/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=30) as resp:
        return resp.status, json.loads(resp.read())


def _poll(base, task_id, timeout_s, interval):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{base}/status/{task_id}", timeout=15
            ) as resp:
                d = json.loads(resp.read())
            if d.get("image_url"):
                return True, d
        except urllib.error.URLError:
            pass
        time.sleep(interval)
    return False, {"status": "timeout"}


def run(base, workload, poll_timeout, poll_interval):
    records = []
    prev_model = None
    for req in workload:
        rid = req["seq"]
        t0 = time.time()
        try:
            code, body = _post_generate(base, req, rid)
        except Exception as e:  # noqa: BLE001 - bench should not die on one req
            records.append({**req, "ok": False, "error": f"post:{e}"})
            continue
        if code >= 400:
            records.append({**req, "ok": False, "error": f"http{code}:{body}"})
            continue
        ok, st = _poll(base, body["task_id"], poll_timeout, poll_interval)
        latency = time.time() - t0
        records.append(
            {
                "seq": req["seq"],
                "model": req["model"],
                "res": f"{req['width']}x{req['height']}",
                "steps": req["num_inference_steps"],
                "cold": req["cold"],
                "initial": req["initial"],
                "switched_from": prev_model
                if (req["cold"] or req["initial"])
                else None,
                "ok": ok,
                "latency_s": round(latency, 3),
                "task_id": body["task_id"],
            }
        )
        prev_model = req["model"]
        print(
            f"[{rid:3d}] {req['model']:5s} {req['width']}x{req['height']:<5d} "
            f"steps={req['num_inference_steps']:<3d} "
            f"{'COLD' if req['cold'] else ('INIT' if req['initial'] else 'warm')} "
            f"-> {latency:6.2f}s {'OK' if ok else 'TIMEOUT'}",
            flush=True,
        )
    return records


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return round(xs[k], 3)


def analyze(records, slo_interactive, slo_relaxed):
    ok = [r for r in records if r.get("ok")]
    warm = [r for r in ok if not r["cold"] and not r["initial"]]
    cold = [r for r in ok if r["cold"]]
    lat = lambda rs: [r["latency_s"] for r in rs]  # noqa: E731

    # cold-start tax: match each cold req to a warm req w/ same model+res+steps
    warm_idx = defaultdict(list)
    for r in warm:
        warm_idx[(r["model"], r["res"], r["steps"])].append(r["latency_s"])
    taxes = []
    for c in cold:
        key = (c["model"], c["res"], c["steps"])
        if warm_idx.get(key):
            taxes.append(round(c["latency_s"] - statistics.mean(warm_idx[key]), 3))

    def slo(rs, thr):
        if not rs:
            return None
        return round(100.0 * sum(1 for r in rs if r["latency_s"] <= thr) / len(rs), 1)

    return {
        "counts": {
            "total": len(records),
            "ok": len(ok),
            "warm": len(warm),
            "cold": len(cold),
            "failed": len(records) - len(ok),
        },
        "warm_latency_s": {
            "p50": _pct(lat(warm), 50),
            "p95": _pct(lat(warm), 95),
            "p99": _pct(lat(warm), 99),
            "max": round(max(lat(warm)), 3) if warm else None,
        },
        "cold_latency_s": {
            "p50": _pct(lat(cold), 50),
            "p95": _pct(lat(cold), 95),
            "p99": _pct(lat(cold), 99),
            "max": round(max(lat(cold)), 3) if cold else None,
        },
        "cold_start_tax_s": {
            "n_matched": len(taxes),
            "mean": round(statistics.mean(taxes), 3) if taxes else None,
            "p50": _pct(taxes, 50),
            "p95": _pct(taxes, 95),
            "max": round(max(taxes), 3) if taxes else None,
        },
        "slo": {
            "interactive_threshold_s": slo_interactive,
            "relaxed_threshold_s": slo_relaxed,
            "interactive_pass_pct_all": slo(ok, slo_interactive),
            "interactive_pass_pct_warm": slo(warm, slo_interactive),
            "interactive_pass_pct_cold": slo(cold, slo_interactive),
            "relaxed_pass_pct_all": slo(ok, slo_relaxed),
            "relaxed_pass_pct_cold": slo(cold, slo_relaxed),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://[::1]:8000")
    ap.add_argument("--trace-dir", default="data")
    ap.add_argument("--models", default="sd3,flux")
    ap.add_argument("--window-start", type=int, default=0)
    ap.add_argument("--max-requests", type=int, default=80)
    ap.add_argument(
        "--pattern", choices=["trace", "alternate"], default="trace"
    )
    ap.add_argument("--alt-steps", type=int, default=20)
    ap.add_argument("--alt-res", default="512x512")
    ap.add_argument(
        "--alt-block",
        type=int,
        default=3,
        help="alternate pattern: same-model run length (1st=cold, rest=warm)",
    )
    ap.add_argument("--poll-timeout", type=int, default=180)
    ap.add_argument("--poll-interval", type=float, default=0.1,
                    help="status poll granularity (s); 0.1 resolves sub-2s cold/warm deltas")
    ap.add_argument("--slo-interactive", type=float, default=10.0)
    ap.add_argument("--slo-relaxed", type=float, default=30.0)
    ap.add_argument("--out", default="coldstart_slo_result.json")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    wl = build_workload(
        a.trace_dir,
        models,
        a.window_start,
        a.max_requests,
        a.pattern,
        a.alt_steps,
        a.alt_res,
        a.alt_block,
    )
    n_sw = sum(1 for r in wl if r["cold"])
    print(
        f"workload[{a.pattern}]: {len(wl)} reqs from {models}, "
        f"{n_sw} model-switches",
        flush=True,
    )
    recs = run(a.base_url, wl, a.poll_timeout, a.poll_interval)
    summary = analyze(recs, a.slo_interactive, a.slo_relaxed)
    with open(a.out, "w") as fh:
        json.dump({"summary": summary, "records": recs}, fh, indent=2)
    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print(f"\nfull records -> {a.out}")


if __name__ == "__main__":
    main()
