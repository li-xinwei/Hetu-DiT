# Hetu Benchmark — methodology (v1)

The complete UX/SLO-centric benchmark for Hetu-DiT as a **multimodel
serving system under industrial high-concurrency + load imbalance**.
`scripts/hetu_benchmark.py` is the single runnable artifact; this file
is the rationale and the run book.

## 1. Why these metrics (not throughput / mean latency)

Raw throughput and mean latency do not measure user experience. From the
serving literature and production guides:

- **Goodput@SLO** — completed requests/s that met their latency SLO — is
  the headline number that ties resource use to UX and revenue
  (DistServe OSDI'24; Anyscale/NVIDIA benchmarking guides).
- **Tail, not mean** — a 200 ms mean TTFT routinely hides a 3 s p99;
  always report p50/p90/p95/p99 separately (Anyscale; NVIDIA NIM).
- **SLO attainment %** per model and per SLO class is how production
  systems are graded (DiffServe MLSys'25 "SLO violation ratio";
  MoDM/HADIS tail-latency-under-rate).
- **Fairness / performance isolation** — Jain's index over per-model
  SLO-attainment; a heavy model must not blow a cold model's SLO
  (Clockwork NSDI'20; Shepherd; InferFair FGCS'23: +1.7× fairness).
- **Realistic bursty arrivals** — real Azure GPT traffic is Gamma-
  distributed; smaller shape α ⇒ higher coefficient-of-variation ⇒
  burstier (BurstGPT, 10.3 M-trace Azure dataset). Poisson is the mild
  baseline; deterministic adversarial scenarios are the regression core.
- **Capacity = the SLO knee** — sweep offered load, find the max RPS
  sustaining ≥99 % goodput; classify overload as **graceful** (admission
  shed, bounded latency, goodput plateaus) vs **collapse** (goodput
  craters, GPU idle — the §8.32 failure) (DiffServe/MoDM methodology).
- **Switch / cold-start tax** is first-class for *multimodel* serving:
  model-swap wall time (measured here: hot 0 s, sd3→flux ≈9 s,
  flux→sd3 ≈6 s) and how often it lands on the user's critical path.

Sources: DistServe https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf ·
AlpaServe https://www.usenix.org/system/files/osdi23-li-zhuohan.pdf ·
Clockwork https://web.eecs.umich.edu/~mosharaf/Readings/Clockwork.pdf ·
InferFair https://dl.acm.org/doi/10.1016/j.future.2023.08.020 ·
DiffServe https://arxiv.org/pdf/2411.15381 · MoDM https://arxiv.org/html/2503.11972v2 ·
BurstGPT https://arxiv.org/abs/2401.17644 ·
Revisiting SLO & Goodput https://arxiv.org/html/2410.14257v1 ·
Anyscale metrics https://docs.anyscale.com/llm/serving/benchmarking/metrics

## 2. What it measures (per model + global)

`/task_timeline` (dispatcher TaskHandle submit/start/done) + `/dispatch_stats`
give authoritative per-request truth — **no dependence on the §8.41-buggy
`/status` PNG-glob**:

- `e2e = done − submit` (what the user feels), `queue = start − submit`
  (waiting behind others), `service = done − start` (switch tax + infer).
- goodput@SLO (rps), SLO-attainment %, throughput, completion ratio.
- e2e p50/p95/p99/max, queue p95, service p50, tail ratio p99/p50
  (predictability — Clockwork).
- Jain fairness index over per-model SLO-attainment.
- starvation-freedom: min per-model completion ratio (the §8.41 gate).
- per-scenario PASS/FAIL against a structured gate encoding the *actual*
  property under test (not "every model 100 %").

## 3. Workload (aggregates the full taxonomy + stochastic + sweep)

- **taxonomy** — 14 deterministic load-imbalance scenarios, per-scenario
  drain-isolated: A skew (steady/cold-burst/gpu-hog/hotset-shift/zipf),
  B temporal (flash-crowd/sync-burst/idle-then-hit), C work-size
  (res-step-mix/cheap-vs-costly), D switch (alternate/thrash-sweep),
  F SLO-fairness (starvation-free HARD GATE), E1 multi-GPU (gated).
- **capacity** — Gamma-burst rate sweep ⇒ goodput@SLO-vs-load curve,
  capacity knee, graceful/collapse classification.
- **burst** — BurstGPT-style Gamma arrivals (shape α = burstiness,
  `--skew` heavy-model probability), seeded/reproducible.

## 4. Hetu Latency Score (0–100) — latency only, robustness is a GATE

The score IS latency quality. Robustness (no OOM / no collapse /
starvation-freedom / fairness) is a **binary GATE only**: any scenario
verdict==FAIL invalidates the run (score = INVALID) — you cannot trust a
latency number from a server that didn't stay up. When the gate passes:

`55·SLO-attainment + 30·latency-knee(rps@p99≤10s, /1.0) +
15·tail-predictability(1/(p99÷p50 spread))`.

All three terms are latency-derived; throughput/fairness/graceful are
NOT weighted in. The headline output is the **latency-vs-load curve**
(e2e p99 swept over offered load) + the per-scenario / per-model e2e
p50/p95/p99 with queue-vs-service decomposition.

## 5. Run it — anytime

Local (server already up on :8000):

```
python3 scripts/hetu_benchmark.py --mode all --out-dir ~/hetu_bench
# or a fast subset:
python3 scripts/hetu_benchmark.py --mode taxonomy \
    --only A1_steady_skew,A2_cold_burst,F2_starvation_free
```

One command, provisioned real hardware (Lambda A100-40GB, end-to-end:
launch → env → models → server → benchmark → REPORT → teardown):

```
bash scripts/run_hetu_benchmark.sh        # needs ~/.lambdalabs/key, HF_TOKEN
```

Output: `HETU_REPORT.json` (machine-readable, versioned) + a console
summary with the Hetu Score, per-scenario verdicts, and the capacity
curve. Deterministic (fixed seeds) so versions are comparable.
`HETU_BENCH_VERSION=1`.
