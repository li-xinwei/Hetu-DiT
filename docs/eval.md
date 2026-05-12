# Warm-pool eval harness

This eval system answers the architectural question: **does a pre-warm pool
of N pods, costing N × GPU-hour, buy enough P95-latency / SLO improvement
over a fully on-demand baseline to be worth recommending to TridentServe
users?**

The components are gated, additive, and reusable. Production behavior is
unchanged when `HETU_COLDSTART_TRACE` is unset and `/metrics` is unqueried.

## Components

| Path | Purpose |
| --- | --- |
| `hetu_dit/cstrace.py` | `cst_print` (existing) + `cst_request` (new) |
| `hetu_dit/metrics.py` | In-process counters + ring buffer for `/metrics` |
| `hetu_dit/entrypoint/api_server.py` | `cst_request("start")` on /generate; `GET /metrics` |
| `hetu_dit/engine/async_serving_engine.py` | Per-dispatch markers (`hit_type=ready\|warm\|l2\|cold`) |
| `scripts/workload_gen.py` | Emit trace files in 5 patterns (cold/burst/poisson/steady/mixed) |
| `scripts/eval_sim.py` | Pure-Python event-driven simulator |
| `scripts/parse-coldstart-trace.py --json` | JSON output mode for downstream tools |
| `scripts/cost_model.py` | GPU-hour cost + SLO target/attainment helpers |
| `scripts/aggregate_runs.py` | Walks results root → `per_run.csv` + `per_run.json` |
| `scripts/gen_report.py` | Renders `report.md` + `plot_slo_vs_pool.png` + `plot_cost_vs_slo.png` |
| `scripts/run_eval_matrix.sh` | Orchestrator: drives the workload × pool matrix |
| `scripts/eval_matrix.yaml` | Cell definitions for `small` (sim validation) and `full` (real-mode) sets |
| `data/optimal_latencies.json` | SLO reference latencies (calibrate against B5 hot cluster) |

## Hit-type taxonomy

The dispatcher classifies each request into exactly one of four buckets:

| hit_type | Meaning |
| --- | --- |
| **ready** | `_find_ready_executor` matched — same executor still bound, fastest hit |
| **warm** | `_find_warm_l2_executor` matched a **warm-tagged + L2-parked** executor (true pre-warm pool hit) |
| **l2** | `_find_warm_l2_executor` matched any L2-parked executor (single-pod L2 mechanism — saves only the GPU model load step) |
| **cold** | Fell through to `switch_parallel_env` (legacy reconfigure path) |

In the report, `warm_hit_rate = warm / served` and
`warm_or_l2_hit_rate = (warm + l2) / served`. Ready hits are not counted
toward the warm-pool win — they are the second-and-later requests served by
an already-active executor, which would happen even without a warm pool.

## Quick start (sim mode, no GPUs required)

```bash
# Phase 4 small set: 6 cells, runs in <30 seconds on a laptop.
bash scripts/run_eval_matrix.sh \
    --matrix scripts/eval_matrix.yaml \
    --set small --mode sim \
    --out results/eval-sim-$(date +%Y%m%d-%H%M%S)

# View report
open results/eval-sim-*/report/report.md
```

The orchestrator runs four steps per cell: `workload_gen.py` →
`eval_sim.py` → `aggregate_runs.py` → `gen_report.py`.

## Calibration

`data/optimal_latencies.json` holds the SLO reference: optimal-parallelism
single-request latency on a hot cluster. The SLO threshold is
`multiplier × reference` (default multiplier = 2.5, matching TridentServe).
Recalibrate against a B5 (always-hot) run before any Phase 5 real-mode
mentor artifact.

## Real-mode (Phase 5)

`scripts/run_eval_matrix.sh --mode real` is stubbed but not yet wired —
the per-cell loop needs to call `scripts/run-runpod-paths.sh` for the K8s
apply+drive+teardown sequence and poll `/metrics` at the end. The cstrace
markers, aggregator, cost model, and report generator are mode-agnostic,
so the only Phase 5 work is the orchestrator's real-mode branch.

## Verification

| Check | Command | Pass criterion |
| --- | --- | --- |
| Syntax | `python3 -m compileall scripts/ hetu_dit/cstrace.py hetu_dit/metrics.py hetu_dit/entrypoint/api_server.py hetu_dit/engine/async_serving_engine.py` | exit 0 |
| Unit tests still green | `pytest tests/unit/ -q` | all pass |
| Workload roundtrip | `workload_gen.py --pattern poisson --rate 1 --duration 60 --out /tmp/w.trace` then parse via `api_benchmark_example_with_trace.py:load_requests_from_trace` | 58 ± 3σ requests, all fields populated |
| Sim end-to-end | `run_eval_matrix.sh --set small --mode sim --out /tmp/e` | 6 cells, `report.md` + 2 PNGs non-empty |
| Pareto sanity | inspect `plot_cost_vs_slo.png` | warm=0 top-left, warm≥2 bottom-right |
| Back-compat | `aggregate_runs.py --root results/runpod-h100-2026-05-06/results` | 14 legacy rows, `hit_type=unknown`, no crashes |
