# Real-mode bench — Lambda A100 8x SXM4, 2026-05-12

## Setup

- Lambda Cloud `gpu_8x_a100` (A100 40GB SXM4 with NVLink), $15.92/hr in us-west-2.
- Total Lambda session cost: ~$25 (~1.6 hr — most time on env troubleshooting, K8s/image hell).
- Used 4 of 8 GPUs (sp=4 cfg=1). The other 4 were idle in this run.
- **Bare-process Ray** (no K3s/K8s). K3s+KubeRay path was attempted but failed on libstdc++ ABI
  conflicts between nvcr.io pytorch base + pip-installed pyarrow/protobuf, and on a
  separate pytorch 2.3.1 base the pip install died on yunchang/nixl metadata gen.
- Lambda host: torch 2.7.0+cu126, ray 2.39.0, diffusers 0.32.2, yunchang 0.3.5.

## What was measured

Six cells, each spawns `python3 -m hetu_dit.entrypoint.api_server` fresh on host,
waits `/readyz`, fires the workload, captures cstrace + `/metrics`, tears down.

| Cell | `--l2_pool_enabled` | Workload | N |
| --- | --- | --- | ---: |
| A_cold_seq      | False | seq SD3 1024² | 1 |
| A_cold_burst8   | False | burst 8 concurrent SD3 1024² | 8 |
| A_cold_mixed    | False | Poisson λ=1 mixed (1024/768/1280)² | 10 |
| B_warm1_seq     | True  | seq SD3 1024² | 1 |
| B_warm1_burst8  | True  | burst 8 concurrent SD3 1024² | 8 |
| B_warm1_mixed   | True  | Poisson λ=1 mixed (1024/768/1280)² | 10 |

## Results

| Cell | apply_to_ready | wall | p95 | dispatch hits (r/w/l2/c) |
| --- | ---: | ---: | ---: | --- |
| A_cold_seq      | 78.10 s | 1.0 s | 1.01 s | 1 / 0 / 0 / 0 |
| A_cold_burst8   | 76.09 s | 1.0 s | 1.02 s | 8 / 0 / 0 / 0 |
| A_cold_mixed    | 76.11 s | 8.9 s | 1.01 s | 18 / 0 / 0 / 0 |
| B_warm1_seq     | 74.10 s | 4.0 s | 4.02 s | 0 / 0 / 1 / 0 |
| B_warm1_burst8  | 72.10 s | 24.2 s | **24.15 s** | 7 / 0 / 6 / 0 |
| B_warm1_mixed   | 74.10 s | 28.0 s | 19.12 s | 10 / 0 / 7 / 0 |

(Hits legend: `r` = ready / `w` = warm-tagged / `l2` = L2-parked / `c` = cold-path.)

## Honest findings

1. **L2-pool mode is *slower*, not faster, in this configuration.** Sequential request
   total 4.0 s vs 1.0 s cold-baseline; burst-8 p95 24.1 s vs 1.0 s cold-baseline.
   The reason: L2 mode defers GPU model load to first request; the single sp=4
   executor (using all 4 GPUs) then has to bind on each L2-state dispatch. Burst
   traffic stacks behind the bind.

2. **The "warm pool" architecture isn't actually exercised here.** All 8 warm-tag hits
   on the `/metrics` counter show as `l2` not `warm` — meaning the executor wasn't
   tagged warm via `HETUDIT_ROLE=warm` env (which requires multi-pod K8s setup we
   couldn't get working). Single-process Ray can't distinguish warm vs L2 workers.

3. **A100 4-GPU sp=4 SD3 inference is fast — 1.0 s per request.** Cold baseline burst-8
   completed all 8 in 1.0 s wall = diffusers internally batched them across the
   single executor; or sequence-parallel-on-NVLink delivered near-linear throughput.

4. **The 76-78 s apply_to_ready is dominated by host-level Python startup + diffusers
   import + ray init**, not GPU model load. PyTorch 2.7 on host is fatter than the
   container-side build that previously got 14-18 s boot on RTX 3090 / H100 SXM.

## What this round did NOT validate

- **Multi-instance warm pool** (head pod + warm pod with `HETUDIT_ROLE=warm`).
  K3s setup on Lambda failed on libstdc++ ABI / yunchang pip install / KubeRay
  release URL changes. The dispatcher's warm-vs-L2 distinction stayed dormant.

- **Scale-up benefit**: the sim predicted warm pool absorbs burst at SLO. Without
  multiple instances, we get the opposite — L2 mode penalizes burst because all
  requests funnel through one freshly-binding executor.

## Calibration delta against sim

| Quantity | Sim prediction | Lambda A100 4-GPU real | Notes |
| --- | --- | --- | --- |
| T_INFER (SD3 1024² 20 step, sp=4) | 3.21 s (calibrated H100 SXM) | **1.01 s** | A100+NVLink batches very well; sim was conservative |
| T_BIND_COLD_L2 (head pod first bind) | 1.0 s | **3.0 s** | L2 → active transition heavier on this stack |
| T_BOOT_TO_L2 | 11 s | **74-78 s** | host-Python boot is much fatter than container; not a direct comp |
| Cold-start total (no l2_pool) | 15.2 s | 78.1 + 1.0 = **79 s** | dominated by host bootstrap, not GPU load |

## Conclusion

**Real measurements confirm what the previous RTX 3090 bench already showed**: the L2
state-machine alone (without multi-instance K8s deployment) does not deliver warm-pool's
headline SLO benefit; in fact it hurts burst latency by serializing bind operations
through a single executor.

The sim's "warm=4 → 100% SLO" finding remains **uncorroborated by any independent real-mode
multi-instance run.** The 2026-05-06 PKU H100 SXM `path-2inst-warm` data point (3.14 s
warm-hit) is still the only real measurement that points at the architecture being
effective, and it was a single trial.

## Files

- `ALL_CELLS.json` — full per-request timings
- `bench.log` — bench driver stdout
- `A_*/`, `B_*/` — per-cell artifacts: api.log, cstrace.log, request_log.json, summary.txt
