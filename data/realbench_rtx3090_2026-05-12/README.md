# Real-mode bench — 4× RTX 3090 RunPod community pod, 2026-05-12

## What was run

Full Hetu-DiT stack (api_server + Ray actors, no K3s) on a single RunPod
community pod with 4× RTX 3090 24 GB, $0.88/hr ondemand. Total session
≈ 1.5 hr / ≈ $1.50.

Two cells, each spawns a fresh `python3 -m hetu_dit.entrypoint.api_server`,
waits for `/readyz` to be green, fires a sequential `/generate` request,
then a burst of 4 concurrent requests, then captures `/metrics` and tears
the server down. SD3 1024² 20 steps via the real `StableDiffusion3Pipeline`.

## Why bare-process and not K8s

K3s on RunPod community pods is blocked at install time: the pod is an
unprivileged container itself, so K3s's embedded containerd cannot mount
its own overlayfs ("`overlayfs cannot be enabled for /var/lib/rancher/k3s
/agent/containerd ... err: operation not permitted`"). This is a property
of every hosted GPU-pod provider without `--privileged` and not specific
to RunPod.

The pre-warm-pool *dispatcher logic* (`Worker.l2_state` state machine,
`AsyncServingEngine._find_warm_l2_executor`) is at the Ray-actor layer
inside the api_server, so we test it without K8s. What we lose:
multi-pod orchestration, `HETUDIT_ROLE=warm` env-based actor tagging,
cross-pod IP networking.

## Headline numbers (real, RTX 3090 4-GPU pod, sp=4 cfg=1)

| Metric | Cold baseline (no l2_pool) | L2 pool enabled |
| --- | ---: | ---: |
| `/readyz` time (apply → green) | **18.01 s** | **16.01 s** |
| Sequential `/generate` total | **1.01 s** | **3.01 s** |
| Burst-4 concurrent latencies (s) | 1.01, 1.01, 1.01, 7.02 | 3.01, 5.02, 9.02, 8.02 |
| `/metrics` hit_type distribution | 4× `ready` | 4× `l2` |

## Negative finding — what this measurement does *not* validate

The two cells used **a single executor spanning all 4 GPUs** (default
`init_executors()` behaviour at sp=4 cfg=1). Burst traffic queues FIFO
through it in both cells. The L2-pool difference is just *when* the
GPU model load happens:

- **Cold baseline**: model load during boot (18 s) → first request fast (1 s)
- **L2 pool**: boot finishes without GPU load (16 s) → first request pays
  `bind_to_instance` (3 s)

Net total wall-clock-to-first-request is roughly equal: 19 s vs 19 s.

**The actual warm-pool architecture needs multiple instances** (head pod
+ N warm pods, each pre-deployed at L2 state before the workload arrives)
so concurrent burst traffic absorbs into parallel pre-bound pods. That
requires multi-pod K8s deployment — which the 2026-05-06 PKU H100 SXM
4-GPU session did test (path-2inst-warm → 3.14 s warm hit). On a single
unprivileged RunPod pod we cannot reproduce that scenario.

## Cross-validation point against sim

Sim's prediction for the cold baseline on H100 SXM:
`T_BOOT_TO_L2 + T_BIND_COLD_L2 + T_INFER = 11 + 1 + 3.21 = 15.21 s`.

Real measurement on RTX 3090 4-GPU: `18.01 + 1.01 = 19.02 s` (boot
includes both T_BOOT_TO_L2 and T_BIND_COLD_L2 because no l2_pool). Sim
is **24 % optimistic on this hardware** — RTX 3090 PCIe Gen3 is slower
than H100 SXM PCIe Gen5 (factor of ~4× on H2D), which mostly hits the
GPU-load step. The relative ordering of cold > l2-first-req > hot is
preserved.

## Files

- `realbench.py`: bench driver (spawn → /readyz → seq+burst → /metrics → teardown)
- `bench.log`: stdout of the bench run with per-request timings
- `smoke.log`: cstrace markers from a pre-bench smoke api_server boot
- `../../results/eval-real-rtx3090-2026-05-12/`: per-cell artifacts
  (api.log + cstrace.log + request_log.json + summary.txt)

## Honest conclusion

The bare-process real eval **does not validate** the headline pre-warm-pool
claim from the sim (warm=4 → 100 % SLO on bursty workloads). It does
validate:

1. SD3 cold-start total on RTX 3090 4-GPU ≈ 19 s (sim predicted 15 s, 24 %
   gap — calibration constants were H100 SXM).
2. SD3 inference time on RTX 3090 4-GPU ≈ 1 s (much faster than the
   3.21 s sim used, which was the H100 SXM 4-GPU number — RTX 3090 4-GPU
   sp=4 actually does this fast because the pipeline parallel split
   reduces per-GPU work).
3. The L2-split optimization alone (no warm pool) is a wash on
   total-time-to-first-request (just shifts cost from boot to first request).

The warm-pool *architecture* benefit (the actual claim) requires multi-pod
K8s, and the 2026-05-06 H100 SXM session is still the only real
measurement that captures it (3.14 s warm hit vs ~17 s cold). Re-validating
*that* would require either (a) a privileged-mode K8s cluster, (b) a real
mentor-lab cluster (PKU host2 when GPU 7 is fixed), or (c) splitting work
across multiple RunPod pods with manual Ray multi-node setup (which has
its own networking constraints between unprivileged containers).
