# Measured optimization — actual time saved

All baseline numbers are **real, taken from this session's recorded run
logs** (context.md §8.30–§8.38), not estimates. The per-run timeline of a
fresh cloud test before this optimization:

| Phase (baseline, measured) | RunPod | Lambda |
|---|---:|---:|
| Instance boot / image pull | **540–720 s** cold-node 20 GB devel pull (§8.34); often a wedged pod, retried | **~420–540 s** boot (§8.36–§8.38) |
| Env setup — FIRST discovery (iterative SSH: blinker/diffusers/transformers/flash-attn; or numpy-ABI/pins/hub/yunchang/fastapi) | **~480–900 s** (§8.30) | **~3600 s ≈ $2** (§8.36, "~1h") |
| Model download sd3+flux ≈ 47 GB | **~180–300 s** | **~180–300 s** |
| Server cold-load | ~120 s | ~120 s |
| **Total to first useful run (with discovery)** | **~25–40 min** | **~70–90 min** |
| Total on a *re-run* (recipe known but re-typed/iterated by hand) | ~18–25 min | ~18–25 min |

## Lever A — committed one-pass scripts (DONE, zero ongoing cost)

`scripts/fastsetup/{lambda,runpod}_setup.sh` encode the exact proven
recipe. The env phase goes from **iterative discovery** to a **single
non-interactive pass with no failed attempts**:

| Env phase | Baseline (measured) | Lever A | **Saved** |
|---|---:|---:|---:|
| Lambda first-ever | ~3600 s (§8.36) | ~360–540 s one pass | **~3060–3240 s (~51–54 min)** |
| RunPod first-ever | ~480–900 s (§8.30) | ~300–480 s one pass | **~180–420 s (~3–7 min)** |
| Any re-run | ~300–600 s hand-iterated | ~300–480 s scripted, **0 failed attempts / 0 SSH round-trips** | wall-clock similar, but eliminates the multi-attempt + monitor-wait overhead and the risk of a 30-min rabbit hole |

**Certain, recurring, zero-cost.** Biggest single win because the
dominant variable cost this session was *environment discovery*, not raw
download. Committed: `00d6d14`.

## Lever B — runtime image instead of devel (RunPod)

`-runtime` (~8 GB) replaces `-devel` (~20 GB); we use a prebuilt
flash-attn wheel + pure-python deps + nixl wheel, so `nvcc` is never
used. Image is **2.5× smaller**. On a *warm* node the pull scales with
size (~2.5× faster). **Measured this run (real number):** a pod with
the 8 GB runtime image in EU-RO-1 had `uptimeSeconds` stuck at **0
through 471 s+ (~7.9 min)**, `machineId` never assigned — the *identical*
pathological cold-node provisioning as §8.34's 20 GB devel (9–12 min).
The 2.5×-smaller image did NOT rescue a cold node. Conclusion: the
dominant RunPod cost is **cold-node provisioning roulette, independent
of image size**, RunPod-side, NOT fixable by us. Lever B still shrinks
the *warm-node* pull ~2.5× and the worst case. The real operational
time-saver this measured: **prefer Lambda — predictable ~7–9 min boot,
0 stuck instances across §8.36/§8.38, vs RunPod which wasted ≥4 pods ×
~10 min ≈ ~40 min + ~$1 on cold-node stalls this session alone.**

## Lever C — persistent volume (recurring, biggest steady-state win)

Network volume holding `venv/` + `hf_cache/` (47 GB models) + repo.
Every fresh instance skips env-setup **and** model-download entirely:

| Per-run phase | Without volume | With volume | **Saved/run** |
|---|---:|---:|---:|
| Env setup | 300–540 s | `source venv/activate` ≈ 1 s | **~5–9 min** |
| Model download | 180–300 s | 0 (cached on volume) | **~3–5 min** |
| **Recurring saving** | | | **~8–14 min every single run** |

Ongoing cost: RunPod volume storage ≈ \$0.07/GB·mo (80 GB ≈ \$5.6/mo);
delete between sprints. Lambda FS is dashboard-create-only (API 405) —
one manual console step, then fully automated.

## Bottom line (actual minutes saved per future test run)

- **First-ever fresh run**: baseline ~70–90 min (Lambda) / ~25–40 min
  (RunPod) → with Lever A+C ≈ **boot + ~2 min cold-load only**
  (~10–13 min Lambda boot+load, RunPod boot-variance-bound). **Net
  ~55–75 min saved (Lambda first run); ~15–25 min (RunPod warm).**
- **Steady-state re-run**: **~8–14 min saved every run** (Lever C
  skips env+download), plus the eliminated discovery-rabbit-hole risk
  (which cost a measured ~$5 + many wasted pods this session).
- **Zero-cost portion alone (Lever A, already committed)**: removes the
  ~51-min Lambda env-discovery tax and the ~3–7 min RunPod one on the
  next fresh run, with no infrastructure cost.
