# Fast cloud setup — eliminate the RunPod/Lambda startup tax

Distilled from the painful §8.30–§8.38 discovery so future test runs do
**not** re-pay it. Two levers:

## Lever A — zero-cost: never rediscover the env again

`lambda_setup.sh` / `runpod_setup.sh` encode the exact, proven dependency
recipe in ONE pass. The env-discovery tax this replaces (measured this
session):

| Phase | Baseline (first discovery) | With committed script |
|---|---|---|
| Lambda env (numpy ABI → pin → hub → yunchang → fastapi → full reinstall, each an SSH round-trip + monitor wait) | **~60 min** (context.md §8.36, "~1h ≈ $2") | **one pass, ~6–9 min** |
| RunPod env (blinker → diffusers/transformers downgrade → flash-attn wheel → scipy/matplotlib) | **~8–15 min iterative** (§8.30) | **one pass, ~5–8 min** |

Net: **~50 min eliminated on a first-ever fresh box; ~3–7 min on a
re-run** (no iteration, no failed attempts). Zero ongoing cost.

## Lever B — RunPod: runtime image, not devel (cold-pull ~2.5× faster)

We install a **prebuilt flash-attn wheel** + pure-python deps + the nixl
wheel — `nvcc` is never invoked. So the ~20 GB `-devel` image is pure
waste; use the ~8 GB `-runtime` image:

```
--image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-runtime-ubuntu22.04
```

Measured baseline (§8.34): cold-node pull of the 20 GB devel image
**9–12 min** (and pods that wedged at `uptimeSeconds=0`). Runtime is
~8 GB ⇒ proportional cold pull **est. ~4–5 min**; see MEASURED.md for the
real timed number from the validation run.

## Lever C — persistent volume: skip env+download entirely (recurring)

RunPod network volume (created via `runpodctl network-volume create
--data-center-id <DC> --name hetudit --size 80`) holding `venv/`,
`hf_cache/` (sd3+flux ≈ 47 GB) and the repo. Build once, then every fresh
pod in that DC just:

```
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf_cache
cd /workspace/Hetu-DiT && git pull
bash scripts/launch_multimodel_paper_set.sh   # ~2 min cold-load only
```

Recurring saving: env-setup (~6–9 min) + model download (~3–5 min) →
**0**; only the unavoidable instance boot + ~2 min model cold-load remain.
Ongoing cost: RunPod volume storage ≈ \$0.07/GB·mo (80 GB ≈ \$5.6/mo) —
delete the volume between testing sprints if idle.

Lambda note: Lambda filesystems are **dashboard-create only** (API POST
/file-systems → HTTP 405). To use Lever C on Lambda: create the FS once in
the Lambda web console, then `ROOT=/mnt/<fs> bash lambda_setup.sh` +
`ROOT=/mnt/<fs> bash lambda_download_models.sh` once; future instances
attach the FS and skip both.

See `MEASURED.md` for the actual before/after numbers from the timed
validation run.
