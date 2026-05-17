# Hetu-DiT Image Layering — Operator Guide

This repo ships two Docker images that compose into the runtime container:

```
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime   (upstream)
                  │
                  ▼
        hetudit-base:<TAG>                        (this repo, Dockerfile.base)
        ├── apt deps (ffmpeg, git, libgl1, ...)
        ├── all Python deps (requirements-base.txt)
        └── flash_attn prebuilt wheel
                  │
                  ▼
        hetudit:<TAG>                             (this repo, Dockerfile)
        └── hetu_dit/ source + `pip install -e . --no-deps`
```

`k8s/raycluster.yaml` only references `hetudit:latest`; the base layer is
pulled in transitively by Docker's `FROM`. Containerd resolves the base
locally if it's already imported (`imagePullPolicy: IfNotPresent`).

---

## Why the split

Pre-D6, `Dockerfile` was monolithic: every change to a single line of
`hetu_dit/worker.py` triggered a `pip install -e .` at build time and a
~4 GB tar to ship out to every dind node.

After D6:
- Base layer (~3.5 GB) changes only when `requirements-base.txt`,
  `Dockerfile.base`, or the upstream PyTorch image changes — typically once
  per quarter.
- App layer (target < 200 MB, hard cap 500 MB) changes on every code edit —
  builds in ~30 s, ships in ~10 s on the LAN.

End-to-end "code change → all nodes have new code" drops from ~5 min to
under 90 s.

---

## When to rebuild what

| You changed... | Rebuild base? | Rebuild app? | Operator command |
|---|---|---|---|
| One line in `hetu_dit/` | No | Yes | `bash scripts/build-app-image.sh` |
| Added `import foo` where `foo` is already in `requirements-base.txt` | No | Yes | `bash scripts/build-app-image.sh` |
| `setup.py:install_requires` — added/removed/repinned a dep | **Yes** | Yes | `bash scripts/build-base-image.sh` then `bash scripts/build-app-image.sh` |
| `requirements-base.txt` (e.g. unblocking a Phase 3-style trap) | **Yes** | Yes | same as above |
| `Dockerfile.base`, `Dockerfile` | depends | depends | rebuild whichever you touched |
| PyTorch / CUDA / cuDNN base image upgrade | **Yes** | Yes | also: re-stage a flash_attn wheel matching the new ABI |

When you change a `setup.py` install_requires entry, mirror the change in
`requirements-base.txt` in the same commit. The two are kept manually in
sync — `setup.py` is the source of truth for "anyone can `pip install -e .`
this repo," and `requirements-base.txt` is a deliberate snapshot that lets
us pin extra constraints (`click<8.2`, `transformers<5`) without
disturbing source-only users.

---

## Operator workflow

### Common case: just changed app code

```bash
# On the build host (e.g. daim216)
bash scripts/build-app-image.sh
# → /tmp/hetudit-app.tar (~50–200 MB)

# Ship to the other node and import on both
scp /tmp/hetudit-app.tar hetudit@daim217:/tmp/
for node in daim216 daim217; do
    ssh hetudit@$node "docker exec xinwei-k8s-host\${node##daim21} \
        ctr -n k8s.io images import /tmp/hetudit-app.tar"
done

kubectl rollout restart raycluster/hetudit
```

### Less common: deps changed, need a base rebuild

```bash
bash scripts/build-base-image.sh
# → /tmp/hetudit-base.tar (~3.5 GB)

# Import via the idempotent helper (safe to re-run)
scp /tmp/hetudit-base.tar hetudit@daim217:/tmp/
for node in daim216 daim217; do
    ssh hetudit@$node "BASE_TAR=/tmp/hetudit-base.tar bash scripts/preimport-base.sh"
done

# Then build + ship the app image as in the common case
bash scripts/build-app-image.sh
# ...
```

### Sanity-check what's loaded in dind containerd

```bash
docker exec xinwei-k8s-host2 ctr -n k8s.io images list -q | grep hetudit
# Expect:
#   hetudit-base:latest
#   hetudit:latest
```

---

## Integration with `bench_coldstart.sh`

The S1 (cold-image) scenario in `scripts/bench_coldstart.sh` evicts
`hetudit:latest` via `crictl rmi`. Pre-D6 this could not be repeated
because there is no registry to pull from.

With D6 it's reproducible:

```bash
# Before each S1 round, the bench harness should:
docker exec xinwei-k8s-host2 crictl rmi hetudit:latest

# Recovery (operator step or follow-up patch to bench_coldstart.sh):
BASE_TAR=/tmp/hetudit-base.tar bash scripts/preimport-base.sh   # idempotent
docker exec xinwei-k8s-host2 ctr -n k8s.io images import /tmp/hetudit-app.tar

# Then the bench can apply raycluster.yaml and time the cold start.
```

The base layer never needs to be re-imported across S1 rounds (preimport
is a no-op once it's there), so the cold-image cost measured by S1 is
strictly the app-tar import + pod-startup time — exactly the workload that
matters for production scale-out.

---

## Limits / future work

- **No private registry** in the current dind setup, so image distribution
  is manual `docker save → scp → ctr import`. When a registry is
  introduced, replace `preimport-base.sh` with a `crictl pull` DaemonSet
  that warms the base on every node automatically.
- **`requirements-base.txt` ↔ `setup.py` drift** is detected only by smoke
  tests today. A pre-commit hook diffing the two could catch this earlier.
- **`hetudit-base:latest` is mutable**; for reproducibility consider tagging
  with date + git short-hash (`hetudit-base:2026-05-03-abc1234`) and
  pinning `BASE_TAG` in `Dockerfile`.
