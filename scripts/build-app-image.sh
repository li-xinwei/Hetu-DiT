#!/usr/bin/env bash
# Build the thin hetudit:latest app image on top of an existing hetudit-base.
#
# Run on the host's outer docker. Fast path: if hetudit-base:${BASE_TAG} is
# already loaded locally, the build only adds the hetu_dit source layer and
# the editable install — typically < 60 s.
#
# Usage:
#   bash scripts/build-app-image.sh
#   APP_TAG=2026-05-03 BASE_TAG=2026-05-03 bash scripts/build-app-image.sh
set -euo pipefail

BASE_TAG="${BASE_TAG:-latest}"
APP_TAG="${APP_TAG:-latest}"
IMAGE="hetudit:${APP_TAG}"
BASE_IMAGE="hetudit-base:${BASE_TAG}"
TAR_OUT="${TAR_OUT:-/tmp/hetudit-app.tar}"
MAX_TAR_MB="${MAX_TAR_MB:-500}"

# Proxy is read from the calling shell — no credentials baked in (see
# scripts/build-base-image.sh for the rationale). Even though this image
# is thin, `pip install -e .` triggers build isolation which still wants
# setuptools from pypi. Use the IP, not `daim116`.
HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
HTTP_PROXY="${HTTP_PROXY//daim116/162.105.146.116}"
HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-$HTTP_PROXY}}"
HTTPS_PROXY="${HTTPS_PROXY//daim116/162.105.146.116}"
NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,daim216,daim217,162.105.146.0/24}}"

cd "$(dirname "$0")/.."

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: base image ${BASE_IMAGE} not found locally.

Build it first via scripts/build-base-image.sh, or pull the existing tar:
  docker load -i /tmp/hetudit-base.tar
EOF
    exit 1
fi

echo "[build-app] building ${IMAGE} on top of ${BASE_IMAGE} ..."
docker build \
    --network=host \
    --build-arg "BASE_TAG=${BASE_TAG}" \
    --build-arg "HTTP_PROXY=${HTTP_PROXY}" \
    --build-arg "HTTPS_PROXY=${HTTPS_PROXY}" \
    --build-arg "NO_PROXY=${NO_PROXY}" \
    -f Dockerfile \
    -t "${IMAGE}" \
    .

echo "[build-app] saving to ${TAR_OUT} ..."
docker save "${IMAGE}" -o "${TAR_OUT}"
TAR_SIZE_MB=$(( $(stat -c %s "${TAR_OUT}" 2>/dev/null || stat -f %z "${TAR_OUT}") / 1024 / 1024 ))

# `docker save` writes a tar containing every layer of the image, including
# the base layers — so the tar size is roughly the full image size, not the
# size of what changed. The real D6 win is the per-layer view: only count
# layers added on top of the base. `docker history` lists newest-first, so
# the first (app_total - base_total) lines are the new ones.
APP_HIST_LINES=$(docker history --format '.' "${IMAGE}" | wc -l)
BASE_HIST_LINES=$(docker history --format '.' "${BASE_IMAGE}" | wc -l)
NEW_HIST_LINES=$(( APP_HIST_LINES - BASE_HIST_LINES ))
NEW_LAYER_BYTES=$(docker history --format '{{.Size}}' --no-trunc "${IMAGE}" \
    | head -n "${NEW_HIST_LINES}" \
    | awk '
        function to_bytes(s,    n) {
            n = s + 0
            if (s ~ /[Gg]B/)      return n * 1024 * 1024 * 1024
            else if (s ~ /[Mm]B/) return n * 1024 * 1024
            else if (s ~ /[Kk]B/) return n * 1024
            else                  return n
        }
        { total += to_bytes($1) } END { printf "%d\n", total }')
NEW_LAYER_MB=$(( NEW_LAYER_BYTES / 1024 / 1024 ))

if [[ "${NEW_LAYER_MB}" -gt "${MAX_TAR_MB}" ]]; then
    cat >&2 <<EOF
ERROR: new app-only layers total ${NEW_LAYER_MB} MB (>${MAX_TAR_MB} MB threshold).

The app should only contribute hetu_dit/ source plus an editable install
record (a few MB). If it ballooned, common causes:
  - Dockerfile lost \`--no-deps\` on \`pip install -e .\`
  - setup.py added a heavy dep that pip resolved against a non-base wheel
  - A new top-level dir (model weights? results/?) ended up in the build
    context — check .dockerignore.

Inspect with: docker history ${IMAGE}
EOF
    exit 1
fi

cat <<EOF
[build-app] done.
  image:           ${IMAGE}
  full tar:        ${TAR_OUT} (${TAR_SIZE_MB} MB; includes base layers)
  app-only layers: ~${NEW_LAYER_MB} MB (containerd will dedup base on import)

Next steps:
  1. Make sure ${BASE_IMAGE} is already imported on the target dind nodes.
     If unsure, run scripts/preimport-base.sh on each node first.
  2. scp ${TAR_OUT} hetudit@<node>:/tmp/
  3. ssh hetudit@<node> 'docker exec xinwei-k8s-host\$N \\
        ctr -n k8s.io images import /tmp/hetudit-app.tar'
  4. kubectl rollout restart raycluster/hetudit  (or recreate the CR)

The full tar is large because docker save bundles every layer, but ctr import
on the target dedups against existing layer blobs — so the on-disk and
network-transfer cost is dominated by the app-only layers, not the tar size.
EOF
