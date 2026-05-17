#!/usr/bin/env bash
# Build the long-lived hetudit-base image and emit a tar for distribution.
#
# Run on the host's outer docker (not inside dind). The resulting tar must be
# scp'd to every cluster node, then imported into dind containerd's k8s.io
# namespace via scripts/preimport-base.sh.
#
# Usage:
#   bash scripts/build-base-image.sh
#   BASE_TAG=2026-05-03 bash scripts/build-base-image.sh
#
# The PKU dind hosts reach docker.io / pypi via an HTTP forward proxy at
# 162.105.146.116:2999. Credentials are operator-managed (ask the cluster
# admin); export HTTP_PROXY/HTTPS_PROXY in your shell before running, or
# inherit them from ~/.bashrc on the build host.
#
# Trigger conditions: requirements-base.txt changed, Dockerfile.base changed,
# or PyTorch base image upgraded. Otherwise reuse the existing tar.
set -euo pipefail

BASE_TAG="${BASE_TAG:-latest}"
IMAGE="hetudit-base:${BASE_TAG}"
TAR_OUT="${TAR_OUT:-/tmp/hetudit-base.tar}"
PYTORCH_BASE="pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime"
PYTORCH_MIRROR="docker.m.daocloud.io/${PYTORCH_BASE}"

# Proxy is read from the calling shell — no credentials are baked into the
# script (Basic-auth strings in source make GitGuardian very unhappy and
# leak the password to anyone who clones the repo).
#
# Important: use the IP for the proxy host, not the `daim116` short name.
# The host resolves daim116 via /etc/hosts, but that file does NOT
# propagate into the docker build sandbox (even with --network=host), and
# the build container's resolver does not always reach the PKU intranet
# DNS for the short name. We rewrite daim116→IP defensively.
HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
HTTP_PROXY="${HTTP_PROXY//daim116/162.105.146.116}"
HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-$HTTP_PROXY}}"
HTTPS_PROXY="${HTTPS_PROXY//daim116/162.105.146.116}"
NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,daim216,daim217,162.105.146.0/24}}"

if [[ -z "${HTTP_PROXY}" ]]; then
    cat >&2 <<'EOF'
WARNING: HTTP_PROXY/http_proxy is not set. The PKU dind hosts need a
forward proxy to reach docker.io and pypi. If your shell already has it
(via ~/.bashrc) re-run inside an interactive shell; otherwise:
  export HTTP_PROXY=http://<user>:<pass>@162.105.146.116:2999
  export HTTPS_PROXY=$HTTP_PROXY
Credentials are operator-managed — ask the cluster admin.
Continuing without a proxy; expect package downloads to fail.
EOF
fi

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# pytorch/pytorch isn't reachable via the upstream proxy on the dind hosts;
# pull from the DaoCloud mirror and re-tag so the FROM line still resolves.
if ! docker image inspect "${PYTORCH_BASE}" >/dev/null 2>&1; then
    echo "[build-base] pulling ${PYTORCH_MIRROR} (mirror) ..."
    docker pull "${PYTORCH_MIRROR}"
    docker tag "${PYTORCH_MIRROR}" "${PYTORCH_BASE}"
fi

echo "[build-base] building ${IMAGE} ..."
docker build \
    --network=host \
    --build-arg "HTTP_PROXY=${HTTP_PROXY}" \
    --build-arg "HTTPS_PROXY=${HTTPS_PROXY}" \
    --build-arg "NO_PROXY=${NO_PROXY}" \
    -f Dockerfile.base \
    -t "${IMAGE}" \
    .

echo "[build-base] saving to ${TAR_OUT} ..."
docker save "${IMAGE}" -o "${TAR_OUT}"
TAR_SIZE_MB=$(( $(stat -c %s "${TAR_OUT}" 2>/dev/null || stat -f %z "${TAR_OUT}") / 1024 / 1024 ))

cat <<EOF
[build-base] done.
  image: ${IMAGE}
  tar:   ${TAR_OUT} (${TAR_SIZE_MB} MB)

Next steps (run on each node where dind hosts the K8s cluster):
  scp ${TAR_OUT} hetudit@<node>:/tmp/
  ssh hetudit@<node> 'BASE_TAR=/tmp/hetudit-base.tar bash ${REPO_ROOT}/scripts/preimport-base.sh'

This image rarely changes; keep the tar around so app-image rebuilds skip this
step. Re-run build-base-image.sh only when requirements-base.txt or
Dockerfile.base changes.
EOF
