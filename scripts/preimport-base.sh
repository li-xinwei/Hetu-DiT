#!/usr/bin/env bash
# Idempotently ensure hetudit-base:${BASE_TAG} is imported into dind
# containerd's k8s.io namespace on this host.
#
# Run once per node after dind bootstrap (or after a teardown that wipes
# containerd state). Safe to re-run — fast no-op if the image is present.
#
# Usage:
#   bash scripts/preimport-base.sh                          # check only
#   BASE_TAR=/tmp/hetudit-base.tar bash scripts/preimport-base.sh
#   BASE_TAG=2026-05-03 DIND_CTR=xinwei-k8s-host1 \
#       BASE_TAR=/tmp/hetudit-base.tar bash scripts/preimport-base.sh
#
# Exit codes:
#   0 = image present (whether already there or successfully imported)
#   1 = image missing and no BASE_TAR provided
#   2 = ctr import failed
set -euo pipefail

BASE_TAG="${BASE_TAG:-latest}"
IMAGE="hetudit-base:${BASE_TAG}"

# Default DIND_CTR by probing what actually exists on this host. The two PKU
# nodes use distinct names (host1 vs host2), so a probe is more robust than a
# hardcoded default that's wrong half the time.
if [[ -z "${DIND_CTR:-}" ]]; then
    for cand in xinwei-k8s-host2 xinwei-k8s-host1; do
        if docker inspect "$cand" >/dev/null 2>&1; then
            DIND_CTR="$cand"
            break
        fi
    done
fi
if [[ -z "${DIND_CTR:-}" ]]; then
    echo "ERROR: no dind container found (tried xinwei-k8s-host{1,2}); set DIND_CTR explicitly." >&2
    exit 1
fi

# `ctr -n k8s.io images list -q` is the source of truth for what kubelet can
# see; `crictl images` works too but adds a runtime hop. Use ctr to keep this
# script independent of crictl config.
image_present() {
    docker exec "$DIND_CTR" ctr -n k8s.io images list -q 2>/dev/null \
        | grep -Fxq "docker.io/library/${IMAGE}" \
        || docker exec "$DIND_CTR" ctr -n k8s.io images list -q 2>/dev/null \
        | grep -Fxq "${IMAGE}"
}

if image_present; then
    echo "[preimport-base] ${IMAGE} already present in ${DIND_CTR} (k8s.io namespace); skipping."
    exit 0
fi

if [[ -z "${BASE_TAR:-}" ]]; then
    cat >&2 <<EOF
ERROR: ${IMAGE} not present in ${DIND_CTR} and BASE_TAR not provided.

Either:
  - rebuild on the host: bash scripts/build-base-image.sh
  - copy from another node: scp hetudit@<other>:/tmp/hetudit-base.tar /tmp/
  - re-run with BASE_TAR pointing at the tar:
      BASE_TAR=/tmp/hetudit-base.tar bash scripts/preimport-base.sh
EOF
    exit 1
fi

if [[ ! -f "$BASE_TAR" ]]; then
    echo "ERROR: BASE_TAR=$BASE_TAR does not exist." >&2
    exit 1
fi

# ctr import inside dind needs the tar to be readable from inside the
# container. /tmp on the host is a tmpfs in the dind container, so we copy
# via stdin to dodge the bind-mount mismatch.
echo "[preimport-base] importing ${BASE_TAR} into ${DIND_CTR}:k8s.io ..."
if ! docker exec -i "$DIND_CTR" ctr -n k8s.io images import - < "$BASE_TAR"; then
    echo "ERROR: ctr import failed." >&2
    exit 2
fi

if image_present; then
    echo "[preimport-base] ${IMAGE} now present; done."
    exit 0
fi

echo "ERROR: ctr import succeeded but ${IMAGE} still not visible in k8s.io namespace." >&2
echo "       Inspect with: docker exec ${DIND_CTR} ctr -n k8s.io images list" >&2
exit 2
