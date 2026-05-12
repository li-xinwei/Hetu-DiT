#!/usr/bin/env bash
# Bootstrap a Lambda Cloud bare-metal node for Hetu-DiT pre-warm pool eval.
#
# Pre-reqs: ubuntu@<lambda-ip> SSH-able. Run from Mac:
#   ssh lambda-hetudit "bash -s" < scripts/bootstrap-lambda.sh
#
# Phases (each idempotent):
#   1. apt deps (k3s install prereqs, jq, tmux)
#   2. install k3s (single-node, bundles containerd, configures nvidia runtime)
#   3. install nvidia container runtime + k3s nvidia.com/gpu device plugin
#   4. install KubeRay operator via Helm
#   5. pre-pull ghcr.io/li-xinwei/hetudit:pku-2026-05-05 via crictl
#   6. download SD3 model to /home/ubuntu/models via hf hub (needs HF_TOKEN)
#   7. clone Hetu-DiT repo for manifest + script access
#
# Output marker: /home/ubuntu/.hetudit-bootstrap-done

set -euo pipefail

MARK=/home/ubuntu/.hetudit-bootstrap-done
[[ -f "$MARK" ]] && { echo "bootstrap already done ($(cat $MARK))"; exit 0; }

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 1. apt deps ---
log "phase 1: apt deps"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    curl jq tmux git python3-pip python3-yaml

# --- 2. install k3s (single-node) ---
log "phase 2: install k3s"
if ! command -v k3s >/dev/null; then
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--write-kubeconfig-mode 644" sh -
fi
mkdir -p ~/.kube && sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config && sudo chown ubuntu:ubuntu ~/.kube/config
export KUBECONFIG=~/.kube/config
kubectl get nodes

# --- 3. nvidia container runtime + device plugin ---
log "phase 3: nvidia container toolkit + k3s gpu plugin"
if ! dpkg -l nvidia-container-toolkit >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
fi
# Configure k3s containerd to use nvidia runtime
if [ ! -f /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl ]; then
    sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd
    sudo cp /var/lib/rancher/k3s/agent/etc/containerd/config.toml /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl 2>/dev/null || true
fi
# Use NVIDIA's runtime-class template — let k3s manage containerd config
sudo nvidia-ctk runtime configure --runtime=containerd --config=/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl --set-as-default
sudo systemctl restart k3s
sleep 5

# Install NVIDIA device plugin DaemonSet
if ! kubectl get ds -n kube-system nvidia-device-plugin-daemonset >/dev/null 2>&1; then
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml
fi
# Wait for node to advertise GPUs
for i in $(seq 1 30); do
    gpus=$(kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null)
    [[ "$gpus" =~ ^[0-9]+$ ]] && [[ "$gpus" -ge 4 ]] && { log "node has $gpus GPUs allocatable"; break; }
    sleep 5
done

# --- 4. KubeRay operator ---
log "phase 4: KubeRay operator"
if ! command -v helm >/dev/null; then
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | sudo bash
fi
if ! helm list -n kuberay-operator 2>/dev/null | grep -q kuberay; then
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 2>/dev/null || true
    helm repo update
    helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1 \
        --namespace kuberay-operator --create-namespace --wait
fi

# --- 5. pre-pull hetudit image ---
log "phase 5: pre-pull hetudit image (~10 GB)"
IMAGE=ghcr.io/li-xinwei/hetudit:pku-2026-05-05
sudo k3s crictl pull "$IMAGE"

# --- 6. download SD3 model ---
log "phase 6: download SD3 model (~17 GB)"
mkdir -p /home/ubuntu/models /home/ubuntu/results /home/ubuntu/profile-cache
if [ ! -f /home/ubuntu/models/stable-diffusion-3-medium-diffusers/model_index.json ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: HF_TOKEN env not set; cannot download gated SD3 model"
        echo "Run: HF_TOKEN=hf_xxx bash scripts/bootstrap-lambda.sh"
        exit 1
    fi
    pip install --quiet --user huggingface_hub
    HF_TOKEN="$HF_TOKEN" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/stable-diffusion-3-medium-diffusers',
    local_dir='/home/ubuntu/models/stable-diffusion-3-medium-diffusers',
    local_dir_use_symlinks=False,
    max_workers=8)
"
fi

# --- 7. clone Hetu-DiT repo ---
log "phase 7: clone Hetu-DiT for manifest + scripts"
if [ ! -d /home/ubuntu/Hetu-DiT ]; then
    git clone https://github.com/li-xinwei/Hetu-DiT.git /home/ubuntu/Hetu-DiT
    cd /home/ubuntu/Hetu-DiT && git checkout xinwei/d2-l2-warmpool
fi

date > "$MARK"
log "bootstrap done — marker $MARK"
