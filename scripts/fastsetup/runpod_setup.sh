#!/usr/bin/env bash
# RunPod one-pass env setup. Use the RUNTIME image (NOT -devel): we install
# a prebuilt flash-attn wheel + pure-python deps + nixl wheel — nvcc is
# never used, so runtime (~8GB) replaces devel (~20GB) and cold-node image
# pull is ~2.5x faster (§8.34 measured 9-12min devel -> est ~4-5min runtime).
# Recommended --image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-runtime-ubuntu22.04
# Usage: HF_TOKEN=hf_xxx bash runpod_setup.sh [BRANCH]
set -euo pipefail
BRANCH="${1:-xinwei/multimodel-base}"
: "${HF_TOKEN:?set HF_TOKEN}"
t0=$(date +%s)
apt-get update -qq && apt-get install -qq -y git tmux build-essential 2>&1 | tail -1
rm -f /usr/lib/python3/dist-packages/blinker*.egg-info 2>/dev/null || true
pip install -q --upgrade blinker
ROOT="${ROOT:-/workspace}"; mkdir -p "$ROOT"; cd "$ROOT"
[ -d Hetu-DiT ] || git clone --depth=1 -b "$BRANCH" https://github.com/li-xinwei/Hetu-DiT
cd Hetu-DiT && git pull -q || true
pip install -q -e .
# torch 2.4 on this image is too OLD for new diffusers/transformers
# (PEP604 infer_schema bug, §8.30) -> pin the compatible pair:
pip install -q scipy matplotlib 'diffusers==0.32.0' 'transformers<4.50'
pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
python3 -c 'from hetu_dit.entrypoint.api_server import app; from hetu_dit.engine.serial_dispatcher import SerialModelDispatcher; print("IMPORTS_OK")'
echo "ENV_SETUP_DONE in $(( $(date +%s)-t0 ))s"
