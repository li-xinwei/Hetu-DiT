#!/usr/bin/env bash
# Lambda Stack (Ubuntu 22.04 / Python 3.10 / torch 2.7 / CUDA 12.8) one-pass
# env setup for hetu_dit multimodel. Distilled from the §8.36-§8.38
# discovery (which cost ~1h of iterative SSH debugging — now ZERO).
# Usage: HF_TOKEN=hf_xxx bash lambda_setup.sh [BRANCH]
set -euo pipefail
BRANCH="${1:-xinwei/multimodel-base}"
: "${HF_TOKEN:?set HF_TOKEN env var (do NOT hardcode)}"
ROOT="${ROOT:-$HOME}"            # override ROOT=/mnt/fs for a persistent volume
VENV="$ROOT/venv"; REPO="$ROOT/work/Hetu-DiT"; HFC="$ROOT/hf_cache"
t0=$(date +%s)
python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
pip install -q -U pip
mkdir -p "$ROOT/work"; [ -d "$REPO" ] || git clone --depth=1 -b "$BRANCH" \
  https://github.com/li-xinwei/Hetu-DiT "$REPO"
cd "$REPO" && git pull -q || true
# flash-attn prebuilt wheel FIRST: `pip install -e .` pulls yunchang==0.3.5
# which requires flash_attn; without the wheel pre-satisfied pip builds it
# from sdist (no nvcc on Lambda Stack) -> fails -> set -e aborts the whole
# recipe atomically (the §8.39 silent-bootstrap bug).
pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
pip install -q -e .
# The §8.36 fixes, applied in ONE pass (no rediscovery):
#  - numpy/scipy ABI pin (Lambda Stack system numpy is 2.x; scipy/diffusers
#    ext built for a different ABI -> "numpy.dtype size changed")
#  - diffusers0.32/transformers4.49 are the mutually-compatible pair
#  - huggingface_hub<1.0 (transformers 4.49 requires it; the new `hf` CLI
#    pulls 1.x which breaks transformers)
#  - flash-attn prebuilt wheel for torch2.7/cu12/abiTRUE/cp310
#  - explicit full dep list bypasses the broken system-flatbuffers resolver
pip install -q --force-reinstall --no-cache-dir "numpy==1.26.4" "scipy==1.13.1"
pip install -q accelerate "diffusers==0.32.0" "transformers==4.49.0" \
  "huggingface_hub>=0.26,<1.0" sentencepiece beautifulsoup4 "yunchang==0.3.5" \
  flask "opencv-python==4.9.0.80" "ray==2.39.0" "fastapi==0.110.0" \
  "uvicorn==0.28.0" aiohttp pulp matplotlib pytest
pip install -q nixl || echo "nixl optional (adjust_strategy=base ok without)"
python -c 'from hetu_dit.entrypoint.api_server import app; from hetu_dit.engine.serial_dispatcher import SerialModelDispatcher; print("IMPORTS_OK")'
echo "ENV_SETUP_DONE in $(( $(date +%s)-t0 ))s  (venv=$VENV)"
