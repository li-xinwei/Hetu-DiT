#!/usr/bin/env bash
# Download the TridentServe model set into HF cache (idempotent; on a
# persistent volume this runs ONCE, future instances reuse it -> 0 dl time).
# Usage: HF_TOKEN=hf_xxx bash lambda_download_models.sh
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN}"
ROOT="${ROOT:-$HOME}"; source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/hf_cache"; mkdir -p "$HF_HOME"
t0=$(date +%s)
# Use `huggingface-cli` (the env pins huggingface_hub<1.0; the unified
# `hf` CLI only exists in hub>=1.0). Pass each --exclude separately;
# never a bare positional glob (treated as a filename -> 404).
huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers \
  --exclude "*.fp16.safetensors" --token "$HF_TOKEN"
huggingface-cli download black-forest-labs/FLUX.1-dev --exclude "*.onnx" \
  --exclude "flux1-dev.safetensors" --exclude "flux1-dev.gguf" \
  --exclude "*.fp8.*" --token "$HF_TOKEN"
echo "MODELS_DOWNLOADED in $(( $(date +%s)-t0 ))s  ($(du -sh "$HF_HOME"|cut -f1))"
