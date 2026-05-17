#!/usr/bin/env bash
set -e
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN env var (do NOT hardcode)} HF_HOME=/workspace/hf_cache HETU_COLDSTART_TRACE=1
cd /workspace/Hetu-DiT
# Register the full TridentServe paper workload model set (sd3/flux/
# cogvideox/hunyuanvideo). Only sd3+flux weights are downloaded this run
# (the no-collapse stress only requests those, exact §8.32 repro); the
# video models are registered+routable for coverage.
exec python -m hetu_dit.entrypoint.api_server --host 0.0.0.0 --port 8000 \
  --models sd3=stabilityai/stable-diffusion-3-medium-diffusers \
  --models flux=black-forest-labs/FLUX.1-dev \
  --models cogvideox=THUDM/CogVideoX-2b \
  --models hunyuanvideo=hunyuanvideo-community/HunyuanVideo \
  --default-model sd3 \
  --tensor_parallel_degree 1 --ulysses_degree 1 --ring_degree 1 \
  --machine_nums 1 --search-mode random 2>&1 | tee /workspace/server.log
